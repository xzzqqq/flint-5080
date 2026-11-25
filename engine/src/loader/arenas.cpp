#include "loader/arenas.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <unordered_map>

#include "loader/safetensors.h"

namespace flint {
namespace {
using D = TensorDType;
using Shape = std::vector<int64_t>;
constexpr size_t kAlignment = 256;
size_t Align(size_t offset) { return (offset + kAlignment - 1) & ~(kAlignment - 1); }

void Cuda(cudaError_t error, const char* op) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(error));
}
void* AllocateDevice(size_t bytes) {
  void* ptr = nullptr;
  Cuda(cudaMalloc(&ptr, bytes), "allocate device arena");
  return ptr;
}

// Enumerate the model once. An empty setter denotes a host-only codebook marker.
struct Binding {
  D dtype;
  Shape shape;
  std::function<void(const void*)> set;
};
using Schema = std::unordered_map<std::string, Binding>;
class Fields {
 public:
  Fields(Schema& schema, std::string prefix) : schema_(schema), prefix_(std::move(prefix)) {}
  template <class T>
  void Add(const std::string& name, D dtype, Shape shape, const T*& dst) {
    Insert(name, {dtype, std::move(shape),
                  [&dst](const void* address) { dst = static_cast<const T*>(address); }});
  }
  void Matrix(const std::string& name, uint32_t k, uint32_t n, uint8_t k2, Exl3MatrixView& dst) {
    dst.k = k;
    dst.n = n;
    dst.k2 = k2;
    Add(name + ".trellis", D::kI16, {k / 16, n / 16, 8 * k2}, dst.trellis);
    Add(name + ".suh", D::kF16, {k}, dst.suh);
    Add(name + ".svh", D::kF16, {n}, dst.svh);
    Insert(name + ".mul1", {D::kI32, {}, {}});
  }

 private:
  void Insert(const std::string& name, Binding binding) {
    if (!schema_.emplace(prefix_ + name, std::move(binding)).second)
      throw std::logic_error("duplicate model binding " + prefix_ + name);
  }
  Schema& schema_;
  std::string prefix_;
};

Schema WeightSchema(ModelWeightsView& model) {
  Schema schema;
  Fields global(schema, "model.llm.");
  global.Add("embed.weight", D::kBF16, {201024, 2304}, model.embed_weight);
  global.Add("embed_norm.weight", D::kBF16, {2304}, model.embed_norm_weight);
  global.Add("norm.weight", D::kBF16, {2304}, model.final_norm_weight);
  global.Add("unembed.weight", D::kBF16, {201024, 2304}, model.unembed_weight);
  global.Add("attn_res.output_query", D::kF32, {2304}, model.output_query);
  global.Add("middle_cycle_cross_pass_queries", D::kF32, {24, 2304}, model.cross_pass_queries);
  global.Add("middle_cycle_cross_pass_kv_queries", D::kF32, {24, 2304},
             model.cross_pass_kv_queries);
  for (unsigned i = 0; i < FLINT_NUM_PHYSICAL_LAYERS; ++i) {
    auto& layer = model.layers[i];
    layer.is_full_attention = model.config.is_full_layer[i];
    layer.is_loop_layer = i >= model.config.loop_core_start && i < model.config.loop_core_end;
    const auto id = std::to_string(i);
    Fields f(schema, "model.llm.layers." + id + ".");
    if (layer.is_full_attention) {
      f.Add("q_input_norm.weight", D::kBF16, {2304}, layer.q_input_norm);
      f.Add("k_input_norm.weight", D::kBF16, {2304}, layer.k_input_norm);
      f.Add("v_input_norm.weight", D::kBF16, {2304}, layer.v_input_norm);
      f.Add("attn.k_sconv.static_w", D::kBF16, {4, 1024}, layer.k_static_w);
      f.Add("attn.v_sconv.static_w", D::kBF16, {4, 1024}, layer.v_static_w);
      f.Add("attn.k_sconv.weight_proj", D::kBF16, {32, 2304}, layer.k_weight_proj);
      f.Add("attn.v_sconv.weight_proj", D::kBF16, {32, 2304}, layer.v_weight_proj);
    } else {
      f.Add("attn_norm.weight", D::kBF16, {2304}, layer.attn_norm);
      f.Add("attn.k_sconv.weight", D::kBF16, {1024, 1, 4}, layer.k_sconv);
      f.Add("attn.v_sconv.weight", D::kBF16, {1024, 1, 4}, layer.v_sconv);
    }
    f.Add("mlp_norm.weight", D::kBF16, {2304}, layer.mlp_norm);
    f.Add("attn_sconv.weight", D::kBF16, {2304, 1, 4}, layer.attn_sconv);
    f.Add("mlp_sconv.weight", D::kBF16, {2304, 1, 4}, layer.mlp_sconv);
    f.Add("attn.wr_du.weight", D::kBF16, {384, 2304}, layer.wr_du);
    f.Add("attn.rel_logits_proj.proj", D::kBF16, {16, 1024}, layer.rel_logits_proj);
    f.Add("attn.q_l2_clip.scale", D::kBF16, {24, 128}, layer.q_l2_clip_scale);
    f.Add("attn.rel_l2_clip.scale", D::kBF16, {24, 16}, layer.rel_l2_clip_scale);
    f.Add("attn.attn_sink_proj.weight", D::kBF16, {24, 2304}, layer.attn_sink_proj);
    if (i > 0)
      global.Add("attn_res.attn_queries." + std::to_string(i - 1), D::kF32,
                 layer.is_full_attention ? Shape{3, 2304} : Shape{2304}, layer.attn_query);
    global.Add("attn_res.mlp_queries." + id, D::kF32, {2304}, layer.mlp_query);
    f.Add("mlp.gate.weight", D::kBF16, {128, 2304}, layer.router_weight);
    f.Add("mlp.gate.bias", D::kF32, {128}, layer.router_bias);
    Fields attn(schema, id + ".self_attn.");
    attn.Matrix("q_proj.linear", 2304, 3072, 12, layer.q_proj);
    attn.Matrix("k_proj.linear", 2304, 1024, 12, layer.k_proj);
    attn.Matrix("v_proj.linear", 2304, 1024, 12, layer.v_proj);
    attn.Matrix("o_proj.gate_proj", 2304, 3072, 12, layer.o_gate_proj);
    attn.Matrix("o_proj.linear", 3072, 2304, 12, layer.o_proj);
    Fields experts(schema, id + ".experts.");
    for (int j = 0; j < FLINT_NUM_EXPERTS; ++j) {
      experts.Matrix("gate_up_proj." + std::to_string(j), 2304, 2304, 6 + j % 2,
                     layer.gate_up_proj[j]);
      experts.Matrix("down_proj." + std::to_string(j), 1152, 2304, 6 + j % 2, layer.down_proj[j]);
    }
    if (layer.is_loop_layer) {
      const char* names[] = {"attn", "k", "v", "mlp"};
      for (size_t j = 0; j < std::size(names); ++j) {
        auto prefix = std::string("controllers.") + names[j] + ".";
        auto& ctl = layer.controllers[j];
        f.Add(prefix + "down_weight", D::kBF16, {64, 4608}, ctl.down_weight);
        f.Add(prefix + "forget_weight", D::kBF16, {2304, 64}, ctl.forget_weight);
        f.Add(prefix + "gain_weight", D::kBF16, {2304, 64}, ctl.gain_weight);
      }
    }
  }
  return schema;
}

struct Copy {
  std::span<const std::byte> source;
  size_t offset;
  std::function<void(const void*)> set;
};
std::vector<Copy> PlanWeights(const SafetensorsFile& file, ModelWeightsView& model) {
  static_assert(std::endian::native == std::endian::little);
  auto schema = WeightSchema(model);
  std::vector<Copy> plan;
  for (const auto& tensor : file.tensors()) {
    auto node = schema.extract(tensor.name);
    if (node.empty()) throw std::runtime_error("unexpected tensor " + tensor.name);
    auto& binding = node.mapped();
    if (binding.dtype != tensor.dtype || binding.shape != tensor.shape)
      throw std::runtime_error("wrong dtype or shape for " + tensor.name);
    auto bytes = file.Bytes(tensor);
    if (binding.set) {
      size_t offset = Align(model.arena_bytes);
      plan.push_back({bytes, offset, std::move(binding.set)});
      model.arena_bytes = offset + bytes.size();
    } else {
      uint32_t magic;
      std::memcpy(&magic, bytes.data(), sizeof(magic));
      if (magic != FLINT_EXL3_MUL1_MAGIC)
        throw std::runtime_error("bad mul1 magic for " + tensor.name);
      ++model.verified_mul1_count;
    }
  }
  if (!schema.empty()) throw std::runtime_error("missing tensor " + schema.begin()->first);
  if (model.verified_mul1_count != 6264 ||
      plan.size() + model.verified_mul1_count != file.tensors().size() ||
      file.tensors().size() != 25582)
    throw std::logic_error("incomplete checkpoint accounting");
  model.mapped_tensor_count = file.tensors().size();
  return plan;
}

struct Event {
  cudaEvent_t event{};
  Event() { Cuda(cudaEventCreate(&event), "create transfer event"); }
  ~Event() { cudaEventDestroy(event); }
};

float Upload(const std::vector<Copy>& plan, std::byte* device, cudaStream_t stream) {
  constexpr size_t capacity = 64ull << 20;
  void* host;
  Cuda(cudaMallocHost(&host, capacity), "allocate pinned staging buffer");
  std::unique_ptr<void, decltype(&cudaFreeHost)> staging(host, cudaFreeHost);
  auto* buffer = static_cast<std::byte*>(host);
  size_t sent = 0, used = 0;
  float elapsed = 0;
  Event start, finish;
  auto flush = [&] {
    Cuda(cudaEventRecord(start.event, stream), "record transfer start");
    Cuda(cudaMemcpyAsync(device + sent, buffer, used, cudaMemcpyHostToDevice, stream),
         "upload weights");
    Cuda(cudaEventRecord(finish.event, stream), "record transfer finish");
    Cuda(cudaEventSynchronize(finish.event), "wait for staging buffer");
    float ms;
    Cuda(cudaEventElapsedTime(&ms, start.event, finish.event), "measure transfer");
    elapsed += ms;
    sent += used;
    used = 0;
  };
  for (const auto& copy : plan) {
    copy.set(device + copy.offset);
    // Each staging chunk starts on a 256-byte boundary, so the padding always fits.
    size_t pad = copy.offset - (sent + used);
    std::memset(buffer + used, 0, pad);
    used += pad;
    auto source = copy.source;
    while (!source.empty()) {
      size_t n = std::min(source.size(), capacity - used);
      std::memcpy(buffer + used, source.data(), n);
      used += n;
      source = source.subspan(n);
      if (used == capacity) flush();
    }
  }
  if (used) flush();
  return elapsed;
}

struct KvLayout {
  size_t swa, per_token, kv_conv, sublayer_conv;
  size_t Fixed() const { return swa + kv_conv + sublayer_conv; }
};
KvLayout Layout(const ModelConfig& c) {
  if (c.num_key_value_heads != 8 || c.head_dim != 128 || c.hidden_size != 2304 ||
      c.sconv_kernel_size != 4 || c.sliding_window_size != 1024)
    throw std::invalid_argument("unsupported KV geometry");
  size_t channels = c.num_key_value_heads * c.head_dim;
  size_t kv_pair = 2 * channels * sizeof(uint16_t);
  size_t state = FLINT_NUM_LOGICAL_VISITS * 2 * (c.sconv_kernel_size - 1) * sizeof(float);
  return {FLINT_NUM_SWA_VISITS * c.sliding_window_size * kv_pair, FLINT_NUM_FULL_VISITS * kv_pair,
          state * channels, state * c.hidden_size};
}
}  // namespace

std::array<VisitPlanEntry, FLINT_NUM_LOGICAL_VISITS> BuildVisitPlan(const ModelConfig& c) {
  if (c.loop_core_start != 6 || c.loop_core_end != 18 || c.loop_num_passes != 3 ||
      c.num_hidden_layers != 24 || c.n_layers_per_attn_res_block != 6)
    throw std::invalid_argument("unsupported visit schedule");
  std::array<VisitPlanEntry, FLINT_NUM_LOGICAL_VISITS> plan{};
  uint8_t idx = 0, swa = 0, full = 0;
  auto append = [&](uint32_t begin, uint32_t end, uint8_t pass) {
    for (uint8_t i = begin; i < end; ++i) {
      auto global = c.is_full_layer[i];
      if (global != (i % 6 == 5)) throw std::invalid_argument("invalid attention schedule");
      plan[idx] = {idx,
                   i,
                   pass,
                   global,
                   i % c.n_layers_per_attn_res_block == 0,
                   pass > 0,
                   global ? full++ : swa++};
      ++idx;
    }
  };
  append(0, c.loop_core_start, 0);
  for (uint8_t pass = 0; pass < c.loop_num_passes; ++pass)
    append(c.loop_core_start, c.loop_core_end, pass);
  append(c.loop_core_end, c.num_hidden_layers, 0);
  return plan;
}

WeightsArena WeightsArena::Load(const std::filesystem::path& dir, cudaStream_t stream) {
  WeightsArena result;
  auto& view = result.view_;
  view.config = LoadModelConfig(dir / "config.json");
  auto file = SafetensorsFile::Open(dir / "model.safetensors");
  auto plan = PlanWeights(file, view);
  result.allocation_.reset(AllocateDevice(view.arena_bytes));
  view.arena_base = result.allocation_.get();
  result.transfer_ms_ = Upload(plan, static_cast<std::byte*>(view.arena_base), stream);
  return result;
}

uint32_t KvStateArena::MaxFullSeqLenForBudget(const ModelConfig& c, size_t budget) {
  auto l = Layout(c);
  if (budget < l.Fixed()) return 0;
  return std::min<size_t>(UINT32_MAX, (budget - l.Fixed()) / l.per_token);
}
KvStateArena KvStateArena::Allocate(const ModelConfig& c, uint32_t tokens) {
  if (!tokens) throw std::invalid_argument("empty full KV context");
  auto l = Layout(c);
  KvStateArena result;
  auto& v = result.view_;
  v.arena_bytes = l.Fixed() + tokens * l.per_token;
  result.allocation_.reset(AllocateDevice(v.arena_bytes));
  auto* base = static_cast<std::byte*>(result.allocation_.get());
  v.arena_base = v.swa_kv = base;
  v.full_kv = base + l.swa;
  auto* conv = base + l.swa + tokens * l.per_token;
  v.kv_conv_state = reinterpret_cast<float*>(conv);
  v.sublayer_conv_state = reinterpret_cast<float*>(conv + l.kv_conv);
  v.max_full_seq_len = tokens;
  v.swa_window = c.sliding_window_size;
  v.num_swa_visits = FLINT_NUM_SWA_VISITS;
  v.num_full_visits = FLINT_NUM_FULL_VISITS;
  v.num_total_visits = FLINT_NUM_LOGICAL_VISITS;
  v.num_kv_heads = c.num_key_value_heads;
  v.head_dim = c.head_dim;
  v.conv_state_len = c.sconv_kernel_size - 1;
  return result;
}
void KvStateArena::Reset(cudaStream_t stream) {
  Cuda(cudaMemsetAsync(view_.arena_base, 0, view_.arena_bytes, stream), "reset KV state");
}
ScratchArena ScratchArena::Allocate(size_t bytes) {
  if (!bytes) throw std::invalid_argument("empty scratch arena");
  ScratchArena result;
  result.allocation_.reset(AllocateDevice(bytes));
  result.view_ = {result.allocation_.get(), bytes};
  return result;
}
}  // namespace flint

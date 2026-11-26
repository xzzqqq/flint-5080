#include "loader/arenas.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "common/cuda_check.h"
#include "loader/kv_layout.h"
#include "loader/safetensors.h"

namespace flint {
DeviceBuffer AllocateDeviceBuffer(size_t bytes) {
  void* ptr = nullptr;
  CheckCuda(cudaMalloc(&ptr, bytes), "allocate device arena");
  return DeviceBuffer(ptr);
}

namespace {
using D = TensorDType;
using Shape = std::vector<int64_t>;
constexpr size_t kAlignment = 256;
size_t Align(size_t offset) { return (offset + kAlignment - 1) & ~(kAlignment - 1); }

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
  global.Add("embed.weight", D::kBF16, {model::kVocabSize, model::kHiddenSize}, model.embed_weight);
  global.Add("embed_norm.weight", D::kBF16, {model::kHiddenSize}, model.embed_norm_weight);
  global.Add("norm.weight", D::kBF16, {model::kHiddenSize}, model.final_norm_weight);
  global.Add("unembed.weight", D::kBF16, {model::kVocabSize, model::kHiddenSize},
             model.unembed_weight);
  global.Add("attn_res.output_query", D::kF32, {model::kHiddenSize}, model.output_query);
  global.Add("middle_cycle_cross_pass_queries", D::kF32,
             {2 * (model::kLoopCoreEnd - model::kLoopCoreStart), model::kHiddenSize},
             model.cross_pass_queries);
  global.Add("middle_cycle_cross_pass_kv_queries", D::kF32,
             {2 * (model::kLoopCoreEnd - model::kLoopCoreStart), model::kHiddenSize},
             model.cross_pass_kv_queries);
  for (unsigned i = 0; i < model::kNumPhysicalLayers; ++i) {
    auto& layer = model.layers[i];
    layer.is_full_attention = model.config.is_full_layer[i];
    layer.is_loop_layer = i >= model.config.loop_core_start && i < model.config.loop_core_end;
    const auto id = std::to_string(i);
    Fields f(schema, "model.llm.layers." + id + ".");
    if (layer.is_full_attention) {
      f.Add("q_input_norm.weight", D::kBF16, {model::kHiddenSize}, layer.q_input_norm);
      f.Add("k_input_norm.weight", D::kBF16, {model::kHiddenSize}, layer.k_input_norm);
      f.Add("v_input_norm.weight", D::kBF16, {model::kHiddenSize}, layer.v_input_norm);
      f.Add("attn.k_sconv.static_w", D::kBF16, {model::kSconvKernelSize, model::kKvDim},
            layer.k_static_w);
      f.Add("attn.v_sconv.static_w", D::kBF16, {model::kSconvKernelSize, model::kKvDim},
            layer.v_static_w);
      f.Add("attn.k_sconv.weight_proj", D::kBF16, {model::kDynConvProjDim, model::kHiddenSize},
            layer.k_weight_proj);
      f.Add("attn.v_sconv.weight_proj", D::kBF16, {model::kDynConvProjDim, model::kHiddenSize},
            layer.v_weight_proj);
    } else {
      f.Add("attn_norm.weight", D::kBF16, {model::kHiddenSize}, layer.attn_norm);
      f.Add("attn.k_sconv.weight", D::kBF16, {model::kKvDim, 1, model::kSconvKernelSize},
            layer.k_sconv);
      f.Add("attn.v_sconv.weight", D::kBF16, {model::kKvDim, 1, model::kSconvKernelSize},
            layer.v_sconv);
    }
    f.Add("mlp_norm.weight", D::kBF16, {model::kHiddenSize}, layer.mlp_norm);
    f.Add("attn_sconv.weight", D::kBF16, {model::kHiddenSize, 1, model::kSconvKernelSize},
          layer.attn_sconv);
    f.Add("mlp_sconv.weight", D::kBF16, {model::kHiddenSize, 1, model::kSconvKernelSize},
          layer.mlp_sconv);
    f.Add("attn.wr_du.weight", D::kBF16, {model::kRelProjDim, model::kHiddenSize}, layer.wr_du);
    f.Add("attn.rel_logits_proj.proj", D::kBF16, {model::kRelDim, model::kRelExtent},
          layer.rel_logits_proj);
    f.Add("attn.q_l2_clip.scale", D::kBF16, {model::kNumQHeads, model::kHeadDim},
          layer.q_l2_clip_scale);
    f.Add("attn.rel_l2_clip.scale", D::kBF16, {model::kNumQHeads, model::kRelDim},
          layer.rel_l2_clip_scale);
    f.Add("attn.attn_sink_proj.weight", D::kBF16, {model::kNumQHeads, model::kHiddenSize},
          layer.attn_sink_proj);
    if (i > 0)
      global.Add("attn_res.attn_queries." + std::to_string(i - 1), D::kF32,
                 layer.is_full_attention ? Shape{3, model::kHiddenSize} : Shape{model::kHiddenSize},
                 layer.attn_query);
    global.Add("attn_res.mlp_queries." + id, D::kF32, {model::kHiddenSize}, layer.mlp_query);
    f.Add("mlp.gate.weight", D::kBF16, {model::kNumExperts, model::kHiddenSize},
          layer.router_weight);
    f.Add("mlp.gate.bias", D::kF32, {model::kNumExperts}, layer.router_bias);
    Fields attn(schema, id + ".self_attn.");
    attn.Matrix("q_proj.linear", model::kHiddenSize, model::kQDim, 12, layer.q_proj);
    attn.Matrix("k_proj.linear", model::kHiddenSize, model::kKvDim, 12, layer.k_proj);
    attn.Matrix("v_proj.linear", model::kHiddenSize, model::kKvDim, 12, layer.v_proj);
    attn.Matrix("o_proj.gate_proj", model::kHiddenSize, model::kQDim, 12, layer.o_gate_proj);
    attn.Matrix("o_proj.linear", model::kQDim, model::kHiddenSize, 12, layer.o_proj);
    Fields experts(schema, id + ".experts.");
    for (int j = 0; j < model::kNumExperts; ++j) {
      experts.Matrix("gate_up_proj." + std::to_string(j), model::kHiddenSize, model::kGateUpSize,
                     6 + j % 2, layer.gate_up_proj[j]);
      experts.Matrix("down_proj." + std::to_string(j), model::kIntermediateSize, model::kHiddenSize,
                     6 + j % 2, layer.down_proj[j]);
    }
    if (layer.is_loop_layer) {
      const char* names[] = {"attn", "k", "v", "mlp"};
      for (size_t j = 0; j < std::size(names); ++j) {
        auto prefix = std::string("controllers.") + names[j] + ".";
        auto& ctl = layer.controllers[j];
        f.Add(prefix + "down_weight", D::kBF16, {model::kLoopGateRank, model::kMessageGateInDim},
              ctl.down_weight);
        f.Add(prefix + "forget_weight", D::kBF16, {model::kHiddenSize, model::kLoopGateRank},
              ctl.forget_weight);
        f.Add(prefix + "gain_weight", D::kBF16, {model::kHiddenSize, model::kLoopGateRank},
              ctl.gain_weight);
      }
    }
  }
  return schema;
}

Schema MtpSchema(MtpWeightsView& w) {
  Schema schema;
  Fields f(schema, "");
  f.Add("fuse_weight", D::kBF16, {4, 4}, w.fuse_weight);
  f.Add("mask_embed", D::kBF16, {2304}, w.mask_embed);
  f.Add("norm.weight", D::kBF16, {2304}, w.norm);
  f.Add("markov_head.dflare_norm.weight", D::kBF16, {2304}, w.dflare_norm);
  f.Add("markov_head.hidden_out.weight", D::kBF16, {2304, 256}, w.hidden_out);
  f.Add("markov_head.hidden_out.bias", D::kBF16, {2304}, w.hidden_bias);
  f.Add("markov_head.markov_w1.weight", D::kBF16, {201024, 256}, w.markov_w1);
  f.Add("markov_head.rnn_proj.weight", D::kBF16, {768, 5120}, w.rnn_proj);
  f.Add("markov_head.rnn_proj.bias", D::kBF16, {768}, w.rnn_bias);
  for (int i = 0; i < 3; ++i) {
    Fields layer(schema, "layers." + std::to_string(i) + ".");
    auto& l = w.layers[i];
    layer.Add("dflare_norm.weight", D::kBF16, {2304}, l.dflare_norm);
    layer.Add("input_layernorm.weight", D::kBF16, {2304}, l.input_layernorm);
    layer.Add("post_attention_layernorm.weight", D::kBF16, {2304}, l.post_attention_layernorm);
    layer.Add("self_attn.q_proj.weight", D::kBF16, {3072, 2304}, l.q_proj);
    layer.Add("self_attn.k_proj.weight", D::kBF16, {1024, 2304}, l.k_proj);
    layer.Add("self_attn.v_proj.weight", D::kBF16, {1024, 2304}, l.v_proj);
    layer.Add("self_attn.o_proj.weight", D::kBF16, {2304, 3072}, l.o_proj);
    layer.Add("self_attn.k_proj_ctx.weight", D::kBF16, {1024, 2304}, l.k_proj_ctx);
    layer.Add("self_attn.v_proj_ctx.weight", D::kBF16, {1024, 2304}, l.v_proj_ctx);
    layer.Add("self_attn.k_norm_ctx.weight", D::kBF16, {128}, l.k_norm_ctx);
    layer.Add("self_attn.q_norm.weight", D::kBF16, {128}, l.q_norm);
    layer.Add("self_attn.k_norm.weight", D::kBF16, {128}, l.k_norm);
    layer.Add("mlp.gate_proj.weight", D::kBF16, {9216, 2304}, l.gate_proj);
    layer.Add("mlp.up_proj.weight", D::kBF16, {9216, 2304}, l.up_proj);
    layer.Add("mlp.down_proj.weight", D::kBF16, {2304, 9216}, l.down_proj);
  }
  return schema;
}

struct Copy {
  std::span<const std::byte> source;
  size_t offset;
  std::function<void(const void*)> set;
};
void PlanWeights(const SafetensorsFile& file, ModelWeightsView& model, Schema schema,
                 std::vector<Copy>& plan) {
  static_assert(std::endian::native == std::endian::little);

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
}

constexpr size_t kExpertCount = FLINT_NUM_PHYSICAL_LAYERS * 2 * FLINT_NUM_EXPERTS;

std::vector<Exl3ExpertDesc> BuildExpertTable(const ModelWeightsView& model) {
  std::vector<Exl3ExpertDesc> table;
  table.reserve(kExpertCount);
  for (const auto& layer : model.layers)
    for (auto projection : {std::span(layer.gate_up_proj), std::span(layer.down_proj)})
      for (const auto& matrix : projection) {
        if (matrix.k2 != 6 && matrix.k2 != 7)
          throw std::runtime_error("unsupported EXL3 expert bitrate");
        table.push_back({matrix.trellis, matrix.suh, matrix.svh, matrix.k2});
      }
  return table;
}

struct Event {
  cudaEvent_t event{};
  Event() { CheckCuda(cudaEventCreate(&event), "create transfer event"); }
  ~Event() { cudaEventDestroy(event); }
};

float Upload(const std::vector<Copy>& plan, std::byte* device, cudaStream_t stream) {
  constexpr size_t capacity = 64ull << 20;
  void* host;
  CheckCuda(cudaMallocHost(&host, capacity), "allocate pinned staging buffer");
  std::unique_ptr<void, decltype(&cudaFreeHost)> staging(host, cudaFreeHost);
  auto* buffer = static_cast<std::byte*>(host);
  size_t sent = 0, used = 0;
  float elapsed = 0;
  Event start, finish;
  auto flush = [&] {
    CheckCuda(cudaEventRecord(start.event, stream), "record transfer start");
    CheckCuda(cudaMemcpyAsync(device + sent, buffer, used, cudaMemcpyHostToDevice, stream),
              "upload weights");
    CheckCuda(cudaEventRecord(finish.event, stream), "record transfer finish");
    CheckCuda(cudaEventSynchronize(finish.event), "wait for staging buffer");
    float ms;
    CheckCuda(cudaEventElapsedTime(&ms, start.event, finish.event), "measure transfer");
    elapsed += ms;
    sent += used;
    used = 0;
  };
  for (const auto& copy : plan) {
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

}  // namespace

WeightsArena WeightsArena::Load(const std::filesystem::path& dir, cudaStream_t stream,
                                bool load_mtp) {
  WeightsArena result;
  auto& view = result.view_;
  view.config = LoadModelConfig(dir / "config.json");
  auto file = SafetensorsFile::Open(dir / "model.safetensors");
  const auto& metadata = file.metadata();
  if (auto tag = metadata.find("layout_tag"); tag != metadata.end() && tag->second != "raw_exl3")
    throw std::runtime_error("unsupported weight layout_tag: " + tag->second);
  std::vector<Copy> plan;
  PlanWeights(file, view, WeightSchema(view), plan);
  view.mapped_tensor_count = file.tensors().size();
  std::optional<SafetensorsFile> draft;
  std::array<float, 16> alpha{};
  if (load_mtp && view.config.has_mtp && std::filesystem::exists(dir / "model-mtp.safetensors")) {
    draft.emplace(SafetensorsFile::Open(dir / "model-mtp.safetensors"));
    PlanWeights(*draft, view, MtpSchema(view.mtp), plan);
    auto src = draft->Bytes(*draft->Find("fuse_weight"));
    for (size_t i = 0; i < 16; ++i) {
      uint16_t bits;
      std::memcpy(&bits, src.data() + 2 * i, 2);
      alpha[i] = std::bit_cast<float>(uint32_t(bits) << 16);
    }
    for (size_t i = 0; i < 16; i += 4) {
      auto* row = alpha.data() + i;
      float mx = *std::max_element(row, row + 4);
      for (float& x : std::span(row, 4)) x = std::exp(x - mx);
      float sum = (row[0] + row[2]) + (row[1] + row[3]);
      for (float& x : std::span(row, 4)) x /= sum;
    }
    size_t offset = Align(view.arena_bytes);
    plan.push_back({std::as_bytes(std::span(alpha)), offset, [&](const void* ptr) {
                      view.mtp.fuse_alpha = static_cast<const float*>(ptr);
                    }});
    view.arena_bytes = offset + sizeof(alpha);
    view.has_mtp = 1;
    view.mapped_tensor_count += draft->tensors().size();
  }
  const size_t table_offset = Align(view.arena_bytes);
  view.arena_bytes = table_offset + kExpertCount * sizeof(Exl3ExpertDesc);
  result.allocation_ = AllocateDeviceBuffer(view.arena_bytes);
  view.arena_base = result.allocation_.get();
  auto* base = static_cast<std::byte*>(view.arena_base);
  for (const auto& copy : plan) copy.set(base + copy.offset);
  const auto table = BuildExpertTable(view);
  view.expert_table = reinterpret_cast<const Exl3ExpertDesc*>(base + table_offset);
  plan.push_back({std::as_bytes(std::span(table)), table_offset, {}});
  result.transfer_ms_ = Upload(plan, base, stream);
  return result;
}

uint32_t KvStateArena::MaxFullSeqLenForBudget(const ModelConfig& c, size_t budget, bool with_mtp) {
  auto l = ComputeKvArenaSizes(c, with_mtp);
  if (budget < l.Fixed()) return 0;
  return std::min<size_t>(UINT32_MAX, (budget - l.Fixed()) / l.per_token);
}
KvStateArena KvStateArena::Allocate(const ModelConfig& c, uint32_t tokens, bool with_mtp) {
  if (!tokens) throw std::invalid_argument("empty full KV context");
  auto l = ComputeKvArenaSizes(c, with_mtp);
  KvStateArena result;
  auto& v = result.view_;
  v.arena_bytes = l.Fixed() + tokens * l.per_token;
  result.allocation_ = AllocateDeviceBuffer(v.arena_bytes);
  auto* base = static_cast<std::byte*>(result.allocation_.get());
  v.arena_base = v.swa_kv = base;
  v.full_kv = base + l.swa;
  auto* conv = base + l.swa + tokens * l.per_token;
  v.kv_conv_state = reinterpret_cast<float*>(conv);
  v.sublayer_conv_state = reinterpret_cast<float*>(conv + l.kv_conv);
  if (with_mtp) {
    v.has_mtp_kv = 1;
    v.mtp_swa_window = c.mtp.sliding_window_size + c.mtp.block_size;
    v.mtp_full_kv = conv - tokens * l.mtp_per_token;
    v.mtp_swa_kv = conv + l.kv_conv + l.sublayer_conv;
  }
  v.max_full_seq_len = tokens;
  v.swa_window = c.sliding_window_size;
  v.num_swa_visits = model::kNumSwaVisits;
  v.num_full_visits = model::kNumFullVisits;
  v.num_total_visits = model::kNumLogicalVisits;
  v.num_kv_heads = c.num_key_value_heads;
  v.hidden_size = c.hidden_size;
  v.head_dim = c.head_dim;
  v.conv_state_len = c.sconv_kernel_size - 1;
  return result;
}
void KvStateArena::Reset(cudaStream_t stream) {
  CheckCuda(cudaMemsetAsync(view_.arena_base, 0, view_.arena_bytes, stream), "reset KV state");
}
ScratchArena ScratchArena::Allocate(size_t bytes) {
  if (!bytes) throw std::invalid_argument("empty scratch arena");
  ScratchArena result;
  result.allocation_ = AllocateDeviceBuffer(bytes);
  result.view_ = {result.allocation_.get(), bytes};
  return result;
}
}  // namespace flint

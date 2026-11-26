#include "loader/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <tuple>

#include "common/json.h"
#include "loader/model_constants.h"

namespace flint {
namespace {
using namespace std::string_view_literals;

void Check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

using json::Json;
using json::Parser;
using D = TensorDType;
struct TypeInfo {
  std::string_view name;
  size_t bytes;
};
constexpr TypeInfo kTypes[] = {{"BF16", 2}, {"F16", 2}, {"F32", 4}, {"F64", 8},
                               {"I16", 2},  {"I32", 4}, {"I64", 8}, {"U8", 1},
                               {"U16", 2},  {"U32", 4}, {"BOOL", 1}};
D ParseType(const std::string& name) {
  for (size_t i = 0; i < std::size(kTypes); ++i)
    if (kTypes[i].name == name) return static_cast<D>(i);
  throw std::runtime_error("unsupported tensor dtype " + name);
}

struct Fd {
  int value;
  ~Fd() {
    if (value >= 0) close(value);
  }
};
}  // namespace

size_t TensorDTypeBytes(D dtype) {
  const auto i = static_cast<size_t>(dtype);
  Check(i < std::size(kTypes), "unsupported tensor dtype");
  return kTypes[i].bytes;
}

void SafetensorsFile::Unmap::operator()(const std::byte* p) const {
  munmap(const_cast<std::byte*>(p), size);
}

SafetensorsFile SafetensorsFile::Open(const std::filesystem::path& path) {
  Fd fd{open(path.c_str(), O_RDONLY | O_CLOEXEC)};
  struct stat st{};
  if (fd.value < 0 || fstat(fd.value, &st) < 0)
    throw std::system_error(errno, std::generic_category(), path.string());
  Check(S_ISREG(st.st_mode) && st.st_size >= 8, "invalid safetensors file " + path.string());
  const size_t size = st.st_size;
  void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd.value, 0);
  if (addr == MAP_FAILED) throw std::system_error(errno, std::generic_category(), path.string());
  SafetensorsFile file;
  file.mapping_ = decltype(mapping_)(static_cast<const std::byte*>(addr), Unmap{size});
  const auto* bytes = static_cast<const unsigned char*>(addr);
  uint64_t len = 0;
  for (int i = 0; i < 8; ++i) len |= uint64_t(bytes[i]) << (8 * i);
  Check(len && len <= 100000000 && len <= size - 8, "invalid safetensors header length");
  file.payload_offset_ = 8 + len;
  auto root = Parser({static_cast<const char*>(addr) + 8, len}).Parse();
  for (const auto& [name, desc] : root.as<Json::Object>()) {
    if (name == "__metadata__") {
      for (const auto& [key, value] : desc.as<Json::Object>())
        file.metadata_.emplace(key, value.as<std::string>());
      continue;
    }
    TensorEntry e{name, ParseType(desc.at("dtype").as<std::string>()), {}, 0, 0};
    const auto& offsets = desc.at("data_offsets").as<Json::Array>();
    Check(offsets.size() == 2, "invalid data_offsets for " + name);
    e.data_offset = offsets[0].number<size_t>();
    const size_t end = offsets[1].number<size_t>();
    Check(end >= e.data_offset && end <= file.payload_bytes(), "offset out of range for " + name);
    for (const auto& dim : desc.at("shape").as<Json::Array>()) {
      const auto d = dim.number<int64_t>();
      Check(d >= 0, "negative dimension for " + name);
      e.shape.push_back(d);
    }
    size_t count = TensorDTypeBytes(e.dtype);
    if (std::find(e.shape.begin(), e.shape.end(), 0) != e.shape.end()) count = 0;
    for (size_t dim : e.shape) {
      Check(!dim || count <= SIZE_MAX / dim, "shape overflow for " + name);
      count *= dim;
    }
    e.nbytes = end - e.data_offset;
    Check(e.nbytes == count, "shape/byte mismatch for " + name);
    file.tensors_.push_back(std::move(e));
  }
  std::sort(file.tensors_.begin(), file.tensors_.end(), [](const auto& a, const auto& b) {
    return std::tie(a.data_offset, a.nbytes) < std::tie(b.data_offset, b.nbytes);
  });
  size_t end = 0;
  for (size_t i = 0; i < file.tensors_.size(); ++i) {
    auto& e = file.tensors_[i];
    Check(e.data_offset == end, "gap or overlap at " + e.name);
    end += e.nbytes;
    file.index_.emplace(e.name, i);
  }
  Check(end == file.payload_bytes(), "unmapped trailing payload");
  return file;
}

const TensorEntry* SafetensorsFile::Find(std::string_view name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}
std::span<const std::byte> SafetensorsFile::Bytes(const TensorEntry& e) const {
  Check(mapping_ && e.data_offset <= payload_bytes() && e.nbytes <= payload_bytes() - e.data_offset,
        "tensor outside mapped payload");
  return {mapping_.get() + payload_offset_ + e.data_offset, e.nbytes};
}

namespace {
template <class T>
T Expected(const Json& obj, const char* key, T expected) {
  const auto actual = obj.at(key).number<T>();
  Check(actual == expected, std::string("unsupported config value for ") + key);
  return actual;
}
}  // namespace
ModelConfig LoadModelConfig(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  Check(file.is_open(), "cannot open " + path.string());
  std::string input{std::istreambuf_iterator<char>(file), {}};
  const auto root = Parser(input).Parse();
  const auto& t = root.at("text_config");
  const auto& loop = t.at("loop_config");
  const auto& quant = root.at("quantization_config");
  Check(t.at("model_type").as<std::string>() == "inkling2_loop_text", "unsupported text model");
  Check(!loop.at("is_varloop").as<bool>(), "unsupported is_varloop");
  Check(quant.at("quant_method").as<std::string>() == "exl3" &&
            quant.at("codebook").as<std::string>() == "mul1",
        "unsupported quantization");
  const auto defaults = model::MakeDefaultInklingConfig();
  ModelConfig c{};
  c.hidden_size = Expected(t, "hidden_size", defaults.hidden_size);
  c.num_hidden_layers = Expected(t, "num_hidden_layers", defaults.num_hidden_layers);
  c.vocab_size = Expected(t, "vocab_size", defaults.vocab_size);
  c.unpadded_vocab_size = Expected(t, "unpadded_vocab_size", defaults.unpadded_vocab_size);
  c.num_attention_heads = Expected(t, "num_attention_heads", defaults.num_attention_heads);
  c.num_key_value_heads = Expected(t, "num_key_value_heads", defaults.num_key_value_heads);
  c.head_dim = Expected(t, "head_dim", defaults.head_dim);
  c.sliding_window_size = Expected(t, "sliding_window_size", defaults.sliding_window_size);
  c.d_rel = Expected(t, "d_rel", defaults.d_rel);
  c.rel_extent = Expected(t, "rel_extent", defaults.rel_extent);
  c.log_scaling_n_floor = Expected(t, "log_scaling_n_floor", defaults.log_scaling_n_floor);
  c.sconv_kernel_size = Expected(t, "sconv_kernel_size", defaults.sconv_kernel_size);
  c.n_layers_per_attn_res_block =
      Expected(t, "n_layers_per_attn_res_block", defaults.n_layers_per_attn_res_block);
  c.n_routed_experts = Expected(t, "n_routed_experts", defaults.n_routed_experts);
  c.num_experts_per_tok = Expected(t, "num_experts_per_tok", defaults.num_experts_per_tok);
  c.intermediate_size = Expected(t, "intermediate_size", defaults.intermediate_size);
  c.log_scaling_alpha = Expected(t, "log_scaling_alpha", defaults.log_scaling_alpha);
  c.rms_norm_eps = Expected(t, "rms_norm_eps", defaults.rms_norm_eps);
  c.qk_l2_clip_threshold = Expected(t, "qk_l2_clip_threshold", defaults.qk_l2_clip_threshold);
  c.rel_l2_clip_threshold = Expected(t, "rel_l2_clip_threshold", defaults.rel_l2_clip_threshold);
  c.route_scale = Expected(t, "route_scale", defaults.route_scale);
  c.mlp_gate_softcap = Expected(t, "mlp_gate_softcap", defaults.mlp_gate_softcap);
  c.mlp_up_softcap = Expected(t, "mlp_up_softcap", defaults.mlp_up_softcap);
  c.final_logit_softcapping =
      Expected(t, "final_logit_softcapping", defaults.final_logit_softcapping);
  c.loop_core_start = Expected(loop, "core_start", defaults.loop_core_start);
  c.loop_core_end = Expected(loop, "core_end", defaults.loop_core_end);
  c.loop_num_passes = Expected(loop, "num_passes", defaults.loop_num_passes);
  c.loop_gate_rank = Expected(loop, "gate_rank", defaults.loop_gate_rank);
  c.eos_token_id = Expected(root, "eos_token_id", defaults.eos_token_id);
  // The locked checkpoint omits pad_token_id; its tokenizer uses <|endoftext|>.
  c.pad_token_id = defaults.pad_token_id;
  for (const Json* obj : {&root, &t})
    if (obj->as<Json::Object>().contains("pad_token_id"))
      Expected(*obj, "pad_token_id", c.pad_token_id);
  Expected(t, "swa_head_dim", c.head_dim);
  Expected(t, "swa_num_attention_heads", c.num_attention_heads);
  Expected(t, "swa_num_key_value_heads", c.num_key_value_heads);
  Expected(t, "n_shared_experts", 0u);
  Check(!t.at("q_bias").as<bool>() && !t.at("o_bias").as<bool>() &&
            t.at("attn_res_normalize_values").as<bool>() && t.at("use_gate_bias").as<bool>() &&
            t.at("use_value_norm_no_gamma").as<bool>(),
        "unsupported attention/router configuration");
  std::fill(std::begin(c.is_full_layer), std::end(c.is_full_layer), 1);
  for (const auto& id : t.at("local_layer_ids").as<Json::Array>()) {
    auto i = id.number<uint32_t>();
    Check(i < c.num_hidden_layers && c.is_full_layer[i], "invalid local layer id");
    c.is_full_layer[i] = 0;
  }
  for (unsigned i = 0; i < c.num_hidden_layers; ++i)
    Check(c.is_full_layer[i] == defaults.is_full_layer[i], "unsupported local layer schedule");
  if (root.as<Json::Object>().contains("mtp_config")) {
    const auto& m = root.at("mtp_config");
    auto& d = c.mtp;
    d.hidden_size = Expected(m, "hidden_size", model::kMtpConfig.hidden_size);
    d.num_hidden_layers = Expected(m, "num_hidden_layers", model::kMtpConfig.num_hidden_layers);
    d.num_attention_heads =
        Expected(m, "num_attention_heads", model::kMtpConfig.num_attention_heads);
    d.num_key_value_heads =
        Expected(m, "num_key_value_heads", model::kMtpConfig.num_key_value_heads);
    d.head_dim = Expected(m, "head_dim", model::kMtpConfig.head_dim);
    d.intermediate_size = Expected(m, "intermediate_size", model::kMtpConfig.intermediate_size);
    d.max_position_embeddings =
        Expected(m, "max_position_embeddings", model::kMtpConfig.max_position_embeddings);
    d.sliding_window_size =
        Expected(m, "sliding_window_size", model::kMtpConfig.sliding_window_size);
    d.block_size = Expected(m, "block_size", model::kMtpConfig.block_size);
    d.markov_rank = Expected(m, "markov_rank", model::kMtpConfig.markov_rank);
    d.rms_norm_eps = Expected(m, "rms_norm_eps", model::kMtpConfig.rms_norm_eps);
    d.rope_theta = Expected(m, "rope_theta", model::kMtpConfig.rope_theta);
    for (auto& id : m.at("local_layer_ids").as<Json::Array>()) {
      auto i = id.number<unsigned>();
      Check(i < 3 && !d.is_local_layer[i], "invalid draft local layer");
      d.is_local_layer[i] = 1;
    }
    const auto& ids = m.at("target_layer_ids").as<Json::Array>();
    Check(ids.size() == 4, "invalid draft tap count");
    for (unsigned i = 0; i < 4; ++i) d.target_layer_ids[i] = ids[i].number<unsigned>();
    Check(std::equal(std::begin(d.target_layer_ids), std::end(d.target_layer_ids),
                     model::kMtpConfig.target_layer_ids) &&
              std::equal(std::begin(d.is_local_layer), std::end(d.is_local_layer),
                         model::kMtpConfig.is_local_layer),
          "unsupported draft schedule");
    c.has_mtp = 1;
  }
  return c;
}
}  // namespace flint

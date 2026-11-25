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
  ModelConfig c{};
  c.hidden_size = Expected(t, "hidden_size", 2304u);
  c.num_hidden_layers = Expected(t, "num_hidden_layers", 24u);
  c.vocab_size = Expected(t, "vocab_size", 201024u);
  c.unpadded_vocab_size = Expected(t, "unpadded_vocab_size", 200058u);
  c.num_attention_heads = Expected(t, "num_attention_heads", 24u);
  c.num_key_value_heads = Expected(t, "num_key_value_heads", 8u);
  c.head_dim = Expected(t, "head_dim", 128u);
  c.sliding_window_size = Expected(t, "sliding_window_size", 1024u);
  c.d_rel = Expected(t, "d_rel", 16u);
  c.rel_extent = Expected(t, "rel_extent", 1024u);
  c.log_scaling_n_floor = Expected(t, "log_scaling_n_floor", 8192u);
  c.sconv_kernel_size = Expected(t, "sconv_kernel_size", 4u);
  c.n_layers_per_attn_res_block = Expected(t, "n_layers_per_attn_res_block", 6u);
  c.n_routed_experts = Expected(t, "n_routed_experts", 128u);
  c.num_experts_per_tok = Expected(t, "num_experts_per_tok", 8u);
  c.intermediate_size = Expected(t, "intermediate_size", 1152u);
  c.log_scaling_alpha = Expected(t, "log_scaling_alpha", 0.1f);
  c.rms_norm_eps = Expected(t, "rms_norm_eps", 1e-6f);
  c.qk_l2_clip_threshold = Expected(t, "qk_l2_clip_threshold", 22.627416997969522f);
  c.rel_l2_clip_threshold = Expected(t, "rel_l2_clip_threshold", 8.f);
  c.route_scale = Expected(t, "route_scale", 8.f);
  c.mlp_gate_softcap = Expected(t, "mlp_gate_softcap", 4.f);
  c.mlp_up_softcap = Expected(t, "mlp_up_softcap", 25.f);
  c.final_logit_softcapping = Expected(t, "final_logit_softcapping", 30.f);
  c.loop_core_start = Expected(loop, "core_start", 6u);
  c.loop_core_end = Expected(loop, "core_end", 18u);
  c.loop_num_passes = Expected(loop, "num_passes", 3u);
  c.loop_gate_rank = Expected(loop, "gate_rank", 64u);
  c.eos_token_id = Expected(root, "eos_token_id", 200006u);
  // The locked checkpoint omits pad_token_id; its tokenizer uses <|endoftext|>.
  c.pad_token_id = 199999;
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
    Check(c.is_full_layer[i] == (i % 6 == 5), "unsupported local layer schedule");
  return c;
}
}  // namespace flint

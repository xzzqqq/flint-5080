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
#include <variant>

namespace flint {
namespace {
using namespace std::string_view_literals;

void Check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

// Number tokens retain their spelling so sizes and offsets never round through a double.
struct Json {
  using Object = std::map<std::string, Json, std::less<>>;
  using Array = std::vector<Json>;
  std::variant<std::nullptr_t, bool, std::string_view, std::string, Array, Object> value;

  template <class T>
  const T& as() const {
    if (auto p = std::get_if<T>(&value)) return *p;
    throw std::runtime_error("unexpected JSON value type");
  }
  const Json& at(std::string_view key) const {
    const auto& obj = as<Object>();
    auto it = obj.find(key);
    if (it == obj.end()) throw std::runtime_error("missing JSON field " + std::string(key));
    return it->second;
  }
  template <class T>
  T number() const {
    const auto token = as<std::string_view>();
    T result{};
    const auto end = token.data() + token.size();
    const auto [ptr, ec] = std::from_chars(token.data(), end, result);
    Check(ec == std::errc{} && ptr == end, "invalid numeric value");
    return result;
  }
};

class Parser {
 public:
  explicit Parser(std::string_view input) : input_(input) {}
  Json Parse() {
    Json root = Value(0);
    Space();
    if (pos_ != input_.size()) Invalid();
    return root;
  }

 private:
  [[noreturn]] void Invalid() const {
    throw std::runtime_error("invalid JSON at byte " + std::to_string(pos_));
  }
  char Peek() const { return pos_ < input_.size() ? input_[pos_] : '\0'; }
  void Space() {
    while (" \r\n\t"sv.find(Peek()) != std::string_view::npos) ++pos_;
  }
  bool Eat(char c) {
    if (Peek() != c) return false;
    ++pos_;
    return true;
  }
  void Need(char c) {
    if (!Eat(c)) Invalid();
  }
  unsigned Hex4() {
    if (input_.size() - pos_ < 4) Invalid();
    unsigned code;
    auto end = input_.data() + pos_ + 4;
    auto [ptr, ec] = std::from_chars(end - 4, end, code, 16);
    if (ec != std::errc{} || ptr != end) Invalid();
    pos_ += 4;
    return code;
  }
  static void AppendUtf8(std::string& out, unsigned cp) {
    if (cp < 128) {
      out += static_cast<char>(cp);
      return;
    }
    int extra = cp < 0x800 ? 1 : cp < 0x10000 ? 2 : 3;
    out += static_cast<char>((0xffu << (7 - extra)) | (cp >> (6 * extra)));
    while (extra > 0) {
      --extra;
      out += static_cast<char>(0x80 | ((cp >> (6 * extra)) & 63));
    }
  }
  void Escape(std::string& out) {
    const char c = Peek();
    ++pos_;
    auto idx = "\"\\/bfnrt"sv.find(c);
    if (idx != std::string_view::npos) {
      out += "\"\\/\b\f\n\r\t"[idx];
      return;
    }
    if (c != 'u') Invalid();
    unsigned cp = Hex4();
    if (cp >= 0xd800 && cp <= 0xdbff) {
      Need('\\');
      Need('u');
      unsigned low = Hex4();
      if (low < 0xdc00 || low > 0xdfff) Invalid();
      cp = 0x10000 + (cp - 0xd800) * 1024 + low - 0xdc00;
    } else if (cp >= 0xdc00 && cp <= 0xdfff) {
      Invalid();
    }
    AppendUtf8(out, cp);
  }
  std::string String() {
    Need('"');
    std::string out;
    while (!Eat('"')) {
      auto c = static_cast<unsigned char>(Peek());
      if (c < 32) Invalid();
      ++pos_;
      if (c == '\\') {
        Escape(out);
      } else if (c < 128) {
        out += static_cast<char>(c);
      } else {
        // Reject overlong sequences, surrogate scalars and code points above U+10FFFF.
        if (c < 0xc2 || c > 0xf4) Invalid();
        int n = c < 0xe0 ? 1 : c < 0xf0 ? 2 : 3;
        unsigned cp = c & (0x7f >> (n + 1));
        for (int i = 0; i < n; ++i) {
          auto tail = static_cast<unsigned char>(Peek());
          if ((tail & 0xc0) != 0x80) Invalid();
          cp = (cp << 6) | (tail & 63);
          ++pos_;
        }
        if ((n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) || (cp >= 0xd800 && cp <= 0xdfff) ||
            cp > 0x10ffff)
          Invalid();
        out.append(input_.substr(pos_ - n - 1, n + 1));
      }
    }
    return out;
  }
  void Digits() {
    size_t start = pos_;
    while (Peek() >= '0' && Peek() <= '9') ++pos_;
    if (pos_ == start) Invalid();
  }
  Json Value(int depth) {
    if (depth > 128) Invalid();
    Space();
    if (Eat('{')) {
      Json::Object obj;
      Space();
      if (!Eat('}')) {
        do {
          Space();
          auto key = String();
          Space();
          Need(':');
          if (!obj.emplace(std::move(key), Value(depth + 1)).second) Invalid();
          Space();
        } while (Eat(','));
        Need('}');
      }
      return {std::move(obj)};
    }
    if (Eat('[')) {
      Json::Array arr;
      Space();
      if (!Eat(']')) {
        do {
          arr.push_back(Value(depth + 1));
          Space();
        } while (Eat(','));
        Need(']');
      }
      return {std::move(arr)};
    }
    if (Peek() == '"') return {String()};
    for (auto literal : {"true"sv, "false"sv, "null"sv}) {
      if (input_.substr(pos_).starts_with(literal)) {
        pos_ += literal.size();
        if (literal == "null") return {nullptr};
        return {literal == "true"};
      }
    }
    const auto start = pos_;
    Eat('-');
    if (!Eat('0')) Digits();
    if (Eat('.')) Digits();
    if (Eat('e') || Eat('E')) {
      if (!Eat('+')) Eat('-');
      Digits();
    }
    return {input_.substr(start, pos_ - start)};
  }
  std::string_view input_;
  size_t pos_ = 0;
};

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

#pragma once
#include <charconv>
#include <concepts>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace flint::json {
using namespace std::string_view_literals;
// Number tokens retain their spelling so sizes and offsets never round through a double.
struct Json {
  using Object = std::map<std::string, Json, std::less<>>;
  using Array = std::vector<Json>;
  std::variant<std::nullptr_t, bool, std::string_view, std::string, Array, Object, uint64_t,
               int64_t, double>
      value;
  Json() : value(nullptr) {}
  Json(std::nullptr_t) : value(nullptr) {}
  Json(bool v) : value(v) {}
  Json(const char* v) : value(std::string(v)) {}
  Json(std::string v) : value(std::move(v)) {}
  Json(std::string_view numeric_token) : value(numeric_token) {}
  Json(Array v) : value(std::move(v)) {}
  Json(Object v) : value(std::move(v)) {}
  template <std::unsigned_integral T>
  Json(T v) : value(uint64_t(v)) {}
  template <std::signed_integral T>
  Json(T v) : value(int64_t(v)) {}
  Json(double v) : value(v) {}
  template <class T>
  T get(std::string_view key, T fallback) const {
    if (!as<Object>().contains(key)) return fallback;
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
      return at(key).number<T>();
    else
      return at(key).as<T>();
  }

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
    if (ec != std::errc{} || ptr != end) throw std::runtime_error("invalid numeric value");
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

}  // namespace flint::json

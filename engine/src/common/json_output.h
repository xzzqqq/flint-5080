#pragma once
#include <cmath>
#include <iomanip>
#include <sstream>

#include "common/json.h"

namespace flint {
using json::Json;
using Object = Json::Object;
using Array = Json::Array;
inline void WriteJson(std::ostream& out, const Json& j) {
  std::visit(
      [&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, std::string>) {
          out << '"';
          for (unsigned char c : value) {
            if (c == '"' || c == '\\')
              out << '\\' << c;
            else if (c < 32)
              out << "\\u00" << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
            else
              out << c;
          }
          out << '"';
        } else if constexpr (std::is_same_v<T, Object> || std::is_same_v<T, Array>) {
          constexpr bool object = std::is_same_v<T, Object>;
          out << (object ? '{' : '[');
          bool first = true;
          for (const auto& item : value) {
            if (!first) out << ',';
            first = false;
            if constexpr (object) {
              WriteJson(out, item.first);
              out << ':';
              WriteJson(out, item.second);
            } else
              WriteJson(out, item);
          }
          out << (object ? '}' : ']');
        } else if constexpr (std::is_same_v<T, std::nullptr_t>)
          out << "null";
        else if constexpr (std::is_same_v<T, bool>)
          out << (value ? "true" : "false");
        else if constexpr (std::is_same_v<T, double>) {
          if (std::isfinite(value))
            out << std::setprecision(17) << value;
          else
            out << "null";
        } else
          out << value;
      },
      j.value);
}
inline std::string Encode(const Json& j) {
  std::ostringstream out;
  WriteJson(out, j);
  return out.str();
}
}  // namespace flint

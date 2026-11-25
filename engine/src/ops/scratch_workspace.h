#pragma once

#include <stdexcept>

#include "loader/model_views.h"

namespace flint::ops_detail {

class ScratchWorkspace {
 public:
  explicit ScratchWorkspace(ScratchArenaView v) : rest(v) {}
  template <class T>
  T* Take(size_t n) {
    if (n > rest.bytes / sizeof(T)) throw std::length_error("operator scratch exhausted");
    size_t bytes = (sizeof(T) * n + 255) & ~size_t(255);
    if (bytes > rest.bytes) throw std::length_error("operator scratch alignment exceeds capacity");
    auto ptr = static_cast<T*>(rest.device_ptr);
    rest.bytes -= bytes;
    rest.device_ptr = static_cast<char*>(rest.device_ptr) + bytes;
    return ptr;
  }
  ScratchArenaView rest;
};
}  // namespace flint::ops_detail

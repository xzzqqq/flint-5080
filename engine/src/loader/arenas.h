#ifndef FLINT_ARENAS_H
#define FLINT_ARENAS_H

#include <cuda_runtime_api.h>

#include <filesystem>
#include <memory>
#include <utility>

#include "loader/model_views.h"

namespace flint {

namespace detail {
struct DeviceFree {
  void operator()(void* ptr) const noexcept { cudaFree(ptr); }
};
}  // namespace detail

using DeviceBuffer = std::unique_ptr<void, detail::DeviceFree>;
[[nodiscard]] DeviceBuffer AllocateDeviceBuffer(size_t bytes);

namespace detail {
template <class View>
class DeviceArena {
 public:
  const View& view() const { return view_; }

 protected:
  DeviceArena() = default;
  DeviceArena(DeviceArena&& other) noexcept { *this = std::move(other); }
  DeviceArena& operator=(DeviceArena&& other) noexcept {
    if (this != &other) {
      allocation_ = std::move(other.allocation_);
      view_ = std::exchange(other.view_, {});
    }
    return *this;
  }
  DeviceBuffer allocation_;
  View view_{};
};
}  // namespace detail

class WeightsArena : public detail::DeviceArena<ModelWeightsView> {
 public:
  static WeightsArena Load(const std::filesystem::path& checkpoint_dir,
                           cudaStream_t stream = nullptr);
  float transfer_ms() const { return transfer_ms_; }

 private:
  WeightsArena() = default;
  float transfer_ms_ = 0;
};

class KvStateArena : public detail::DeviceArena<KvStateArenaView> {
 public:
  static KvStateArena Allocate(const ModelConfig& config, uint32_t max_full_seq_len);
  static uint32_t MaxFullSeqLenForBudget(const ModelConfig& config, size_t budget_bytes);
  void Reset(cudaStream_t stream = nullptr);

 private:
  KvStateArena() = default;
};

class ScratchArena : public detail::DeviceArena<ScratchArenaView> {
 public:
  static ScratchArena Allocate(size_t bytes = 450ull * 1024 * 1024);

 private:
  ScratchArena() = default;
};
}  // namespace flint
#endif

#include "forge_smoke.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "forward_abi.h"

namespace flint {
namespace {
constexpr size_t kArenaBytes = 4 * 1024 * 1024;
constexpr size_t kLeakLimitBytes = 4 * 1024 * 1024;
constexpr int kElementCount = 4097;  // Exercise the partial final CUDA block.
constexpr float kBias = 0.25f;

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

struct ScratchArena {
  ScratchArena() = default;
  ScratchArena(const ScratchArena&) = delete;
  ScratchArena& operator=(const ScratchArena&) = delete;

  ~ScratchArena() {
    if (data != nullptr) {
      cudaFree(data);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  float* data{};
  cudaStream_t stream{};
};

class Module {
 public:
  explicit Module(const std::filesystem::path& path)
      : handle_(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
    if (!handle_) {
      throw std::runtime_error(dlerror());
    }
  }
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  ~Module() {
    if (handle_) {
      dlclose(handle_);
    }
  }

  template <typename Function>
  [[nodiscard]] Function Symbol(const char* name) const {
    dlerror();
    void* address = dlsym(handle_, name);
    if (const char* error = dlerror(); error != nullptr) {
      throw std::runtime_error(error);
    }
    return reinterpret_cast<Function>(address);
  }

  void Close() {
    if (dlclose(handle_) != 0) {
      throw std::runtime_error(dlerror());
    }
    handle_ = nullptr;
  }

 private:
  void* handle_;
};

size_t FreeDeviceBytes() {
  size_t available, total;
  CheckCuda(cudaMemGetInfo(&available, &total));
  return available;
}

void VerifyOutput(std::span<const float> input, const float* device_output,
                  const ForwardSmokeOutput& result) {
  std::vector<float> output(input.size());
  CheckCuda(cudaMemcpy(output.data(), device_output, input.size_bytes(), cudaMemcpyDeviceToHost));
  float checksum = 0;
  for (size_t i = 0; i < input.size(); ++i) {
    const float expected = input[i] + kBias;
    if (std::bit_cast<uint32_t>(expected) != std::bit_cast<uint32_t>(output[i])) {
      throw std::runtime_error("smoke add mismatch at index " + std::to_string(i));
    }
    checksum += expected;
  }
  if (result.checksum != checksum || !std::isfinite(result.elapsed_ms) || result.elapsed_ms < 0) {
    throw std::runtime_error("invalid forward smoke summary");
  }
}

float RunForward(const Module& module, const ForwardWorkspace& workspace,
                 std::span<const float> input) {
  const auto version =
      module.Symbol<decltype(&flint_forward_abi_version)>("flint_forward_abi_version");
  if (version() != FLINT_FORWARD_ABI_VERSION) {
    throw std::runtime_error("forward ABI mismatch");
  }
  const auto create = module.Symbol<decltype(&flint_forward_create)>("flint_forward_create");
  const auto destroy = module.Symbol<decltype(&flint_forward_destroy)>("flint_forward_destroy");
  const auto run = module.Symbol<decltype(&flint_forward_run_smoke)>("flint_forward_run_smoke");
  const std::unique_ptr<ForwardContext, decltype(destroy)> context(
      create(nullptr, nullptr, nullptr, workspace.cuda_stream), destroy);
  if (!context) {
    throw std::runtime_error("forward context creation failed");
  }

  const ForwardOptions options{};
  ForwardSmokeOutput result{};
  CheckCuda(static_cast<cudaError_t>(run(context.get(), &workspace, &options,
                                         static_cast<int32_t>(input.size()), kBias, &result)));
  CheckCuda(static_cast<cudaError_t>(result.status));
  VerifyOutput(input, static_cast<const float*>(workspace.device_ptr) + input.size(), result);
  return result.elapsed_ms;
}
}  // namespace

void RunForgeSmoke(const std::filesystem::path& forward_path) {
  cudaDeviceProp gpu{};
  CheckCuda(cudaSetDevice(0));
  CheckCuda(cudaGetDeviceProperties(&gpu, 0));
  ScratchArena arena;
  CheckCuda(cudaStreamCreateWithFlags(&arena.stream, cudaStreamNonBlocking));
  CheckCuda(cudaMalloc(&arena.data, kArenaBytes));
  std::vector<float> input(kElementCount);
  for (int i = 0; i < kElementCount; ++i) {
    input[i] = static_cast<float>(i % 257 - 128);
  }
  CheckCuda(cudaMemcpyAsync(arena.data, input.data(), std::span(input).size_bytes(),
                            cudaMemcpyHostToDevice, arena.stream));
  CheckCuda(cudaStreamSynchronize(arena.stream));

  const ForwardWorkspace workspace{
      .device_ptr = arena.data, .bytes = kArenaBytes, .cuda_stream = arena.stream};
  const size_t free_before = FreeDeviceBytes();
  Module module(forward_path);
  const float elapsed_ms = RunForward(module, workspace, input);
  module.Close();
  const size_t free_after = FreeDeviceBytes();
  const size_t loss = free_before > free_after ? free_before - free_after : 0;
  if (loss > kLeakLimitBytes) {
    throw std::runtime_error("WARN: forward unload lost " + std::to_string(loss) + " VRAM bytes");
  }
  std::cout << gpu.name << ", " << input.size() << " exact outputs, " << elapsed_ms
            << " ms, VRAM loss " << loss << " bytes\n";
}
}  // namespace flint

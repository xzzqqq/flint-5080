#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <memory>
#include <new>
#include <span>

#include "forge/forward_abi.h"
#include "ops/op_params.h"

struct ForwardContext {
  ForwardContext() = default;
  ForwardContext(const ForwardContext&) = delete;
  ForwardContext& operator=(const ForwardContext&) = delete;

  ~ForwardContext() {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
  }

  cudaEvent_t start{};
  cudaEvent_t stop{};
};

namespace {
[[nodiscard]] cudaError_t RunTimedKernel(const ForwardContext& context, cudaStream_t stream,
                                         const SmokeAddParams& params, float& elapsed_ms) {
  if (const auto status = cudaEventRecord(context.start, stream); status != cudaSuccess) {
    return status;
  }
  if (const int status = flint_launch_smoke_add_f32(&params, stream); status != cudaSuccess) {
    return static_cast<cudaError_t>(status);
  }
  if (const auto status = cudaEventRecord(context.stop, stream); status != cudaSuccess) {
    return status;
  }
  if (const auto status = cudaEventSynchronize(context.stop); status != cudaSuccess) {
    return status;
  }
  return cudaEventElapsedTime(&elapsed_ms, context.start, context.stop);
}

[[nodiscard]] cudaError_t ReadChecksum(std::span<const float> device_values, float& checksum) {
  std::array<float, 256> buffer;
  while (!device_values.empty()) {
    const auto chunk = device_values.first(std::min(buffer.size(), device_values.size()));
    if (const auto status =
            cudaMemcpy(buffer.data(), chunk.data(), chunk.size_bytes(), cudaMemcpyDeviceToHost);
        status != cudaSuccess) {
      return status;
    }
    for (float value : std::span(buffer).first(chunk.size())) {
      checksum += value;
    }
    device_values = device_values.subspan(chunk.size());
  }
  return cudaSuccess;
}
}  // namespace

uint32_t flint_forward_abi_version() { return FLINT_FORWARD_ABI_VERSION; }

ForwardContext* flint_forward_create() {
  std::unique_ptr<ForwardContext> context(new (std::nothrow) ForwardContext);
  if (!context) {
    return nullptr;
  }
  if (cudaEventCreate(&context->start) != cudaSuccess) {
    return nullptr;
  }
  if (cudaEventCreate(&context->stop) != cudaSuccess) {
    return nullptr;
  }
  return context.release();
}

void flint_forward_destroy(ForwardContext* context) { delete context; }

int32_t flint_forward_run_smoke(ForwardContext* context, const ForwardWorkspace* workspace,
                                const ForwardOptions* options, int32_t element_count, float bias,
                                ForwardSmokeOutput* result) {
  if (result == nullptr) {
    return cudaErrorInvalidValue;
  }
  *result = {.status = cudaErrorInvalidValue};
  if (!context || !workspace || !workspace->device_ptr || !options || element_count <= 0 ||
      static_cast<size_t>(element_count) > workspace->bytes / (2 * sizeof(float))) {
    return result->status;
  }

  auto* input = static_cast<float*>(workspace->device_ptr);
  const SmokeAddParams params{
      .input = input, .output = input + element_count, .n = element_count, .bias = bias};
  const auto stream = static_cast<cudaStream_t>(workspace->cuda_stream);
  result->status = RunTimedKernel(*context, stream, params, result->elapsed_ms);
  if (result->status == cudaSuccess) {
    result->status =
        ReadChecksum({params.output, static_cast<size_t>(element_count)}, result->checksum);
  }
  return result->status;
}

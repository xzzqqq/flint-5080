#include "stub_ops.h"

namespace {
constexpr int kThreadsPerBlock = 256;

__global__ void SmokeAdd(SmokeAddParams params) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < static_cast<unsigned int>(params.n)) {
    params.output[index] = params.input[index] + params.bias;
  }
}
}  // namespace

extern "C" int flint_launch_smoke_add_f32(const SmokeAddParams* params, void* stream) {
  if (params == nullptr || params->n <= 0 || params->input == nullptr ||
      params->output == nullptr) {
    return cudaErrorInvalidValue;
  }
  const int blocks = 1 + (params->n - 1) / kThreadsPerBlock;
  SmokeAdd<<<blocks, kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(*params);
  return cudaGetLastError();
}

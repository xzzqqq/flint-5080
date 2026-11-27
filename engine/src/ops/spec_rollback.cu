#include <cuda_bf16.h>

#include "spec_rollback.h"
namespace {
using B = __nv_bfloat16;
__global__ void SwapSlots(B* ring, B* backup, int pos, int count, int first, bool restore) {
  int plane = blockIdx.y, j = first + blockIdx.x;
  int slot = (pos + j) % 1024;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    auto& live = ring[(plane * 1024 + slot) * 1024 + i];
    auto& saved = backup[(plane * count + j) * 1024 + i];
    if (restore)
      live = saved;
    else
      saved = live;
  }
}
__global__ void Commit(const float* trace, float* state, int channels, int count, int accepted) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < channels)
    for (int j = 0; j < 3; ++j) state[c * 3 + j] = trace[c * (3 + count) + accepted + j];
}
}  // namespace

int flint_swa_kv_backup(void* ring, void* backup, int pos, int count, void* stream) {
  if (count < 1 || count > 8 || pos < 0) return cudaErrorInvalidValue;
  SwapSlots<<<dim3(count, 80), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<B*>(ring), static_cast<B*>(backup), pos, count, 0, false);
  return cudaGetLastError();
}
int flint_swa_kv_restore(void* ring, void* backup, int pos, int count, int accepted, void* stream) {
  if (count < 1 || count > 8 || pos < 0 || accepted < 0 || accepted > count)
    return cudaErrorInvalidValue;
  if (accepted == count) return cudaSuccess;
  SwapSlots<<<dim3(count - accepted, 80), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<B*>(ring), static_cast<B*>(backup), pos, count, accepted, true);
  return cudaGetLastError();
}
int flint_sconv_commit_trace(const float* trace, float* state, int channels, int count,
                             int accepted, void* stream) {
  if (count < 1 || count > 8 || accepted < 0 || accepted > count || channels < 1)
    return cudaErrorInvalidValue;
  Commit<<<(channels + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      trace, state, channels, count, accepted);
  return cudaGetLastError();
}

#pragma once
// Phase 1 reference parity: reduce four contiguous lanes before the binary tree.
namespace flint::row {
inline int Threads(int width) { return width == 2304 ? 512 : width == 128 ? 32 : 256; }
template <class Value>
__device__ float Sum(int width, Value value, float* buf) {
  int lane = threadIdx.x;
  float sum = 0;
  if (width == 2304 || width == 128) {
    float columns[4] = {};
    for (int base = 4 * lane; base < width; base += 4 * blockDim.x) {
#pragma unroll
      for (int j = 0; j < 4; ++j) columns[j] = __fadd_rn(columns[j], value(base + j));
    }
    for (float x : columns) sum += x;
  } else {
    for (int i = lane; i < width; i += blockDim.x) sum += value(i);
  }
  buf[lane] = sum;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d /= 2) {
    if (lane < d) buf[lane] += buf[lane + d];
    __syncthreads();
  }
  sum = buf[0];
  __syncthreads();
  return sum;
}
template <class Value>
__device__ float InvRms(int width, Value value, float* buf) {
  float ss = Sum(
      width,
      [&](int i) {
        float x = value(i);
        return __fmul_rn(x, x);
      },
      buf);
  return rsqrtf(__fadd_rn(__fmul_rn(ss, 1.f / width), 1e-6f));
}
}  // namespace flint::row

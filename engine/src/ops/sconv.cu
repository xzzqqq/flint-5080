#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "sconv_ops.h"

namespace {
__device__ float Read(const void* x, int i, FlintType type) {
  if (type == FLINT_F32) return static_cast<const float*>(x)[i];
  if (type == FLINT_F16) return __half2float(static_cast<const __half*>(x)[i]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(x)[i]);
}
__device__ void Write(void* y, int i, float v, FlintType type) {
  if (type == FLINT_F32)
    static_cast<float*>(y)[i] = v;
  else if (type == FLINT_F16)
    static_cast<__half*>(y)[i] = __float2half_rn(v);
  else
    static_cast<__nv_bfloat16*>(y)[i] = __float2bfloat16_rn(v);
}

__global__ void Conv(ConvParams p, bool dynamic) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= p.channels) return;
  float* history = p.state + 3 * c;
  float a = history[0], b = history[1], d = history[2];
  for (int t = 0; t < p.tokens; ++t) {
    int i = t * p.channels + c;
    float x = Read(p.x, i, p.input);
    float window[4] = {a, b, d, x}, sum = 0;
    for (int j = 0; j < 4; ++j) {
      int lag = dynamic ? j : 3 - j;
      float w = dynamic ? Read(p.weight, lag * p.channels + c, FLINT_BF16) +
                              Read(p.delta, (t * 8 + c / 128) * 4 + lag, FLINT_BF16)
                        : Read(p.weight, 4 * c + j, FLINT_BF16);
      sum = __fadd_rn(sum, __fmul_rn(w, window[3 - lag]));
    }
    Write(p.y, i, (sum + x) * (dynamic ? 1.f / 3.f : 1.f / 2.2360679775f), p.output);
    a = b;
    b = d;
    d = x;
  }
  history[0] = a;
  history[1] = b;
  history[2] = d;
}
int Launch(ConvParams p, void* s, bool dynamic) {
  Conv<<<(p.channels + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(p, dynamic);
  return cudaGetLastError();
}
}  // namespace
int flint_static_sconv(ConvParams p, void* s) { return Launch(p, s, false); }
int flint_dynamic_sconv(ConvParams p, void* s) {
  if (p.channels != 1024) return cudaErrorInvalidValue;
  return Launch(p, s, true);
}

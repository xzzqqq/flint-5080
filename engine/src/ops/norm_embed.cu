#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "norm_ops.h"
#include "row_reduction.cuh"

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

__global__ void Add(const float* partial, const void* bf16, void* out, int count, FlintType type) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) Write(out, i, partial[i] + Read(bf16, i, FLINT_BF16), type);
}

__global__ void Norm(NormParams p) {
  __shared__ float sums[512];
  int t = blockIdx.x, lane = threadIdx.x, row = p.ids ? p.ids[t] : t;
  float inv = flint::row::InvRms(
      p.width, [&](int i) { return Read(p.x, row * p.width + i, p.input); }, sums);
  for (int i = lane; i < p.width; i += blockDim.x) {
    float y = Read(p.x, row * p.width + i, p.input) * inv;
    Write(p.y, t * p.width + i, p.weight ? y * Read(p.weight, i, FLINT_BF16) : y, p.output);
  }
}
int Launch(NormParams p, void* s) {
  Norm<<<p.rows, flint::row::Threads(p.width), 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
__global__ void Head(HeadParams p) {
  __shared__ float maxima[256];
  __shared__ int indices[256];
  int t = blockIdx.x, lane = threadIdx.x, id = lane;
  float best = -__int_as_float(0x7f800000);
  for (int i = lane; i < p.vocab; i += blockDim.x) {
    float y = 30.f * tanhf(Read(p.x, t * p.stride + i, p.input) * (1.f / 30.f));
    p.logits[t * p.vocab + i] = y;
    if (y > best) {
      best = y;
      id = i;
    }
  }
  maxima[lane] = best;
  indices[lane] = id;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (lane < d && (maxima[lane + d] > maxima[lane] ||
                     (maxima[lane + d] == maxima[lane] && indices[lane + d] < indices[lane]))) {
      maxima[lane] = maxima[lane + d];
      indices[lane] = indices[lane + d];
    }
    __syncthreads();
  }
  if (!lane) p.ids[t] = indices[0];
}
}  // namespace
int flint_embed_rmsnorm(NormParams p, void* s) {
  if (!p.ids || !p.weight) return cudaErrorInvalidValue;
  return Launch(p, s);
}
int flint_rmsnorm(NormParams p, void* s) {
  p.ids = nullptr;
  return Launch(p, s);
}
int flint_unweighted_rmsnorm(NormParams p, void* s) {
  p.ids = nullptr;
  p.weight = nullptr;
  return Launch(p, s);
}
int flint_logit_softcap_argmax(HeadParams p, void* s) {
  Head<<<p.rows, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}

int flint_add_residual(const float* partial, const void* bf16, void* out, int count, FlintType type,
                       void* s) {
  Add<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(partial, bf16, out, count,
                                                                     type);
  return cudaGetLastError();
}

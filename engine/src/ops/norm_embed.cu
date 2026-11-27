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
  int t = blockIdx.y, first = blockIdx.x * blockDim.x + threadIdx.x;
  HeadWinner best{-__int_as_float(0x7f800000), first};
  for (int i = first; i < p.vocab; i += gridDim.x * blockDim.x) {
    float y = 30.f * tanhf(Read(p.x, t * p.stride + i, p.input) * (1.f / 30.f));
    if (p.logits) p.logits[t * p.vocab + i] = y;
    if (y > best.value) best = {y, i};
  }
  best = flint::head::BlockWinner(best);
  if (threadIdx.x == 0) {
    if (p.reduce_scratch)
      static_cast<HeadWinner*>(p.reduce_scratch)[t * gridDim.x + blockIdx.x] = best;
    else
      p.ids[t] = best.token;
  }
}
__global__ void ReduceHead(const HeadWinner* winners, int* ids) {
  auto best = flint::head::BlockWinner(winners[blockIdx.y * FLINT_HEAD_BLOCKS + threadIdx.x]);
  if (threadIdx.x == 0) ids[blockIdx.y] = best.token;
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
  int blocks = p.reduce_scratch ? FLINT_HEAD_BLOCKS : 1;
  Head<<<dim3(blocks, p.rows), 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  auto status = cudaGetLastError();
  if (status || !p.reduce_scratch) return status;
  return flint_reduce_head_winners(static_cast<const HeadWinner*>(p.reduce_scratch), p.ids, p.rows,
                                   s);
}
int flint_reduce_head_winners(const HeadWinner* winners, int* ids, int rows, void* s) {
  ReduceHead<<<dim3(1, rows), FLINT_HEAD_BLOCKS, 0, static_cast<cudaStream_t>(s)>>>(winners, ids);
  return cudaGetLastError();
}

int flint_add_residual(const float* partial, const void* bf16, void* out, int count, FlintType type,
                       void* s) {
  Add<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(partial, bf16, out, count,
                                                                     type);
  return cudaGetLastError();
}

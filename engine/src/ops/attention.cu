#include <cuda_bf16.h>

#include "op_params.h"
#include "row_reduction.cuh"

namespace {
using B = __nv_bfloat16;
__device__ float Get(const void* x, size_t i) {
  return __bfloat162float(static_cast<const B*>(x)[i]);
}
__device__ float Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ void Put(void* y, size_t i, float x) { static_cast<B*>(y)[i] = __float2bfloat16_rn(x); }
__device__ float Reduce(float* buf, float v, int n, bool maximum = false) {
  int i = threadIdx.x;
  buf[i] = v;
  __syncthreads();
  for (int d = n / 2; d; d /= 2) {
    if (i < d) buf[i] = maximum ? fmaxf(buf[i], buf[i + d]) : buf[i] + buf[i + d];
    __syncthreads();
  }
  float result = buf[0];
  __syncthreads();
  return result;
}
__global__ void Clip(ClipParams p, int heads, int dim, float threshold) {
  __shared__ float buf[128];
  auto value = [&](int c) {
    int i = blockIdx.x * dim + c;
    return Get(p.x, i) * (p.scale ? Get(p.scale, i % (heads * dim)) : 1.f);
  };
  float norm = sqrtf(flint::row::Sum(
      dim,
      [&](int c) {
        float x = value(c);
        return __fmul_rn(x, x);
      },
      buf));
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    // Scalar / tensor in the reference rounds the reciprocal before multiplying.
    float clipped = Round(value(c) * fminf(1.f, threshold * __frcp_rn(norm + 1e-6f)));
    if (p.log_scale)
      clipped *= 1.f + 0.1f * logf(fmaxf(1.f, (p.position + blockIdx.x / heads + 1) / 8192.f));
    Put(p.y, blockIdx.x * dim + c, clipped);
  }
}
int LaunchClip(ClipParams p, void* s, int heads, int dim, float threshold) {
  Clip<<<p.tokens * heads, dim == 128 ? 32 : dim, 0, static_cast<cudaStream_t>(s)>>>(p, heads, dim,
                                                                                     threshold);
  return cudaGetLastError();
}
__global__ void Cache(CacheParams p) {
  int t = blockIdx.x + max(0, p.tokens - p.capacity), slot = (p.position + t) % p.capacity;
  for (int c = threadIdx.x; c < 1024; c += blockDim.x) {
    size_t dest = (size_t)slot * 1024 + c, src = (size_t)t * 1024 + c;
    static_cast<B*>(p.cache_k)[dest] = static_cast<const B*>(p.k)[src];
    static_cast<B*>(p.cache_v)[dest] = static_cast<const B*>(p.v)[src];
  }
}
__device__ int Begin(AttentionParams p, int pos) {
  return p.sliding ? max(0, pos + 1 - p.capacity) : 0;
}
__global__ void Score(AttentionParams p) {
  __shared__ float buf[128];
  int h = blockIdx.y, t = blockIdx.z, pos = p.position + t, begin = Begin(p, pos),
      key = begin + blockIdx.x;
  if (key > pos) return;
  int c = threadIdx.x, dist = pos - key;
  float dot = flint::row::Sum(
      128,
      [&](int i) {
        return Get(p.q, (t * 24 + h) * 128 + i) *
               Get(p.cache_k, ((size_t)(key % p.capacity) * 8 + h / 3) * 128 + i);
      },
      buf);
  if (!c) {
    float bias = dist < 1024 ? Get(p.rel, (t * 24 + h) * 1024 + dist) : 0;
    p.scores[(size_t)(t * 24 + h) * p.capacity + blockIdx.x] =
        Round(Round(Round(dot) * 0.08838834764831845f) + bias);
  }
}
__global__ void Attend(AttentionParams p) {
  __shared__ float buf[128];
  int row = blockIdx.x, t = row / 24, h = row % 24, c = threadIdx.x;
  int pos = p.position + t, begin = Begin(p, pos), n = pos - begin + 1;
  float* scores = p.scores + (size_t)row * p.capacity;
  float sink = Get(p.sink, row), peak = sink;
  for (int j = c; j < n; j += 128) peak = fmaxf(peak, scores[j]);
  peak = Reduce(buf, peak, 128, true);
  // The sink is the last softmax column. Preserve its lane in the
  // 32-lane reduction, including when a row spans multiple warp iterations.
  for (int j = c; j < n; j += 128) scores[j] = expf(scores[j] - peak);
  __syncthreads();
  float denom = 0;
  if (c < 32)
    for (int j = c; j <= n; j += 32) denom += j == n ? expf(sink - peak) : scores[j];
  denom = Reduce(buf, denom, 128);
  for (int j = c; j < n; j += 128) scores[j] = Round(scores[j] / denom);
  __syncthreads();
  // The reference reduces the strided key axis with four accumulators.
  // At 64 visible keys it also splits the reduction across four warps.
  int groups = n >= 64 ? 4 : 1;
  float group[4] = {};
  for (int g = 0; g < groups; ++g) {
    float part[4] = {};
    for (int j = g; j < n; j += groups) {
      auto index = ((size_t)((begin + j) % p.capacity) * 8 + h / 3) * 128 + c;
      part[(j / groups) % 4] += scores[j] * Get(p.cache_v, index);
    }
    group[g] = ((part[0] + part[1]) + part[2]) + part[3];
  }
  float out = (group[0] + group[2]) + (group[1] + group[3]);
  Put(p.y, row * 128 + c, out);
}
__global__ void Gate(GateMulParams p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) Put(p.y, i, Get(p.x, i) * (1.f / (1.f + expf(-Get(p.gate, i)))));
}
}  // namespace
int flint_q_norm_clip(ClipParams p, void* s) {
  return LaunchClip(p, s, 24, 128, 22.627416997969522f);
}
int flint_k_l2_clip(ClipParams p, void* s) {
  p.scale = nullptr;
  p.log_scale = 0;
  return LaunchClip(p, s, 8, 128, 22.627416997969522f);
}
int flint_rel_norm_clip(ClipParams p, void* s) { return LaunchClip(p, s, 24, 16, 8.f); }
int flint_kv_cache_write(CacheParams p, void* s) {
  if (p.position < 0 || p.tokens < 1 || p.capacity < 1 ||
      (!p.sliding && p.position > p.capacity - p.tokens))
    return cudaErrorInvalidValue;
  int retained = p.tokens < p.capacity ? p.tokens : p.capacity;
  Cache<<<retained, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_attention_forward(AttentionParams p, void* s) {
  if (p.tokens < 1 || p.position < 0 || p.capacity < 1 ||
      (!p.sliding && p.position > p.capacity - p.tokens))
    return cudaErrorInvalidValue;
  int visible = p.position + p.tokens;
  if (visible > p.capacity) visible = p.capacity;
  Score<<<dim3(visible, 24, p.tokens), 32, 0, static_cast<cudaStream_t>(s)>>>(p);
  auto status = cudaGetLastError();
  if (status) return status;
  Attend<<<p.tokens * 24, 128, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_gated_mul_sigmoid(GateMulParams p, void* s) {
  Gate<<<(p.count + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}

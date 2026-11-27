#include <cuda_bf16.h>
#include <math_constants.h>

#include "attention_ops.h"
#include "row_reduction.cuh"

namespace {
using B = __nv_bfloat16;
__device__ float Get(const void* x, size_t i) {
  return __bfloat162float(static_cast<const B*>(x)[i]);
}
__device__ float Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ void Put(void* y, size_t i, float x) { static_cast<B*>(y)[i] = __float2bfloat16_rn(x); }
__device__ int Position(int pos, const int32_t* ptr, int offset) {
  return ptr ? *ptr + offset : pos;
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
      clipped *= 1.f + 0.1f * logf(fmaxf(1.f, (Position(p.position, p.d_pos, p.pos_offset) +
                                               blockIdx.x / heads + 1) /
                                                  8192.f));
    Put(p.y, blockIdx.x * dim + c, clipped);
  }
}
int LaunchClip(ClipParams p, void* s, int heads, int dim, float threshold) {
  Clip<<<p.tokens * heads, dim == 128 ? 32 : dim, 0, static_cast<cudaStream_t>(s)>>>(p, heads, dim,
                                                                                     threshold);
  return cudaGetLastError();
}
__global__ void Cache(CacheParams p) {
  int t = blockIdx.x + max(0, p.tokens - p.capacity),
      slot = (Position(p.position, p.d_pos, p.pos_offset) + t) % p.capacity;
  for (int c = threadIdx.x; c < 1024; c += blockDim.x) {
    size_t dest = (size_t)slot * 1024 + c, src = (size_t)t * 1024 + c;
    static_cast<B*>(p.cache_k)[dest] = static_cast<const B*>(p.k)[src];
    static_cast<B*>(p.cache_v)[dest] = static_cast<const B*>(p.v)[src];
  }
}
// One warp owns (query token, Q head) and four consecutive output channels per lane.
// The CTA stages eight keys of one KV head for all its query warps.
constexpr int kKeyTile = 8, kQueryTile = 8;
struct alignas(8) Channels {
  B x[4];
};
enum class Mode { Target, Draft, Prewritten };

template <Mode mode>
__global__ void Flash(AttentionParams p, int first) {
  constexpr bool draft = mode == Mode::Draft;
  constexpr bool prewritten = mode == Mode::Prewritten;
  __shared__ __align__(16) B keys[kKeyTile][128], values[kKeyTile][128];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int tile = blockIdx.y * kQueryTile, row = tile + warp / 3;
  int kv_head = blockIdx.x, head = kv_head * 3 + warp % 3, query = row * 24 + head;
  int base = Position(p.position, p.d_pos, p.pos_offset);
  int split = blockIdx.z, splits = gridDim.z;
  int tile_end = base + min(tile + kQueryTile, p.tokens);
  int tile_begin = draft ? (p.sliding ? max(0, base - 1024) : first)
                         : (p.sliding ? max(0, base + tile + 1 - p.capacity) : 0);
  // Assign absolute key tiles to splits. Every causal row then reduces its
  // prefix in the same order, including partial and wrapped verify blocks.
  int first_tile = tile_begin / kKeyTile;
  int start = (first_tile + (split - first_tile % splits + splits) % splits) * kKeyTile;
  int end = tile_end;
  int pos = base + row;
  int begin = !draft && p.sliding ? max(0, pos + 1 - p.capacity) : tile_begin;

  bool active = row < p.tokens;
  float peak = !draft && split == 0 && active ? Get(p.sink, query) : -CUDART_INF_F;
  float denom = !draft && split == 0 ? 1.f : 0.f;
  float acc[4] = {}, q[4];
  if (active && start < end) {
    Channels pack = reinterpret_cast<const Channels*>(p.q)[query * 32 + lane];
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = float(pack.x[i]);
  }

  // Cache-only clients historically round normalized probabilities to BF16.
  // Replay their logits after finding the denominator, without allocating scores.
  for (int pass = 0; pass < (prewritten ? 2 : 1); ++pass) {
    for (int key_tile = start; key_tile < end; key_tile += kKeyTile * splits) {
      int count = min(kKeyTile, end - key_tile);
      for (int i = threadIdx.x; i < count * 16; i += blockDim.x) {
        int j = i / 16, key = key_tile + j;
        bool fresh = mode == Mode::Target && key >= base;
        int slot = fresh ? key - base : key % p.capacity;
        size_t vec = (size_t(slot) * 8 + kv_head) * 16 + i % 16;
        reinterpret_cast<uint4*>(keys[j])[i % 16] =
            static_cast<const uint4*>(fresh ? p.k_new : p.cache_k)[vec];
        if (!prewritten || pass)
          reinterpret_cast<uint4*>(values[j])[i % 16] =
              static_cast<const uint4*>(fresh ? p.v_new : p.cache_v)[vec];
      }
      __syncthreads();
      if (active) {
        int lo = max(0, begin - key_tile);
        int hi = draft ? count : min(count, pos - key_tile + 1);
        for (int j = lo; j < hi; ++j) {
          Channels k = reinterpret_cast<const Channels*>(keys[j])[lane];
          float dot = 0.f;
#pragma unroll
          for (int i = 0; i < 4; ++i) dot += q[i] * float(k.x[i]);
#pragma unroll
          for (int d = 16; d; d /= 2) dot += __shfl_down_sync(0xffffffff, dot, d);
          float score = 0;
          if (lane == 0) {
            constexpr float scale = 0.08838834764831845f;
            if constexpr (draft) {
              score = dot * scale;
            } else {
              int dist = pos - (key_tile + j);
              float bias = dist < 1024 ? Get(p.rel, query * 1024 + dist) : 0.f;
              score = Round(Round(Round(dot) * scale) + bias);
            }
          }
          score = __shfl_sync(0xffffffff, score, 0);
          float old_scale = 1.f, weight;
          if (prewritten && pass) {
            weight = Round(expf(score - peak) / denom);
          } else {
            float next = fmaxf(peak, score);
            old_scale = expf(peak - next);
            weight = expf(score - next);
            denom = fmaf(denom, old_scale, weight);
            peak = next;
          }
          if (!prewritten || pass) {
            Channels v = reinterpret_cast<const Channels*>(values[j])[lane];
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = fmaf(weight, float(v.x[i]), acc[i] * old_scale);
          }
        }
      }
      __syncthreads();
    }
  }
  if (!active) return;
  if (splits == 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      Put(p.y, query * 128 + lane * 4 + i, prewritten ? acc[i] : acc[i] / denom);
  } else {
    float* part = p.scores + (size_t(split) * p.tokens * 24 + query) * FLINT_ATTENTION_PARTIAL;
    if (!lane) {
      part[0] = peak;
      part[1] = denom;
    }
    // Combine skips the numerator of empty splits.
    if (denom) {
#pragma unroll
      for (int i = 0; i < 4; ++i) part[2 + lane * 4 + i] = acc[i];
    }
  }
}

__global__ void Combine(AttentionParams p, int splits) {
  __shared__ float scales[32], denominator;
  int c = threadIdx.x, query = blockIdx.y * 24 + blockIdx.x;
  const float* part = p.scores + query * FLINT_ATTENTION_PARTIAL;
  size_t stride = size_t(p.tokens) * 24 * FLINT_ATTENTION_PARTIAL;
  if (c < 32) {
    float peak = c < splits ? part[c * stride] : -CUDART_INF_F;
    float largest = peak;
#pragma unroll
    for (int d = 16; d; d /= 2) largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, d));
    float scale = expf(peak - largest);
    scales[c] = scale;
    float sum = c < splits ? part[c * stride + 1] * scale : 0.f;
#pragma unroll
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (!c) denominator = sum;
  }
  __syncthreads();
  float sum = 0.f;
  for (int s = 0; s < splits; ++s)
    if (scales[s]) sum = fmaf(scales[s], part[s * stride + 2 + c], sum);
  Put(p.y, query * 128 + c, sum / denominator);
}

template <Mode mode>
int LaunchFlash(AttentionParams p, int first, int splits, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  int rows = p.tokens < kQueryTile ? p.tokens : kQueryTile;
  dim3 grid(8, (p.tokens + kQueryTile - 1) / kQueryTile, splits);
  Flash<mode><<<grid, rows * 3 * 32, 0, s>>>(p, first);
  auto status = cudaGetLastError();
  if (status || splits == 1) return status;
  Combine<<<dim3(24, p.tokens), 128, 0, s>>>(p, splits);
  return cudaGetLastError();
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
int flint_attention_splits(int capacity) { return capacity > 8192 ? 32 : 16; }
int flint_bidirectional_attention(AttentionParams p, int first, int splits, void* s) {
  return LaunchFlash<Mode::Draft>(p, first, splits, s);
}
int flint_attention_forward(AttentionParams p, void* s) {
  if (p.tokens < 1 || p.position < 0 || p.capacity < 1 ||
      (!p.sliding && p.position > p.capacity - p.tokens) || bool(p.k_new) != bool(p.v_new))
    return cudaErrorInvalidValue;
  if (!p.k_new) return LaunchFlash<Mode::Prewritten>(p, 0, 1, s);
  int splits = flint_attention_splits(p.capacity);
  if (splits > 1 && !p.scores) return cudaErrorInvalidValue;
  return LaunchFlash<Mode::Target>(p, 0, splits, s);
}
int flint_gated_mul_sigmoid(GateMulParams p, void* s) {
  Gate<<<(p.count + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}

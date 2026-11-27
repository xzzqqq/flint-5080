#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "loop_res_ops.h"
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

__device__ float State(ResidualParams p, int b, int t, int i) {
  if (b == p.num_blocks) return Read(p.partial, t * p.width + i, p.partial_type);
  return Read(p.blocks, (b * p.tokens + t) * p.width + i, p.block_type);
}
constexpr int kStates = 8;
constexpr int kWarps = 8;

template <int Queries>
__global__ void BlockResidual(ResidualParams p, int query_major) {
  const int t = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int nb = p.num_blocks + 1;
  // Row zero accumulates squared norms. Each subsequent row holds one query's dots.
  float sums[Queries + 1][kStates] = {};
  for (int c = threadIdx.x; c < p.width; c += blockDim.x) {
    float query[Queries];
#pragma unroll
    for (int q = 0; q < Queries; ++q) query[q] = p.query[q * p.width + c];
#pragma unroll
    for (int b = 0; b < kStates; ++b) {
      if (b < nb) {
        float x = State(p, b, t, c);
        sums[0][b] = fmaf(x, x, sums[0][b]);
#pragma unroll
        for (int q = 0; q < Queries; ++q) sums[q + 1][b] = fmaf(x, query[q], sums[q + 1][b]);
      }
    }
  }

  __shared__ float partials[Queries + 1][kWarps][kStates];
#pragma unroll
  for (int b = 0; b < kStates; ++b) {
    if (b < nb) {
#pragma unroll
      for (int r = 0; r <= Queries; ++r) {
        float v = sums[r][b];
#pragma unroll
        for (int d = 16; d; d /= 2) v += __shfl_down_sync(0xffffffff, v, d);
        if (lane == 0) partials[r][warp][b] = v;
      }
    }
  }
  __syncthreads();

  __shared__ float weights[Queries][kStates];
  // Each eight-lane group forms a complete softmax, including masked states.
  if (threadIdx.x < Queries * kStates) {
    int q = threadIdx.x / kStates, b = lane % kStates;
    unsigned mask = 0xffu << (lane - b);
    float inv = 0.f, score = -CUDART_INF_F;
    if (b < nb) {
      float ss = 0.f, dot = 0.f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        ss += partials[0][w][b];
        dot += partials[q + 1][w][b];
      }
      inv = rsqrtf(ss / p.width + 1e-6f);
      score = dot * inv * rsqrtf(float(p.width));
    }
    float peak = score;
    for (int d = kStates / 2; d; d /= 2)
      peak = fmaxf(peak, __shfl_xor_sync(mask, peak, d, kStates));
    float e = expf(score - peak), total = e;
    for (int d = kStates / 2; d; d /= 2) total += __shfl_xor_sync(mask, total, d, kStates);
    weights[q][b] = (e / total) * inv;
  }
  __syncthreads();

  for (int c = threadIdx.x; c < p.width; c += blockDim.x) {
    float out[Queries] = {};
    for (int b = 0; b < nb; ++b) {
      float x = State(p, b, t, c);
#pragma unroll
      for (int q = 0; q < Queries; ++q) out[q] = fmaf(weights[q][b], x, out[q]);
    }
#pragma unroll
    for (int q = 0; q < Queries; ++q) {
      int row = query_major ? q * p.tokens + t : t * Queries + q;
      p.y[row * p.width + c] = out[q];
    }
  }
}

__global__ void NormStates(ResidualParams p, float* states) {
  __shared__ float buf[512];
  int b = blockIdx.x % (p.num_blocks + 1), t = blockIdx.x / (p.num_blocks + 1);
  auto value = [&](int i) { return State(p, b, t, i); };
  float inv = flint::row::InvRms(p.width, value, buf);
  for (int i = threadIdx.x; i < p.width; i += blockDim.x)
    states[blockIdx.x * p.width + i] = value(i) * inv;
}
__global__ void Softmax(float* scores, int states, float scale) {
  int lane = threadIdx.x;
  scores += blockIdx.x * states;
  float v = lane < states ? scores[lane] * scale : -CUDART_INF_F;
  float peak = v;
  for (int d = 4; d; d /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xff, peak, d, 8));
  float e = expf(v - peak), sum = e;
  for (int d = 4; d; d /= 2) sum += __shfl_xor_sync(0xff, sum, d, 8);
  if (lane < states) scores[lane] = e / sum;
}

__global__ void Prep(PrepParams p) {
  __shared__ float buf[512];
  int t = blockIdx.x / 2;
  const float* src = (blockIdx.x % 2 ? p.message : p.x) + t * p.width;
  float inv = flint::row::InvRms(p.width, [&](int i) { return src[i]; }, buf);
  for (int i = threadIdx.x; i < p.width; i += blockDim.x)
    Write(p.y, blockIdx.x * p.width + i, src[i] * inv, FLINT_BF16);
}
__global__ void Mix(MixParams p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count)
    p.y[i] = __fadd_rn(__fmul_rn(1.f - tanhf(p.forget[i]), p.x[i]),
                       __fmul_rn(tanhf(p.gain[i]), p.message[i]));
}
__global__ void Merge(MergeParams p) {
  int row = blockIdx.x;
  const float* src = (row == 1 || row == 2 ? p.kv : p.qm) + (2 * p.layer + (row >= 2)) * p.width;
  for (int i = threadIdx.x; i < p.width; i += blockDim.x) p.y[row * p.width + i] = src[i];
}
}  // namespace
int flint_message_gate_prep(PrepParams p, void* s) {
  Prep<<<p.tokens * 2, flint::row::Threads(p.width), 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_message_gate_mix(MixParams p, void* s) {
  Mix<<<(p.count + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_merge_cross_pass_queries(MergeParams p, void* s) {
  if (p.layer < 0 || p.layer >= 12) return cudaErrorInvalidValue;
  Merge<<<4, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}

int flint_block_residual_fused(ResidualParams p, int query_major, void* s) {
  if (p.num_blocks < 0 || p.num_blocks >= kStates || p.width <= 0 || p.tokens <= 0)
    return cudaErrorInvalidValue;
  auto stream = static_cast<cudaStream_t>(s);
  switch (p.queries) {
    case 1:
      BlockResidual<1><<<p.tokens, kWarps * 32, 0, stream>>>(p, query_major);
      break;
    case 2:
      BlockResidual<2><<<p.tokens, kWarps * 32, 0, stream>>>(p, query_major);
      break;
    case 3:
      BlockResidual<3><<<p.tokens, kWarps * 32, 0, stream>>>(p, query_major);
      break;
    case 4:
      BlockResidual<4><<<p.tokens, kWarps * 32, 0, stream>>>(p, query_major);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

int flint_residual_norm_states(ResidualParams p, float* states, void* s) {
  NormStates<<<p.tokens*(p.num_blocks + 1), flint::row::Threads(p.width), 0,
               static_cast<cudaStream_t>(s)>>>(p, states);
  return cudaGetLastError();
}
int flint_residual_softmax(float* scores, int rows, int states, float scale, void* s) {
  if (states > 8) return cudaErrorInvalidValue;
  Softmax<<<rows, 8, 0, static_cast<cudaStream_t>(s)>>>(scores, states, scale);
  return cudaGetLastError();
}

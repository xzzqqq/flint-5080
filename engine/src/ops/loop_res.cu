#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "op_params.h"
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

__device__ float Reduce(float v, float* buf) {
  int i = threadIdx.x;
  buf[i] = v;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d /= 2) {
    if (i < d) buf[i] += buf[i + d];
    __syncthreads();
  }
  float sum = buf[0];
  __syncthreads();
  return sum;
}
__device__ float State(ResidualParams p, int b, int t, int i) {
  if (b == p.num_blocks) return Read(p.partial, t * p.width + i, p.partial_type);
  return Read(p.blocks, (b * p.tokens + t) * p.width + i, p.block_type);
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

__global__ void Scores(ResidualParams p) {
  __shared__ float buf[256];
  int row = blockIdx.x, b = row % (p.num_blocks + 1), q = row / (p.num_blocks + 1) % p.queries,
      t = row / (p.num_blocks + 1) / p.queries;
  float ss = 0;
  for (int i = threadIdx.x; i < p.width; i += 256) {
    float x = State(p, b, t, i);
    ss += x * x;
  }
  float inv = rsqrtf(Reduce(ss, buf) / p.width + 1e-6f);
  float dot = 0;
  for (int i = threadIdx.x; i < p.width; i += 256)
    dot += p.query[q * p.width + i] * (State(p, b, t, i) * inv);
  float score = Reduce(dot, buf) * rsqrtf((float)p.width);
  if (!threadIdx.x) {
    p.scratch[2 * row] = inv;
    p.scratch[2 * row + 1] = score;
  }
}
__global__ void Combine(ResidualParams p) {
  int row = blockIdx.x;
  float* stats = p.scratch + 2 * row * (p.num_blocks + 1);
  float peak = stats[1];
  for (int b = 1; b <= p.num_blocks; ++b) peak = fmaxf(peak, stats[2 * b + 1]);
  float total = 0;
  for (int b = 0; b <= p.num_blocks; ++b) total += expf(stats[2 * b + 1] - peak);
  for (int i = threadIdx.x; i < p.width; i += 256) {
    float y = 0;
    for (int b = 0; b <= p.num_blocks; ++b)
      y += (State(p, b, row / p.queries, i) * stats[2 * b]) *
           (expf(stats[2 * b + 1] - peak) / total);
    p.y[row * p.width + i] = y;
  }
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
int flint_block_attn_residual(ResidualParams p, void* s) {
  Scores<<<p.tokens * p.queries*(p.num_blocks + 1), 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  auto err = cudaGetLastError();
  if (err) return err;
  Combine<<<p.tokens * p.queries, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
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

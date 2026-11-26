#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include "dspark_ops.h"
#include "row_reduction.cuh"

namespace {
using B = __nv_bfloat16;
__device__ float read(const void* p, size_t i) {
  return __bfloat162float(static_cast<const B*>(p)[i]);
}
__device__ void write(void* p, size_t i, float x) {
  static_cast<B*>(p)[i] = __float2bfloat16_rn(x);
}
__device__ float round_bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__global__ void Mix(const void* taps, const float* a, const void* w, void* out) {
  __shared__ float mixed[2304], buf[512];
  int t = blockIdx.x;
  for (int i = threadIdx.x; i < 2304; i += blockDim.x) {
    float sum = 0;
    for (int j = 0; j < 4; ++j) sum += __fmul_rn(read(taps, (size_t(t) * 4 + j) * 2304 + i), a[j]);
    mixed[i] = sum;
  }
  __syncthreads();
  float inv = flint::row::InvRms(2304, [&](int i) { return mixed[i]; }, buf);
  for (int i = threadIdx.x; i < 2304; i += blockDim.x)
    write(out, t * 2304 + i, mixed[i] * inv * read(w, i));
}
__global__ void Embed(const void* e, const void* norm, const void* mask, int anchor, void* out) {
  __shared__ float buf[512];
  int t = blockIdx.x;
  auto value = [&](int i) { return read(e, anchor * 2304 + i); };
  float inv = t ? 0.f : flint::row::InvRms(2304, value, buf);
  for (int i = threadIdx.x; i < 2304; i += blockDim.x)
    write(out, t * 2304 + i, t ? read(mask, i) : value(i) * inv * read(norm, i));
}
__global__ void Rope(void* x, const void* w, int heads, int pos) {
  __shared__ float buf[32];
  int base = blockIdx.x * 128;
  float inv = flint::row::InvRms(128, [&](int i) { return read(x, base + i); }, buf);
  for (int i = threadIdx.x; i < 64; i += 32) {
    float a = round_bf(read(x, base + i) * inv * read(w, i));
    float b = round_bf(read(x, base + 64 + i) * inv * read(w, 64 + i));
    float phase = (pos + int(blockIdx.x) / heads) * (1.f / powf(1e6f, float(i) / 64));
    float s, c;
    sincosf(phase, &s, &c);
    write(x, base + i, __fmul_rn(a, c) - __fmul_rn(b, s));
    write(x, base + i + 64, __fmul_rn(b, c) + __fmul_rn(a, s));
  }
}
__global__ void WriteKv(DraftKv cache, const B* k, const B* v, int pos) {
  int t = blockIdx.x, slot = (pos + t) % cache.capacity;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    static_cast<B*>(cache.k)[slot * 1024 + i] = k[t * 1024 + i];
    static_cast<B*>(cache.v)[slot * 1024 + i] = v[t * 1024 + i];
  }
}
// Compute dense FP32 scores and normalize before rounding the value product.
__global__ void Scores(DraftKv cache, const B* q, float* scores, int first, int length, int row) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= length) return;
  auto base = (size_t((first + i) % cache.capacity) * 8 + (row % 24) / 3) * 128;
  float sum = 0;
  for (int d = 0; d < 128; ++d) sum = fmaf(float(q[row * 128 + d]), read(cache.k, base + d), sum);
  scores[i] = sum * 0.08838834764831845f;
}
__global__ void Normalize(float* rows, int length, float scale) {
  float* scores = rows + size_t(blockIdx.x) * length;
  int lane = threadIdx.x;
  float mx = -CUDART_INF_F;
  for (int i = lane; i < length; i += 32) mx = fmaxf(mx, scores[i] * scale);
  for (int d = 16; d; d /= 2) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
  float sum = 0;
  for (int i = lane; i < length; i += 32)
    sum += (scores[i] = expf(__fmul_rn(scores[i], scale) - mx));
  for (int d = 16; d; d /= 2) sum += __shfl_xor_sync(0xffffffff, sum, d);
  for (int i = lane; i < length; i += 32) scores[i] /= sum;
}
__global__ void Expand(const void* cache, float* tmp, int first, int capacity) {
  int t = blockIdx.x, head = blockIdx.y;
  for (int i = threadIdx.x; i < 128; i += blockDim.x)
    tmp[(t * 24 + head) * 128 + i] = read(cache, ((first + t) % capacity * 8 + head / 3) * 128 + i);
}
__global__ void QueryLayout(const B* q, float* tmp) {
  int i = blockIdx.x * 128 + threadIdx.x;
  tmp[i] = float(q[i]);
}
__global__ void OutputLayout(const float* tmp, B* out) {
  int head = blockIdx.x / 8, row = blockIdx.x % 8, d = threadIdx.x;
  out[(row * 24 + head) * 128 + d] = __float2bfloat16_rn(tmp[(head * 8 + row) * 128 + d]);
}
__global__ void Attend(DraftKv cache, const float* scores, B* out, int first, int length, int row) {
  int d = threadIdx.x;
  float sum = 0;
  for (int i = 0; i < length; ++i) {
    auto base = (size_t((first + i) % cache.capacity) * 8 + (row % 24) / 3) * 128;
    sum = fmaf(scores[i], read(cache.v, base + d), sum);
  }
  out[row * 128 + d] = __float2bfloat16_rn(sum);
}
__global__ void Binary(void* x, const void* y, int count, bool silu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    float a = read(x, i), b = read(y, i);
    write(x, i, silu ? (a / (1.f + expf(-a))) * b : a + b);
  }
}
__global__ void Gate(const float* rec, const float* ctx, void* state, void* latent) {
  int i = threadIdx.x;
  auto raw = [&](int offset) { return round_bf(ctx[offset + i] + rec[offset + i]); };
  float g = round_bf(1.f / (1.f + expf(-raw(0)))), c = round_bf(tanhf(raw(256)));
  write(state, i, round_bf(g * read(state, i)) + round_bf(round_bf(1.f - g) * c));
  write(latent, i, tanhf(raw(512)));
}
__global__ void Head(const void* a, const void* b, float* logits, int* id, int vocab) {
  __shared__ float best[256];
  __shared__ int ids[256];
  int lane = threadIdx.x, token = lane;
  float mx = -CUDART_INF_F;
  for (int i = lane; i < vocab; i += 256) {
    float y = round_bf(30.f * tanhf(round_bf(read(a, i) + read(b, i)) / 30.f));
    if (logits) logits[i] = y;
    if (y > mx) {
      mx = y;
      token = i;
    }
  }
  best[lane] = mx;
  ids[lane] = token;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (lane < d && (best[lane + d] > best[lane] ||
                     (best[lane + d] == best[lane] && ids[lane + d] < ids[lane]))) {
      best[lane] = best[lane + d];
      ids[lane] = ids[lane + d];
    }
    __syncthreads();
  }
  if (!lane) *id = ids[0];
}
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

int flint_dspark_mix_rmsnorm(const void* taps, const float* alpha, const void* weight, void* out,
                             int rows, void* stream) {
  Mix<<<rows, 512, 0, static_cast<cudaStream_t>(stream)>>>(taps, alpha, weight, out);
  return cudaGetLastError();
}
int flint_dspark_embed_block(const void* embed, const void* norm, const void* mask, int anchor,
                             void* block, void* stream) {
  Embed<<<8, 512, 0, static_cast<cudaStream_t>(stream)>>>(embed, norm, mask, anchor, block);
  return cudaGetLastError();
}
int flint_dspark_qk_norm_rope(void* x, const void* norm, int rows, int heads, int position,
                              void* stream) {
  Rope<<<rows * heads, 32, 0, static_cast<cudaStream_t>(stream)>>>(x, norm, heads, position);
  return cudaGetLastError();
}
int flint_dspark_kv_write(DraftKv cache, const void* k, const void* v, int rows, int pos,
                          void* stream) {
  WriteKv<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(cache, static_cast<const B*>(k),
                                                               static_cast<const B*>(v), pos);
  return cudaGetLastError();
}
int flint_dspark_block_attention(DraftKv cache, const void* q, void* out, float* scores,
                                 size_t workspace_bytes, void* blas, int ctx, void* stream) {
  int first = cache.local && ctx > 1024 ? ctx - 1024 : 0;
  int length = ctx + 8 - first;
  auto s = static_cast<cudaStream_t>(stream);
  // Materialize one FP32 KV plane at a time. For a long full context, reuse a
  // single score row so workspace remains bounded independently of cache size.
  size_t elems = size_t(length) * (192 + 3072) + 2 * 8 * 3072;
  if (elems <= workspace_bytes / sizeof(float)) {
    auto* expanded = scores + size_t(length) * 192;
    auto* queries = expanded + size_t(length) * 3072;
    auto* result = queries + 8 * 3072;
    QueryLayout<<<192, 128, 0, s>>>(static_cast<const B*>(q), queries);
    Expand<<<dim3(length, 24), 128, 0, s>>>(cache.k, expanded, first, cache.capacity);
    float one = 1, zero = 0;
    auto handle = static_cast<cublasHandle_t>(blas);
    if (cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, length, 8, 128, &one, expanded,
                                  3072, 128, queries, 3072, 128, &zero, scores, length, 8 * length,
                                  24) != CUBLAS_STATUS_SUCCESS)
      return cudaErrorUnknown;
    Normalize<<<192, 32, 0, s>>>(scores, length, 0.08838834764831845f);
    Expand<<<dim3(length, 24), 128, 0, s>>>(cache.v, expanded, first, cache.capacity);
    if (cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 128, 8, length, &one, expanded,
                                  3072, 128, scores, length, 8 * length, &zero, result, 128, 1024,
                                  24) != CUBLAS_STATUS_SUCCESS)
      return cudaErrorUnknown;
    OutputLayout<<<192, 128, 0, s>>>(result, static_cast<B*>(out));
    return cudaGetLastError();
  }
  if (size_t(length) > workspace_bytes / sizeof(float)) return cudaErrorMemoryAllocation;
  for (int row = 0; row < 192; ++row) {
    Scores<<<(length + 127) / 128, 128, 0, s>>>(cache, static_cast<const B*>(q), scores, first,
                                                length, row);
    Normalize<<<1, 32, 0, s>>>(scores, length, 1.f);
    Attend<<<1, 128, 0, s>>>(cache, scores, static_cast<B*>(out), first, length, row);
  }
  return cudaGetLastError();
}
int flint_dspark_silu_mul(void* gate, const void* up, int count, void* stream) {
  Binary<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, count, true);
  return cudaGetLastError();
}
int flint_dspark_add_bf16(void* x, const void* y, int count, void* stream) {
  Binary<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(x, y, count, false);
  return cudaGetLastError();
}
int flint_dspark_markov_gate(const float* recurrent, const float* context, void* state,
                             void* latent, void* stream) {
  Gate<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(recurrent, context, state, latent);
  return cudaGetLastError();
}
int flint_dspark_add_softcap_argmax(const void* a, const void* b, float* logits, int32_t* id,
                                    int vocab, void* stream) {
  Head<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(a, b, logits, id, vocab);
  return cudaGetLastError();
}
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

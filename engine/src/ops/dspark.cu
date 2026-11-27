#include <cuda_bf16.h>
#include <math_constants.h>

#include "attention_ops.h"
#include "dspark_ops.h"
#include "norm_ops.h"
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
__global__ void Embed(const void* e, const void* norm, const void* mask, int anchor, void* out,
                      const int32_t* d_anchor) {
  if (d_anchor) anchor = *d_anchor;
  __shared__ float buf[512];
  int t = blockIdx.x;
  auto value = [&](int i) { return read(e, anchor * 2304 + i); };
  float inv = t ? 0.f : flint::row::InvRms(2304, value, buf);
  for (int i = threadIdx.x; i < 2304; i += blockDim.x)
    write(out, t * 2304 + i, t ? read(mask, i) : value(i) * inv * read(norm, i));
}
__global__ void Rope(void* x, const void* w, int heads, int pos, const int32_t* d_pos) {
  if (d_pos) pos = *d_pos;
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
__global__ void WriteKv(DraftKv cache, const B* k, const B* v, int pos, const int32_t* d_pos) {
  if (d_pos) pos = *d_pos;
  int t = blockIdx.x, slot = (pos + t) % cache.capacity;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    static_cast<B*>(cache.k)[slot * 1024 + i] = k[t * 1024 + i];
    static_cast<B*>(cache.v)[slot * 1024 + i] = v[t * 1024 + i];
  }
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
__global__ void GatherMarkov(B* dst, const B* w1, const int32_t* prev, int32_t fallback) {
  int row = prev ? *prev : fallback;
  dst[threadIdx.x] = w1[row * 256 + threadIdx.x];
}
__global__ void Head(const void* a, const void* b, const void* offset, float* logits, int* id,
                     int vocab, HeadWinner* winners) {
  int first = blockIdx.x * blockDim.x + threadIdx.x;
  HeadWinner best{-CUDART_INF_F, first};
  for (int i = first; i < vocab; i += gridDim.x * blockDim.x) {
    float bias = read(b, i);
    if (offset) bias = round_bf(bias + read(offset, i));
    float y = round_bf(30.f * tanhf(round_bf(read(a, i) + bias) / 30.f));
    if (logits) logits[i] = y;
    if (y > best.value) best = {y, i};
  }
  best = flint::head::BlockWinner(best);
  if (threadIdx.x == 0) {
    if (winners)
      winners[blockIdx.x] = best;
    else
      *id = best.token;
  }
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
                             void* block, void* stream, const int32_t* d_anchor) {
  Embed<<<8, 512, 0, static_cast<cudaStream_t>(stream)>>>(embed, norm, mask, anchor, block,
                                                          d_anchor);
  return cudaGetLastError();
}
int flint_dspark_qk_norm_rope(void* x, const void* norm, int rows, int heads, int position,
                              void* stream, const int32_t* d_pos) {
  Rope<<<rows * heads, 32, 0, static_cast<cudaStream_t>(stream)>>>(x, norm, heads, position, d_pos);
  return cudaGetLastError();
}
int flint_dspark_kv_write(DraftKv cache, const void* k, const void* v, int rows, int pos,
                          void* stream, const int32_t* d_pos) {
  WriteKv<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      cache, static_cast<const B*>(k), static_cast<const B*>(v), pos, d_pos);
  return cudaGetLastError();
}
int flint_dspark_block_attention(DraftKv cache, const void* q, void* out, float* scores,
                                 size_t workspace_bytes, void*, int ctx, void* stream,
                                 const int32_t* d_ctx) {
  int splits = flint_attention_splits(cache.capacity);
  size_t bytes = size_t(splits) * 8 * 24 * FLINT_ATTENTION_PARTIAL * sizeof(float);
  if (d_ctx && (!scores || bytes > workspace_bytes)) return cudaErrorInvalidValue;
  if (!scores || bytes > workspace_bytes) splits = 1;
  AttentionParams p{q, cache.k, cache.v, nullptr, nullptr, out, scores, 8, ctx, cache.capacity};
  p.d_pos = d_ctx;
  p.sliding = cache.local;
  return flint_bidirectional_attention(p, 0, splits, stream);
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
int flint_dspark_gather_markov_w1(void* dst, const void* w1, const int32_t* prev, int32_t fallback,
                                  void* stream) {
  GatherMarkov<<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<B*>(dst), static_cast<const B*>(w1), prev, fallback);
  return cudaGetLastError();
}
int flint_dspark_add_softcap_argmax(const void* a, const void* b, const void* offset, float* logits,
                                    int32_t* id, int vocab, void* reduce_scratch, void* stream) {
  auto winners = static_cast<HeadWinner*>(reduce_scratch);
  int blocks = winners ? FLINT_HEAD_BLOCKS : 1;
  Head<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(a, b, offset, logits, id, vocab,
                                                              winners);
  auto status = cudaGetLastError();
  if (status || !winners) return status;
  return flint_reduce_head_winners(winners, id, 1, stream);
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

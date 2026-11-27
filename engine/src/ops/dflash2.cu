#include <cuda_bf16.h>
#include <math_constants.h>

#include <cub/block/block_radix_sort.cuh>

#include "attention_ops.h"
#include "common/splitmix64.h"
#include "dflash2_ops.h"
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
__global__ void Embed(const void* e, const void* norm, void* out, const int32_t* d_anchor) {
  int anchor = blockIdx.x ? 200056 : *d_anchor;
  __shared__ float buf[512];
  int t = blockIdx.x;
  auto value = [&](int i) { return read(e, anchor * 2304 + i); };
  float inv = flint::row::InvRms(2304, value, buf);
  for (int i = threadIdx.x; i < 2304; i += blockDim.x)
    write(out, t * 2304 + i, value(i) * inv * read(norm, i));
}
__global__ void Rope(void* x, const void* w, int heads, int pos, const int32_t* d_pos) {
  if (d_pos) pos = *d_pos;
  __shared__ float buf[32];
  int base = blockIdx.x * 128;
  float inv = flint::row::InvRms(128, [&](int i) { return read(x, base + i); }, buf);
  for (int i = threadIdx.x; i < 64; i += 32) {
    float a = round_bf(read(x, base + i) * inv * read(w, i));
    float b = round_bf(read(x, base + 64 + i) * inv * read(w, 64 + i));
    float phase = (pos + int(blockIdx.x) / heads) * (1.f / powf(10000.f, float(i) / 64));
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
}  // namespace
int flint_dflash2_embed_block(const void* embed, const void* norm, void* block, void* stream,
                              const int32_t* d_anchor) {
  Embed<<<8, 512, 0, static_cast<cudaStream_t>(stream)>>>(embed, norm, block, d_anchor);
  return cudaGetLastError();
}
int flint_dflash2_qk_norm_rope(void* x, const void* norm, int rows, int heads, int position,
                               void* stream, const int32_t* d_pos) {
  Rope<<<rows * heads, 32, 0, static_cast<cudaStream_t>(stream)>>>(x, norm, heads, position, d_pos);
  return cudaGetLastError();
}
int flint_dflash2_kv_write(DraftKv cache, const void* k, const void* v, int rows, int pos,
                           void* stream, const int32_t* d_pos) {
  WriteKv<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      cache, static_cast<const B*>(k), static_cast<const B*>(v), pos, d_pos);
  return cudaGetLastError();
}
int flint_dflash2_block_attention(DraftKv cache, const void* q, void* out, float* scores,
                                  size_t workspace_bytes, int ctx, void* stream,
                                  const int32_t* d_ctx) {
  int splits = flint_attention_splits(cache.capacity);
  size_t bytes = size_t(splits) * 8 * 32 * FLINT_ATTENTION_PARTIAL * sizeof(float);
  if (d_ctx && (!scores || bytes > workspace_bytes)) return cudaErrorInvalidValue;
  if (!scores || bytes > workspace_bytes) splits = 1;
  AttentionParams p{q, cache.k, cache.v, nullptr, nullptr, out, scores, 8, ctx, cache.capacity};
  p.d_pos = d_ctx;
  p.sliding = 1;
  return flint_bidirectional_attention(p, 0, splits, stream);
}
int flint_dflash2_silu_mul(void* gate, const void* up, int count, void* stream) {
  Binary<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, count, true);
  return cudaGetLastError();
}
int flint_dflash2_add_bf16(void* x, const void* y, int count, void* stream) {
  Binary<<<(count + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(x, y, count, false);
  return cudaGetLastError();
}

namespace {
__global__ void GroupedConv(const B* input, const B* delta, const B* base, int side, B* output) {
  int row = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= 2304) return;
  int group = c / 16;
  int b = side * 2 * 2304 + c, d = (row * 4 + side * 2) * 144 + group;
  float current = float(base[b]) + float(delta[d]);
  float y = current * float(input[row * 2304 + c]);
  if (row)
    y += (float(base[b + 2304]) + float(delta[d + 144])) * float(input[(row - 1) * 2304 + c]);
  output[row * 2304 + c] = __float2bfloat16_rn(y);
}

// Each tile emits its best 16. Packed keys order BF16 values and then inverted IDs.
constexpr int kTileSize = 1024, kVocab = 200058;
constexpr int kFirstTiles = (kVocab + kTileSize - 1) / kTileSize;
constexpr int kMergeTiles = (kFirstTiles * kSelectorTopK + kTileSize - 1) / kTileSize;
template <bool first, bool final>
__global__ void TopTiles(const B* logits, const uint64_t* input, int width, uint64_t* output,
                         ForwardDraftResult* result, float* unary) {
  using Sort = cub::BlockRadixSort<uint64_t, 128, 8>;
  __shared__ typename Sort::TempStorage tmp;
  uint64_t keys[8];
  for (int i = 0; i < 8; ++i) {
    int col = blockIdx.x * kTileSize + threadIdx.x + i * 128;
    keys[i] = 0;
    if (col >= width) continue;
    if constexpr (first) {
      unsigned bits = __bfloat16_as_ushort(logits[blockIdx.y * width + col]);
      if ((bits & 0x7fff) == 0) bits = 0;
      unsigned ordered = bits ^ ((bits & 0x8000) ? 0xffff : 0x8000);
      keys[i] = (uint64_t(ordered) << 32) | (~uint32_t(col));
    } else
      keys[i] = input[blockIdx.y * width + col];
  }
  Sort(tmp).SortDescendingBlockedToStriped(keys);
  if (threadIdx.x < 16) {
    int dest = (blockIdx.y * gridDim.x + blockIdx.x) * 16 + threadIdx.x;
    if constexpr (final) {
      int id = ~uint32_t(keys[0]);
      result->candidate_ids[dest] = id;
      unary[dest] = float(logits[blockIdx.y * kVocab + id]);
    } else
      output[dest] = keys[0];
  }
}
__global__ void Edges(const B* proj, const B* a, const B* b, const ForwardDraftSpec* spec,
                      const ForwardDraftResult* result, const float* unary, float* scores) {
  int e = blockIdx.z, p = blockIdx.y, c = blockIdx.x, lane = threadIdx.x;
  int index = (e * 16 + p) * 16 + c;
  if (!e && p) {
    if (!lane) scores[index] = 0;
    return;
  }
  int prev = e ? result->candidate_ids[(e - 1) * 16 + p] : spec->anchor;
  int next = result->candidate_ids[e * 16 + c];
  float dot = 0;
  for (int j = lane; j < 256; j += 32)
    dot += float(a[prev * 256 + j]) * float(proj[e * 256 + j]) * float(b[next * 256 + j]);
  for (int d = 16; d; d /= 2) dot += __shfl_down_sync(0xffffffff, dot, d);
  if (!lane) scores[index] = 30.f * tanhf((unary[e * 16 + c] + dot) / 30.f);
}
__global__ void Walk(const ForwardDraftSpec* spec, const float* scores,
                     ForwardDraftResult* result) {
  int prev = 0;
  uint64_t seed = spec->seed;
  for (int e = 0; e < 7; ++e) {
    auto row = scores + (e * 16 + prev) * 16;
    auto probs = result->q_probs + e * 16;
    int best = 0;
    for (int c = 1; c < 16; ++c)
      if (row[c] > row[best]) best = c;
    if (spec->temperature > 0) {
      double sum = 0;
      for (int c = 0; c < 16; ++c) {
        probs[c] = expf((row[c] - row[best]) / spec->temperature);
        sum += probs[c];
      }
      double mass = 0;
      for (int c = 0; c < 16; ++c) {
        probs[c] = float(probs[c] / sum);
        mass += probs[c];
      }
      double draw = flint::SplitMixUniform(seed) * mass;
      best = 0;
      for (int c = 0; c < 16; ++c) {
        if (probs[c] > 0) best = c;
        if ((draw -= probs[c]) < 0) break;
      }
    } else
      for (int c = 0; c < 16; ++c) probs[c] = c == best;
    prev = best;
    result->draft_ids[e] = result->candidate_ids[e * 16 + best];
  }
}
}  // namespace

int flint_dflash2_conv(const void* input, const void* delta, const void* base, int side,
                       void* output, void* stream) {
  GroupedConv<<<dim3(9, 8), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const B*>(input), static_cast<const B*>(delta), static_cast<const B*>(base), side,
      static_cast<B*>(output));
  return cudaGetLastError();
}
size_t flint_dflash2_topk_scratch_words() {
  return kDraftTokens * kSelectorTopK * (kFirstTiles + kMergeTiles);
}
int flint_dflash2_select(const void* logits, const void* projected, const void* predecessor,
                         const void* successor, const ForwardDraftSpec* spec,
                         ForwardDraftResult* result, uint64_t* scratch, float* unary, float* scores,
                         void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  auto stage2 = scratch + kDraftTokens * kFirstTiles * kSelectorTopK;
  auto raw = static_cast<const B*>(logits);
  TopTiles<true, false><<<dim3(kFirstTiles, kDraftTokens), 128, 0, s>>>(raw, nullptr, kVocab,
                                                                        scratch, result, unary);
  TopTiles<false, false><<<dim3(kMergeTiles, kDraftTokens), 128, 0, s>>>(
      raw, scratch, kFirstTiles * kSelectorTopK, stage2, result, unary);
  TopTiles<false, true>
      <<<dim3(1, 7), 128, 0, s>>>(raw, stage2, kMergeTiles * kSelectorTopK, nullptr, result, unary);
  Edges<<<dim3(16, 16, 7), 32, 0, s>>>(
      static_cast<const B*>(projected), static_cast<const B*>(predecessor),
      static_cast<const B*>(successor), spec, result, unary, scores);
  Walk<<<1, 1, 0, s>>>(spec, scores, result);
  return cudaGetLastError();
}

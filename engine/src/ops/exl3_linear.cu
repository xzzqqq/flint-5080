#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_ops.h"

namespace {
constexpr int kDenseKSplits = 12;
constexpr int kTileRows = 16;
constexpr int kColumnWarps = 4;
constexpr int kExpertKSplits = 12;

__device__ float Read(const void* x, size_t i, FlintType type) {
  if (type == FLINT_F32) return static_cast<const float*>(x)[i];
  if (type == FLINT_F16) return __half2float(static_cast<const __half*>(x)[i]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(x)[i]);
}
__device__ void Write(void* y, size_t i, float v, FlintType type) {
  if (type == FLINT_F32)
    static_cast<float*>(y)[i] = v;
  else if (type == FLINT_F16)
    static_cast<__half*>(y)[i] = __float2half_rn(v);
  else
    static_cast<__nv_bfloat16*>(y)[i] = __float2bfloat16_rn(v);
}

template <int K2>
__device__ __half DecodeValue(const unsigned* words, int step) {
  int start = ((step + 1) * K2 / 2 - 16 + 128 * K2) % (128 * K2);
  unsigned window =
      __funnelshift_l(words[(start / 32 + 1) % (4 * K2)], words[start / 32], start % 32);
  unsigned product = (window >> 16) * 0x83DCD12Du;
  int sum = __dp4a(product, 0x01010101u, 1024u);
  return __float2half_rn(sum * __half2float(__ushort_as_half(0x1eee)) +
                         __half2float(__ushort_as_half(0xc931)));
}
template <int K2>
__global__ void Decode(DecodeParams p) {
  int tile = blockIdx.x, step = threadIdx.x, n = p.n;
  auto words = reinterpret_cast<const unsigned*>(p.trellis) + tile * (4 * K2);
  int lane = step / 8, j = step % 8;
  int r = (tile / (n / 16)) * 16 + 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1);
  int c = (tile % (n / 16)) * 16 + lane / 4 + 8 * (j / 4);
  static_cast<__half*>(p.inner)[r * n + c] = DecodeValue<K2>(words, step);
}

__device__ unsigned PackedWord(const short* src, int i) {
  if ((reinterpret_cast<size_t>(src) & 3) == 0) return reinterpret_cast<const unsigned*>(src)[i];
  auto h = reinterpret_cast<const unsigned short*>(src) + 2 * i;
  return h[0] | (unsigned(h[1]) << 16);
}

template <bool SingleRow>
__device__ float& ExpandedWeight(float (&weight)[16][17], int k, int col) {
  if constexpr (!SingleRow) return weight[k][col];
  // The four K rows written at each decode step occupy disjoint eight-bank
  // spans. Shift even rows into the padding to compensate for stride 17.
  int parity = ((k >> 1) ^ (k >> 3)) & 1;
  int row = ((k & 4) << 1) | ((k & 8) >> 1) | ((k & 1) << 1) | parity;
  return weight[row][col + 1 - parity];
}

// Each lane decodes eight consecutive steps from the cyclic bitstream, then
// exchanges the expanded weights with the other lanes through shared memory.
template <int K2, bool SingleRow>
__device__ void ExpandTile(const short* src, float (&weight)[16][17]) {
  constexpr int words = 4 * K2;
  int lane = threadIdx.x % 32;
  unsigned history, upcoming;
  if constexpr (K2 == 12) {
    // Two lanes cover three words. Assemble the 16 preceding bits and
    // the 48 new bits as a pair of registers for each lane.
    int base = (lane / 2) * 3;
    bool odd = lane & 1;
    unsigned a = PackedWord(src, odd ? base + 1 : (base ? base - 1 : words - 1));
    unsigned b = PackedWord(src, base + (odd ? 2 : 0));
    unsigned peer = __shfl_xor_sync(0xffffffff, a, 1);
    history = odd ? a : __funnelshift_l(b, a, 16);
    upcoming = odd ? b : __funnelshift_l(peer, b, 16);
  } else {
    unsigned word = lane < words ? PackedWord(src, lane) : 0;
    int bit = lane * (4 * K2), idx = bit / 32;
    unsigned prev = __shfl_sync(0xffffffff, word, idx ? idx - 1 : words - 1);
    unsigned cur = __shfl_sync(0xffffffff, word, idx);
    unsigned next = __shfl_sync(0xffffffff, word, idx + 1);
    history = __funnelshift_l(cur, prev, bit & 31);
    upcoming = __funnelshift_l(next, cur, bit & 31);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int advance = (j + 1) * K2 / 2;
    unsigned code;
    if constexpr (K2 == 12)
      code = advance < 16 ? (history >> (16 - advance)) & 0xffff
                          : __funnelshift_lc(upcoming, history, advance - 16) & 0xffff;
    else
      code = __funnelshift_l(upcoming, history, advance) & 0xffff;
    unsigned sum = __dp4a(code * 0x83DCD12Du, 0x01010101u, 1024u);
    constexpr float scale = 0x1.bb8p-8f, bias = -0x1.4c4p+3f;
    int k = 2 * (lane % 4) + (j & 1) + ((j & 2) * 4);
    ExpandedWeight<SingleRow>(weight, k, lane / 4 + (j / 4) * 8) =
        __half2float(__float2half_rn(fmaf(float(sum), scale, bias)));
  }
  __syncwarp();
}

// The two half-warps share one output column. For a single input row they
// split the K work, otherwise each half-warp owns one row of each pair.
template <int K2, int TM>
__device__ __forceinline__ void DotTiles(const short* trellis, const float* x, int rows, int k,
                                         int n, int tile, int begin, int end,
                                         float (&weight)[16][17], float (&sums)[(TM + 1) / 2]) {
  int lane = threadIdx.x % 32, half = lane / 16, col = lane % 16;
  for (int kt = begin; kt < end; ++kt) {
    auto* src = trellis + (size_t(kt) * (n / 16) + tile) * (8 * K2);
    ExpandTile<K2, TM == 1>(src, weight);
#pragma unroll
    for (int pair = 0; pair < (TM + 1) / 2; ++pair) {
      int row = TM == 1 ? 0 : 2 * pair + half;
      if constexpr (TM > 1) {
        if (row >= rows) break;
      }
      constexpr int count = TM == 1 ? 8 : 16;
      int first = TM == 1 ? half * 8 : 0;
      auto* input = x + size_t(row) * k + kt * 16;
#pragma unroll
      for (int j = first; j < first + count; j += 4) {
        float4 v = *reinterpret_cast<const float4*>(input + j);
        sums[pair] = fmaf(v.x, ExpandedWeight<TM == 1>(weight, j, col), sums[pair]);
        sums[pair] = fmaf(v.y, ExpandedWeight<TM == 1>(weight, j + 1, col), sums[pair]);
        sums[pair] = fmaf(v.z, ExpandedWeight<TM == 1>(weight, j + 2, col), sums[pair]);
        sums[pair] = fmaf(v.w, ExpandedWeight<TM == 1>(weight, j + 3, col), sums[pair]);
      }
    }
    __syncwarp();
  }
  if constexpr (TM == 1) sums[0] += __shfl_xor_sync(0xffffffff, sums[0], 16);
}

__global__ void IndexedGemv(ExpertSelection selection, const float* x, float* partial, int k,
                            int n) {
  __shared__ float weight[kColumnWarps][16][17];
  int slot = blockIdx.z, split = blockIdx.y;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int tile = blockIdx.x * kColumnWarps + warp;
  int chunk = k / 16 / kExpertKSplits, begin = split * chunk;
  const auto& expert = selection.table[selection.ids[slot]];
  x += slot * k;
  float sum[1] = {};
  if (expert.k2 == 6)
    DotTiles<6, 1>(expert.trellis, x, 1, k, n, tile, begin, begin + chunk, weight[warp], sum);
  else
    DotTiles<7, 1>(expert.trellis, x, 1, k, n, tile, begin, begin + chunk, weight[warp], sum);
  if (lane < 16) partial[(slot * kExpertKSplits + split) * n + tile * 16 + lane] = sum[0];
}

template <int K2, int TM, int splits>
__device__ __forceinline__ void GroupedRows(GroupedExl3GemmParams p, const short* trellis,
                                            int begin, int rows, float (&weight)[16][17]) {
  int tile = blockIdx.x * kColumnWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  float sums[(TM + 1) / 2] = {};
  int split = splits > 1 ? blockIdx.z : 0;
  int chunk = p.k / 16 / splits;
  // When a large batch uses one output plane, perform the same four
  // reductions in registers as the verify kernel's four parallel splits.
  constexpr int inner_splits = splits == 1 ? FLINT_GROUPED_SMALL_SPLITS : 1;
  for (int part = 0; part < inner_splits; ++part) {
    float partial[(TM + 1) / 2] = {};
    int begin_k = split * chunk + part * chunk / inner_splits;
    int end_k = split * chunk + (part + 1) * chunk / inner_splits;
    DotTiles<K2, TM>(trellis, p.rotated + size_t(begin) * p.k, rows, p.k, p.n, tile, begin_k, end_k,
                     weight, partial);
#pragma unroll
    for (int pair = 0; pair < (TM + 1) / 2; ++pair) sums[pair] += partial[pair];
  }
#pragma unroll
  for (int pair = 0; pair < (TM + 1) / 2; ++pair) {
    int row = 2 * pair + lane / 16;
    if (row >= rows) break;
    p.projected[(size_t(begin + row) * splits + split) * p.n + tile * 16 + lane % 16] = sums[pair];
  }
}

template <int K2, bool small, int splits>
__device__ __forceinline__ void GroupedDot(GroupedExl3GemmParams p, const short* trellis, int begin,
                                           int rows, float (&weight)[16][17]) {
  // A later draft can select this expert too. Keep the dot reduction of
  // an earlier row independent of how many other rows route here.
  if (rows <= 4)
    GroupedRows<K2, 4, splits>(p, trellis, begin, rows, weight);
  else if (small || rows <= 8)
    GroupedRows<K2, 8, splits>(p, trellis, begin, rows, weight);
  else
    GroupedRows<K2, 16, splits>(p, trellis, begin, rows, weight);
}

template <bool small, int splits>
__device__ __forceinline__ void GroupedGemm(GroupedExl3GemmParams p) {
  __shared__ float weight[kColumnWarps][16][17];
  int tile = blockIdx.y;
  if (tile >= *p.num_tiles) return;
  int warp = threadIdx.x / 32;
  const auto& expert = p.table[p.tile_expert[tile]];
  int begin = p.tile_begin[tile], rows = p.tile_end[tile] - begin;
  if (expert.k2 == 6)
    GroupedDot<6, small, splits>(p, expert.trellis, begin, rows, weight[warp]);
  else
    GroupedDot<7, small, splits>(p, expert.trellis, begin, rows, weight[warp]);
}

template <int splits>
__global__ __launch_bounds__(128, 10) void GroupedGemmSmall(GroupedExl3GemmParams p) {
  GroupedGemm<true, splits>(p);
}
__global__ void GroupedGemmLarge(GroupedExl3GemmParams p) { GroupedGemm<false, 1>(p); }

template <int K2, int TM>
__global__ void DenseMatmul(DenseExl3Params p) {
  constexpr bool prefill = TM == 16;
  constexpr int warps = prefill ? kDenseKSplits : kColumnWarps;
  constexpr int row_pairs = (TM + 1) / 2;
  __shared__ float weight[warps][16][17];
  __shared__ float partial[prefill ? warps * TM * 16 : 1];
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32, col = lane % 16;
  int tile = prefill ? blockIdx.x % (p.n / 16) : blockIdx.x * warps + warp;
  int first_row = prefill ? (blockIdx.x / (p.n / 16)) * TM : 0;
  int tiles = p.k / 16;
  int split = prefill ? warp : blockIdx.y;
  // Prefill keeps its K partitions in one CTA and writes each output once.
  int chunk = tiles / (prefill ? warps : p.splits);
  int extra = prefill ? tiles % warps : 0;
  int begin = split * chunk + min(split, extra);
  int end = begin + chunk + (split < extra);
  float sums[row_pairs] = {};
  DotTiles<K2, TM>(p.trellis, p.rotated + size_t(first_row) * p.k, p.rows - first_row, p.k, p.n,
                   tile, begin, end, weight[warp], sums);
#pragma unroll
  for (int pair = 0; pair < row_pairs; ++pair) {
    int r = 2 * pair + lane / 16;
    if (first_row + r >= p.rows) break;
    if constexpr (prefill)
      partial[(warp * TM + r) * 16 + col] = sums[pair];
    else
      p.projected[(size_t(r) * p.splits + split) * p.n + tile * 16 + col] = sums[pair];
  }
  if constexpr (prefill) {
    __syncthreads();
    int r = threadIdx.x / 16;
    if (r < TM && first_row + r < p.rows) {
      float value = 0;
#pragma unroll
      for (int w = 0; w < warps; ++w) value += partial[(w * TM + r) * 16 + col];
      p.projected[size_t(first_row + r) * p.n + tile * 16 + col] = value;
    }
  }
}

template <int K2>
void LaunchDense(DenseExl3Params p, cudaStream_t stream) {
  dim3 grid(p.n / 64, p.splits);
  if (p.rows == 1) {
    DenseMatmul<K2, 1><<<grid, 128, 0, stream>>>(p);
  } else if (p.rows <= 8) {
    DenseMatmul<K2, 8><<<grid, 128, 0, stream>>>(p);
  } else {
    size_t blocks = ((size_t(p.rows) + 15) / 16) * (p.n / 16);
    DenseMatmul<K2, 16><<<blocks, 32 * kDenseKSplits, 0, stream>>>(p);
  }
}

__global__ void Wht(WhtParams p, bool scale_in, ExpertSelection experts, bool broadcast, int splits,
                    const int* row_map) {
  int slot = blockIdx.x / (p.width / 128);
  if (experts.table) {
    const auto& m = experts.table[experts.ids[slot]];
    p.scale = scale_in ? m.suh : m.svh;
  }
  __shared__ float values[128];
  size_t i = size_t(blockIdx.x) * 128 + threadIdx.x;
  int column = (blockIdx.x % (p.width / 128)) * 128 + threadIdx.x;
  float scale = Read(p.scale, column, FLINT_F16);
  float value;
  if (splits > 1) {
    const float* parts = static_cast<const float*>(p.x) + size_t(slot) * splits * p.width + column;
    value = 0;
    for (int j = 0; j < splits; ++j) value += parts[size_t(j) * p.width];
  } else {
    size_t source = broadcast ? column : i;
    if (scale_in && row_map) source = size_t(row_map[slot]) * p.width + column;
    value = Read(p.x, source, p.input);
  }
  if (scale_in) value *= scale;
  value *= 0.0883883461356163f;
#pragma unroll
  for (int stride = 1; stride < 32; stride *= 2) {
    float peer = __shfl_xor_sync(0xffffffff, value, stride);
    value = threadIdx.x & stride ? peer - value : value + peer;
  }
#pragma unroll
  for (int stride = 32; stride < 128; stride *= 2) {
    values[threadIdx.x] = value;
    __syncthreads();
    float peer = values[threadIdx.x ^ stride];
    value = threadIdx.x & stride ? peer - value : value + peer;
    if (stride == 32) __syncthreads();
  }
  Write(p.y, i, scale_in ? value : value * scale, p.output);
}
int Rotate(WhtParams p, void* s, bool in, ExpertSelection expert = {}, bool broadcast = false,
           int splits = 1, const int* row_map = nullptr) {
  if (p.rows <= 0 || p.width <= 0 || p.width % 128) return cudaErrorInvalidValue;
  Wht<<<p.rows*(p.width / 128), 128, 0, static_cast<cudaStream_t>(s)>>>(p, in, expert, broadcast,
                                                                        splits, row_map);
  return cudaGetLastError();
}
__global__ void Cast(CastParams p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) Write(p.y, i, Read(p.x, i, p.input), p.output);
}
}  // namespace
int flint_exl3_dense_splits(int rows, int k) {
  if (rows > 8 || k <= 0 || k % 128) return 1;
  return k % (16 * kDenseKSplits) == 0 ? kDenseKSplits : 8;
}
int flint_exl3_dense_matmul(DenseExl3Params p, void* s) {
  if (p.rows <= 0 || p.k <= 0 || p.k % 128 || p.n <= 0 || p.n % 128 || p.splits <= 0 ||
      (p.rows > 8 && p.splits != 1) || (p.k / 16) % p.splits)
    return cudaErrorInvalidValue;
  auto stream = static_cast<cudaStream_t>(s);
  switch (p.k2) {
    case 6:
      LaunchDense<6>(p, stream);
      break;
    case 7:
      LaunchDense<7>(p, stream);
      break;
    case 12:
      LaunchDense<12>(p, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
int flint_wht128_scale_out_split(WhtParams p, int splits, void* s) {
  if (splits <= 0 || p.input != FLINT_F32) return cudaErrorInvalidValue;
  return Rotate(p, s, false, {}, false, splits);
}
int flint_exl3_indexed_gemv(ExpertSelection experts, const float* rotated, float* partial, int k,
                            int n, void* s) {
  if (n <= 0 || n % 64 || k <= 0 || k % (16 * kExpertKSplits)) return cudaErrorInvalidValue;
  IndexedGemv<<<dim3(n / 64, kExpertKSplits, 8), 128, 0, static_cast<cudaStream_t>(s)>>>(
      experts, rotated, partial, k, n);
  return cudaGetLastError();
}
int flint_exl3_indexed_rotate(ExpertSelection experts, WhtParams p, int input, int broadcast,
                              void* s) {
  return Rotate(p, s, input, experts, broadcast, input ? 1 : kExpertKSplits);
}
int flint_exl3_grouped_rotate(ExpertSelection experts, WhtParams p, int input, const int* row_map,
                              int splits, void* s) {
  return Rotate(p, s, input, experts, false, splits, row_map);
}
int flint_exl3_grouped_gemm_tile_rows(void) { return kTileRows; }
int flint_exl3_grouped_gemm(GroupedExl3GemmParams p, void* s) {
  if (p.max_tiles <= 0 || p.n <= 0 || p.n % 64 || p.k <= 0 || p.k % 16)
    return cudaErrorInvalidValue;
  bool small = p.max_rows_per_tile > 0 && p.max_rows_per_tile <= 8;
  if (p.splits != 1 && (!small || p.splits != FLINT_GROUPED_SMALL_SPLITS || (p.k / 16) % p.splits))
    return cudaErrorInvalidValue;
  dim3 grid(p.n / 64, p.max_tiles, p.splits);
  auto stream = static_cast<cudaStream_t>(s);
  if (p.splits == FLINT_GROUPED_SMALL_SPLITS)
    GroupedGemmSmall<FLINT_GROUPED_SMALL_SPLITS><<<grid, 128, 0, stream>>>(p);
  else if (small)
    GroupedGemmSmall<1><<<grid, 128, 0, stream>>>(p);
  else
    GroupedGemmLarge<<<grid, 128, 0, stream>>>(p);
  return cudaGetLastError();
}
int flint_cast(CastParams p, void* s) {
  Cast<<<(p.count + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_exl3_decode_inner_f16(DecodeParams p, void* s) {
  if (p.k <= 0 || p.n <= 0 || p.k % 16 || p.n % 16) return cudaErrorInvalidValue;
  int tiles = p.k / 16 * (p.n / 16);
  switch (p.k2) {
    case 6:
      Decode<6><<<tiles, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
      break;
    case 7:
      Decode<7><<<tiles, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
      break;
    case 12:
      Decode<12><<<tiles, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
int flint_wht128_scale_in(WhtParams p, void* s) { return Rotate(p, s, true); }
int flint_wht128_scale_out(WhtParams p, void* s) { return Rotate(p, s, false); }

int flint_exl3_indexed_gemv_k_splits(void) { return kExpertKSplits; }

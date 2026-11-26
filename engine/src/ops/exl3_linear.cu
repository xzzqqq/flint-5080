#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_ops.h"

namespace {
// Phase 1 reference parity: preserve the 12-way split-K and 4x32 WHT reduction order.
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

template <int K2>
__device__ void LoadTrellisTile(const short* trellis, unsigned* tile, int index, int lane) {
  constexpr int words = 4 * K2;
  auto src = reinterpret_cast<const unsigned*>(trellis) + index * words;
  tile[lane] = src[lane];
  if (lane + 16 < words) tile[lane + 16] = src[lane + 16];
}

template <int K2>
__device__ float TileWeight(const unsigned* tile, int row, int lane) {
  int step = ((lane % 8) * 4 + (row % 8) / 2) * 8 + (lane / 8) * 4 + (row / 8) * 2 + row % 2;
  return __half2float(DecodeValue<K2>(tile, step));
}

// Each half warp expands a 16-column trellis tile. Keep the reference
// cuBLAS split-K layout: 12 contiguous partial sums of 96 or 192 inputs.
template <int K2>
__device__ void ExpertDot(const short* trellis, const float* x, float* partial, int k, int n) {
  constexpr int words_per_tile = 4 * K2;
  __shared__ unsigned tiles[8][words_per_tile];
  int lane = threadIdx.x % 16, tile = threadIdx.x / 16;
  int column = blockIdx.x * 128 + threadIdx.x;
  int chunk = k / kExpertKSplits, start = blockIdx.y * chunk;
  float sum = 0;
  for (int base = start; base < start + chunk; base += 16) {
    int index = (base / 16) * (n / 16) + column / 16;
    LoadTrellisTile<K2>(trellis, tiles[tile], index, lane);
    __syncwarp();
#pragma unroll
    for (int row = 0; row < 16; ++row) {
      sum = fmaf(x[base + row], TileWeight<K2>(tiles[tile], row, lane), sum);
    }
    __syncwarp();
  }
  partial[blockIdx.y * n + column] = sum;
}
__global__ void IndexedGemv(ExpertSelection selection, const float* x, float* partial, int k,
                            int n) {
  int slot = blockIdx.z;
  const auto& expert = selection.table[selection.ids[slot]];
  x += slot * k;
  partial += slot * kExpertKSplits * n;
  if (expert.k2 == 6)
    ExpertDot<6>(expert.trellis, x, partial, k, n);
  else
    ExpertDot<7>(expert.trellis, x, partial, k, n);
}

constexpr int kTileRows = 16;

template <int K2>
__device__ void GroupedDot(GroupedExl3GemmParams p, const short* trellis, int begin, int end) {
  __shared__ unsigned tiles[8][4 * K2];
  __shared__ float x_smem[kTileRows][16];
  int lane = threadIdx.x % 16, tile = threadIdx.x / 16;
  int col = blockIdx.x * 128 + threadIdx.x;
  int chunk = p.k / kExpertKSplits;
  float total[kTileRows] = {};
  for (int split = 0; split < kExpertKSplits; ++split) {
    float part[kTileRows] = {};
    for (int base = split * chunk; base < (split + 1) * chunk; base += 16) {
      LoadTrellisTile<K2>(trellis, tiles[tile], (base / 16) * (p.n / 16) + col / 16, lane);
#pragma unroll
      for (int r = tile; r < kTileRows; r += 8)
        x_smem[r][lane] = begin + r < end ? p.rotated[size_t(begin + r) * p.k + base + lane] : 0.f;
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        float weight = TileWeight<K2>(tiles[tile], j, lane);
#pragma unroll
        for (int r = 0; r < kTileRows; ++r) part[r] = fmaf(x_smem[r][j], weight, part[r]);
      }
      __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kTileRows; ++r) total[r] = __fadd_rn(total[r], part[r]);
  }
#pragma unroll
  for (int r = 0; r < kTileRows; ++r)
    if (begin + r < end) p.projected[size_t(begin + r) * p.n + col] = total[r];
}

__global__ void GroupedGemm(GroupedExl3GemmParams p) {
  int tile = blockIdx.y;
  if (tile >= *p.num_tiles) return;
  const auto& expert = p.table[p.tile_expert[tile]];
  int begin = p.tile_begin[tile], end = p.tile_end[tile];
  if (expert.k2 == 6)
    GroupedDot<6>(p, expert.trellis, begin, end);
  else
    GroupedDot<7>(p, expert.trellis, begin, end);
}

__global__ void Wht(WhtParams p, bool scale_in, ExpertSelection experts, bool broadcast,
                    bool split_input, const int* row_map) {
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
  if (split_input) {
    const float* splits =
        static_cast<const float*>(p.x) + size_t(slot) * kExpertKSplits * p.width + column;
    value = 0;
    for (int j = 0; j < kExpertKSplits; ++j) value += splits[j * p.width];
  } else {
    size_t source = broadcast ? column : i;
    if (scale_in && row_map) source = size_t(row_map[slot]) * p.width + column;
    value = Read(p.x, source, p.input);
  }
  values[threadIdx.x] = value * (scale_in ? scale : 1.f);
  __syncthreads();
  // The reference's padded FP32 GEMM splits K into four 32-element reductions.
  float sum = 0;
  for (int begin = 0; begin < 128; begin += 32) {
    float part = 0;
    for (int j = begin; j < begin + 32; ++j) {
      float h = (__popc(j & threadIdx.x) & 1) ? -0.0883883461356163f : 0.0883883461356163f;
      part = fmaf(values[j], h, part);
    }
    sum = __fadd_rn(sum, part);
  }
  Write(p.y, i, scale_in ? sum : sum * scale, p.output);
}
int Rotate(WhtParams p, void* s, bool in, ExpertSelection expert = {}, bool broadcast = false,
           bool split_input = false, const int* row_map = nullptr) {
  if (p.rows <= 0 || p.width <= 0 || p.width % 128) return cudaErrorInvalidValue;
  Wht<<<p.rows*(p.width / 128), 128, 0, static_cast<cudaStream_t>(s)>>>(p, in, expert, broadcast,
                                                                        split_input, row_map);
  return cudaGetLastError();
}
__global__ void Cast(CastParams p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) Write(p.y, i, Read(p.x, i, p.input), p.output);
}
}  // namespace
int flint_exl3_indexed_gemv(ExpertSelection experts, const float* rotated, float* partial, int k,
                            int n, void* s) {
  IndexedGemv<<<dim3(n / 128, kExpertKSplits, 8), 128, 0, static_cast<cudaStream_t>(s)>>>(
      experts, rotated, partial, k, n);
  return cudaGetLastError();
}
int flint_exl3_indexed_rotate(ExpertSelection experts, WhtParams p, int input, int broadcast,
                              void* s) {
  return Rotate(p, s, input, experts, broadcast, !input);
}
int flint_exl3_grouped_rotate(ExpertSelection experts, WhtParams p, int input, const int* row_map,
                              void* s) {
  return Rotate(p, s, input, experts, false, false, row_map);
}
int flint_exl3_grouped_gemm_tile_rows(void) { return kTileRows; }
int flint_exl3_grouped_gemm(GroupedExl3GemmParams p, void* s) {
  if (p.max_tiles <= 0 || p.n <= 0 || p.n % 128 || p.k <= 0 || p.k % (16 * kExpertKSplits))
    return cudaErrorInvalidValue;
  GroupedGemm<<<dim3(p.n / 128, p.max_tiles), 128, 0, static_cast<cudaStream_t>(s)>>>(p);
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

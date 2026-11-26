#include <cuda_bf16.h>

#include "moe_ops.h"

namespace {
using B = __nv_bfloat16;
__global__ void Route(RouterParams p) {
  __shared__ float rank[128], sigmoid[128], cutoff;
  int t = blockIdx.x, i = threadIdx.x;
  sigmoid[i] = 1.f / (1.f + expf(-p.logits[t * 128 + i]));
  float key = sigmoid[i] + p.bias[i];
  rank[i] = key;
  __syncthreads();
  int ahead = 0;
  for (int j = 0; j < 128; ++j) ahead += rank[j] > key || (rank[j] == key && j < i);
  if (ahead == 7) cutoff = key;
  __syncthreads();
  // CUDA topk(sorted=false) emits strict winners in index order, followed by
  // winners tied at the cutoff. The lowest indices win cutoff ties.
  int* ids = p.ids + t * 8;
  if (ahead < 8) {
    int slot = 0;
    for (int j = 0; j < 128; ++j) {
      if (key > cutoff)
        slot += j < i && rank[j] > cutoff;
      else
        slot += rank[j] > cutoff || (j < i && rank[j] == cutoff);
    }
    ids[slot] = i;
  }
  __syncthreads();
  if (i < 8) {
    float pair[4];
    for (int j = 0; j < 4; ++j) pair[j] = sigmoid[ids[j]] + sigmoid[ids[j + 4]];
    float total = (pair[0] + pair[2]) + (pair[1] + pair[3]);
    p.weights[t * 8 + i] = sigmoid[ids[i]] / (total + 1e-20f) * 8.f;
  }
}
__global__ void GroupTokens(MoeGroupParams p) {
  __shared__ int counts[128], cursor[128], first_tile[128];
  int e = threadIdx.x, slots = p.tokens * 8;
  counts[e] = 0;
  __syncthreads();
  for (int s = e; s < slots; s += 128) atomicAdd(&counts[p.ids[s]], 1);
  __syncthreads();
  if (e == 0) {
    int row = 0, tile = 0;
    for (int i = 0; i < 128; ++i) {
      cursor[i] = row;
      first_tile[i] = tile;
      row += counts[i];
      tile += (counts[i] + p.tile_rows - 1) / p.tile_rows;
    }
    *p.num_tiles = tile;
  }
  __syncthreads();
  int begin = cursor[e], end = begin + counts[e], tile = first_tile[e];
  for (int row = begin; row < end; row += p.tile_rows, ++tile) {
    p.tile_expert[tile] = e;
    p.tile_begin[tile] = row;
    p.tile_end[tile] = min(row + p.tile_rows, end);
  }
  __syncthreads();
  for (int s = e; s < slots; s += 128) {
    int expert = p.ids[s], row = atomicAdd(&cursor[expert], 1);
    p.grouped_expert[row] = expert;
    p.grouped_token[row] = s / 8;
    p.slot_to_grouped[s] = row;
  }
}
__global__ void Swiglu(SwigluParams p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.rows * p.width) return;
  int j = (i / p.width) * 2 * p.width + i % p.width;
  const B* x = static_cast<const B*>(p.gate_up);
  float g = __bfloat162float(x[j]), u = __bfloat162float(x[j + p.width]);
  static_cast<B*>(p.y)[i] = __float2bfloat16_rn((4.f * tanhf(g / 4.f)) * (1.f / (1.f + expf(-g))) *
                                                (25.f * tanhf(u * (1.f / 25.f))));
}
__global__ void Accum(AccumParams p) {
  __shared__ int slots[8];
  int t = blockIdx.x, lane = threadIdx.x;
  if (lane < 8) {
    int order = lane;
    if (p.ids) {
      const int* ids = p.ids + t * 8;
      order = 0;
      for (int j = 0; j < 8; ++j) order += ids[j] < ids[lane];
    }
    slots[order] = lane;
  }
  __syncthreads();
  for (int c = lane; c < p.width; c += blockDim.x) {
    float sum = 0;
    for (int order = 0; order < 8; ++order) {
      int slot = t * 8 + slots[order];
      int row = p.row_map ? p.row_map[slot] : slot;
      float x =
          __bfloat162float(static_cast<const B*>(p.expert_outputs)[size_t(row) * p.width + c]);
      sum = __fadd_rn(sum, __fmul_rn(x, p.weights[slot]));
    }
    static_cast<B*>(p.y)[size_t(t) * p.width + c] = __float2bfloat16_rn(sum);
  }
}
}  // namespace
int flint_moe_group_tokens(MoeGroupParams p, void* s) {
  if (p.tokens <= 0 || p.tile_rows <= 0) return cudaErrorInvalidValue;
  GroupTokens<<<1, 128, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_moe_router_top8(RouterParams p, void* s) {
  Route<<<p.tokens, 128, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_softcapped_swiglu(SwigluParams p, void* s) {
  Swiglu<<<(p.rows * p.width + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_moe_weighted_accum(AccumParams p, void* s) {
  Accum<<<p.tokens, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}

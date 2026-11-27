#pragma once
#include "op_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const void *x, *weight;
  void* y;
  const int* ids;
  int rows, width;
  FlintType input, output;
} NormParams;
enum { FLINT_HEAD_BLOCKS = 256 };
typedef struct {
  float value;
  int token;
} HeadWinner;

typedef struct {
  const void* x;
  float* logits;
  int* ids;
  int rows, vocab, stride;
  FlintType input;
#ifdef __cplusplus
  void* reduce_scratch = nullptr;
#else
  void* reduce_scratch;
#endif
} HeadParams;
int flint_reduce_head_winners(const HeadWinner* winners, int* ids, int rows, void* stream);
int flint_add_residual(const float* partial, const void* bf16, void* out, int count,
                       FlintType output, void* stream);
int flint_embed_rmsnorm(NormParams p, void* cuda_stream);
int flint_rmsnorm(NormParams p, void* cuda_stream);
int flint_unweighted_rmsnorm(NormParams p, void* cuda_stream);
int flint_logit_softcap_argmax(HeadParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
namespace flint::head {
// Both heads compare their final logits, including any model-specific rounding.
__device__ inline HeadWinner BlockWinner(HeadWinner winner) {
  __shared__ HeadWinner candidates[256];
  unsigned lane = threadIdx.x;
  candidates[lane] = winner;
  __syncthreads();
  for (unsigned d = 128; d; d >>= 1) {
    if (lane < d) {
      auto other = candidates[lane + d];
      if (other.value > winner.value || (other.value == winner.value && other.token < winner.token))
        candidates[lane] = winner = other;
    }
    __syncthreads();
  }
  return winner;  // Lane zero holds the block winner.
}
}  // namespace flint::head
#endif

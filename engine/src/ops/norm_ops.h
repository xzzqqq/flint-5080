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
typedef struct {
  const void* x;
  float* logits;
  int* ids;
  int rows, vocab, stride;
  FlintType input;
} HeadParams;
int flint_add_residual(const float* partial, const void* bf16, void* out, int count,
                       FlintType output, void* stream);
int flint_embed_rmsnorm(NormParams p, void* cuda_stream);
int flint_rmsnorm(NormParams p, void* cuda_stream);
int flint_unweighted_rmsnorm(NormParams p, void* cuda_stream);
int flint_logit_softcap_argmax(HeadParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

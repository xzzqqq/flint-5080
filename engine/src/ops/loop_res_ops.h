#pragma once
#include "op_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* blocks: contiguous [num_blocks,tokens,width]; partial: [tokens,width]. */
typedef struct {
  const void *blocks, *partial;
  const float* query;
  float* y;
  int num_blocks, tokens, width, queries;
  FlintType block_type, partial_type;
} ResidualParams;
typedef struct {
  const float *x, *message;
  void* y;
  int tokens, width;
} PrepParams;
typedef struct {
  const float *x, *message, *forget, *gain;
  float* y;
  int count;
} MixParams;
typedef struct {
  const float *qm, *kv;
  float* y;
  int layer, width;
} MergeParams;
int flint_residual_norm_states(ResidualParams p, float* states, void* stream);
int flint_residual_softmax(float* scores, int rows, int states, float scale, void* stream);
int flint_message_gate_prep(PrepParams p, void* cuda_stream);
int flint_message_gate_mix(MixParams p, void* cuda_stream);
int flint_merge_cross_pass_queries(MergeParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

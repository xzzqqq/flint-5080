#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Slots follow CUDA unsorted top-k order: strict winners by ID, then cutoff ties. */
typedef struct {
  const float *logits, *bias;
  int* ids;
  float* weights;
  int tokens;
} RouterParams;
typedef struct {
  const void* gate_up;
  void* y;
  int rows, width;
} SwigluParams;
/* expert_outputs: [tokens,8,width], in the same slot order as weights. */
typedef struct {
  const void* expert_outputs;
  const float* weights;
  void* y;
  int tokens, width;
  const int* ids; /* Optional. Accumulate in ascending expert order when provided. */
} AccumParams;
int flint_moe_router_top8(RouterParams p, void* cuda_stream);
int flint_softcapped_swiglu(SwigluParams p, void* cuda_stream);
int flint_moe_weighted_accum(AccumParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

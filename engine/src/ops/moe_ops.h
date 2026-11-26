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
typedef struct {
  const int* ids;                           /* [tokens * 8], token-major */
  int* grouped_expert;                      /* expert ID per grouped row */
  int* grouped_token;                       /* source token per grouped row */
  int* slot_to_grouped;                     /* inverse permutation of token-major slots */
  int *tile_expert, *tile_begin, *tile_end; /* half-open grouped row ranges */
  int* num_tiles;
  int tokens, tile_rows;
} MoeGroupParams;
typedef struct {
  const void* expert_outputs;
  const float* weights;
  void* y;
  int tokens, width;
  const int* ids;     /* Optional. Accumulate in ascending expert order when provided. */
  const int* row_map; /* Optional token-major slot -> expert_outputs row. */
} AccumParams;
int flint_moe_group_tokens(MoeGroupParams p, void* cuda_stream);
int flint_moe_router_top8(RouterParams p, void* cuda_stream);
int flint_softcapped_swiglu(SwigluParams p, void* cuda_stream);
int flint_moe_weighted_accum(AccumParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

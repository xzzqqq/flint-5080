#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const void *x, *scale;
  void* y;
  int tokens, position, log_scale;
  const int32_t* d_pos;
} ClipParams;
/* Caller selects one visit's cache planes. For ring writes, only the last capacity slots survive.
 */
typedef struct {
  const void *k, *v;
  void *cache_k, *cache_v;
  int tokens, position, capacity, sliding;
  const int32_t* d_pos;
} CacheParams;
/* scores: tokens * 24 * capacity floats. Cache must retain every visible key. */
typedef struct {
  const void *q, *cache_k, *cache_v, *rel, *sink;
  void* y;
  float* scores;
  int tokens, position, capacity, sliding;
  const int32_t* d_pos;
} AttentionParams;
typedef struct {
  const void *x, *gate;
  void* y;
  int count;
} GateMulParams;
int flint_q_norm_clip(ClipParams p, void* cuda_stream);
int flint_k_l2_clip(ClipParams p, void* cuda_stream);
int flint_rel_norm_clip(ClipParams p, void* cuda_stream);
int flint_kv_cache_write(CacheParams p, void* cuda_stream);
int flint_attention_forward(AttentionParams p, void* cuda_stream);
int flint_gated_mul_sigmoid(GateMulParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

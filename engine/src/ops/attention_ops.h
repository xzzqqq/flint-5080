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
  int pos_offset;  // Chunk offset when d_pos supplies the request position.
} ClipParams;
/* Caller selects one visit's cache planes. For ring writes, only the last capacity slots survive.
 */
typedef struct {
  const void *k, *v;
  void *cache_k, *cache_v;
  int tokens, position, capacity, sliding;
  const int32_t* d_pos;
  int pos_offset;  // Chunk offset when d_pos supplies the request position.
} CacheParams;
/* Split scratch: splits * tokens * 24 * FLINT_ATTENTION_PARTIAL floats.
 * Direct output and callers using only the prewritten cache need no scratch. */
enum { FLINT_ATTENTION_PARTIAL = 130 };
typedef struct {
  const void *q, *cache_k, *cache_v, *rel, *sink;
  void* y;
  float* scores;
  int tokens, position, capacity, sliding;
  const int32_t* d_pos;
#ifdef __cplusplus
  const void *k_new = nullptr, *v_new = nullptr;
#else
  const void *k_new, *v_new;
#endif
  int pos_offset;
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
int flint_attention_splits(int capacity);
int flint_bidirectional_attention(AttentionParams p, int first, int splits, void* cuda_stream);
int flint_attention_forward(AttentionParams p, void* cuda_stream);
int flint_gated_mul_sigmoid(GateMulParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

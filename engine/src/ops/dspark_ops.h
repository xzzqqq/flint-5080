#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int flint_dspark_mix_rmsnorm(const void* taps, const float* alpha, const void* weight, void* out,
                             int rows, void* stream);
int flint_dspark_embed_block(const void* embed, const void* norm, const void* mask, int anchor,
                             void* block, void* stream);
int flint_dspark_qk_norm_rope(void* x, const void* norm, int rows, int heads, int position,
                              void* stream);
typedef struct DraftKv {
  void *k, *v;
  int capacity, local;
} DraftKv;
int flint_dspark_kv_write(DraftKv cache, const void* k, const void* v, int rows, int pos,
                          void* stream);
int flint_dspark_block_attention(DraftKv cache, const void* q, void* out, float* scores,
                                 size_t workspace_bytes, void* blas, int ctx, void* stream);
int flint_dspark_silu_mul(void* gate, const void* up, int count, void* stream);
int flint_dspark_add_bf16(void* x, const void* y, int count, void* stream);
int flint_dspark_markov_gate(const float* recurrent, const float* context, void* state,
                             void* latent, void* stream);
int flint_dspark_add_softcap_argmax(const void* a, const void* b, float* logits, int32_t* id,
                                    int vocab, void* stream);
int flint_swa_kv_backup(void* ring, void* backup, int pos, int count, void* stream);
int flint_swa_kv_restore(void* ring, void* backup, int pos, int count, int accepted, void* stream);
int flint_sconv_commit_trace(const float* trace, float* state, int channels, int count,
                             int accepted, void* stream);
#ifdef __cplusplus
}
#endif

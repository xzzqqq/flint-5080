#pragma once
#include <stddef.h>
#include <stdint.h>

#include "draft_types.h"
#ifdef __cplusplus
extern "C" {
#define FLINT_DFLASH2_OPTIONAL = nullptr
#else
#define FLINT_DFLASH2_OPTIONAL
#endif
int flint_dflash2_embed_block(const void* embed, const void* norm, void* block, void* stream,
                              const int32_t* d_anchor);
int flint_dflash2_qk_norm_rope(void* x, const void* norm, int rows, int heads, int position,
                               void* stream, const int32_t* d_pos FLINT_DFLASH2_OPTIONAL);
typedef struct DraftKv {
  void *k, *v;
  int capacity;
} DraftKv;
int flint_dflash2_kv_write(DraftKv cache, const void* k, const void* v, int rows, int pos,
                           void* stream, const int32_t* d_pos FLINT_DFLASH2_OPTIONAL);
int flint_dflash2_block_attention(DraftKv cache, const void* q, void* out, float* scores,
                                  size_t workspace_bytes, int ctx, void* stream,
                                  const int32_t* d_ctx FLINT_DFLASH2_OPTIONAL);
int flint_dflash2_silu_mul(void* gate, const void* up, int count, void* stream);
int flint_dflash2_add_bf16(void* x, const void* y, int count, void* stream);
int flint_dflash2_conv(const void* input, const void* delta, const void* base, int side,
                       void* output, void* stream);
size_t flint_dflash2_topk_scratch_words(void);
int flint_dflash2_select(const void* logits, const void* projected, const void* predecessor,
                         const void* successor, const ForwardDraftSpec* spec,
                         ForwardDraftResult* result, uint64_t* scratch, float* unary, float* scores,
                         void* stream);
#ifdef __cplusplus
}
#endif

#undef FLINT_DFLASH2_OPTIONAL

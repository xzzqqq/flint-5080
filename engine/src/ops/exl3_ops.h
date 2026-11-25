#pragma once
#include "op_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* EXL3 trellis is [k/16,n/16,8*k2] I16; inner, suh and svh are F16. */
typedef struct {
  const short* trellis;
  void* inner;
  int k, n, k2;
} DecodeParams;
typedef struct {
  const void *x, *scale;
  void* y;
  int rows, width;
  FlintType input, output;
} WhtParams;
typedef struct {
  const void* x;
  void* y;
  int count;
  FlintType input, output;
} CastParams;
typedef struct {
  const short* trellis;
  const void *suh, *svh;
  int k2;
} ExpertProjection;
typedef struct {
  const ExpertProjection* table;
  const int* ids;
} ExpertSelection;
int flint_exl3_indexed_gemv(ExpertSelection experts, const float* rotated, float* partial, int k,
                            int n, void* stream);
int flint_exl3_indexed_rotate(ExpertSelection expert, WhtParams p, int input, int broadcast,
                              void* stream);
int flint_exl3_decode_inner_f16(DecodeParams p, void* cuda_stream);
int flint_wht128_scale_in(WhtParams p, void* cuda_stream);
int flint_wht128_scale_out(WhtParams p, void* cuda_stream);
int flint_cast(CastParams p, void* cuda_stream);
int flint_exl3_indexed_gemv_k_splits(void);

#ifdef __cplusplus
}
#endif

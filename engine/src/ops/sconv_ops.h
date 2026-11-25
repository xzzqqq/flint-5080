#pragma once
#include "op_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Static weight [channels,4] is oldest first. Dynamic weight [4,channels]
   and delta [tokens,8,4] are newest first. State [channels,3] is oldest first. */
typedef struct {
  const void *x, *weight, *delta;
  float* state;
  void* y;
  int tokens, channels;
  FlintType input, output;
} ConvParams;
int flint_static_sconv(ConvParams p, void* cuda_stream);
int flint_dynamic_sconv(ConvParams p, void* cuda_stream);

#ifdef __cplusplus
}
#endif

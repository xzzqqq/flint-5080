#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SmokeAddParams {
  const float* input;
  float* output;
  int n;
  float bias;
} SmokeAddParams;
/* Hidden and cache elements are BF16. The host selects this visit's state. */
typedef struct StubParams {
  const int* ids;
  const uint16_t* input;
  uint16_t *output, *k, *v;
  float *conv, *logits;
  int* next;
  unsigned width, count, vocab, position, visit;
} StubParams;
int flint_stub_embed(StubParams params, void* stream);
int flint_stub_visit(StubParams params, void* stream);
int flint_stub_head(StubParams params, void* stream);
int flint_launch_smoke_add_f32(const SmokeAddParams* params, void* cuda_stream);

#ifdef __cplusplus
}
#endif

#ifndef FLINT_OP_PARAMS_H
#define FLINT_OP_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SmokeAddParams {
  const float* input;
  float* output;
  int n;
  float bias;
} SmokeAddParams;

/* The stub reserves one F32 recurrent value per visit in each state arena. */
typedef struct StubParams {
  const int* ids;
  const float* input;
  float *output, *kv, *conv, *logits;
  int* next;
  const unsigned short* weight;
  unsigned width, count, vocab, position, visit;
} StubParams;
int flint_stub_embed(StubParams params, void* stream);
int flint_stub_visit(StubParams params, void* stream);
int flint_stub_head(StubParams params, void* stream);

/* Returns the CUDA launch status (zero on success). */
int flint_launch_smoke_add_f32(const SmokeAddParams* params, void* cuda_stream);

#ifdef __cplusplus
}
#endif
#endif

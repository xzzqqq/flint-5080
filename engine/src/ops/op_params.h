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

/* Returns the CUDA launch status (zero on success). */
int flint_launch_smoke_add_f32(const SmokeAddParams* params, void* cuda_stream);

#ifdef __cplusplus
}
#endif
#endif

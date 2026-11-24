#ifndef FLINT_FORWARD_ABI_H
#define FLINT_FORWARD_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLINT_FORWARD_ABI_VERSION 1u

typedef struct ForwardContext ForwardContext;

/* Two adjacent arrays of n_elems floats: initialized input, then output. */
typedef struct ForwardWorkspace {
  void* device_ptr;
  size_t bytes;
  void* cuda_stream;
} ForwardWorkspace;

typedef struct ForwardOptions {
  uint8_t use_cuda_graph;
  uint8_t use_speculative;
  uint8_t return_logits;
} ForwardOptions;

typedef struct ForwardSmokeOutput {
  float elapsed_ms;
  float checksum;
  int32_t status;
} ForwardSmokeOutput;

uint32_t flint_forward_abi_version(void);
ForwardContext* flint_forward_create(void);
/* Synchronous. Returns zero on success. Each context runs one call at a time. */
int32_t flint_forward_run_smoke(ForwardContext* context, const ForwardWorkspace* workspace,
                                const ForwardOptions* options, int32_t n_elems, float bias,
                                ForwardSmokeOutput* result);
void flint_forward_destroy(ForwardContext* context);

#ifdef __cplusplus
}
#endif
#endif

#ifndef FLINT_FORWARD_ABI_H
#define FLINT_FORWARD_ABI_H

#include <stddef.h>
#include <stdint.h>

#include "loader/model_views.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FLINT_FORWARD_ABI_VERSION 2u

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
  uint8_t enable_profiling;
  uint8_t profiler_range;
} ForwardOptions;

#define FLINT_MAX_STAGE_TIMINGS 64
typedef struct ForwardStageTiming {
  char name[32];
  float elapsed_ms;
  uint64_t traffic_bytes;
} ForwardStageTiming;

/* M1.3 hidden buffers are F32 [seq_len, hidden_size]; logits cover the last token.
   Block streams are reserved for the M1.4 residual attention implementation. */
typedef struct ForwardRunSpec {
  uint32_t begin_visit, end_visit;
  uint8_t run_embed, run_lm_head, is_decode, reset_kv;
  uint32_t seq_len, pos_offset;
  const int32_t* d_input_ids;
  const void* d_hidden_in;
  const void* d_block_in;
  uint32_t num_block_in;
  void* d_hidden_out;
  void* d_block_out;
  uint32_t* num_block_out;
  float* d_logits_out;
  int32_t* d_next_token;
} ForwardRunSpec;

typedef struct ForwardRunResult {
  int32_t status;
  float elapsed_ms;
  uint64_t logical_bytes;
  int32_t next_token_id;
  uint32_t num_stages;
  ForwardStageTiming stages[FLINT_MAX_STAGE_TIMINGS];
} ForwardRunResult;

typedef struct ForwardSmokeOutput {
  float elapsed_ms;
  float checksum;
  int32_t status;
} ForwardSmokeOutput;

uint32_t flint_forward_abi_version(void);
ForwardContext* flint_forward_create(const ModelWeightsView* weights, const KvStateArenaView* kv,
                                     const ScratchArenaView* scratch, void* cuda_stream);
int32_t flint_forward_run_layers(ForwardContext* context, const ForwardRunSpec* spec,
                                 const ForwardOptions* options, ForwardRunResult* result);
/* Synchronous. Returns zero on success. Each context runs one call at a time. */
int32_t flint_forward_run_smoke(ForwardContext* context, const ForwardWorkspace* workspace,
                                const ForwardOptions* options, int32_t n_elems, float bias,
                                ForwardSmokeOutput* result);
void flint_forward_destroy(ForwardContext* context);

#ifdef __cplusplus
}
#endif
#endif

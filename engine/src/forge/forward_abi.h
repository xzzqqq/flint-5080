#ifndef FLINT_FORWARD_ABI_H
#define FLINT_FORWARD_ABI_H

#include "model/forward_types.h"
#include "ops/draft_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FLINT_FORWARD_ABI_VERSION 6u

typedef struct ForwardContext ForwardContext;

/* Two adjacent arrays of n_elems floats: initialized input, then output. */
typedef struct ForwardWorkspace {
  void* device_ptr;
  size_t bytes;
  void* cuda_stream;
} ForwardWorkspace;

typedef struct ForwardSmokeOutput {
  float elapsed_ms;
  float checksum;
  int32_t status;
} ForwardSmokeOutput;

uint32_t flint_forward_abi_version(void);
/* The host may reuse all scratch bytes between runs, including across loaded modules. */
ForwardContext* flint_forward_create(const ModelWeightsView* weights, const KvStateArenaView* kv,
                                     const ScratchArenaView* scratch, void* cuda_stream);
int32_t flint_forward_run_layers(ForwardContext* context, const ForwardRunSpec* spec,
                                 const ForwardOptions* options, ForwardRunResult* result);
/* Optional activation capture. */
int32_t flint_forward_run_layers_capture(ForwardContext* context, const ForwardRunSpec* spec,
                                         const ForwardOptions* options, ForwardRunResult* result,
                                         const ForwardCapture* capture);
/* Synchronous. Returns zero on success. Each context runs one call at a time. */
int32_t flint_forward_run_smoke(ForwardContext* context, const ForwardWorkspace* workspace,
                                const ForwardOptions* options, int32_t n_elems, float bias,
                                ForwardSmokeOutput* result);
int32_t flint_forward_store_context(ForwardContext* c, const void* taps, uint32_t count,
                                    uint32_t pos);
/* Host spec and result. Returns after all seven conditional proposal rows are ready. */
int32_t flint_forward_draft(ForwardContext* c, const ForwardDraftSpec* spec,
                            ForwardDraftResult* result);
int32_t flint_forward_verify(ForwardContext* c, const ForwardRunSpec* spec,
                             const ForwardOptions* opts, ForwardRunResult* result);
int32_t flint_forward_commit(ForwardContext* c, uint32_t pos, uint32_t len, uint32_t accepted);
/* Optional graph diagnostics. */
ForwardGraphStats flint_forward_graph_stats(const ForwardContext* context);
void flint_forward_destroy(ForwardContext* context);

#ifdef __cplusplus
}
#endif
#endif

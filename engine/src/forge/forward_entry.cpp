#include "common/cuda_check.h"
#include "forge/forward_abi.h"
#include "model/inkling2_forward.h"
struct ForwardContext {
  flint::InklingRunner runner;
};
uint32_t flint_forward_abi_version() { return FLINT_FORWARD_ABI_VERSION; }
ForwardContext* flint_forward_create(const ModelWeightsView* weights, const KvStateArenaView* kv,
                                     const ScratchArenaView* scratch, void* stream) {
  if (!weights || !kv || !scratch) return nullptr;
  try {
    return new ForwardContext{{*weights, *kv, *scratch, static_cast<cudaStream_t>(stream)}};
  } catch (const std::exception&) {
    return nullptr;
  }
}
void flint_forward_destroy(ForwardContext* c) { delete c; }
ForwardGraphStats flint_forward_graph_stats(const ForwardContext* c) {
  return c->runner.graph_stats();
}
int32_t flint_forward_run_layers_capture(ForwardContext* c, const ForwardRunSpec* s,
                                         const ForwardOptions* o, ForwardRunResult* r,
                                         const ForwardCapture* capture) {
  if (!r) return cudaErrorInvalidValue;
  *r = {.status = cudaErrorInvalidValue, .next_token_id = -1};
  if (!c || !s || !o) return r->status;
  try {
    *r = c->runner.Run(*s, *o, capture);
  } catch (const flint::CudaError& err) {
    r->status = err.code();
  } catch (const std::length_error&) {
    r->status = cudaErrorMemoryAllocation;
  } catch (const std::invalid_argument&) {
    r->status = cudaErrorInvalidValue;
  } catch (const std::exception&) {
    r->status = cudaErrorUnknown;
  }
  return r->status;
}
int32_t flint_forward_run_layers(ForwardContext* c, const ForwardRunSpec* s,
                                 const ForwardOptions* o, ForwardRunResult* r) {
  return flint_forward_run_layers_capture(c, s, o, r, nullptr);
}

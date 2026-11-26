#include <optional>

#include "common/cuda_check.h"
#include "forge/forward_abi.h"
#include "loader/scratch_workspace.h"
#include "model/dspark_forward.h"
#include "model/inkling2_forward.h"
namespace {
ScratchArenaView TargetScratch(ScratchArenaView scratch, bool mtp) {
  if (!mtp) return scratch;
  flint::ScratchWorkspace ws(scratch);
  ws.Take<char>(32ul << 20);
  return ws.rest;
}
}  // namespace
struct ForwardContext {
  flint::InklingRunner runner;
  std::optional<flint::DsparkRunner> draft;
  ForwardContext(const ModelWeightsView& w, const KvStateArenaView& kv, ScratchArenaView ws,
                 cudaStream_t s)
      : runner(w, kv, TargetScratch(ws, w.has_mtp), s) {
    if (w.has_mtp)
      draft.emplace(w, kv, ScratchArenaView{ws.device_ptr, 32ul << 20}, runner.stream());
  }
};
uint32_t flint_forward_abi_version() { return FLINT_FORWARD_ABI_VERSION; }
ForwardContext* flint_forward_create(const ModelWeightsView* weights, const KvStateArenaView* kv,
                                     const ScratchArenaView* scratch, void* stream) {
  if (!weights || !kv || !scratch) return nullptr;
  try {
    return new ForwardContext(*weights, *kv, *scratch, static_cast<cudaStream_t>(stream));
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
namespace {
template <class F>
int DraftCall(ForwardContext* c, F call) {
  if (!c || !c->draft) return cudaErrorInvalidValue;
  try {
    return call();
  } catch (const flint::CudaError& e) {
    return e.code();
  } catch (const std::length_error&) {
    return cudaErrorMemoryAllocation;
  } catch (const std::invalid_argument&) {
    return cudaErrorInvalidValue;
  } catch (const std::exception&) {
    return cudaErrorUnknown;
  }
}
}  // namespace
int32_t flint_forward_store_context(ForwardContext* c, const void* taps, uint32_t count,
                                    uint32_t pos) {
  return DraftCall(c, [&]() {
    c->draft->StoreContextKv(taps, count, pos);
    return 0;
  });
}
int32_t flint_forward_draft(ForwardContext* c, int32_t anchor, const void* tap, uint32_t pos,
                            int32_t* h_ids) {
  return DraftCall(c, [&]() {
    c->draft->DraftGreedy(anchor, tap, pos, h_ids);
    return 0;
  });
}
int32_t flint_forward_verify(ForwardContext* c, const ForwardRunSpec* spec,
                             const ForwardOptions* opts, ForwardRunResult* result) {
  return DraftCall(c, [&]() {
    if (!spec || !opts || !result) return int(cudaErrorInvalidValue);
    *result = c->runner.RunVerify(*spec, *opts);
    return 0;
  });
}
int32_t flint_forward_commit(ForwardContext* c, uint32_t pos, uint32_t len, uint32_t accepted) {
  return DraftCall(c, [&]() {
    c->runner.CommitVerify(pos, len, accepted);
    return 0;
  });
}

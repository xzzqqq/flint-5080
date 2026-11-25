#include <algorithm>
#include <iterator>

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
  auto stats = c->runner.graph_stats();
  return {stats.captures, stats.launches};
}
int32_t flint_forward_run_layers_capture(ForwardContext* c, const ForwardRunSpec* s,
                                         const ForwardOptions* o, ForwardRunResult* r,
                                         const ForwardCapture* capture) {
  if (!r) return cudaErrorInvalidValue;
  *r = {.status = cudaErrorInvalidValue, .next_token_id = -1};
  if (!c || !s || !o) return r->status;
  try {
    flint::ForwardBatch batch{s->begin_visit,  s->end_visit,   s->run_embed,    s->run_lm_head,
                              s->is_decode,    s->reset_kv,    s->seq_len,      s->pos_offset,
                              s->d_input_ids,  s->d_hidden_in, s->d_block_in,   s->num_block_in,
                              s->d_hidden_out, s->d_block_out, s->d_logits_out, s->d_next_token};
    flint::ForwardRunOptions options{o->use_cuda_graph, o->use_speculative, o->return_logits,
                                     o->enable_profiling, o->profiler_range};
    flint::CaptureBuffers buffers{};
    if (capture) {
      buffers.embed = capture->embed;
      std::copy(std::begin(capture->steps), std::end(capture->steps), buffers.steps);
      std::copy(std::begin(capture->taps), std::end(capture->taps), buffers.taps);
    }
    auto output = c->runner.Run(batch, options, capture ? &buffers : nullptr);
    r->elapsed_ms = output.elapsed_ms;
    r->logical_bytes = output.logical_bytes;
    r->next_token_id = output.next_token_id;
    r->num_block_out = output.num_block_out;
    static_assert(std::size(output.stages) == FLINT_MAX_STAGE_TIMINGS);
    r->num_stages = output.num_stages;
    for (unsigned i = 0; i < output.num_stages; ++i) {
      auto& src = output.stages[i];
      auto& dst = r->stages[i];
      std::copy(std::begin(src.name), std::end(src.name), dst.name);
      dst.elapsed_ms = src.elapsed_ms;
      dst.traffic_bytes = src.traffic_bytes;
    }
    r->status = cudaSuccess;
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

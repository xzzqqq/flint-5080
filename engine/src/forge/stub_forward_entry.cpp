#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <new>
#include <span>

#include "common/cuda_check.h"
#include "common/cuda_profiler.h"
#include "forge/forward_abi.h"
#include "loader/kv_layout.h"
#include "model/visit_plan.h"
#include "ops/stub_ops.h"
using flint::CheckCuda;

struct StubGraphKey {
  uint32_t begin, end, seq_len, position, blocks;
  uint8_t embed, head, decode, reset, graph, speculative, logits, profile, profiler_range;
  const int32_t* ids;
  const void *hidden_in, *block_in;
  void *hidden_out, *block_out;
  float* logits_out;
  int32_t* next_token;
  bool operator==(const StubGraphKey&) const = default;
};
struct ForwardContext {
  ForwardContext() = default;
  ForwardContext(const ForwardContext&) = delete;
  ForwardContext& operator=(const ForwardContext&) = delete;

  ~ForwardContext() {
    if (graph) cudaGraphExecDestroy(graph);
    if (source_graph) cudaGraphDestroy(source_graph);
    for (auto event : marks)
      if (event) cudaEventDestroy(event);
    if (owned_stream) cudaStreamDestroy(owned_stream);
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
  }

  ForwardGraphStats graph_stats{};
  cudaGraph_t source_graph{};
  flint::VisitPlan plan = flint::BuildVisitPlan(flint::model::MakeDefaultInklingConfig());
  const KvStateArenaView* kv{};
  const ScratchArenaView* scratch{};
  cudaStream_t stream{}, owned_stream{};
  cudaGraphExec_t graph{};
  StubGraphKey captured{};
  ForwardRunResult graph_plan{};
  std::array<cudaEvent_t, FLINT_MAX_STAGE_TIMINGS + 1> marks{};
  cudaEvent_t start{};
  cudaEvent_t stop{};
};

namespace {
[[nodiscard]] cudaError_t RunTimedKernel(const ForwardContext& context, cudaStream_t stream,
                                         const SmokeAddParams& params, float& elapsed_ms) {
  if (const auto status = cudaEventRecord(context.start, stream); status != cudaSuccess) {
    return status;
  }
  if (const int status = flint_launch_smoke_add_f32(&params, stream); status != cudaSuccess) {
    return static_cast<cudaError_t>(status);
  }
  if (const auto status = cudaEventRecord(context.stop, stream); status != cudaSuccess) {
    return status;
  }
  if (const auto status = cudaEventSynchronize(context.stop); status != cudaSuccess) {
    return status;
  }
  return cudaEventElapsedTime(&elapsed_ms, context.start, context.stop);
}

[[nodiscard]] cudaError_t ReadChecksum(std::span<const float> device_values, float& checksum) {
  std::array<float, 256> buffer;
  while (!device_values.empty()) {
    const auto chunk = device_values.first(std::min(buffer.size(), device_values.size()));
    if (const auto status =
            cudaMemcpy(buffer.data(), chunk.data(), chunk.size_bytes(), cudaMemcpyDeviceToHost);
        status != cudaSuccess) {
      return status;
    }
    for (float value : std::span(buffer).first(chunk.size())) {
      checksum += value;
    }
    device_values = device_values.subspan(chunk.size());
  }
  return cudaSuccess;
}
}  // namespace

uint32_t flint_forward_abi_version() { return FLINT_FORWARD_ABI_VERSION; }

ForwardContext* flint_forward_create(const ModelWeightsView*, const KvStateArenaView* kv,
                                     const ScratchArenaView* scratch, void* stream) {
  std::unique_ptr<ForwardContext> context(new (std::nothrow) ForwardContext);
  if (!context) {
    return nullptr;
  }
  if (cudaEventCreate(&context->start) != cudaSuccess) {
    return nullptr;
  }
  if (cudaEventCreate(&context->stop) != cudaSuccess) {
    return nullptr;
  }
  if (scratch) {
    for (auto& mark : context->marks)
      if (cudaEventCreate(&mark) != cudaSuccess) return nullptr;
  }
  context->kv = kv;
  context->scratch = scratch;
  if (!stream &&
      cudaStreamCreateWithFlags(&context->owned_stream, cudaStreamNonBlocking) != cudaSuccess)
    return nullptr;
  context->stream = stream ? static_cast<cudaStream_t>(stream) : context->owned_stream;
  return context.release();
}

ForwardGraphStats flint_forward_graph_stats(const ForwardContext* context) {
  return context->graph_stats;
}

void flint_forward_destroy(ForwardContext* context) { delete context; }

int32_t flint_forward_run_smoke(ForwardContext* context, const ForwardWorkspace* workspace,
                                const ForwardOptions* options, int32_t element_count, float bias,
                                ForwardSmokeOutput* result) {
  if (result == nullptr) {
    return cudaErrorInvalidValue;
  }
  *result = {.status = cudaErrorInvalidValue};
  if (!context || !workspace || !workspace->device_ptr || !options || element_count <= 0 ||
      static_cast<size_t>(element_count) > workspace->bytes / (2 * sizeof(float))) {
    return result->status;
  }

  auto* input = static_cast<float*>(workspace->device_ptr);
  const SmokeAddParams params{
      .input = input, .output = input + element_count, .n = element_count, .bias = bias};
  const auto stream = static_cast<cudaStream_t>(workspace->cuda_stream);
  result->status = RunTimedKernel(*context, stream, params, result->elapsed_ms);
  if (result->status == cudaSuccess) {
    result->status =
        ReadChecksum({params.output, static_cast<size_t>(element_count)}, result->checksum);
  }
  return result->status;
}

namespace {

// Each mark terminates one stage and starts the next. External events survive graph capture.
void Execute(ForwardContext& c, const ForwardRunSpec& s, const ForwardOptions& o,
             ForwardRunResult& r, bool capturing) {
  auto mark = [&] {
    if (o.enable_profiling)
      CheckCuda(
          cudaEventRecordWithFlags(c.marks[r.num_stages], c.stream,
                                   capturing ? cudaEventRecordExternal : cudaEventRecordDefault));
  };
  auto stage = [&](const char* name, uint64_t bytes) {
    r.logical_bytes += bytes;
    if (o.enable_profiling) {
      auto& timing = r.stages[r.num_stages++];
      std::snprintf(timing.name, sizeof(timing.name), "%s", name);
      timing.traffic_bytes = bytes;
      mark();
    }
  };
  StubParams p{};
  p.width = c.kv->hidden_size;
  p.vocab = flint::model::kUnpaddedVocabSize;
  p.count = s.seq_len * p.width;
  p.position = s.pos_offset;
  p.ids = s.d_input_ids;
  p.input = static_cast<const uint16_t*>(s.d_hidden_in);
  auto* out = static_cast<uint16_t*>(s.d_hidden_out);
  p.output = s.begin_visit == s.end_visit ? out : static_cast<uint16_t*>(c.scratch->device_ptr);
  if (s.reset_kv) CheckCuda(cudaMemsetAsync(c.kv->arena_base, 0, c.kv->arena_bytes, c.stream));
  if (s.d_block_out)
    CheckCuda(cudaMemsetAsync(s.d_block_out, 0,
                              size_t(p.count) * sizeof(uint16_t) * flint::model::kCarryPlanes,
                              c.stream));
  mark();
  if (s.run_embed) {
    CheckCuda(flint_stub_embed(p, c.stream));
    p.input = p.output;
    stage("embed", 972288);
  } else if (s.begin_visit == s.end_visit && p.input != out) {
    CheckCuda(cudaMemcpyAsync(out, p.input, p.count * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
                              c.stream));
  }
  for (unsigned v = s.begin_visit; v < s.end_visit; ++v) {
    p.visit = v;
    auto cache = flint::VisitKvCache(*c.kv, c.plan[v]);
    // Keep one recurrent scalar per visit at the start of each cache plane.
    p.k = cache.k;
    p.v = cache.v;
    p.conv = flint::VisitKvConvState(*c.kv, v);
    if (v + 1 == s.end_visit) p.output = out;
    CheckCuda(flint_stub_visit(p, c.stream));
    p.input = p.output;
    char name[32];
    std::snprintf(name, sizeof(name), "visit_%02u", v);
    stage(name, (1248365568ull + 1232253888ull) / flint::model::kNumLogicalVisits);
  }
  if (s.run_lm_head) {
    p.input = out;
    p.logits = o.return_logits ? s.d_logits_out : nullptr;
    p.next = s.d_next_token;
    CheckCuda(flint_stub_head(p, c.stream));
    stage("lm_head", 926318592);
  }
}

void PrepareGraph(ForwardContext& c, const ForwardRunSpec& s, const ForwardOptions& o) {
  StubGraphKey key{s.begin_visit,      s.end_visit,      s.seq_len,         s.pos_offset,
                   s.num_block_in,     s.run_embed,      s.run_lm_head,     s.is_decode,
                   s.reset_kv,         o.use_cuda_graph, o.use_speculative, o.return_logits,
                   o.enable_profiling, o.profiler_range, s.d_input_ids,     s.d_hidden_in,
                   s.d_block_in,       s.d_hidden_out,   s.d_block_out,     s.d_logits_out,
                   s.d_next_token};
  if (c.graph) {
    if (key == c.captured) return;
    CheckCuda(cudaGraphExecDestroy(c.graph));
    c.graph = nullptr;
    CheckCuda(cudaGraphDestroy(c.source_graph));
    c.source_graph = nullptr;
  }
  c.graph_plan = {};
  cudaGraph_t graph{};
  CheckCuda(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
  try {
    Execute(c, s, o, c.graph_plan, true);
  } catch (...) {
    cudaStreamEndCapture(c.stream, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  CheckCuda(cudaStreamEndCapture(c.stream, &graph));
  auto status = cudaGraphInstantiate(&c.graph, graph, 0);
  if (status != cudaSuccess) {
    cudaGraphDestroy(graph);
    CheckCuda(status);
  }
  c.source_graph = graph;
  ++c.graph_stats.captures;
  c.captured = key;
}
}  // namespace

int32_t flint_forward_run_layers(ForwardContext* c, const ForwardRunSpec* s,
                                 const ForwardOptions* o, ForwardRunResult* r) {
  if (!r) return cudaErrorInvalidValue;
  *r = {.status = cudaErrorInvalidValue, .next_token_id = -1};
  if (!c || !s || !o || !c->scratch || !c->scratch->device_ptr || !s->seq_len ||
      s->begin_visit > s->end_visit || s->end_visit > c->plan.size() || !s->d_hidden_out ||
      (s->run_embed ? !s->d_input_ids : !s->d_hidden_in) ||
      (s->run_lm_head && o->return_logits && !s->d_logits_out) || (!c->kv || s->num_block_in))
    return r->status;
  size_t width = c->kv->hidden_size;
  if (width * s->seq_len > std::min<size_t>(UINT32_MAX - 256, c->scratch->bytes / sizeof(uint16_t)))
    return r->status;
  if (s->pos_offset > c->kv->max_full_seq_len ||
      s->seq_len > c->kv->max_full_seq_len - s->pos_offset)
    return r->status;
  bool profiling = false;
  try {
    bool graph = o->use_cuda_graph;
    if (graph) {
      PrepareGraph(*c, *s, *o);
      *r = c->graph_plan;
    }
    if (o->profiler_range) {
      CheckCuda(cudaProfilerStart());
      profiling = true;
    }
    CheckCuda(cudaEventRecord(c->start, c->stream));
    if (graph) {
      CheckCuda(cudaGraphLaunch(c->graph, c->stream));
      ++c->graph_stats.launches;
    } else
      Execute(*c, *s, *o, *r, false);
    r->next_token_id = -1;
    if (s->run_lm_head && s->d_next_token)
      CheckCuda(cudaMemcpyAsync(&r->next_token_id, s->d_next_token, sizeof(int),
                                cudaMemcpyDeviceToHost, c->stream));
    CheckCuda(cudaEventRecord(c->stop, c->stream));
    CheckCuda(cudaEventSynchronize(c->stop));
    CheckCuda(cudaEventElapsedTime(&r->elapsed_ms, c->start, c->stop));
    for (unsigned i = 0; i < r->num_stages; ++i)
      CheckCuda(cudaEventElapsedTime(&r->stages[i].elapsed_ms, c->marks[i], c->marks[i + 1]));
    r->status = cudaSuccess;
  } catch (const flint::CudaError& error) {
    r->status = error.code();
  } catch (const std::length_error&) {
    r->status = cudaErrorMemoryAllocation;
  } catch (const std::invalid_argument&) {
    r->status = cudaErrorInvalidValue;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "forward execution failed: %s\n", e.what());
    r->status = cudaErrorUnknown;
  }
  if (profiling) {
    auto status = cudaProfilerStop();
    if (!r->status) r->status = status;
  }
  return r->status;
}
int32_t flint_forward_store_context(ForwardContext*, const void* taps, uint32_t count,
                                    uint32_t pos) {
  return cudaErrorNotSupported;
}
int32_t flint_forward_draft(ForwardContext*, const ForwardDraftSpec* spec,
                            ForwardDraftResult* result) {
  return cudaErrorNotSupported;
}
int32_t flint_forward_verify(ForwardContext*, const ForwardRunSpec* spec,
                             const ForwardOptions* opts, ForwardRunResult* result) {
  return cudaErrorNotSupported;
}
int32_t flint_forward_commit(ForwardContext*, uint32_t pos, uint32_t len, uint32_t accepted) {
  return cudaErrorNotSupported;
}

int32_t flint_forward_spec_round(ForwardContext*, const ForwardSpecRoundSpec*,
                                 const ForwardOptions*, ForwardSpecRoundResult*) {
  return cudaErrorNotSupported;
}

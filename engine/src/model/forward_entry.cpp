#include <cuda_runtime.h>
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
// CUDA runtime-only toolkit installations export both APIs without the profiler header.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <tuple>
#include <vector>

#include "forge/forward_abi.h"
#include "model/inkling2_forward.h"
#include "ops/scratch_workspace.h"

struct ForwardContext {
  ForwardContext() = default;
  ForwardContext(const ForwardContext&) = delete;
  ForwardContext& operator=(const ForwardContext&) = delete;

  ~ForwardContext() {
    if (graph) cudaGraphExecDestroy(graph);
    if (source_graph) cudaGraphDestroy(source_graph);
    gemm.reset();
    if (experts) cudaFree(experts);
    if (host_token) cudaFreeHost(host_token);
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

  std::unique_ptr<flint::Gemm> gemm;
  ScratchArenaView activations{};
  int *next_token{}, *host_token{};
  int32_t* d_decode_pos{};
  ExpertProjection* experts{};
  ForwardGraphStats graph_stats{};
  uint32_t graph_blocks{};
  cudaGraph_t source_graph{};
  std::array<VisitPlanEntry, 48> plan{};
  const ModelWeightsView* weights{};
  const KvStateArenaView* kv{};
  const ScratchArenaView* scratch{};
  cudaStream_t stream{}, owned_stream{};
  cudaGraphExec_t graph{};
  ForwardRunSpec captured{};
  ForwardOptions captured_options{};
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

ForwardContext* flint_forward_create(const ModelWeightsView* weights, const KvStateArenaView* kv,
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
  context->weights = weights;
  context->kv = kv;
  context->scratch = scratch;
  if (!stream &&
      cudaStreamCreateWithFlags(&context->owned_stream, cudaStreamNonBlocking) != cudaSuccess)
    return nullptr;
  context->stream = stream ? static_cast<cudaStream_t>(stream) : context->owned_stream;
  if (weights) {
    try {
      if (!kv || !scratch) return nullptr;
      if (cudaMallocHost(&context->host_token, sizeof(int)) != cudaSuccess) return nullptr;
      context->plan = flint::BuildVisitPlan(weights->config);
      flint::ops_detail::ScratchWorkspace ws(*scratch);
      constexpr size_t bytes = 4 << 20;
      auto workspace = ws.Take<char>(bytes);
      context->next_token = ws.Take<int>(1);
      context->d_decode_pos = ws.Take<int32_t>(1);
      context->activations = ws.rest;
      std::vector<ExpertProjection> descriptors;
      descriptors.reserve(2 * FLINT_NUM_EXPERTS * FLINT_NUM_PHYSICAL_LAYERS);
      for (const auto& layer : weights->layers)
        for (const auto* projection : {layer.gate_up_proj, layer.down_proj})
          for (unsigned e = 0; e < FLINT_NUM_EXPERTS; ++e) {
            const auto& m = projection[e];
            if (m.k2 != 6 && m.k2 != 7) return nullptr;
            descriptors.push_back({m.trellis, m.suh, m.svh, m.k2});
          }
      auto table_bytes = descriptors.size() * sizeof(ExpertProjection);
      if (cudaMalloc(&context->experts, table_bytes) != cudaSuccess ||
          cudaMemcpyAsync(context->experts, descriptors.data(), table_bytes, cudaMemcpyHostToDevice,
                          context->stream) != cudaSuccess ||
          cudaStreamSynchronize(context->stream) != cudaSuccess)
        return nullptr;
      context->gemm = std::make_unique<flint::Gemm>(context->stream, workspace, bytes);
    } catch (const std::exception&) {
      return nullptr;
    }
  }
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
void Require(int status) {
  if (status != cudaSuccess) throw static_cast<cudaError_t>(status);
}

// Each mark terminates one stage and starts the next. External events survive graph capture.
void Execute(ForwardContext& c, const ForwardRunSpec& s, const ForwardOptions& o,
             ForwardRunResult& r, bool capturing) {
  auto mark = [&] {
    if (o.enable_profiling)
      Require(
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
  p.width = 2304;
  p.vocab = 201024;
  p.count = s.seq_len * p.width;
  p.position = s.pos_offset;
  p.ids = s.d_input_ids;
  p.input = static_cast<const float*>(s.d_hidden_in);
  auto* out = static_cast<float*>(s.d_hidden_out);
  p.output = s.begin_visit == s.end_visit ? out : static_cast<float*>(c.scratch->device_ptr);
  if (c.kv) {
    p.kv = static_cast<float*>(c.kv->swa_kv);
    p.conv = c.kv->kv_conv_state;
    if (s.reset_kv) {
      Require(cudaMemsetAsync(p.kv, 0, 48 * sizeof(float), c.stream));
      Require(cudaMemsetAsync(p.conv, 0, 48 * sizeof(float), c.stream));
    }
  }
  mark();
  if (s.run_embed) {
    Require(flint_stub_embed(p, c.stream));
    p.input = p.output;
    stage("embed", 972288);
  } else if (s.begin_visit == s.end_visit && p.input != out) {
    Require(
        cudaMemcpyAsync(out, p.input, p.count * sizeof(float), cudaMemcpyDeviceToDevice, c.stream));
  }
  for (unsigned v = s.begin_visit; v < s.end_visit; ++v) {
    p.visit = v;
    if (v + 1 == s.end_visit) p.output = out;
    Require(flint_stub_visit(p, c.stream));
    p.input = p.output;
    char name[32];
    std::snprintf(name, sizeof(name), "visit_%02u", v);
    stage(name, (1248365568ull + 1232253888ull) / 48);
  }
  if (s.run_lm_head) {
    p.input = out;
    p.logits = o.return_logits ? s.d_logits_out : nullptr;
    p.next = s.d_next_token;
    Require(flint_stub_head(p, c.stream));
    stage("lm_head", 926318592);
  }
}

void PrepareGraph(ForwardContext& c, const ForwardRunSpec& s, const ForwardOptions& o) {
  auto key = [](const ForwardRunSpec& spec, const ForwardOptions& options) {
    return std::tuple(spec.begin_visit, spec.end_visit, spec.run_embed, spec.run_lm_head,
                      options.return_logits, spec.d_input_ids, spec.d_hidden_in, spec.d_block_in,
                      spec.num_block_in, spec.d_hidden_out, spec.d_block_out, spec.d_logits_out,
                      spec.d_next_token);
  };
  if (c.graph) {
    bool hit = c.weights ? key(s, o) == key(c.captured, c.captured_options)
                         : !std::memcmp(&s, &c.captured, sizeof(s)) &&
                               !std::memcmp(&o, &c.captured_options, sizeof(o));
    if (hit) return;
    Require(cudaGraphExecDestroy(c.graph));
    c.graph = nullptr;
    Require(cudaGraphDestroy(c.source_graph));
    c.source_graph = nullptr;
  }
  c.graph_plan = {};
  cudaGraph_t graph{};
  Require(cudaStreamBeginCapture(c.stream, cudaStreamCaptureModeThreadLocal));
  try {
    if (c.weights) {
      auto spec = s;
      spec.reset_kv = 0;
      spec.num_block_out = &c.graph_blocks;
      flint::RunInkling(*c.gemm, c.plan, *c.weights, *c.kv, c.activations, spec, o, c.graph_plan,
                        c.marks.data(), nullptr, c.experts, c.d_decode_pos);
    } else {
      Execute(c, s, o, c.graph_plan, true);
    }
  } catch (...) {
    cudaStreamEndCapture(c.stream, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  Require(cudaStreamEndCapture(c.stream, &graph));
  auto status = cudaGraphInstantiate(&c.graph, graph, 0);
  if (status != cudaSuccess) {
    cudaGraphDestroy(graph);
    Require(status);
  }
  c.source_graph = graph;
  ++c.graph_stats.captures;
  std::memcpy(&c.captured, &s, sizeof(s));
  c.captured_options = o;
}
}  // namespace

int32_t flint_forward_run_layers_capture(ForwardContext* c, const ForwardRunSpec* s,
                                         const ForwardOptions* o, ForwardRunResult* r,
                                         const ForwardCapture* capture) {
  if (!r) return cudaErrorInvalidValue;
  *r = {.status = cudaErrorInvalidValue, .next_token_id = -1};
  if (!c || !s || !o || !c->scratch || !c->scratch->device_ptr || !s->seq_len ||
      s->begin_visit > s->end_visit || s->end_visit > 48 || !s->d_hidden_out ||
      (s->run_embed ? !s->d_input_ids : !s->d_hidden_in) ||
      (s->run_lm_head && o->return_logits && !s->d_logits_out) || (!c->weights && s->num_block_in))
    return r->status;
  size_t width = c->weights ? c->weights->config.hidden_size : 2304;
  if (width * s->seq_len > std::min<size_t>(UINT32_MAX - 256, c->scratch->bytes / sizeof(float)))
    return r->status;
  auto spec = *s;
  if (c->weights && spec.run_lm_head && !spec.d_next_token) spec.d_next_token = c->next_token;
  s = &spec;
  if (s->num_block_out) *s->num_block_out = 0;
  bool profiling = false;
  try {
    bool decode = c->weights && spec.is_decode && spec.seq_len == 1;
    bool graph = o->use_cuda_graph && (!c->weights || (decode && !capture && !o->enable_profiling));
    if (decode && spec.pos_offset >= c->kv->max_full_seq_len) return r->status;
    if (graph) {
      PrepareGraph(*c, *s, *o);
      *r = c->graph_plan;
    }
    if (o->profiler_range) {
      Require(cudaProfilerStart());
      profiling = true;
    }
    Require(cudaEventRecord(c->start, c->stream));
    if (decode)
      Require(cudaMemcpyAsync(c->d_decode_pos, &spec.pos_offset, sizeof(int32_t),
                              cudaMemcpyHostToDevice, c->stream));
    if (graph) {
      if (decode && spec.reset_kv)
        Require(cudaMemsetAsync(c->kv->arena_base, 0, c->kv->arena_bytes, c->stream));
      Require(cudaGraphLaunch(c->graph, c->stream));
      ++c->graph_stats.launches;
      if (c->weights && spec.num_block_out) *spec.num_block_out = c->graph_blocks;
    } else if (c->weights)
      flint::RunInkling(*c->gemm, c->plan, *c->weights, *c->kv, c->activations, *s, *o, *r,
                        c->marks.data(), capture, c->experts, decode ? c->d_decode_pos : nullptr);
    else
      Execute(*c, *s, *o, *r, false);
    r->next_token_id = -1;
    int* token = c->weights ? c->host_token : &r->next_token_id;
    if (s->run_lm_head && s->d_next_token)
      Require(
          cudaMemcpyAsync(token, s->d_next_token, sizeof(int), cudaMemcpyDeviceToHost, c->stream));
    Require(cudaEventRecord(c->stop, c->stream));
    Require(cudaEventSynchronize(c->stop));
    if (s->run_lm_head && s->d_next_token) r->next_token_id = *token;
    Require(cudaEventElapsedTime(&r->elapsed_ms, c->start, c->stop));
    for (unsigned i = 0; i < r->num_stages; ++i)
      Require(cudaEventElapsedTime(&r->stages[i].elapsed_ms, c->marks[i], c->marks[i + 1]));
    r->status = cudaSuccess;
  } catch (cudaError_t error) {
    r->status = error;
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

int32_t flint_forward_run_layers(ForwardContext* c, const ForwardRunSpec* s,
                                 const ForwardOptions* o, ForwardRunResult* r) {
  return flint_forward_run_layers_capture(c, s, o, r, nullptr);
}

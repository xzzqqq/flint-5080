#include "model/inkling2_forward.h"

#include <cuda_runtime.h>

#include "common/cuda_check.h"
#include "loader/kv_layout.h"
#include "ops/norm_ops.h"
#include "ops/sconv_ops.h"

#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
// CUDA runtime-only toolkit installations export both APIs without the profiler header.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif

#include <cstdio>
#include <vector>

#include "model/visit_plan.h"
#include "ops/scratch_workspace.h"
#include "ops/submodules.h"

namespace flint {

using ops_detail::ScratchWorkspace;
namespace {
uint64_t MatrixBytes(const Exl3MatrixView& m) {
  return uint64_t(m.k) * m.n * m.k2 / 16 + 2ull * (m.k + m.n);
}
// Single-token logical weight traffic (model.md 6.1). Average the two expert
// bitrates and count only controllers and vocabulary rows used by this visit.
uint64_t VisitBytes(const ModelConfig& c, const LayerWeightsView& l, VisitPlanEntry v) {
  uint64_t h = c.hidden_size, kv = c.num_key_value_heads * c.head_dim,
           heads = c.num_attention_heads;
  uint64_t rel = heads * c.d_rel;
  uint64_t bytes = MatrixBytes(l.q_proj) + MatrixBytes(l.k_proj) + MatrixBytes(l.v_proj) +
                   MatrixBytes(l.o_gate_proj) + MatrixBytes(l.o_proj);
  bytes += 2 * (h * rel + c.d_rel * c.rel_extent + heads * c.head_dim + rel +
                heads * h);  // relative, clip and sink
  bytes += 2 * (2 * (kv + h) * c.sconv_kernel_size +
                h * (v.is_full_attention ? 4 : 2));  // convolutions and norms
  if (v.is_full_attention) bytes += 2 * 2 * c.num_key_value_heads * c.sconv_kernel_size * h;
  bytes += 2 * c.n_routed_experts * h + 4 * c.n_routed_experts;  // router and bias
  bytes +=
      4 * h * (1 + (v.physical_layer ? (v.is_full_attention ? 3 : 1) : 0));  // residual queries
  if (v.has_messages) bytes += 4 * 4 * h + 4 * 2 * 4 * h * c.loop_gate_rank;
  uint64_t pair = 0;
  for (int e = 0; e < 2; ++e) pair += MatrixBytes(l.gate_up_proj[e]) + MatrixBytes(l.down_proj[e]);
  return bytes + pair * c.num_experts_per_tok / 2;
}
struct GraphKey {
  ForwardBatch batch;
  ForwardRunOptions options;
  bool operator==(const GraphKey&) const = default;
};
}  // namespace
struct InklingRunner::Impl {
  const ModelWeightsView& weights;
  const KvStateArenaView& kv;
  VisitPlan plan;
  ScratchArenaView activations{};
  std::unique_ptr<Gemm> gemm;
  ExpertProjection* d_experts{};
  int32_t *d_next_token{}, *d_decode_pos{}, *host_token{};
  cudaStream_t stream{}, owned_stream{};
  cudaEvent_t start{}, stop{};
  std::array<cudaEvent_t, std::extent_v<decltype(ForwardOutput::stages)> + 1> marks{};
  cudaGraphExec_t graph{};
  cudaGraph_t source_graph{};
  GraphKey key{};
  ForwardOutput graph_output{};
  GraphStats stats{};
  Impl(const ModelWeightsView& w, const KvStateArenaView& k)
      : weights(w), kv(k), plan(BuildVisitPlan(w.config)) {}
  ~Impl() {
    if (graph) cudaGraphExecDestroy(graph);
    if (source_graph) cudaGraphDestroy(source_graph);
    gemm.reset();
    if (host_token) cudaFreeHost(host_token);
    for (auto e : marks)
      if (e) cudaEventDestroy(e);
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (owned_stream) cudaStreamDestroy(owned_stream);
  }
  void RunInkling(const ForwardBatch&, const ForwardRunOptions&, ForwardOutput&,
                  const CaptureBuffers*);
  void PrepareGraph(const ForwardBatch&, const ForwardRunOptions&);
};
InklingRunner::InklingRunner(const ModelWeightsView& w, const KvStateArenaView& kv,
                             ScratchArenaView scratch, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(w, kv)) {
  auto& c = *impl_;
  if (!scratch.device_ptr) throw std::invalid_argument("missing scratch");
  if (!stream) CheckCuda(cudaStreamCreateWithFlags(&c.owned_stream, cudaStreamNonBlocking));
  c.stream = stream ? stream : c.owned_stream;
  CheckCuda(cudaEventCreate(&c.start));
  CheckCuda(cudaEventCreate(&c.stop));
  for (auto& e : c.marks) CheckCuda(cudaEventCreate(&e));
  CheckCuda(cudaHostAlloc(&c.host_token, sizeof(int32_t), cudaHostAllocDefault));
  ScratchWorkspace ws(scratch);
  constexpr size_t bytes = 4 << 20;
  auto workspace = ws.Take<char>(bytes);
  c.d_next_token = ws.Take<int32_t>(1);
  c.d_decode_pos = ws.Take<int32_t>(1);
  std::vector<ExpertProjection> descriptors;
  descriptors.reserve(2 * model::kNumExperts * model::kNumPhysicalLayers);
  for (const auto& layer : w.layers)
    for (const auto* projection : {layer.gate_up_proj, layer.down_proj})
      for (int e = 0; e < model::kNumExperts; ++e) {
        const auto& m = projection[e];
        if (m.k2 != 6 && m.k2 != 7) throw std::invalid_argument("unsupported expert bitrate");
        descriptors.push_back({m.trellis, m.suh, m.svh, m.k2});
      }
  c.d_experts = ws.Take<ExpertProjection>(descriptors.size());
  c.activations = ws.rest;
  CheckCuda(cudaMemcpyAsync(c.d_experts, descriptors.data(),
                            descriptors.size() * sizeof(ExpertProjection), cudaMemcpyHostToDevice,
                            c.stream));
  CheckCuda(cudaStreamSynchronize(c.stream));
  c.gemm = std::make_unique<Gemm>(c.stream, workspace, bytes);
}
InklingRunner::~InklingRunner() = default;
GraphStats InklingRunner::graph_stats() const { return impl_->stats; }
void InklingRunner::Impl::RunInkling(const ForwardBatch& s, const ForwardRunOptions& o,
                                     ForwardOutput& result, const CaptureBuffers* capture) {
  const auto& g = *gemm;
  const auto& w = weights;
  const auto* d_pos = s.is_decode && s.seq_len == 1 ? d_decode_pos : nullptr;
  const int t = s.seq_len, h = w.config.hidden_size, n = t * h;
  ScratchWorkspace ws(activations);
  auto* hidden = static_cast<uint16_t*>(s.d_hidden_out);
  auto* carry = ws.Take<uint16_t>(size_t(model::kCarryPlanes) * n);
  auto block = [&](unsigned idx) { return carry + size_t(idx) * n; };
  auto copy = [&](void* dst, const void* src, size_t bytes) {
    if (dst && dst != src)
      CheckCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
  };
  size_t plane = size_t(n) * 2;
  unsigned count = s.num_block_in;
  unsigned expected = 0;
  for (unsigned v = 0; v < s.begin_visit; ++v) {
    if (plan[v].is_loop_pass_start) expected = 1;
    expected += plan[v].is_block_boundary;
  }
  if (count != expected || (count && !s.d_block_in) || (s.run_embed && s.begin_visit) ||
      (s.run_lm_head && s.end_visit != plan.size()))
    throw std::invalid_argument("invalid carry or range");
  if (s.reset_kv) CheckCuda(cudaMemsetAsync(kv.arena_base, 0, kv.arena_bytes, stream));
  if (count)
    copy(carry, s.d_block_in, model::kCarryPlanes * plane);
  else
    CheckCuda(cudaMemsetAsync(carry, 0, model::kCarryPlanes * plane, stream));
  auto stage = [&](const char* name, uint64_t bytes) {
    result.logical_bytes += bytes;
    if (!o.enable_profiling) return;
    auto& st = result.stages[result.num_stages++];
    std::snprintf(st.name, sizeof(st.name), "%s", name);
    st.traffic_bytes = bytes;
    CheckCuda(cudaEventRecord(marks[result.num_stages], stream));
  };
  if (o.enable_profiling) CheckCuda(cudaEventRecord(marks[0], stream));
  if (s.run_embed) {
    CheckCuda(flint_embed_rmsnorm(
        {w.embed_weight, w.embed_norm_weight, hidden, s.d_input_ids, t, h, FLINT_BF16, FLINT_BF16},
        stream));
    if (capture) copy(capture->embed, hidden, plane);
    stage("embed", 4u * h);
  } else
    copy(hidden, s.d_hidden_in, plane);

  // Residual attention returns token-major [T,Q,H]. Gather each query into
  // a contiguous [T,H] plane for the individual controllers and projections.
  auto* packed = ws.Take<float>(size_t(4) * n);
  auto* queries = ws.Take<float>(4 * h);
  auto* messages = ws.Take<float>(size_t(4) * n);
  auto* attn = ws.Take<float>(size_t(3) * n);
  auto* normalized = ws.Take<uint16_t>(size_t(3) * n);
  auto* partial = ws.Take<float>(n);
  auto* tmp = ws.Take<uint16_t>(n);
  auto aggregate = [&](const uint16_t* blocks, int nb, const void* x, FlintType type,
                       const float* query, int q, float* dst) {
    RunBlockResidual(g, ws.rest,
                     {blocks, x, query, q == 1 ? dst : packed, nb, t, h, q, FLINT_BF16, type});
    if (q > 1)
      for (int j = 0; j < q; ++j)
        CheckCuda(cudaMemcpy2DAsync(dst + size_t(j) * n, h * 4, packed + j * h, q * h * 4, h * 4, t,
                                    cudaMemcpyDeviceToDevice, stream));
  };
  auto norm = [&](const void* x, FlintType type, const void* weight, uint16_t* dst) {
    CheckCuda(flint_rmsnorm({x, weight, dst, nullptr, t, h, type, FLINT_BF16}, stream));
  };
  for (unsigned v = s.begin_visit; v < s.end_visit; ++v) {
    auto visit = plan[v];
    auto& l = w.layers[visit.physical_layer];
    if (visit.is_loop_pass_start) {
      copy(block(6), hidden, plane);
      copy(block(7), block(2), plane);
      copy(hidden, block(4), plane);
      copy(block(0), block(5), plane);
      count = 1;
    }
    if (capture && visit.mtp_tap_idx >= 0) copy(capture->taps[visit.mtp_tap_idx], hidden, plane);
    int nq = visit.is_full_attention ? 3 : 1;
    if (visit.physical_layer) aggregate(carry, count, hidden, FLINT_BF16, l.attn_query, nq, attn);
    if (visit.has_messages) {
      CheckCuda(
          flint_merge_cross_pass_queries({w.cross_pass_queries, w.cross_pass_kv_queries, queries,
                                          int(visit.physical_layer - w.config.loop_core_start), h},
                                         stream));
      aggregate(block(7), 1, block(6), FLINT_BF16, queries, 4, messages);
    }
    const void* norms[] = {l.q_input_norm, l.k_input_norm, l.v_input_norm};
    for (int j = 0; j < (visit.has_messages ? 3 : nq); ++j) {
      auto* src = attn + (nq == 1 ? 0 : j * n);
      if (visit.has_messages) {
        RunMessageGate(g, ws.rest, l.controllers[j], src, messages + j * n, partial, t);
        src = partial;
      }
      if (visit.physical_layer)
        norm(src, FLINT_F32, nq == 1 ? l.attn_norm : norms[j], normalized + j * n);
      else
        norm(hidden, FLINT_BF16, l.attn_norm, normalized);
    }
    int stride = (visit.has_messages || nq == 3) ? n : 0;
    AttentionBatch batch{
        normalized, normalized + stride, normalized + 2 * stride, tmp, t, int(s.pos_offset), visit,
        d_pos};
    (visit.is_full_attention ? RunFullAttention : RunSlidingAttention)(g, ws.rest, l, kv, batch);
    auto conv = [&](const void* weight, float* state) {
      CheckCuda(flint_static_sconv({tmp, weight, nullptr, state, tmp, t, h, FLINT_BF16, FLINT_BF16},
                                   stream));
    };
    conv(l.attn_sconv, VisitSublayerConvState(kv, v));
    CheckCuda(flint_cast({tmp, partial, n, FLINT_BF16, FLINT_F32}, stream));
    if (visit.is_block_boundary)
      copy(block(count++), hidden, plane);
    else
      CheckCuda(flint_add_residual(partial, hidden, partial, n, FLINT_F32, stream));
    aggregate(carry, count, partial, FLINT_F32, l.mlp_query, 1, attn);
    if (visit.has_messages)
      RunMessageGate(g, ws.rest, l.controllers[3], attn, messages + 3 * n, attn, t);
    norm(attn, FLINT_F32, l.mlp_norm, normalized);
    RunSparseMoeBlock(g, ws.rest, l, normalized, tmp, t, nullptr, nullptr, nullptr,
                      d_experts + visit.physical_layer * 2 * model::kNumExperts);
    conv(l.mlp_sconv, VisitSublayerConvState(kv, v, 1));
    CheckCuda(flint_add_residual(partial, tmp, hidden, n, FLINT_BF16, stream));
    if (visit.is_prologue_end) {
      copy(block(4), hidden, plane);
      copy(block(5), carry, plane);
    }
    if (capture) copy(capture->steps[v], hidden, plane);
    char name[32];
    std::snprintf(name, sizeof(name), "visit_%02u", v);
    stage(name, VisitBytes(w.config, l, visit));
  }
  copy(s.d_block_out, carry, plane * model::kCarryPlanes);
  result.num_block_out = count;
  if (s.run_lm_head) {
    aggregate(carry, count, hidden, FLINT_BF16, w.output_query, 1, attn);
    norm(attn, FLINT_F32, w.final_norm_weight, hidden);
    if (capture) copy(capture->taps[3], hidden, plane);
    int vocab = w.config.unpadded_vocab_size;
    auto* raw = ws.Take<float>(vocab);
    auto* rounded = ws.Take<uint16_t>(vocab);
    // Match the fixture's BF16 linear output before applying softcap in FP32.
    RunRowMatmul(g, ws.rest, hidden + (t - 1) * h, w.unembed_weight, raw, 1, vocab, h, FLINT_BF16,
                 FLINT_F32, true);
    CheckCuda(flint_cast({raw, rounded, vocab, FLINT_F32, FLINT_BF16}, stream));
    CheckCuda(flint_logit_softcap_argmax({rounded, o.return_logits ? s.d_logits_out : raw,
                                          s.d_next_token, 1, vocab, vocab, FLINT_BF16},
                                         stream));
    stage("lm_head", 2ull * h * vocab + 6ull * h);
  }
}

void InklingRunner::Impl::PrepareGraph(const ForwardBatch& batch, const ForwardRunOptions& opts) {
  // Position is uploaded before each launch. KV reset also runs outside capture.
  auto captured = batch;
  captured.pos_offset = 0;
  captured.reset_kv = 0;
  GraphKey requested{captured, opts};
  if (graph && key == requested) return;
  if (graph) {
    CheckCuda(cudaGraphExecDestroy(graph));
    graph = nullptr;
  }
  if (source_graph) {
    CheckCuda(cudaGraphDestroy(source_graph));
    source_graph = nullptr;
  }
  graph_output = {};
  CheckCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  try {
    RunInkling(captured, opts, graph_output, nullptr);
  } catch (...) {
    cudaStreamEndCapture(stream, &source_graph);
    throw;
  }
  CheckCuda(cudaStreamEndCapture(stream, &source_graph));
  CheckCuda(cudaGraphInstantiate(&graph, source_graph, 0));
  key = requested;
  ++stats.captures;
}
ForwardOutput InklingRunner::Run(const ForwardBatch& batch, const ForwardRunOptions& o,
                                 const CaptureBuffers* capture) {
  auto& c = *impl_;
  auto s = batch;
  if (!s.seq_len || s.begin_visit > s.end_visit || s.end_visit > c.plan.size() || !s.d_hidden_out ||
      (s.run_embed ? !s.d_input_ids : !s.d_hidden_in) ||
      (s.run_lm_head && o.return_logits && !s.d_logits_out) ||
      size_t(s.seq_len) * c.weights.config.hidden_size > INT32_MAX ||
      s.pos_offset > c.kv.max_full_seq_len || s.seq_len > c.kv.max_full_seq_len - s.pos_offset)
    throw std::invalid_argument("invalid forward batch");
  if (s.run_lm_head && !s.d_next_token) s.d_next_token = c.d_next_token;
  bool decode = s.is_decode && s.seq_len == 1;
  bool graph = o.use_cuda_graph && decode && !capture && !o.enable_profiling;
  ForwardOutput r{};
  if (graph) {
    c.PrepareGraph(s, o);
    r = c.graph_output;
  }
  if (o.profiler_range) CheckCuda(cudaProfilerStart());
  try {
    CheckCuda(cudaEventRecord(c.start, c.stream));
    if (decode)
      CheckCuda(cudaMemcpyAsync(c.d_decode_pos, &s.pos_offset, sizeof(int32_t),
                                cudaMemcpyHostToDevice, c.stream));
    if (graph) {
      if (s.reset_kv) CheckCuda(cudaMemsetAsync(c.kv.arena_base, 0, c.kv.arena_bytes, c.stream));
      CheckCuda(cudaGraphLaunch(c.graph, c.stream));
      ++c.stats.launches;
    } else
      c.RunInkling(s, o, r, capture);
    if (s.run_lm_head)
      CheckCuda(cudaMemcpyAsync(c.host_token, s.d_next_token, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, c.stream));
    CheckCuda(cudaEventRecord(c.stop, c.stream));
    CheckCuda(cudaEventSynchronize(c.stop));
    r.next_token_id = s.run_lm_head ? *c.host_token : -1;
    CheckCuda(cudaEventElapsedTime(&r.elapsed_ms, c.start, c.stop));
    for (unsigned i = 0; i < r.num_stages; ++i)
      CheckCuda(cudaEventElapsedTime(&r.stages[i].elapsed_ms, c.marks[i], c.marks[i + 1]));
  } catch (...) {
    if (o.profiler_range) cudaProfilerStop();
    throw;
  }
  if (o.profiler_range) CheckCuda(cudaProfilerStop());
  return r;
}
}  // namespace flint

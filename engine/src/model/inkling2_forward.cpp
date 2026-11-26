#include "model/inkling2_forward.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <type_traits>

#include "common/cuda_check.h"
#include "common/cuda_profiler.h"
#include "loader/kv_layout.h"
#include "loader/scratch_workspace.h"
#include "model/submodules.h"
#include "model/visit_plan.h"
#include "ops/dspark_ops.h"
#include "ops/norm_ops.h"
#include "ops/sconv_ops.h"

namespace flint {

namespace {
static_assert(sizeof(Exl3ExpertDesc) == sizeof(ExpertProjection) &&
              alignof(Exl3ExpertDesc) == alignof(ExpertProjection) &&
              offsetof(Exl3ExpertDesc, trellis) == offsetof(ExpertProjection, trellis) &&
              offsetof(Exl3ExpertDesc, suh) == offsetof(ExpertProjection, suh) &&
              offsetof(Exl3ExpertDesc, svh) == offsetof(ExpertProjection, svh) &&
              offsetof(Exl3ExpertDesc, k2) == offsetof(ExpertProjection, k2));
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
  // Decode position and KV reset are applied on the stream outside the graph.
  uint32_t begin_visit, end_visit, seq_len, num_block_in;
  uint8_t run_embed, run_lm_head, is_decode;
  const int32_t* input_ids;
  const void *hidden_in, *block_in;
  void *hidden_out, *block_out;
  float* logits_out;
  int32_t* next_token;
  uint8_t use_cuda_graph, use_speculative, return_logits, enable_profiling, profiler_range;

  bool operator==(const GraphKey&) const = default;
};
}  // namespace
struct InklingRunner::Impl {
  const ModelWeightsView& weights;
  const KvStateArenaView& kv;
  VisitPlan plan;
  uint16_t* backup{};
  float *kv_trace{}, *sub_trace{};
  uint32_t verify_pos{}, verify_len{};
  bool recording{};
  ScratchArenaView activations{};
  std::unique_ptr<Gemm> gemm;
  int32_t *d_next_token{}, *d_decode_pos{}, *host_token{};
  cudaStream_t stream{}, owned_stream{};
  cudaEvent_t start{}, stop{};
  std::array<cudaEvent_t, std::extent_v<decltype(ForwardRunResult::stages)> + 1> marks{};
  cudaGraphExec_t graph{};
  cudaGraph_t source_graph{};
  GraphKey key{};
  ForwardRunResult graph_output{};
  ForwardGraphStats stats{};
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
  void RunInkling(const ForwardRunSpec&, const ForwardOptions&, ForwardRunResult&,
                  const ForwardCapture*);
  void PrepareGraph(const ForwardRunSpec&, const ForwardOptions&);
};
InklingRunner::InklingRunner(const ModelWeightsView& w, const KvStateArenaView& kv,
                             ScratchArenaView scratch, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(w, kv)) {
  auto& c = *impl_;
  if (!w.expert_table) throw std::invalid_argument("missing expert table");
  if (!scratch.device_ptr) throw std::invalid_argument("missing scratch");
  if (!stream) CheckCuda(cudaStreamCreateWithFlags(&c.owned_stream, cudaStreamNonBlocking));
  c.stream = stream ? stream : c.owned_stream;
  CheckCuda(cudaEventCreate(&c.start));
  CheckCuda(cudaEventCreate(&c.stop));
  for (auto& e : c.marks) CheckCuda(cudaEventCreate(&e));
  CheckCuda(cudaHostAlloc(&c.host_token, sizeof(int32_t), cudaHostAllocDefault));
  // Graphs retain these addresses. Each run initializes their contents before use.
  ScratchWorkspace ws(scratch);
  constexpr size_t bytes = 4 << 20;
  auto workspace = ws.Take<char>(bytes);
  c.d_next_token = ws.Take<int32_t>(1);
  c.d_decode_pos = ws.Take<int32_t>(1);
  if (kv.has_mtp_kv) {
    c.backup = ws.Take<uint16_t>(8 * 2 * model::kNumSwaVisits * model::kKvDim);
    c.kv_trace = ws.Take<float>(11 * 2 * model::kNumLogicalVisits * model::kKvDim);
    c.sub_trace = ws.Take<float>(11 * 2 * model::kNumLogicalVisits * model::kHiddenSize);
  }
  c.activations = ws.rest;
  c.gemm = std::make_unique<Gemm>(c.stream, workspace, bytes);
}
cudaStream_t InklingRunner::stream() const { return impl_->stream; }
InklingRunner::~InklingRunner() = default;
ForwardGraphStats InklingRunner::graph_stats() const { return impl_->stats; }
void InklingRunner::Impl::RunInkling(const ForwardRunSpec& s, const ForwardOptions& o,
                                     ForwardRunResult& result, const ForwardCapture* capture) {
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
  auto tap = [&](int i) {
    if (capture) copy(capture->taps[i], hidden, plane);
    if (s.d_taps_out)
      CheckCuda(cudaMemcpy2DAsync(static_cast<uint16_t*>(s.d_taps_out) + i * h, 4 * h * 2, hidden,
                                  h * 2, h * 2, t, cudaMemcpyDeviceToDevice, stream));
  };
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
    if (visit.mtp_tap_idx >= 0) tap(visit.mtp_tap_idx);
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
    AttentionBatch batch{normalized,
                         normalized + stride,
                         normalized + 2 * stride,
                         tmp,
                         t,
                         int(s.pos_offset),
                         visit,
                         d_pos,
                         recording ? kv_trace + 2 * v * model::kKvDim * (3 + t) : nullptr};
    (visit.is_full_attention ? RunFullAttention : RunSlidingAttention)(g, ws.rest, l, kv, batch);
    auto conv = [&](const void* weight, float* state, int sublayer) {
      CheckCuda(
          flint_static_sconv({tmp, weight, nullptr, state, tmp, t, h, FLINT_BF16, FLINT_BF16,
                              recording ? sub_trace + (2 * v + sublayer) * h * (3 + t) : nullptr},
                             stream));
    };
    conv(l.attn_sconv, VisitSublayerConvState(kv, v), 0);
    CheckCuda(flint_cast({tmp, partial, n, FLINT_BF16, FLINT_F32}, stream));
    if (visit.is_block_boundary)
      copy(block(count++), hidden, plane);
    else
      CheckCuda(flint_add_residual(partial, hidden, partial, n, FLINT_F32, stream));
    aggregate(carry, count, partial, FLINT_F32, l.mlp_query, 1, attn);
    if (visit.has_messages)
      RunMessageGate(g, ws.rest, l.controllers[3], attn, messages + 3 * n, attn, t);
    norm(attn, FLINT_F32, l.mlp_norm, normalized);
    const auto* experts = w.expert_table + visit.physical_layer * 2 * model::kNumExperts;
    RunSparseMoeBlock(g, ws.rest, l, normalized, tmp, t,
                      reinterpret_cast<const ExpertProjection*>(experts));
    conv(l.mlp_sconv, VisitSublayerConvState(kv, v, 1), 1);
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
  if (s.run_lm_head || (s.d_taps_out && s.end_visit == plan.size())) {
    aggregate(carry, count, hidden, FLINT_BF16, w.output_query, 1, attn);
    norm(attn, FLINT_F32, w.final_norm_weight, hidden);
    tap(3);
    if (!s.run_lm_head) return;
    int rows = s.all_token_logits ? t : 1;
    int vocab = w.config.unpadded_vocab_size;
    // The parity GEMM uses 32-row tiles. Reuse one tile even when all logits
    // from a long prefill are requested.
    int tile = std::min(rows, 32);
    auto* raw = ws.Take<float>(size_t(tile) * vocab);
    auto* rounded = ws.Take<uint16_t>(size_t(tile) * vocab);
    for (int row = 0; row < rows; row += tile) {
      int count = std::min(tile, rows - row);
      RunParityRowMatmul(g, ws.rest, hidden + size_t(t - rows + row) * h, w.unembed_weight, raw,
                         count, vocab, h, FLINT_BF16, FLINT_F32, true);
      CheckCuda(flint_cast({raw, rounded, count * vocab, FLINT_F32, FLINT_BF16}, stream));
      float* logits = o.return_logits ? s.d_logits_out + size_t(row) * vocab : raw;
      CheckCuda(flint_logit_softcap_argmax(
          {rounded, logits, s.d_next_token + row, count, vocab, vocab, FLINT_BF16}, stream));
    }
    stage("lm_head", 2ull * h * vocab + 6ull * h);
  }
}

void InklingRunner::Impl::PrepareGraph(const ForwardRunSpec& batch, const ForwardOptions& opts) {
  // Position is uploaded before each launch. KV reset also runs outside capture.
  auto captured = batch;
  captured.pos_offset = 0;
  captured.reset_kv = 0;
  const GraphKey requested{.begin_visit = batch.begin_visit,
                           .end_visit = batch.end_visit,
                           .seq_len = batch.seq_len,
                           .num_block_in = batch.num_block_in,
                           .run_embed = batch.run_embed,
                           .run_lm_head = batch.run_lm_head,
                           .is_decode = batch.is_decode,
                           .input_ids = batch.d_input_ids,
                           .hidden_in = batch.d_hidden_in,
                           .block_in = batch.d_block_in,
                           .hidden_out = batch.d_hidden_out,
                           .block_out = batch.d_block_out,
                           .logits_out = batch.d_logits_out,
                           .next_token = batch.d_next_token,
                           .use_cuda_graph = opts.use_cuda_graph,
                           .use_speculative = opts.use_speculative,
                           .return_logits = opts.return_logits,
                           .enable_profiling = opts.enable_profiling,
                           .profiler_range = opts.profiler_range};
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
ForwardRunResult InklingRunner::Run(const ForwardRunSpec& batch, const ForwardOptions& o,
                                    const ForwardCapture* capture) {
  auto& c = *impl_;
  auto s = batch;
  if (!s.seq_len || s.begin_visit > s.end_visit || s.end_visit > c.plan.size() || !s.d_hidden_out ||
      (s.run_embed ? !s.d_input_ids : !s.d_hidden_in) ||
      (s.run_lm_head && o.return_logits && !s.d_logits_out) ||
      size_t(s.seq_len) * c.weights.config.hidden_size > INT32_MAX ||
      s.pos_offset > c.kv.max_full_seq_len || s.seq_len > c.kv.max_full_seq_len - s.pos_offset)
    throw std::invalid_argument("invalid forward batch");
  if (c.verify_len && !c.recording) throw std::logic_error("commit pending verification first");
  if (s.run_lm_head && s.all_token_logits && s.seq_len > 1 && !s.d_next_token)
    throw std::invalid_argument("missing all-row token output");
  if (s.run_lm_head && !s.d_next_token) s.d_next_token = c.d_next_token;
  bool decode = s.is_decode && s.seq_len == 1;
  bool graph = o.use_cuda_graph && decode && !capture && !s.d_taps_out && !c.recording &&
               !o.enable_profiling;
  ForwardRunResult r{};
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
      CheckCuda(cudaMemcpyAsync(c.host_token,
                                s.d_next_token + (s.all_token_logits ? s.seq_len - 1 : 0),
                                sizeof(int32_t), cudaMemcpyDeviceToHost, c.stream));
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
ForwardRunResult InklingRunner::RunVerify(ForwardRunSpec s, const ForwardOptions& o) {
  auto& c = *impl_;
  if (!c.backup || c.verify_len || !s.seq_len || s.seq_len > 8 || s.reset_kv ||
      s.pos_offset > c.kv.max_full_seq_len || s.seq_len > c.kv.max_full_seq_len - s.pos_offset)
    throw std::invalid_argument("invalid verify block");
  s.begin_visit = 0;
  s.end_visit = model::kNumLogicalVisits;
  s.run_embed = s.run_lm_head = s.all_token_logits = 1;
  CheckCuda(flint_swa_kv_backup(c.kv.swa_kv, c.backup, s.pos_offset, s.seq_len, c.stream));
  c.verify_pos = s.pos_offset;
  c.verify_len = s.seq_len;
  c.recording = true;
  try {
    auto result = Run(s, o);
    c.recording = false;
    return result;
  } catch (...) {
    c.recording = false;
    CommitVerify(s.pos_offset, s.seq_len, 0);
    throw;
  }
}
void InklingRunner::CommitVerify(uint32_t pos, uint32_t len, uint32_t accepted) {
  auto& c = *impl_;
  if (!len || len != c.verify_len || pos != c.verify_pos || accepted > len)
    throw std::invalid_argument("invalid verify commit");
  if (accepted < len) {
    CheckCuda(flint_swa_kv_restore(c.kv.swa_kv, c.backup, pos, len, accepted, c.stream));
  }
  if (accepted) {
    CheckCuda(flint_sconv_commit_trace(c.kv_trace, c.kv.kv_conv_state,
                                       2 * model::kNumLogicalVisits * model::kKvDim, len, accepted,
                                       c.stream));
    CheckCuda(flint_sconv_commit_trace(c.sub_trace, c.kv.sublayer_conv_state,
                                       2 * model::kNumLogicalVisits * model::kHiddenSize, len,
                                       accepted, c.stream));
  }
  c.verify_len = 0;
}
}  // namespace flint

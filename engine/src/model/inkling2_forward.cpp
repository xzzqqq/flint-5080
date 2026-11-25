#include "model/inkling2_forward.h"

#include <cstdio>

#include "ops/scratch_workspace.h"
#include "ops/submodules.h"

namespace flint {
using ops_detail::CheckCuda;
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
}  // namespace
void RunInkling(const Gemm& g, const std::array<VisitPlanEntry, 48>& plan,
                const ModelWeightsView& w, const KvStateArenaView& kv, ScratchArenaView scratch,
                const ForwardRunSpec& s, const ForwardOptions& o, ForwardRunResult& result,
                const cudaEvent_t* marks, const ForwardCapture* capture,
                const ExpertProjection* experts, const int32_t* d_pos) {
  const int t = s.seq_len, h = w.config.hidden_size, n = t * h;
  const auto stream = g.stream();
  ScratchWorkspace ws(scratch);
  auto* hidden = static_cast<uint16_t*>(s.d_hidden_out);
  auto* carry = ws.Take<uint16_t>(size_t(FLINT_CARRY_PLANES) * n);
  auto block = [&](unsigned idx) { return carry + size_t(idx) * n; };
  auto copy = [&](void* dst, const void* src, size_t bytes) {
    if (dst && dst != src)
      CheckCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
  };
  size_t plane = size_t(n) * 2;
  unsigned count = s.num_block_in;
  unsigned expected = 0;
  for (unsigned v = 0; v < s.begin_visit; ++v) {
    if (v == 18 || v == 30) expected = 1;
    expected += plan[v].is_block_boundary;
  }
  if (count != expected || (count && !s.d_block_in) || (s.run_embed && s.begin_visit) ||
      (s.run_lm_head && s.end_visit != 48) || s.pos_offset > kv.max_full_seq_len ||
      s.seq_len > kv.max_full_seq_len - s.pos_offset)
    throw std::invalid_argument("invalid carry or range");
  if (s.reset_kv) CheckCuda(cudaMemsetAsync(kv.arena_base, 0, kv.arena_bytes, stream));
  if (count)
    copy(carry, s.d_block_in, FLINT_CARRY_PLANES * plane);
  else
    CheckCuda(cudaMemsetAsync(carry, 0, FLINT_CARRY_PLANES * plane, stream));
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
    RunBlockResidual(
        g, ws.rest,
        {blocks, x, query, q == 1 ? dst : packed, nullptr, nb, t, h, q, FLINT_BF16, type});
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
    if (visit.has_messages && visit.physical_layer == w.config.loop_core_start) {
      copy(block(6), hidden, plane);
      copy(block(7), block(2), plane);
      copy(hidden, block(4), plane);
      copy(block(0), block(5), plane);
      count = 1;
    }
    if (capture && (v == 5 || v == 23 || v == 41)) copy(capture->taps[(v - 5) / 18], hidden, plane);
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
    float* history = kv.sublayer_conv_state + size_t(v) * 2 * h * kv.conv_state_len;
    auto conv = [&](const void* weight, float* state) {
      CheckCuda(flint_static_sconv({tmp, weight, nullptr, state, tmp, t, h, FLINT_BF16, FLINT_BF16},
                                   stream));
    };
    conv(l.attn_sconv, history);
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
                      experts + visit.physical_layer * 2 * FLINT_NUM_EXPERTS);
    conv(l.mlp_sconv, history + h * kv.conv_state_len);
    CheckCuda(flint_add_residual(partial, tmp, hidden, n, FLINT_BF16, stream));
    if (v == 5) {
      copy(block(4), hidden, plane);
      copy(block(5), carry, plane);
    }
    if (capture) copy(capture->steps[v], hidden, plane);
    char name[32];
    std::snprintf(name, sizeof(name), "visit_%02u", v);
    stage(name, VisitBytes(w.config, l, visit));
  }
  copy(s.d_block_out, carry, plane * FLINT_CARRY_PLANES);
  if (s.num_block_out) *s.num_block_out = count;
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
}  // namespace flint

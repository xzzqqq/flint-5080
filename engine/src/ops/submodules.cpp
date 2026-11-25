#include "ops/submodules.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "ops/scratch_workspace.h"
namespace flint {
using ops_detail::CheckCuda;
using ops_detail::ScratchWorkspace;
void RunExl3Linear(const Gemm& g, ScratchArenaView scratch, const Exl3MatrixView& mat,
                   const void* x, void* y, int rows, FlintType input, FlintType output) {
  ScratchWorkspace ws(scratch);
  auto s = g.stream();
  int k = mat.k, n = mat.n;
  auto inner = ws.Take<uint16_t>(k * n);
  auto weight = ws.Take<float>(k * n);
  auto rotated = ws.Take<float>((size_t)rows * k);
  auto projected = ws.Take<float>((size_t)rows * n);
  CheckCuda(flint_exl3_decode_inner_f16({mat.trellis, inner, k, n, mat.k2}, s));
  CheckCuda(flint_cast({inner, weight, k * n, FLINT_F16, FLINT_F32}, s));
  CheckCuda(flint_wht128_scale_in({x, mat.suh, rotated, rows, k, input, FLINT_F32}, s));
  RunRowMatmul(g, ws.rest, rotated, weight, projected, rows, n, k, FLINT_F32, FLINT_F32, false);
  CheckCuda(flint_wht128_scale_out({projected, mat.svh, y, rows, n, FLINT_F32, output}, s));
}
namespace {
void AttentionChunk(const Gemm& g, ScratchArenaView scratch, const LayerWeightsView& l,
                    const KvStateArenaView& cache, AttentionBatch b, bool full) {
  int t = b.tokens, cap = full ? cache.max_full_seq_len : cache.swa_window;
  auto s = g.stream();
  ScratchWorkspace ws(scratch);
  auto q = ws.Take<uint16_t>(size_t(t) * 3072), k = ws.Take<uint16_t>(size_t(t) * 1024),
       v = ws.Take<uint16_t>(size_t(t) * 1024);
  auto rel = ws.Take<uint16_t>(size_t(t) * 384), bias = ws.Take<uint16_t>(size_t(t) * 24 * 1024),
       sink = ws.Take<uint16_t>(size_t(t) * 24);
  auto gated = ws.Take<uint16_t>(size_t(t) * 3072), gate = ws.Take<uint16_t>(size_t(t) * 3072);
  auto delta = ws.Take<uint16_t>(t * 32);
  auto scores = ws.Take<float>(size_t(cap) * 24);
  RunExl3Linear(g, ws.rest, l.q_proj, b.q, q, t);
  CheckCuda(flint_q_norm_clip({q, l.q_l2_clip_scale, q, t, b.position, full}, s));
  auto conv_state = cache.kv_conv_state + size_t(b.visit.visit_idx) * 2 * 1024 * 3;
  for (int j = 0; j < 2; ++j) {
    const void* input = j ? b.v : b.k;
    auto kv = j ? v : k;
    RunExl3Linear(g, ws.rest, j ? l.v_proj : l.k_proj, input, kv, t);
    ConvParams conv{kv,        j ? l.v_sconv : l.k_sconv,
                    delta,     conv_state + j * 1024 * 3,
                    kv,        t,
                    1024,      FLINT_BF16,
                    FLINT_BF16};
    if (full) {
      RunRowMatmul(g, ws.rest, input, j ? l.v_weight_proj : l.k_weight_proj, delta, t, 32, 2304,
                   FLINT_BF16, FLINT_BF16, true);
      conv.weight = j ? l.v_static_w : l.k_static_w;
      CheckCuda(flint_dynamic_sconv(conv, s));
    } else
      CheckCuda(flint_static_sconv(conv, s));
  }
  CheckCuda(flint_k_l2_clip({k, nullptr, k, t, b.position, 0}, s));
  CheckCuda(
      flint_unweighted_rmsnorm({v, nullptr, v, nullptr, t * 8, 128, FLINT_BF16, FLINT_BF16}, s));
  RunRowMatmul(g, ws.rest, b.q, l.wr_du, rel, t, 384, 2304, FLINT_BF16, FLINT_BF16, true);
  CheckCuda(flint_rel_norm_clip({rel, l.rel_l2_clip_scale, rel, t, b.position, full}, s));
  RunRowMatmul(g, ws.rest, rel, l.rel_logits_proj, bias, t * 24, 1024, 16, FLINT_BF16, FLINT_BF16,
               false);
  RunRowMatmul(g, ws.rest, b.q, l.attn_sink_proj, sink, t, 24, 2304, FLINT_BF16, FLINT_BF16, true);
  auto ck = static_cast<uint16_t*>(full ? cache.full_kv : cache.swa_kv) +
            size_t(b.visit.kv_slot_idx) * 2 * cap * 1024;
  auto cv = ck + size_t(cap) * 1024;
  // Interleave writes and reads so later prefill tokens cannot evict a visible key.
  for (int row = 0; row < t; ++row) {
    CheckCuda(flint_kv_cache_write(
        {k + row * 1024, v + row * 1024, ck, cv, 1, b.position + row, cap, !full}, s));
    CheckCuda(
        flint_attention_forward({q + row * 3072, ck, cv, bias + row * 24 * 1024, sink + row * 24,
                                 gated + row * 3072, scores, 1, b.position + row, cap, !full},
                                s));
  }
  RunExl3Linear(g, ws.rest, l.o_gate_proj, b.q, gate, t);
  CheckCuda(flint_gated_mul_sigmoid({gated, gate, gated, t * 3072}, s));
  RunExl3Linear(g, ws.rest, l.o_proj, gated, b.y, t);
}
void Attention(const Gemm& g, ScratchArenaView scratch, const LayerWeightsView& l,
               const KvStateArenaView& cache, AttentionBatch b, bool full) {
  int t = b.tokens, cap = full ? cache.max_full_seq_len : cache.swa_window;
  if (t <= 0 || b.position < 0 || (full && b.position > cap - t) ||
      full != bool(l.is_full_attention) || full != bool(b.visit.is_full_attention) ||
      b.visit.visit_idx >= cache.num_total_visits ||
      b.visit.kv_slot_idx >= (full ? cache.num_full_visits : cache.num_swa_visits))
    throw std::invalid_argument("invalid attention batch or visit");
  // Bound relative-logit and activation scratch independently of prompt length.
  for (int row = 0; row < t;) {
    int count = std::min(32, t - row);
    size_t offset = size_t(row) * 2304;
    AttentionChunk(
        g, scratch, l, cache,
        {static_cast<const uint16_t*>(b.q) + offset, static_cast<const uint16_t*>(b.k) + offset,
         static_cast<const uint16_t*>(b.v) + offset, static_cast<uint16_t*>(b.y) + offset, count,
         b.position + row, b.visit},
        full);
    row += count;
  }
}
}  // namespace
void RunSlidingAttention(const Gemm& g, ScratchArenaView w, const LayerWeightsView& l,
                         const KvStateArenaView& c, AttentionBatch b) {
  Attention(g, w, l, c, b, false);
}
void RunFullAttention(const Gemm& g, ScratchArenaView w, const LayerWeightsView& l,
                      const KvStateArenaView& c, AttentionBatch b) {
  Attention(g, w, l, c, b, true);
}
void RunMessageGate(const Gemm& g, ScratchArenaView w, const ControllerWeightsView& ctl,
                    const float* x, const float* msg, float* y, int t) {
  ScratchWorkspace ws(w);
  auto prep = ws.Take<uint16_t>(size_t(t) * 4608), z = ws.Take<uint16_t>(t * 64);
  auto forget = ws.Take<float>(size_t(t) * 2304), gain = ws.Take<float>(size_t(t) * 2304);
  CheckCuda(flint_message_gate_prep({x, msg, prep, t, 2304}, g.stream()));
  RunRowMatmul(g, ws.rest, prep, ctl.down_weight, z, t, 64, 4608, FLINT_BF16, FLINT_BF16, true);
  RunFp32Linear(g, ws.rest, z, FLINT_BF16, ctl.forget_weight, FLINT_BF16, forget, t, 2304, 64);
  RunFp32Linear(g, ws.rest, z, FLINT_BF16, ctl.gain_weight, FLINT_BF16, gain, t, 2304, 64);
  CheckCuda(flint_message_gate_mix({x, msg, forget, gain, y, t * 2304}, g.stream()));
}
void RunSparseMoeBlock(const Gemm& g, ScratchArenaView w, const LayerWeightsView& l, const void* x,
                       void* y, int t, float* logits, int* ids, float* weights) {
  ScratchWorkspace ws(w);
  if (!logits) logits = ws.Take<float>(t * 128);
  if (!ids) ids = ws.Take<int>(t * 8);
  if (!weights) weights = ws.Take<float>(t * 8);
  auto outputs = ws.Take<uint16_t>(size_t(t) * 8 * 2304), gate_up = ws.Take<uint16_t>(2304),
       act = ws.Take<uint16_t>(1152);
  RunFp32Linear(g, ws.rest, x, FLINT_BF16, l.router_weight, FLINT_BF16, logits, t, 128, 2304);
  CheckCuda(flint_moe_router_top8({logits, l.router_bias, ids, weights, t}, g.stream()));
  std::vector<int> selected(size_t(t) * 8);
  CheckCuda(cudaMemcpyAsync(selected.data(), ids, selected.size() * sizeof(int),
                            cudaMemcpyDeviceToHost, g.stream()));
  CheckCuda(cudaStreamSynchronize(g.stream()));
  for (int i = 0; i < t * 8; ++i) {
    int e = selected[i];
    RunExl3Linear(g, ws.rest, l.gate_up_proj[e], static_cast<const uint16_t*>(x) + (i / 8) * 2304,
                  gate_up, 1);
    CheckCuda(flint_softcapped_swiglu({gate_up, act, 1, 1152}, g.stream()));
    RunExl3Linear(g, ws.rest, l.down_proj[e], act, outputs + i * 2304, 1);
  }
  CheckCuda(flint_moe_weighted_accum({outputs, weights, y, t, 2304, ids}, g.stream()));
}
}  // namespace flint

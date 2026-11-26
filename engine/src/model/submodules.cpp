#include "model/submodules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/cuda_check.h"
#include "loader/kv_layout.h"
#include "loader/scratch_workspace.h"
#include "ops/attention_ops.h"
#include "ops/moe_ops.h"
#include "ops/norm_ops.h"
#include "ops/sconv_ops.h"
namespace flint {

void RunParityRowMatmul(const Gemm& g, ScratchArenaView scratch, const void* a, const void* b,
                        void* c, int m, int n, int k, FlintType input, FlintType output,
                        bool transpose) {
  ScratchWorkspace ws(scratch);
  size_t stride_a = size_t(k) * (input == FLINT_F32 ? 4 : 2),
         stride_c = size_t(n) * (output == FLINT_F32 ? 4 : 2);
  auto x = static_cast<const char*>(a);
  auto y = static_cast<char*>(c);
  while (m >= 32) {
    g.Multiply(x, b, y, 32, n, k, input, output, transpose);
    x += 32 * stride_a;
    y += 32 * stride_c;
    m -= 32;
  }
  if (!m) return;
  auto padded = ws.Take<char>(32 * stride_a), out = ws.Take<char>(32 * stride_c);
  CheckCuda(cudaMemcpyAsync(padded, x, m * stride_a, cudaMemcpyDeviceToDevice, g.stream()));
  CheckCuda(cudaMemsetAsync(padded + m * stride_a, 0, (32 - m) * stride_a, g.stream()));
  g.Multiply(padded, b, out, 32, n, k, input, output, transpose);
  CheckCuda(cudaMemcpyAsync(y, out, m * stride_c, cudaMemcpyDeviceToDevice, g.stream()));
}
void RunFp32Linear(const Gemm& g, ScratchArenaView scratch, const void* x, FlintType input,
                   const void* weight, FlintType weight_type, float* out, int rows, int n, int k) {
  ScratchWorkspace ws(scratch);
  auto promote = [&](const void* src, FlintType type, int count) -> const void* {
    if (type == FLINT_F32) return src;
    auto dst = ws.Take<float>(count);
    CheckCuda(flint_cast({src, dst, count, type, FLINT_F32}, g.stream()));
    return dst;
  };
  auto a = promote(x, input, rows * k);
  auto b = promote(weight, weight_type, n * k);
  RunParityRowMatmul(g, ws.rest, a, b, out, rows, n, k, FLINT_F32, FLINT_F32, true);
}

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
  RunParityRowMatmul(g, ws.rest, rotated, weight, projected, rows, n, k, FLINT_F32, FLINT_F32,
                     false);
  CheckCuda(flint_wht128_scale_out({projected, mat.svh, y, rows, n, FLINT_F32, output}, s));
}
namespace {
void AttentionChunk(const Gemm& g, ScratchArenaView scratch, const LayerWeightsView& l,
                    const KvStateArenaView& cache, AttentionBatch b, bool full) {
  int t = b.tokens;
  auto [ck, cv, cap] = VisitKvCache(cache, b.visit);
  auto s = g.stream();
  ScratchWorkspace ws(scratch);
  auto q = ws.Take<uint16_t>(size_t(t) * model::kQDim),
       k = ws.Take<uint16_t>(size_t(t) * model::kKvDim),
       v = ws.Take<uint16_t>(size_t(t) * model::kKvDim);
  auto rel = ws.Take<uint16_t>(size_t(t) * model::kRelProjDim),
       bias = ws.Take<uint16_t>(size_t(t) * model::kNumQHeads * model::kRelExtent),
       sink = ws.Take<uint16_t>(size_t(t) * model::kNumQHeads);
  auto gated = ws.Take<uint16_t>(size_t(t) * model::kQDim),
       gate = ws.Take<uint16_t>(size_t(t) * model::kQDim);
  auto delta = ws.Take<uint16_t>(t * model::kDynConvProjDim);
  auto scores = ws.Take<float>(size_t(cap) * model::kNumQHeads);
  RunExl3Linear(g, ws.rest, l.q_proj, b.q, q, t);
  CheckCuda(flint_q_norm_clip({q, l.q_l2_clip_scale, q, t, b.position, full, b.d_pos}, s));
  for (int j = 0; j < 2; ++j) {
    const void* input = j ? b.v : b.k;
    auto kv = j ? v : k;
    RunExl3Linear(g, ws.rest, j ? l.v_proj : l.k_proj, input, kv, t);
    ConvParams conv{kv,
                    j ? l.v_sconv : l.k_sconv,
                    delta,
                    VisitKvConvState(cache, b.visit.visit_idx, j),
                    kv,
                    t,
                    model::kKvDim,
                    FLINT_BF16,
                    FLINT_BF16};
    if (full) {
      RunParityRowMatmul(g, ws.rest, input, j ? l.v_weight_proj : l.k_weight_proj, delta, t,
                         model::kDynConvProjDim, model::kHiddenSize, FLINT_BF16, FLINT_BF16, true);
      conv.weight = j ? l.v_static_w : l.k_static_w;
      CheckCuda(flint_dynamic_sconv(conv, s));
    } else
      CheckCuda(flint_static_sconv(conv, s));
  }
  CheckCuda(flint_k_l2_clip({k, nullptr, k, t, b.position, 0}, s));
  CheckCuda(flint_unweighted_rmsnorm(
      {v, nullptr, v, nullptr, t * model::kNumKvHeads, model::kHeadDim, FLINT_BF16, FLINT_BF16},
      s));
  RunParityRowMatmul(g, ws.rest, b.q, l.wr_du, rel, t, model::kRelProjDim, model::kHiddenSize,
                     FLINT_BF16, FLINT_BF16, true);
  CheckCuda(flint_rel_norm_clip({rel, l.rel_l2_clip_scale, rel, t, b.position, full, b.d_pos}, s));
  RunParityRowMatmul(g, ws.rest, rel, l.rel_logits_proj, bias, t * model::kNumQHeads,
                     model::kRelExtent, model::kRelDim, FLINT_BF16, FLINT_BF16, false);
  RunParityRowMatmul(g, ws.rest, b.q, l.attn_sink_proj, sink, t, model::kNumQHeads,
                     model::kHiddenSize, FLINT_BF16, FLINT_BF16, true);
  // Interleave writes and reads so later prefill tokens cannot evict a visible key.
  for (int row = 0; row < t; ++row) {
    CheckCuda(flint_kv_cache_write({k + row * model::kKvDim, v + row * model::kKvDim, ck, cv, 1,
                                    b.position + row, cap, !full, b.d_pos},
                                   s));
    CheckCuda(flint_attention_forward(
        {q + row * model::kQDim, ck, cv, bias + row * model::kNumQHeads * model::kRelExtent,
         sink + row * model::kNumQHeads, gated + row * model::kQDim, scores, 1, b.position + row,
         cap, !full, b.d_pos},
        s));
  }
  RunExl3Linear(g, ws.rest, l.o_gate_proj, b.q, gate, t);
  CheckCuda(flint_gated_mul_sigmoid({gated, gate, gated, t * model::kQDim}, s));
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
    int count = std::min(model::kAttentionChunkTokens, t - row);
    size_t offset = size_t(row) * model::kHiddenSize;
    AttentionChunk(
        g, scratch, l, cache,
        {static_cast<const uint16_t*>(b.q) + offset, static_cast<const uint16_t*>(b.k) + offset,
         static_cast<const uint16_t*>(b.v) + offset, static_cast<uint16_t*>(b.y) + offset, count,
         b.position + row, b.visit, b.d_pos},
        full);
    row += count;
  }
}
void GroupedMoe(cudaStream_t stream, ScratchWorkspace ws, const ExpertProjection* experts,
                const void* x, void* y, const int* ids, const float* weights, int t) {
  using namespace model;
  int rows = t * kTopKExperts, tile_rows = flint_exl3_grouped_gemm_tile_rows();
  int max_tiles = std::min(rows, rows / tile_rows + kNumExperts);
  MoeGroupParams group{ids,
                       ws.Take<int>(rows),
                       ws.Take<int>(rows),
                       ws.Take<int>(rows),
                       ws.Take<int>(max_tiles),
                       ws.Take<int>(max_tiles),
                       ws.Take<int>(max_tiles),
                       ws.Take<int>(1),
                       t,
                       tile_rows};
  CheckCuda(flint_moe_group_tokens(group, stream));
  static_assert(kGateUpSize == kHiddenSize && kIntermediateSize <= kHiddenSize);
  size_t elements = size_t(rows) * kHiddenSize;
  auto gate_up = ws.Take<uint16_t>(elements);
  auto act = ws.Take<uint16_t>(size_t(rows) * kIntermediateSize);
  auto rotated = ws.Take<float>(elements);
  auto projected = ws.Take<float>(elements);
  auto linear = [&](const ExpertProjection* table, const void* input, uint16_t* output, int k,
                    const int* source_rows) {
    ExpertSelection selection{table, group.grouped_expert};
    CheckCuda(flint_exl3_grouped_rotate(selection,
                                        {input, nullptr, rotated, rows, k, FLINT_BF16, FLINT_F32},
                                        1, source_rows, stream));
    CheckCuda(
        flint_exl3_grouped_gemm({table, group.tile_expert, group.tile_begin, group.tile_end,
                                 group.num_tiles, rotated, projected, max_tiles, k, kHiddenSize},
                                stream));
    CheckCuda(flint_exl3_grouped_rotate(
        selection, {projected, nullptr, output, rows, kHiddenSize, FLINT_F32, FLINT_BF16}, 0,
        nullptr, stream));
  };
  linear(experts, x, gate_up, kHiddenSize, group.grouped_token);
  CheckCuda(flint_softcapped_swiglu({gate_up, act, rows, kIntermediateSize}, stream));
  linear(experts + kNumExperts, act, gate_up, kIntermediateSize, nullptr);
  CheckCuda(flint_moe_weighted_accum(
      {gate_up, weights, y, t, kHiddenSize, ids, group.slot_to_grouped}, stream));
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
void RunBlockResidual(const Gemm& g, ScratchArenaView scratch, ResidualParams p) {
  // Keep the two per-token FP32 matrix products in the reference orientation.
  // Even one changed BF16 input can select a different expert later in the loop.
  ScratchWorkspace ws(scratch);
  int nb = p.num_blocks + 1, h = p.width, nq = p.queries;
  auto states = ws.Take<float>(size_t(p.tokens) * nb * h);
  auto scores = ws.Take<float>(size_t(p.tokens) * nq * nb);
  CheckCuda(flint_residual_norm_states(p, states, g.stream()));
  for (int t = 0; t < p.tokens; ++t)
    g.Multiply(p.query, states + t * nb * h, scores + t * nq * nb, nq, nb, h, FLINT_F32, FLINT_F32,
               true);
  CheckCuda(
      flint_residual_softmax(scores, p.tokens * nq, nb, 1.f / std::sqrt(float(h)), g.stream()));
  for (int t = 0; t < p.tokens; ++t)
    g.Multiply(scores + t * nq * nb, states + t * nb * h, p.y + t * nq * h, nq, h, nb, FLINT_F32,
               FLINT_F32, false);
}
void RunMessageGate(const Gemm& g, ScratchArenaView w, const ControllerWeightsView& ctl,
                    const float* x, const float* msg, float* y, int t) {
  ScratchWorkspace ws(w);
  auto prep = ws.Take<uint16_t>(size_t(t) * model::kMessageGateInDim),
       z = ws.Take<uint16_t>(t * model::kLoopGateRank);
  auto forget = ws.Take<float>(size_t(t) * model::kHiddenSize),
       gain = ws.Take<float>(size_t(t) * model::kHiddenSize);
  CheckCuda(flint_message_gate_prep({x, msg, prep, t, model::kHiddenSize}, g.stream()));
  RunParityRowMatmul(g, ws.rest, prep, ctl.down_weight, z, t, model::kLoopGateRank,
                     model::kMessageGateInDim, FLINT_BF16, FLINT_BF16, true);
  RunFp32Linear(g, ws.rest, z, FLINT_BF16, ctl.forget_weight, FLINT_BF16, forget, t,
                model::kHiddenSize, model::kLoopGateRank);
  RunFp32Linear(g, ws.rest, z, FLINT_BF16, ctl.gain_weight, FLINT_BF16, gain, t, model::kHiddenSize,
                model::kLoopGateRank);
  CheckCuda(flint_message_gate_mix({x, msg, forget, gain, y, t * model::kHiddenSize}, g.stream()));
}
void RunSparseMoeBlock(const Gemm& g, ScratchArenaView w, const LayerWeightsView& l, const void* x,
                       void* y, int t, const ExpertProjection* experts, float* logits, int* ids,
                       float* weights) {
  if (!experts) throw std::invalid_argument("missing expert projection table");
  ScratchWorkspace ws(w);
  if (!logits) logits = ws.Take<float>(t * model::kNumExperts);
  if (!ids) ids = ws.Take<int>(t * model::kTopKExperts);
  if (!weights) weights = ws.Take<float>(t * model::kTopKExperts);
  RunFp32Linear(g, ws.rest, x, FLINT_BF16, l.router_weight, FLINT_BF16, logits, t,
                model::kNumExperts, model::kHiddenSize);
  CheckCuda(flint_moe_router_top8({logits, l.router_bias, ids, weights, t}, g.stream()));
  if (t == 1) {
    auto outputs = ws.Take<uint16_t>(model::kTopKExperts * model::kHiddenSize);
    auto gate_up = ws.Take<uint16_t>(model::kTopKExperts * model::kGateUpSize),
         act = ws.Take<uint16_t>(model::kTopKExperts * model::kIntermediateSize);
    auto rotated = ws.Take<float>(model::kTopKExperts * model::kHiddenSize),
         partial = ws.Take<float>(model::kTopKExperts * flint_exl3_indexed_gemv_k_splits() *
                                  model::kHiddenSize);
    auto linear = [&](ExpertSelection selection, const void* input, void* output, int k,
                      bool broadcast) {
      constexpr int n = model::kHiddenSize;
      CheckCuda(flint_exl3_indexed_rotate(
          selection, {input, nullptr, rotated, model::kTopKExperts, k, FLINT_BF16, FLINT_F32}, 1,
          broadcast, g.stream()));
      CheckCuda(flint_exl3_indexed_gemv(selection, rotated, partial, k, n, g.stream()));
      CheckCuda(flint_exl3_indexed_rotate(
          selection, {partial, nullptr, output, model::kTopKExperts, n, FLINT_F32, FLINT_BF16}, 0,
          false, g.stream()));
    };
    linear({experts, ids}, x, gate_up, model::kHiddenSize, true);
    CheckCuda(flint_softcapped_swiglu({gate_up, act, model::kTopKExperts, model::kIntermediateSize},
                                      g.stream()));
    linear({experts + model::kNumExperts, ids}, act, outputs, model::kIntermediateSize, false);
    CheckCuda(flint_moe_weighted_accum({outputs, weights, y, t, model::kHiddenSize, ids, nullptr},
                                       g.stream()));
    return;
  }
  // Large configured chunks or smaller scratch arenas may need multiple groups.
  // Budget at most one tile per slot and one alignment gap per scratch allocation.
  constexpr size_t bytes_per_token =
      model::kTopKExperts * (model::kHiddenSize * (2 * sizeof(float) + sizeof(uint16_t)) +
                             model::kIntermediateSize * sizeof(uint16_t) + 6 * sizeof(int));
  constexpr size_t overhead = 11 * 256;
  size_t capacity = ws.rest.bytes > overhead ? (ws.rest.bytes - overhead) / bytes_per_token : 0;
  if (!capacity) throw std::length_error("insufficient grouped MoE scratch");
  for (int row = 0; row < t;) {
    int count = std::min<size_t>(t - row, capacity);
    size_t offset = size_t(row) * model::kHiddenSize;
    GroupedMoe(g.stream(), ws, experts, static_cast<const uint16_t*>(x) + offset,
               static_cast<uint16_t*>(y) + offset, ids + row * model::kTopKExperts,
               weights + row * model::kTopKExperts, count);
    row += count;
  }
}
}  // namespace flint

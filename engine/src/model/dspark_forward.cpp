#include "model/dspark_forward.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "common/cuda_check.h"
#include "loader/scratch_workspace.h"
#include "ops/exl3_ops.h"
#include "ops/norm_ops.h"

namespace flint {
namespace {
thread_local bool default_draft_graph = true;
constexpr int H = 2304, Q = 3072, K = 1024, I = 9216, R = 256, V = 200058;
constexpr int rnn_weights = 3 * R * (2 * H + 2 * R);
constexpr size_t workspace_bytes = 4u << 20;
void* GemmWorkspace(ScratchArenaView& scratch) {
  ScratchWorkspace ws(scratch);
  auto result = ws.Take<char>(workspace_bytes);
  scratch = ws.rest;
  return result;
}
void Copy(void* dst, const void* src, size_t bytes, cudaStream_t s) {
  if (dst) CheckCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s));
}
}  // namespace
ScratchArenaView DsparkRunner::ReserveScratch(ScratchArenaView& shared) {
  // Keep the unfolded workspace for smaller, explicitly configured arenas.
  size_t bytes = shared.bytes >= ((144ull + 64) << 20) ? 144ull << 20 : 32ull << 20;
  ScratchWorkspace ws(shared);
  auto draft = ws.Take<char>(bytes);
  shared = ws.rest;
  return {draft, bytes};
}
DsparkRunner::DsparkRunner(const ModelWeightsView& w, const KvStateArenaView& kv,
                           ScratchArenaView scratch, cudaStream_t stream)
    : w_(w), kv_(kv), scratch_(scratch), gemm_(stream, GemmWorkspace(scratch_), workspace_bytes) {
  if (!w.has_mtp || !kv.has_mtp_kv) throw std::invalid_argument("missing draft arena");
  ScratchWorkspace ws(scratch_);
  if (scratch.bytes >= (128ull << 20)) {
    folded_w_ = ws.Take<uint16_t>(size_t(V) * R);
    folded_b_ = ws.Take<uint16_t>(V);
    rnn_proj_f32_ = ws.Take<float>(rnn_weights);
    rnn_bias_f32_ = ws.Take<float>(3 * R);
    gemm_.Multiply(w.unembed_weight, w.mtp.hidden_out, folded_w_, V, R, H, FLINT_BF16, FLINT_BF16,
                   false);
    gemm_.Multiply(w.mtp.hidden_bias, w.unembed_weight, folded_b_, 1, V, H, FLINT_BF16, FLINT_BF16,
                   true);
    CheckCuda(
        flint_cast({w.mtp.rnn_proj, rnn_proj_f32_, rnn_weights, FLINT_BF16, FLINT_F32}, stream));
    CheckCuda(flint_cast({w.mtp.rnn_bias, rnn_bias_f32_, 3 * R, FLINT_BF16, FLINT_F32}, stream));
  }
  d_anchor_ = ws.Take<int32_t>(2);
  d_ctx_ = d_anchor_ + 1;
  d_ids_ = ws.Take<int32_t>(7);
  scratch_ = ws.rest;
  CheckCuda(cudaHostAlloc(&h_ids_, 7 * sizeof(*h_ids_), cudaHostAllocDefault));
}
DsparkRunner::~DsparkRunner() {
  if (draft_graph_) cudaGraphExecDestroy(draft_graph_);
  if (draft_source_graph_) cudaGraphDestroy(draft_source_graph_);
  if (h_ids_) cudaFreeHost(h_ids_);
}
// Forge loads each candidate in its own DSO. The optional control keeps
// older modules loadable through the unchanged draft ABI.
extern "C" void flint_forward_set_draft_graph(bool enabled) { default_draft_graph = enabled; }
bool DsparkRunner::DefaultDraftGraph() { return default_draft_graph; }

DraftKv DsparkRunner::Cache(int layer) const {
  bool local = w_.config.mtp.is_local_layer[layer];
  int cap = local ? kv_.mtp_swa_window : kv_.max_full_seq_len;
  size_t plane = size_t(cap) * K;
  auto* k = static_cast<uint16_t*>(local ? kv_.mtp_swa_kv : kv_.mtp_full_kv) +
            (layer == 2 ? 2 * plane : 0);
  return {k, k + plane, cap, local};
}
void DsparkRunner::StoreContextKv(const void* taps, uint32_t tokens, uint32_t pos,
                                  DsparkCapture* cap) {
  if (!taps || !tokens || pos > kv_.max_full_seq_len || tokens > kv_.max_full_seq_len - pos)
    throw std::invalid_argument("invalid draft context");
  auto s = gemm_.stream();
  for (unsigned row = 0; row < tokens;) {
    int n = std::min(128u, tokens - row);
    ScratchWorkspace ws(scratch_);
    auto x = ws.Take<uint16_t>(n * H), k = ws.Take<uint16_t>(n * K), v = ws.Take<uint16_t>(n * K);
    auto src = static_cast<const uint16_t*>(taps) + size_t(row) * 4 * H;
    for (int i = 0; i < (cap && cap->mix[3] ? 4 : 3); ++i) {
      const auto& w = w_.mtp;
      auto norm = i == 3 ? w.dflare_norm : w.layers[i].dflare_norm;
      CheckCuda(flint_dspark_mix_rmsnorm(src, w.fuse_alpha + 4 * i, norm, x, n, s));
      if (cap && cap->mix[i])
        Copy(static_cast<uint16_t*>(cap->mix[i]) + size_t(row) * H, x, n * H * 2, s);
      if (i == 3) continue;
      const auto& l = w.layers[i];
      gemm_.Multiply(x, l.k_proj_ctx, k, n, K, H, FLINT_BF16, FLINT_BF16, true);
      gemm_.Multiply(x, l.v_proj_ctx, v, n, K, H, FLINT_BF16, FLINT_BF16, true);
      CheckCuda(flint_dspark_qk_norm_rope(k, l.k_norm_ctx, n, 8, pos + row, s));
      CheckCuda(flint_dspark_kv_write(Cache(i), k, v, n, pos + row, s));
    }
    row += n;
  }
}
void DsparkRunner::ValidateBlock(int32_t anchor, const void* tap, uint32_t ctx,
                                 const int32_t* ids) const {
  if (!tap || !ids || anchor < 0 || anchor >= V || ctx > kv_.max_full_seq_len ||
      kv_.max_full_seq_len - ctx < 8)
    throw std::invalid_argument("invalid draft block");
}
void DsparkRunner::DraftGreedy(int32_t anchor, const void* tap, uint32_t ctx, int32_t* ids,
                               DsparkCapture* cap, bool use_cuda_graph) {
  ValidateBlock(anchor, tap, ctx, ids);
  auto s = gemm_.stream();
  if (!cap && use_cuda_graph) {
    if (!draft_graph_ || draft_graph_tap_ != tap) {
      if (draft_graph_) CheckCuda(cudaGraphExecDestroy(draft_graph_));
      draft_graph_ = nullptr;
      if (draft_source_graph_) CheckCuda(cudaGraphDestroy(draft_source_graph_));
      draft_source_graph_ = nullptr;
      CheckCuda(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      try {
        Draft(anchor, tap, ctx, ids, nullptr, {}, nullptr, true);
      } catch (...) {
        cudaStreamEndCapture(s, &draft_source_graph_);
        throw;
      }
      CheckCuda(cudaStreamEndCapture(s, &draft_source_graph_));
      CheckCuda(cudaGraphInstantiate(&draft_graph_, draft_source_graph_, 0));
      draft_graph_tap_ = tap;
    }
    // The pinned staging buffer is reusable in stream order after the upload.
    h_ids_[0] = anchor;
    h_ids_[1] = ctx;
    CheckCuda(cudaMemcpyAsync(d_anchor_, h_ids_, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, s));
    CheckCuda(cudaGraphLaunch(draft_graph_, s));
  } else {
    Draft(anchor, tap, ctx, ids, nullptr, {}, cap);
    if (cap) return;
  }
  CheckCuda(cudaMemcpyAsync(h_ids_, d_ids_, 7 * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
  CheckCuda(cudaStreamSynchronize(s));
  std::copy_n(h_ids_, 7, ids);
}
void DsparkRunner::DraftWithLogits(int32_t anchor, const void* tap, uint32_t ctx, int32_t* ids,
                                   float* logits, const SampleRow& sample) {
  if (!logits || !sample) throw std::invalid_argument("missing draft logits or sampler");
  ValidateBlock(anchor, tap, ctx, ids);
  Draft(anchor, tap, ctx, ids, logits, sample, nullptr);
}
void DsparkRunner::Draft(int32_t anchor, const void* tap, uint32_t ctx, int32_t* ids, float* logits,
                         const SampleRow& sample, DsparkCapture* cap, bool graph) {
  auto d_anchor = graph ? d_anchor_ : nullptr;
  auto d_ctx = graph ? d_ctx_ : nullptr;
  auto s = gemm_.stream();
  ScratchWorkspace ws(scratch_);
  auto x = ws.Take<uint16_t>(8 * H), h = ws.Take<uint16_t>(8 * H), f = ws.Take<uint16_t>(H);
  const auto& w = w_.mtp;
  auto linear = [&](const void* src, const void* weight, void* dest, int m, int n, int k) {
    gemm_.Multiply(src, weight, dest, m, n, k, FLINT_BF16, FLINT_BF16, true);
  };
  auto norm = [&](const void* weight) {
    CheckCuda(flint_rmsnorm({x, weight, h, nullptr, 8, H, FLINT_BF16, FLINT_BF16}, s));
  };
  CheckCuda(flint_dspark_embed_block(w_.embed_weight, w_.embed_norm_weight, w.mask_embed, anchor, x,
                                     s, d_anchor));
  if (cap) Copy(cap->embed_block, x, 8 * H * 2, s);
  for (int i = 0; i < 3; ++i) {
    const auto& l = w.layers[i];
    ScratchWorkspace layer(ws.rest);
    auto q = layer.Take<uint16_t>(8 * Q), k = layer.Take<uint16_t>(8 * K),
         v = layer.Take<uint16_t>(8 * K);
    auto gate = layer.Take<uint16_t>(8 * I), up = layer.Take<uint16_t>(8 * I);
    norm(l.input_layernorm);
    linear(h, l.q_proj, q, 8, Q, H);
    linear(h, l.k_proj, k, 8, K, H);
    linear(h, l.v_proj, v, 8, K, H);
    CheckCuda(flint_dspark_qk_norm_rope(q, l.q_norm, 8, 24, ctx, s, d_ctx));
    CheckCuda(flint_dspark_qk_norm_rope(k, l.k_norm, 8, 8, ctx, s, d_ctx));
    CheckCuda(flint_dspark_kv_write(Cache(i), k, v, 8, ctx, s, d_ctx));
    CheckCuda(flint_dspark_block_attention(Cache(i), q, gate,
                                           static_cast<float*>(layer.rest.device_ptr),
                                           layer.rest.bytes, gemm_.handle(), ctx, s, d_ctx));
    linear(gate, l.o_proj, h, 8, H, Q);
    CheckCuda(flint_dspark_add_bf16(x, h, 8 * H, s));
    norm(l.post_attention_layernorm);
    linear(h, l.gate_proj, gate, 8, I, H);
    linear(h, l.up_proj, up, 8, I, H);
    CheckCuda(flint_dspark_silu_mul(gate, up, 8 * I, s));
    linear(gate, l.down_proj, h, 8, H, I);
    CheckCuda(flint_dspark_add_bf16(x, h, 8 * H, s));
    if (cap) Copy(cap->layer_out[i], x, 8 * H * 2, s);
  }
  norm(w.norm);
  CheckCuda(flint_dspark_mix_rmsnorm(tap, w.fuse_alpha + 12, w.dflare_norm, f, 1, s));
  if (cap) {
    Copy(cap->head_hidden, h, 8 * H * 2, s);
    Copy(cap->f_rnn, f, H * 2, s);
  }
  auto base = ws.Take<uint16_t>(7 * V), bias_logits = ws.Take<uint16_t>(V);
  linear(h + H, w_.unembed_weight, base, 7, V, H);
  auto state = ws.Take<uint16_t>(2 * R), latent = ws.Take<uint16_t>(R);
  auto features = ws.Take<float>(7 * 2 * H);
  auto context = ws.Take<float>(7 * 3 * R), rec = ws.Take<float>(3 * R),
       rnn_input = ws.Take<float>(2 * R);
  const bool device_greedy = !sample && !cap;
  auto d_logits =
      logits || (cap && (cap->h_top_ids || cap->h_top_logits)) ? ws.Take<float>(V) : nullptr;
  auto winners = ws.Take<HeadWinner>(FLINT_HEAD_BLOCKS);
  auto promote = [&](const void* src, float* dst, int count) {
    CheckCuda(flint_cast({src, dst, count, FLINT_BF16, FLINT_F32}, s));
  };
  auto weights = rnn_proj_f32_;
  auto bias = rnn_bias_f32_;
  if (!weights) {
    weights = ws.Take<float>(rnn_weights);
    bias = ws.Take<float>(3 * R);
    promote(w.rnn_proj, weights, rnn_weights);
    promote(w.rnn_bias, bias, 3 * R);
  }
  for (int i = 0; i < 7; ++i) {
    promote(h + (i + 1) * H, features + i * 2 * H, H);
    promote(f, features + (2 * i + 1) * H, H);
    Copy(context + i * 3 * R, bias, 3 * R * sizeof(float), s);
  }
  // The two FP32 projections share a strided weight matrix. Context includes bias.
  gemm_.LinearStrided(features, weights + 2 * R, context, 7, 3 * R, 2 * H, 2 * H + 2 * R, true);
  if (cap) Copy(cap->context, context, 7 * 3 * R * sizeof(float), s);
  CheckCuda(cudaMemsetAsync(state, 0, R * 2, s));
  std::vector<float> inspected;
  if (cap && (cap->h_top_ids || cap->h_top_logits)) inspected.resize(V);
  int prev = anchor;
  for (int i = 0; i < 7; ++i) {
    CheckCuda(flint_dspark_gather_markov_w1(
        state + R, w.markov_w1, device_greedy && i ? d_ids_ + i - 1 : d_anchor, prev, s));
    promote(state, rnn_input, 2 * R);
    gemm_.LinearStrided(rnn_input, weights, rec, 1, 3 * R, 2 * R, 2 * H + 2 * R, false);
    CheckCuda(flint_dspark_markov_gate(rec, context + i * 3 * R, state, latent, s));
    if (cap) {
      if (cap->states) Copy(static_cast<uint16_t*>(cap->states) + i * R, state, R * 2, s);
      if (cap->latents) Copy(static_cast<uint16_t*>(cap->latents) + i * R, latent, R * 2, s);
    }
    if (folded_w_) {
      linear(latent, folded_w_, bias_logits, 1, V, R);
    } else {
      linear(latent, w.hidden_out, x, 1, H, R);
      CheckCuda(flint_dspark_add_bf16(x, w.hidden_bias, H, s));
      linear(x, w_.unembed_weight, bias_logits, 1, V, H);
    }
    CheckCuda(flint_dspark_add_softcap_argmax(base + i * V, bias_logits, folded_b_, d_logits,
                                              d_ids_ + i, V, winners, s));
    if (device_greedy) continue;
    float* row = logits ? logits + size_t(i) * V : inspected.data();
    if (d_logits)
      CheckCuda(cudaMemcpyAsync(row, d_logits, V * sizeof(float), cudaMemcpyDeviceToHost, s));
    CheckCuda(cudaMemcpyAsync(&ids[i], d_ids_ + i, sizeof(*ids), cudaMemcpyDeviceToHost, s));
    CheckCuda(cudaStreamSynchronize(s));
    if (sample) ids[i] = sample({row, V});
    prev = ids[i];
    if (prev < 0 || prev >= V) throw std::invalid_argument("sampled token outside vocabulary");
    if (!inspected.empty()) {
      std::vector<int> order(V);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + 64, order.end(),
                        [&](int a, int b) { return row[a] != row[b] ? row[a] > row[b] : a < b; });
      for (int j = 0; j < 64; ++j) {
        if (cap->h_top_ids) cap->h_top_ids[i * 64 + j] = order[j];
        if (cap->h_top_logits) cap->h_top_logits[i * 64 + j] = row[order[j]];
      }
    }
  }
}
}  // namespace flint

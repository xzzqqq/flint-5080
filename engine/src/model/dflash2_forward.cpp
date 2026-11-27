#include "model/dflash2_forward.h"

#include <cuda_runtime.h>

#include <algorithm>

#include "common/cuda_check.h"
#include "loader/scratch_workspace.h"
#include "ops/norm_ops.h"

namespace flint {
namespace {

constexpr int H = 2304, Q = 4096, K = 1024, I = 6912, R = 256, V = 200058;
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
ScratchArenaView DFlash2Runner::ReserveScratch(ScratchArenaView& shared) {
  ScratchWorkspace ws(shared);
  auto draft = ws.Take<char>(kScratchBytes);
  shared = ws.rest;
  return {draft, kScratchBytes};
}
DFlash2Runner::DFlash2Runner(const ModelWeightsView& w, const KvStateArenaView& kv,
                             ScratchArenaView scratch, cudaStream_t stream)
    : w_(w), kv_(kv), scratch_(scratch), gemm_(stream, GemmWorkspace(scratch_), workspace_bytes) {
  if (!w.has_draft || !kv.has_draft_kv) throw std::invalid_argument("missing draft arena");
  ScratchWorkspace ws(scratch_);
  spec_ = ws.Take<ForwardDraftSpec>(1);
  result_ = ws.Take<ForwardDraftResult>(1);
  scratch_ = ws.rest;
}
DFlash2Runner::~DFlash2Runner() {
  if (graph_) cudaGraphExecDestroy(graph_);
}
DraftKv DFlash2Runner::Cache(int layer) const {
  int capacity = kv_.draft_swa_window;
  size_t plane = size_t(capacity) * K;
  auto k = static_cast<uint16_t*>(kv_.draft_swa_kv) + 2 * layer * plane;
  return {k, k + plane, capacity};
}
void DFlash2Runner::StoreContextKv(const void* taps, uint32_t tokens, uint32_t pos) {
  if (!taps || !tokens || pos > kv_.max_full_seq_len || tokens > kv_.max_full_seq_len - pos)
    throw std::invalid_argument("invalid draft context");
  auto s = gemm_.stream();
  for (unsigned row = 0; row < tokens;) {
    int n = std::min(128u, tokens - row);
    ScratchWorkspace ws(scratch_);
    auto x = ws.Take<uint16_t>(n * H), k = ws.Take<uint16_t>(n * K), v = ws.Take<uint16_t>(n * K);
    auto src = static_cast<const uint16_t*>(taps) + size_t(row) * 4 * H;
    const auto& w = w_.draft;
    gemm_.Multiply(src, w.fc, x, n, H, 4 * H, FLINT_BF16, FLINT_BF16, true);
    CheckCuda(flint_rmsnorm({x, w.hidden_norm, x, nullptr, n, H, FLINT_BF16, FLINT_BF16}, s));
    for (int i = 0; i < 5; ++i) {
      const auto& l = w.layers[i];
      gemm_.Multiply(x, l.k_proj, k, n, K, H, FLINT_BF16, FLINT_BF16, true);
      gemm_.Multiply(x, l.v_proj, v, n, K, H, FLINT_BF16, FLINT_BF16, true);
      CheckCuda(flint_dflash2_qk_norm_rope(k, l.k_norm, n, 8, pos + row, s));
      CheckCuda(flint_dflash2_kv_write(Cache(i), k, v, n, pos + row, s));
    }
    row += n;
  }
}
void DFlash2Runner::Draft(const ForwardDraftSpec& spec, ForwardDraftResult& result,
                          DFlash2Capture* cap) {
  EnqueueDraft(spec, cap);
  auto s = gemm_.stream();
  CheckCuda(cudaMemcpyAsync(&result, result_, sizeof(result), cudaMemcpyDeviceToHost, s));
  CheckCuda(cudaStreamSynchronize(s));
}
void DFlash2Runner::DraftDevice(const ForwardDraftSpec& spec) { EnqueueDraft(spec, nullptr); }
void DFlash2Runner::EnqueueDraft(const ForwardDraftSpec& spec, DFlash2Capture* cap) {
  if (spec.anchor < 0 || spec.anchor >= V || spec.pos < 0 ||
      uint32_t(spec.pos) > kv_.max_full_seq_len)
    throw std::invalid_argument("invalid draft block");
  auto s = gemm_.stream();
  CheckCuda(cudaMemcpyAsync(spec_, &spec, sizeof(spec), cudaMemcpyHostToDevice, s));
  if (spec.use_cuda_graph && !cap) {
    if (!graph_) {
      cudaGraph_t source{};
      CheckCuda(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      try {
        EnqueueBlock();
      } catch (...) {
        cudaStreamEndCapture(s, &source);
        if (source) cudaGraphDestroy(source);
        throw;
      }
      CheckCuda(cudaStreamEndCapture(s, &source));
      auto status = cudaGraphInstantiate(&graph_, source, 0);
      cudaGraphDestroy(source);
      CheckCuda(status);
    }
    CheckCuda(cudaGraphLaunch(graph_, s));
  } else
    RunBlock(spec_, cap);
}
void DFlash2Runner::RunBlock(const ForwardDraftSpec* spec, DFlash2Capture* cap) {
  auto d_ctx = &spec->pos;
  auto s = gemm_.stream();
  ScratchWorkspace ws(scratch_);
  auto x = ws.Take<uint16_t>(8 * H), h = ws.Take<uint16_t>(8 * H);
  const auto& w = w_.draft;
  auto linear = [&](const void* src, const void* weight, void* dest, int m, int n, int k) {
    gemm_.Multiply(src, weight, dest, m, n, k, FLINT_BF16, FLINT_BF16, true);
  };
  auto norm = [&](const void* weight) {
    CheckCuda(flint_rmsnorm({x, weight, h, nullptr, 8, H, FLINT_BF16, FLINT_BF16}, s));
  };
  CheckCuda(flint_dflash2_embed_block(w_.embed_weight, w_.embed_norm_weight, x, s, &spec->anchor));
  if (cap) Copy(cap->embed_block, x, 8 * H * 2, s);
  for (int i = 0; i < 5; ++i) {
    const auto& l = w.layers[i];
    ScratchWorkspace layer(ws.rest);
    auto q = layer.Take<uint16_t>(8 * Q), k = layer.Take<uint16_t>(8 * K),
         v = layer.Take<uint16_t>(8 * K);
    auto gate = layer.Take<uint16_t>(8 * I), up = layer.Take<uint16_t>(8 * I);
    auto delta = layer.Take<uint16_t>(8 * 576), filtered = layer.Take<uint16_t>(8 * H);
    auto conv = [&](const void* base, int side) {
      CheckCuda(flint_dflash2_conv(h, delta, base, side, filtered, s));
    };
    norm(l.input_layernorm);
    linear(h, l.attention_kernel, delta, 8, 576, H);
    conv(l.attention_base, 0);
    linear(filtered, l.q_proj, q, 8, Q, H);
    linear(filtered, l.k_proj, k, 8, K, H);
    linear(filtered, l.v_proj, v, 8, K, H);
    CheckCuda(flint_dflash2_qk_norm_rope(q, l.q_norm, 8, 32, 0, s, d_ctx));
    CheckCuda(flint_dflash2_qk_norm_rope(k, l.k_norm, 8, 8, 0, s, d_ctx));
    CheckCuda(flint_dflash2_kv_write(Cache(i), k, v, 8, 0, s, d_ctx));
    CheckCuda(flint_dflash2_block_attention(Cache(i), q, gate,
                                            static_cast<float*>(layer.rest.device_ptr),
                                            layer.rest.bytes, 0, s, d_ctx));
    linear(gate, l.o_proj, h, 8, H, Q);
    conv(l.attention_base, 1);
    CheckCuda(flint_dflash2_add_bf16(x, filtered, 8 * H, s));
    norm(l.post_attention_layernorm);
    linear(h, l.mlp_kernel, delta, 8, 576, H);
    conv(l.mlp_base, 0);
    linear(filtered, l.gate_proj, gate, 8, I, H);
    linear(filtered, l.up_proj, up, 8, I, H);
    CheckCuda(flint_dflash2_silu_mul(gate, up, 8 * I, s));
    linear(gate, l.down_proj, h, 8, H, I);
    conv(l.mlp_base, 1);
    CheckCuda(flint_dflash2_add_bf16(x, filtered, 8 * H, s));
    if (cap) {
      Copy(cap->layer_out[i], x, 8 * H * 2, s);
      if (cap->reference_layer_out[i]) Copy(x, cap->reference_layer_out[i], 8 * H * 2, s);
    }
  }
  norm(w.norm);
  if (cap) Copy(cap->head_hidden, h, 8 * H * 2, s);
  auto raw = ws.Take<uint16_t>(7 * V), proj = ws.Take<uint16_t>(7 * R);
  linear(h + H, w_.unembed_weight, raw, 7, V, H);
  linear(h + H, w.projection, proj, 7, R, H);
  auto scratch = ws.Take<uint64_t>(flint_dflash2_topk_scratch_words());
  auto unary = ws.Take<float>(7 * 16), scores = ws.Take<float>(7 * 16 * 16);
  CheckCuda(flint_dflash2_select(raw, proj, w.predecessor, w.successor, spec, result_, scratch,
                                 unary, scores, s));
  if (cap) {
    Copy(cap->unary, unary, 7 * 16 * sizeof(float), s);
    Copy(cap->scores, scores, 7 * 16 * 16 * sizeof(float), s);
  }
}
}  // namespace flint

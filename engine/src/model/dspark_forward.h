#pragma once
#include <functional>
#include <span>

#include "loader/model_views.h"
#include "ops/dspark_ops.h"
#include "ops/gemm_cublas.h"

namespace flint {
// Optional device capture destinations. Top-64 outputs are host arrays [7,64].
struct DsparkCapture {
  void *mix[4]{}, *embed_block{}, *layer_out[3]{}, *head_hidden{}, *f_rnn{};
  float* context{};
  void *states{}, *latents{};
  int* h_top_ids{};
  float* h_top_logits{};
};
class DsparkRunner {
 public:
  static ScratchArenaView ReserveScratch(ScratchArenaView& shared);
  DsparkRunner(const ModelWeightsView&, const KvStateArenaView&, ScratchArenaView, cudaStream_t);
  ~DsparkRunner();
  DsparkRunner(const DsparkRunner&) = delete;
  DsparkRunner& operator=(const DsparkRunner&) = delete;
  void StoreContextKv(const void* taps, uint32_t tokens, uint32_t position,
                      DsparkCapture* = nullptr);
  void DraftGreedy(int32_t anchor, const void* last_tap, uint32_t ctx, int32_t* ids,
                   DsparkCapture* = nullptr, bool use_cuda_graph = DefaultDraftGraph());
  using SampleRow = std::function<int32_t(std::span<const float>)>;
  void DraftWithLogits(int32_t anchor, const void* last_tap, uint32_t ctx, int32_t* ids,
                       float* h_logits, const SampleRow& sample);

 private:
  static bool DefaultDraftGraph();
  void ValidateBlock(int32_t, const void*, uint32_t, const int32_t*) const;
  void Draft(int32_t, const void*, uint32_t, int32_t*, float*, const SampleRow&, DsparkCapture*,
             bool graph = false);
  DraftKv Cache(int layer) const;
  const ModelWeightsView& w_;
  const KvStateArenaView& kv_;
  ScratchArenaView scratch_;
  Gemm gemm_;
  int32_t *d_anchor_{}, *d_ctx_{}, *d_ids_{}, *h_ids_{};
  cudaGraphExec_t draft_graph_{};
  cudaGraph_t draft_source_graph_{};
  const void* draft_graph_tap_{};
  uint16_t *folded_w_{}, *folded_b_{};
  float *rnn_proj_f32_{}, *rnn_bias_f32_{};
};
}  // namespace flint

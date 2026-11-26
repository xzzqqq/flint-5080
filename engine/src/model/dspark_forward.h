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
  DsparkRunner(const ModelWeightsView&, const KvStateArenaView&, ScratchArenaView, cudaStream_t);
  void StoreContextKv(const void* taps, uint32_t tokens, uint32_t position,
                      DsparkCapture* = nullptr);
  void DraftGreedy(int32_t anchor, const void* last_tap, uint32_t ctx, int32_t* ids,
                   DsparkCapture* = nullptr);
  using SampleRow = std::function<int32_t(std::span<const float>)>;
  void DraftWithLogits(int32_t anchor, const void* last_tap, uint32_t ctx, int32_t* ids,
                       float* h_logits, const SampleRow& sample);

 private:
  void Draft(int32_t, const void*, uint32_t, int32_t*, float*, const SampleRow&, DsparkCapture*);
  DraftKv Cache(int layer) const;
  const ModelWeightsView& w_;
  const KvStateArenaView& kv_;
  ScratchArenaView scratch_;
  Gemm gemm_;
};
}  // namespace flint

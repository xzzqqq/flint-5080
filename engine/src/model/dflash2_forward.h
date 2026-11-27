#pragma once
#include "loader/model_views.h"
#include "ops/dflash2_ops.h"
#include "ops/gemm_cublas.h"

namespace flint {
struct DFlash2Capture {
  // Optional teacher-forcing for layer-local error measurement. Capture the
  // computed output before substituting the next layer's reference input.
  const void* reference_layer_out[5]{};
  void *embed_block{}, *layer_out[5]{}, *head_hidden{};
  float *unary{}, *scores{};
};
class DFlash2Runner {
 public:
  static constexpr size_t kScratchBytes = 16ull << 20;
  static ScratchArenaView ReserveScratch(ScratchArenaView& shared);
  DFlash2Runner(const ModelWeightsView&, const KvStateArenaView&, ScratchArenaView, cudaStream_t);
  ~DFlash2Runner();
  DFlash2Runner(const DFlash2Runner&) = delete;
  DFlash2Runner& operator=(const DFlash2Runner&) = delete;
  void StoreContextKv(const void* taps, uint32_t count, uint32_t pos);
  void Draft(const ForwardDraftSpec&, ForwardDraftResult&, DFlash2Capture* = nullptr);

 private:
  void RunBlock(DFlash2Capture*);
  DraftKv Cache(int layer) const;
  const ModelWeightsView& w_;
  const KvStateArenaView& kv_;
  ScratchArenaView scratch_;
  Gemm gemm_;
  ForwardDraftSpec* spec_;
  ForwardDraftResult* result_;
  cudaGraphExec_t graph_{};
};
}  // namespace flint

#pragma once
#include "loader/model_views.h"
#include "ops/exl3_ops.h"
#include "ops/gemm_cublas.h"
#include "ops/loop_res_ops.h"
namespace flint {
void RunExl3Linear(const Gemm&, ScratchArenaView, const Exl3MatrixView&, const void* x, void* y,
                   int rows, FlintType input = FLINT_BF16, FlintType output = FLINT_BF16);
struct AttentionBatch {
  const void *q, *k, *v;
  void* y;
  int tokens, position;
  VisitPlanEntry visit;
  const int32_t* d_pos{};
};
void RunSlidingAttention(const Gemm&, ScratchArenaView, const LayerWeightsView&,
                         const KvStateArenaView&, AttentionBatch);
void RunFullAttention(const Gemm&, ScratchArenaView, const LayerWeightsView&,
                      const KvStateArenaView&, AttentionBatch);
// Optional diagnostic outputs use the same token-major layout as the router.
void RunSparseMoeBlock(const Gemm&, ScratchArenaView, const LayerWeightsView&, const void* x,
                       void* y, int tokens, float* logits = nullptr, int* ids = nullptr,
                       float* weights = nullptr, const ExpertProjection* experts = nullptr);
void RunBlockResidual(const Gemm&, ScratchArenaView, ResidualParams);
void RunMessageGate(const Gemm&, ScratchArenaView, const ControllerWeightsView&, const float* x,
                    const float* message, float* y, int tokens);
}  // namespace flint

#pragma once
#include "forge/forward_abi.h"
#include "model/visit_plan.h"
#include "ops/gemm_cublas.h"
namespace flint {
void RunInkling(const Gemm&, const std::array<VisitPlanEntry, 48>&, const ModelWeightsView&,
                const KvStateArenaView&, ScratchArenaView, const ForwardRunSpec&,
                const ForwardOptions&, ForwardRunResult&, const cudaEvent_t* marks,
                const ForwardCapture* capture, const ExpertProjection* experts,
                const int32_t* d_pos);
}

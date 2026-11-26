#pragma once
#include <cuda_runtime_api.h>

#include <memory>

#include "model/forward_types.h"
namespace flint {
class InklingRunner {
 public:
  InklingRunner(const ModelWeightsView&, const KvStateArenaView&, ScratchArenaView, cudaStream_t);
  ~InklingRunner();
  ForwardRunResult Run(const ForwardRunSpec&, const ForwardOptions&,
                       const ForwardCapture* = nullptr);
  [[nodiscard]] ForwardGraphStats graph_stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace flint

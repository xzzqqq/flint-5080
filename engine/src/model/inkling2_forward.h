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
  [[nodiscard]] cudaStream_t stream() const;
  [[nodiscard]] ScratchArenaView scratch_tail() const;
  ForwardRunResult RunVerify(ForwardRunSpec, const ForwardOptions& = {}, bool device_only = false);
  void CommitVerify(uint32_t pos_offset, uint32_t verify_len, uint32_t accepted_len);
  [[nodiscard]] ForwardGraphStats graph_stats() const;

 private:
  ForwardRunResult Enqueue(const ForwardRunSpec&, const ForwardOptions&, const ForwardCapture*,
                           bool device_only);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace flint

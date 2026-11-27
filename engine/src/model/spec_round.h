#pragma once

#include "model/dflash2_forward.h"
#include "model/inkling2_forward.h"

namespace flint {
class SpeculativeRunner {
 public:
  SpeculativeRunner(const ModelWeightsView&, const KvStateArenaView&,
                    ScratchArenaView draft_scratch, InklingRunner&);
  ~SpeculativeRunner();
  DFlash2Runner& draft();
  ForwardSpecRoundResult Run(const ForwardSpecRoundSpec&, const ForwardOptions&);
  [[nodiscard]] ForwardGraphStats graph_stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

inline ForwardGraphStats operator+(ForwardGraphStats a, ForwardGraphStats b) {
  return {a.captures + b.captures, a.launches + b.launches};
}
}  // namespace flint

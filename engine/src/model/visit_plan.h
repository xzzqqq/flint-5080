#pragma once
#include <array>
#include <stdexcept>

#include "loader/model_views.h"
namespace flint {
inline std::array<VisitPlanEntry, FLINT_NUM_LOGICAL_VISITS> BuildVisitPlan(const ModelConfig& c) {
  if (c.loop_core_start != 6 || c.loop_core_end != 18 || c.loop_num_passes != 3 ||
      c.num_hidden_layers != 24 || c.n_layers_per_attn_res_block != 6)
    throw std::invalid_argument("unsupported visit schedule");
  std::array<VisitPlanEntry, FLINT_NUM_LOGICAL_VISITS> plan{};
  uint8_t idx = 0, swa = 0, full = 0;
  auto append = [&](uint32_t begin, uint32_t end, uint8_t pass) {
    for (uint8_t i = begin; i < end; ++i) {
      auto global = c.is_full_layer[i];
      if (global != (i % 6 == 5)) throw std::invalid_argument("invalid attention schedule");
      plan[idx] = {idx,
                   i,
                   pass,
                   global,
                   i % c.n_layers_per_attn_res_block == 0,
                   pass > 0,
                   global ? full++ : swa++};
      ++idx;
    }
  };
  append(0, c.loop_core_start, 0);
  for (uint8_t pass = 0; pass < c.loop_num_passes; ++pass)
    append(c.loop_core_start, c.loop_core_end, pass);
  append(c.loop_core_end, c.num_hidden_layers, 0);
  return plan;
}

}  // namespace flint

#pragma once
#include <array>
#include <stdexcept>

#include "loader/model_constants.h"
namespace flint {
using VisitPlan = std::array<VisitPlanEntry, model::kNumLogicalVisits>;
inline VisitPlan BuildVisitPlan(const ModelConfig& c) {
  if (c.loop_core_start != model::kLoopCoreStart || c.loop_core_end != model::kLoopCoreEnd ||
      c.loop_num_passes != model::kLoopNumPasses ||
      c.num_hidden_layers != model::kNumPhysicalLayers ||
      c.n_layers_per_attn_res_block != model::kLayersPerResBlock)
    throw std::invalid_argument("unsupported visit schedule");
  VisitPlan plan{};
  uint8_t idx = 0, swa = 0, full = 0;
  auto append = [&](uint32_t begin, uint32_t end, uint8_t pass) {
    for (uint8_t i = begin; i < end; ++i) {
      auto global = c.is_full_layer[i];
      if (global != ((i + 1) % model::kLayersPerResBlock == 0))
        throw std::invalid_argument("invalid attention schedule");
      bool prologue_end = i + 1 == c.loop_core_start;
      // Later taps sample the first block of pass 1 and the last block of pass 2.
      int8_t tap = -1;
      if (prologue_end) tap = 0;
      if (pass && i + 1 == c.loop_core_start + pass * c.n_layers_per_attn_res_block) tap = pass;
      plan[idx] = {idx,
                   i,
                   pass,
                   global,
                   i % c.n_layers_per_attn_res_block == 0,
                   pass > 0,
                   global ? full++ : swa++,
                   pass > 0 && i == c.loop_core_start,
                   prologue_end,
                   tap};
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

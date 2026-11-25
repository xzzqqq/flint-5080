#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

#include "loader/model_constants.h"
namespace flint {
struct KvArenaSizes {
  size_t swa, per_token, kv_conv, sublayer_conv;
  size_t Fixed() const { return swa + kv_conv + sublayer_conv; }
};
inline KvArenaSizes ComputeKvArenaSizes(const ModelConfig& c) {
  if (c.num_key_value_heads != model::kNumKvHeads || c.head_dim != model::kHeadDim ||
      c.hidden_size != model::kHiddenSize || c.sconv_kernel_size != model::kSconvKernelSize ||
      c.sliding_window_size != model::kSlidingWindow)
    throw std::invalid_argument("unsupported KV geometry");
  size_t channels = c.num_key_value_heads * c.head_dim;
  size_t kv_pair = 2 * channels * sizeof(uint16_t);
  size_t state = model::kNumLogicalVisits * 2 * (c.sconv_kernel_size - 1) * sizeof(float);
  return {model::kNumSwaVisits * c.sliding_window_size * kv_pair, model::kNumFullVisits * kv_pair,
          state * channels, state * c.hidden_size};
}

struct KvVisitPointers {
  uint16_t *k, *v;
  int capacity;
};
inline KvVisitPointers VisitKvCache(const KvStateArenaView& kv, const VisitPlanEntry& visit) {
  auto capacity = visit.is_full_attention ? kv.max_full_seq_len : kv.swa_window;
  size_t plane = size_t(capacity) * kv.num_kv_heads * kv.head_dim;
  auto k = static_cast<uint16_t*>(visit.is_full_attention ? kv.full_kv : kv.swa_kv) +
           2 * visit.kv_slot_idx * plane;
  return {k, k + plane, static_cast<int>(capacity)};
}
inline float* VisitKvConvState(const KvStateArenaView& kv, uint32_t visit_idx, uint32_t plane = 0) {
  return kv.kv_conv_state +
         (size_t(visit_idx) * 2 + plane) * kv.num_kv_heads * kv.head_dim * kv.conv_state_len;
}
inline float* VisitSublayerConvState(const KvStateArenaView& kv, uint32_t visit_idx,
                                     uint32_t sublayer = 0) {
  return kv.sublayer_conv_state +
         (size_t(visit_idx) * 2 + sublayer) * kv.hidden_size * kv.conv_state_len;
}
struct KvByteRange {
  size_t offset, bytes;
};
inline std::array<KvByteRange, 4> VisitSnapshotRanges(const KvStateArenaView& kv,
                                                      const VisitPlanEntry& visit,
                                                      size_t live_tokens) {
  auto [k, v, cap] = VisitKvCache(kv, visit);
  size_t dim = size_t(kv.num_kv_heads) * kv.head_dim;
  size_t bytes = std::min<size_t>(cap, live_tokens) * dim * sizeof(*k);
  auto range = [&](const void* p, size_t n) {
    return KvByteRange{
        size_t(static_cast<const std::byte*>(p) - static_cast<const std::byte*>(kv.arena_base)), n};
  };
  return {range(k, bytes), range(v, bytes),
          range(VisitKvConvState(kv, visit.visit_idx), 2 * dim * kv.conv_state_len * sizeof(float)),
          range(VisitSublayerConvState(kv, visit.visit_idx),
                2 * size_t(kv.hidden_size) * kv.conv_state_len * sizeof(float))};
}
}  // namespace flint

#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "common/cuda_check.h"
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
// Packed prefix snapshots contain only fixed state. Full KV remains live in the arena.
inline std::array<std::pair<void*, size_t>, 3> FixedKvRegions(const KvStateArenaView& kv,
                                                              const KvArenaSizes& sizes) {
  return {{{kv.swa_kv, sizes.swa},
           {kv.kv_conv_state, sizes.kv_conv},
           {kv.sublayer_conv_state, sizes.sublayer_conv}}};
}

inline void CopyFixedKvToSnapshot(void* snapshot_base, const KvStateArenaView& kv,
                                  const KvArenaSizes& sizes, cudaStream_t stream) {
  auto* dst = static_cast<std::byte*>(snapshot_base);
  for (auto [src, bytes] : FixedKvRegions(kv, sizes)) {
    CheckCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream),
              "save prefix snapshot");
    dst += bytes;
  }
}

inline void CopySnapshotToFixedKv(const KvStateArenaView& kv, const void* snapshot_base,
                                  const KvArenaSizes& sizes, cudaStream_t stream) {
  auto* src = static_cast<const std::byte*>(snapshot_base);
  for (auto [dst, bytes] : FixedKvRegions(kv, sizes)) {
    CheckCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream),
              "restore prefix snapshot");
    src += bytes;
  }
}

enum class KvSliceKind : uint8_t { kKey, kValue, kKvConv, kSublayerConv };
enum class KvSliceDType : uint8_t { kBF16, kF32 };

struct KvByteRange {
  size_t offset, bytes;
  KvSliceKind kind;
  KvSliceDType dtype;
};
inline std::array<KvByteRange, 4> VisitSnapshotRanges(const KvStateArenaView& kv,
                                                      const VisitPlanEntry& visit,
                                                      size_t live_tokens) {
  auto [k, v, cap] = VisitKvCache(kv, visit);
  size_t dim = size_t(kv.num_kv_heads) * kv.head_dim;
  size_t bytes = std::min<size_t>(cap, live_tokens) * dim * sizeof(*k);
  auto range = [&](const void* p, size_t n, KvSliceKind kind, KvSliceDType dtype) {
    return KvByteRange{
        size_t(static_cast<const std::byte*>(p) - static_cast<const std::byte*>(kv.arena_base)), n,
        kind, dtype};
  };
  return {range(k, bytes, KvSliceKind::kKey, KvSliceDType::kBF16),
          range(v, bytes, KvSliceKind::kValue, KvSliceDType::kBF16),
          range(VisitKvConvState(kv, visit.visit_idx), 2 * dim * kv.conv_state_len * sizeof(float),
                KvSliceKind::kKvConv, KvSliceDType::kF32),
          range(VisitSublayerConvState(kv, visit.visit_idx),
                2 * size_t(kv.hidden_size) * kv.conv_state_len * sizeof(float),
                KvSliceKind::kSublayerConv, KvSliceDType::kF32)};
}
}  // namespace flint

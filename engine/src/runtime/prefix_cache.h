#pragma once

#include <span>
#include <vector>

#include "loader/arenas.h"
#include "loader/kv_layout.h"

namespace flint {
struct PrefixPlan {
  uint32_t matched_tokens = 0;
  uint32_t pin_pos = 0;
  int restore_slot = -1;
  int logits_slot = -1;
  bool reset_kv = true;
  bool use_cached_logits = false;
};

struct PrefixCacheStats {
  uint64_t hits = 0, misses = 0, saved_tokens = 0;
  uint32_t used_slots = 0;
};

// Single worker owns the arena, snapshot metadata, and transfer stream.
class PrefixCache {
 public:
  PrefixCache(uint32_t slots, size_t vocab_size);  // Policy only.
  PrefixCache(uint32_t slots, const ModelConfig& config, uint32_t max_context,
              bool with_draft = false);
  bool enabled() const { return !slots_.empty(); }
  PrefixCacheStats stats() const { return stats_; }
  PrefixPlan Match(std::span<const int32_t> prompt);
  std::span<const float> SlotLogits(int slot) const { return slots_.at(slot).last_logits; }
  void Restore(int slot, const KvStateArenaView& kv, cudaStream_t stream);
  void Save(std::span<const int32_t> tokens, const KvStateArenaView& kv,
            std::span<const float> last_logits, cudaStream_t stream);
  void OnKvWrite(uint32_t write_pos);
  void AppendLive(std::span<const int32_t> tokens);
  void InvalidateAll();

 private:
  struct Slot {
    std::vector<int32_t> tokens;
    std::vector<float> last_logits;
    uint64_t lru_tick = 0;
    bool valid = false, has_logits = false;
  };
  std::vector<Slot> slots_;
  std::vector<int32_t> live_tokens_, last_prompt_;
  uint64_t tick_ = 0;
  PrefixCacheStats stats_;
  KvArenaSizes sizes_{};
  DeviceBuffer snapshots_;
};
}  // namespace flint

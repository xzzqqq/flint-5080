#include "runtime/prefix_cache.h"

#include <algorithm>
#include <limits>

namespace flint {
namespace {
constexpr uint32_t kMinPinGap = 16;
size_t CommonPrefix(std::span<const int32_t> a, std::span<const int32_t> b) {
  return std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first - a.begin();
}
}  // namespace

PrefixCache::PrefixCache(uint32_t slots, size_t vocab_size) : slots_(slots) {
  for (auto& slot : slots_) slot.last_logits.resize(vocab_size);
}
PrefixCache::PrefixCache(uint32_t slots, const ModelConfig& config, uint32_t max_context,
                         bool with_mtp)
    : PrefixCache(slots, config.unpadded_vocab_size) {
  if (!enabled()) return;
  live_tokens_.reserve(max_context);
  last_prompt_.reserve(max_context);
  for (auto& slot : slots_) {
    slot.tokens.reserve(max_context);
    if (with_mtp) slot.last_tap.resize(4 * config.hidden_size);
  }
  sizes_ = ComputeKvArenaSizes(config, with_mtp);
  if (slots > std::numeric_limits<size_t>::max() / sizes_.Fixed())
    throw std::length_error("prefix cache size overflow");
  snapshots_ = AllocateDeviceBuffer(slots * sizes_.Fixed());
}

PrefixPlan PrefixCache::Match(std::span<const int32_t> prompt) {
  PrefixPlan plan;
  if (!enabled()) {
    ++stats_.misses;
    return plan;
  }
  auto live_lcp = CommonPrefix(prompt, live_tokens_);
  size_t best_lcp = std::max(live_lcp, CommonPrefix(prompt, last_prompt_));
  if (live_lcp == live_tokens_.size() && live_lcp < prompt.size()) plan.matched_tokens = live_lcp;
  for (size_t i = 0; i < slots_.size(); ++i) {
    const auto& slot = slots_[i];
    if (!slot.valid) continue;
    auto lcp = CommonPrefix(prompt, slot.tokens);
    best_lcp = std::max(best_lcp, lcp);
    if (lcp != slot.tokens.size() || lcp <= plan.matched_tokens) continue;
    bool exact = lcp == prompt.size();
    if (exact && (!slot.has_logits || (!slot.last_tap.empty() && !slot.has_tap))) continue;
    plan.matched_tokens = lcp;
    plan.restore_slot = exact && live_lcp == lcp && live_tokens_.size() == lcp ? -1 : int(i);
    plan.use_cached_logits = exact;
    plan.logits_slot = exact ? int(i) : -1;
  }
  if (plan.use_cached_logits) slots_[plan.logits_slot].lru_tick = ++tick_;
  if (!prompt.empty()) best_lcp = std::min(best_lcp, prompt.size() - 1);
  if (!plan.use_cached_logits && best_lcp > plan.matched_tokens &&
      best_lcp - plan.matched_tokens >= kMinPinGap)
    plan.pin_pos = best_lcp;
  last_prompt_.assign(prompt.begin(), prompt.end());
  plan.reset_kv = plan.matched_tokens == 0;
  if (plan.reset_kv)
    ++stats_.misses;
  else
    ++stats_.hits;
  stats_.saved_tokens += plan.matched_tokens;
  return plan;
}

void PrefixCache::Restore(int index, const KvStateArenaView& kv, cudaStream_t stream) {
  auto& slot = slots_.at(index);
  if (!slot.valid) throw std::logic_error("invalid prefix snapshot");
  if (snapshots_) {
    auto* src = static_cast<const std::byte*>(snapshots_.get()) + index * sizes_.Fixed();
    CopySnapshotToFixedKv(kv, src, sizes_, stream);
  }
  live_tokens_ = slot.tokens;
  slot.lru_tick = ++tick_;
}

void PrefixCache::Save(std::span<const int32_t> tokens, const KvStateArenaView& kv,
                       std::span<const float> last_logits, cudaStream_t stream,
                       std::span<const uint16_t> last_tap) {
  if (!enabled()) return;
  auto it = std::find_if(slots_.begin(), slots_.end(), [&](const Slot& s) {
    return s.valid && std::ranges::equal(s.tokens, tokens);
  });
  if (it == slots_.end())
    it = std::min_element(slots_.begin(), slots_.end(), [](const Slot& a, const Slot& b) {
      return a.valid != b.valid ? !a.valid : a.lru_tick < b.lru_tick;
    });
  if (!last_tap.empty() && last_tap.size() != it->last_tap.size())
    throw std::invalid_argument("invalid prefix tap size");
  if (!last_logits.empty() && it->last_logits.size() != last_logits.size())
    throw std::invalid_argument("invalid prefix logits size");
  if (snapshots_) {
    auto* dst = static_cast<std::byte*>(snapshots_.get()) + (it - slots_.begin()) * sizes_.Fixed();
    CopyFixedKvToSnapshot(dst, kv, sizes_, stream);
  }
  it->tokens.assign(tokens.begin(), tokens.end());
  it->has_tap = !last_tap.empty();
  std::ranges::copy(last_tap, it->last_tap.begin());
  it->has_logits = !last_logits.empty();
  std::ranges::copy(last_logits, it->last_logits.begin());
  it->lru_tick = ++tick_;
  if (!it->valid) ++stats_.used_slots;
  it->valid = true;
}

void PrefixCache::OnKvWrite(uint32_t write_pos) {
  // All slots share Full KV. A write at p destroys every longer snapshot.
  for (auto& s : slots_)
    if (s.valid && s.tokens.size() > write_pos) {
      s.valid = false;
      --stats_.used_slots;
    }
  if (live_tokens_.size() > write_pos) live_tokens_.resize(write_pos);
}
void PrefixCache::AppendLive(std::span<const int32_t> tokens) {
  if (enabled()) live_tokens_.insert(live_tokens_.end(), tokens.begin(), tokens.end());
}
void PrefixCache::InvalidateAll() {
  for (auto& s : slots_) s.valid = false;
  stats_.used_slots = 0;
  live_tokens_.clear();
  last_prompt_.clear();
}
}  // namespace flint

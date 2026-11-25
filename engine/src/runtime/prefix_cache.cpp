#include "runtime/prefix_cache.h"

#include <algorithm>
#include <limits>

#include "common/cuda_check.h"

namespace flint {
namespace {
constexpr uint32_t kMinPinGap = 16;
size_t CommonPrefix(std::span<const int32_t> a, std::span<const int32_t> b) {
  return std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first - a.begin();
}
void Copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  CheckCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream), "prefix snapshot");
}
}  // namespace

PrefixCache::PrefixCache(uint32_t slots, size_t vocab_size) : slots_(slots) {
  for (auto& slot : slots_) slot.last_logits.resize(vocab_size);
}
PrefixCache::PrefixCache(uint32_t slots, const ModelConfig& config, uint32_t max_context)
    : PrefixCache(slots, config.unpadded_vocab_size) {
  if (!enabled()) return;
  live_tokens_.reserve(max_context);
  last_prompt_.reserve(max_context);
  for (auto& slot : slots_) slot.tokens.reserve(max_context);
  sizes_ = ComputeKvArenaSizes(config);
  if (slots > std::numeric_limits<size_t>::max() / sizes_.Fixed())
    throw std::length_error("prefix cache size overflow");
  void* ptr;
  CheckCuda(cudaMalloc(&ptr, slots * sizes_.Fixed()), "allocate prefix cache");
  device_.reset(ptr);
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
    if (exact && !slot.has_logits) continue;
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
  if (device_) {
    auto* src = static_cast<const std::byte*>(device_.get()) + index * sizes_.Fixed();
    Copy(kv.swa_kv, src, sizes_.swa, stream);
    Copy(kv.kv_conv_state, src + sizes_.swa, sizes_.kv_conv + sizes_.sublayer_conv, stream);
  }
  live_tokens_ = slot.tokens;
  slot.lru_tick = ++tick_;
}

void PrefixCache::Save(std::span<const int32_t> tokens, const KvStateArenaView& kv,
                       std::span<const float> last_logits, cudaStream_t stream) {
  if (!enabled()) return;
  auto it = std::find_if(slots_.begin(), slots_.end(), [&](const Slot& s) {
    return s.valid && std::ranges::equal(s.tokens, tokens);
  });
  if (it == slots_.end())
    it = std::min_element(slots_.begin(), slots_.end(), [](const Slot& a, const Slot& b) {
      return a.valid != b.valid ? !a.valid : a.lru_tick < b.lru_tick;
    });
  if (!last_logits.empty() && it->last_logits.size() != last_logits.size())
    throw std::invalid_argument("invalid prefix logits size");
  if (device_) {
    auto* dst = static_cast<std::byte*>(device_.get()) + (it - slots_.begin()) * sizes_.Fixed();
    Copy(dst, kv.swa_kv, sizes_.swa, stream);
    Copy(dst + sizes_.swa, kv.kv_conv_state, sizes_.kv_conv + sizes_.sublayer_conv, stream);
  }
  it->tokens.assign(tokens.begin(), tokens.end());
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

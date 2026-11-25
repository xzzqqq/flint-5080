#include <iostream>
#include <numeric>

#include "runtime/prefix_cache.h"

namespace {
using namespace flint;
using Tokens = std::vector<int32_t>;
void Require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
void Cold(PrefixPlan p) {
  Require(p.reset_kv && !p.matched_tokens && p.restore_slot == -1 && p.logits_slot == -1 &&
              !p.use_cached_logits,
          "expected cold plan");
}
void Disabled() {
  PrefixCache c(0, 5);
  Tokens prefix{1, 2}, prompt{1, 2, 3};
  c.AppendLive(prefix);
  c.Save(prefix, {}, {}, nullptr);
  Cold(c.Match(prompt));
  Cold(c.Match(prompt));
  Require(!c.enabled() && c.stats().used_slots == 0 && c.stats().hits == 0, "disabled cache");
}
void WatermarkAndLogits() {
  PrefixCache c(3, 2);
  Tokens p{1, 2}, extended{1, 2, 3}, tail{3}, longer{1, 2, 3, 4};
  std::vector<float> logits{2, 1};
  c.AppendLive(p);
  c.Save(p, {}, {}, nullptr);
  Cold(c.Match(p));  // Exact state alone cannot produce the next token.
  auto hit = c.Match(extended);
  Require(hit.matched_tokens == 2 && !hit.reset_kv && hit.restore_slot == -1, "live tie");
  c.Save(p, {}, logits, nullptr);
  auto exact = c.Match(p);
  Require(exact.use_cached_logits && exact.restore_slot == -1 && exact.logits_slot >= 0 &&
              std::ranges::equal(c.SlotLogits(exact.logits_slot), logits),
          "live exact");
  c.AppendLive(tail);
  c.Save(extended, {}, logits, nullptr);
  auto stripped = c.Match(Tokens{1, 2, 7});
  Require(stripped.matched_tokens == 2 && stripped.restore_slot >= 0, "stripped L2");
  exact = c.Match(p);
  Require(
      exact.use_cached_logits && exact.restore_slot == exact.logits_slot && exact.restore_slot >= 0,
      "exact restore");
  c.Restore(exact.restore_slot, {}, nullptr);
  Require(c.Match(p).restore_slot == -1 && c.stats().used_slots == 2, "restore without write");
  hit = c.Match(longer);
  Require(hit.restore_slot >= 0 && hit.matched_tokens == 3, "longer L2 beats live");
  c.OnKvWrite(2);
  hit = c.Match(longer);
  Require(c.stats().used_slots == 1 && hit.matched_tokens == 2 && hit.restore_slot == -1,
          "preserve equal watermark only");
  c.AppendLive(tail);
  c.OnKvWrite(1);
  Require(c.stats().used_slots == 0 && c.Match(extended).matched_tokens == 1, "truncate live");
  c.InvalidateAll();
  Cold(c.Match(extended));
}
void Pinning() {
  PrefixCache c(3, 0);
  Tokens shared(32);
  std::iota(shared.begin(), shared.end(), 0);
  Tokens a = shared, b = shared, d = shared;
  a.push_back(81);
  b.push_back(82);
  d.push_back(83);
  Cold(c.Match(a));
  c.AppendLive(a);
  c.Save(a, {}, {}, nullptr);
  auto plan = c.Match(b);
  Cold(plan);
  Require(plan.pin_pos == shared.size(), "pin divergent prefix");
  c.OnKvWrite(0);
  c.AppendLive(shared);
  c.Save(shared, {}, {}, nullptr);
  c.AppendLive(std::span(b).last(1));
  c.Save(b, {}, {}, nullptr);
  plan = c.Match(d);
  Require(plan.matched_tokens == shared.size() && plan.restore_slot >= 0 && !plan.pin_pos,
          "reuse adaptive checkpoint");
  Require(c.stats().hits == 1 && c.stats().misses == 2 && c.stats().saved_tokens == shared.size(),
          "match counters");
  c.InvalidateAll();
  Cold(c.Match(d));
  Require(c.stats().used_slots == 0 && c.Match(d).pin_pos == shared.size(), "last prompt alone");
  c.InvalidateAll();
  c.Match(std::span(a).first(16));
  Require(!c.Match(std::span(b).first(16)).pin_pos, "capped LCP below minimum gap");
}
void Eviction() {
  PrefixCache c(3, 1);
  Tokens path{1, 2, 3, 4};
  std::span<const int32_t> tokens(path);
  std::vector<float> logits{1};
  for (size_t pos = 0; pos < 3; ++pos) {
    c.OnKvWrite(pos);
    c.AppendLive(tokens.subspan(pos, 1));
    c.Save(tokens.first(pos + 1), {}, logits, nullptr);
  }
  auto a = tokens.first(1);
  c.Restore(c.Match(a).restore_slot, {}, nullptr);
  logits[0] = 2;
  c.Save(a, {}, logits, nullptr);
  c.Restore(c.Match(tokens).restore_slot, {}, nullptr);
  c.OnKvWrite(3);
  c.AppendLive(tokens.last(1));
  c.Save(tokens, {}, logits, nullptr);
  auto exact = c.Match(a);
  Require(exact.use_cached_logits && c.SlotLogits(exact.logits_slot)[0] == 2, "refreshed logits");
  Require(c.stats().used_slots == 3 && c.Match(tokens.first(2)).matched_tokens == 1 &&
              c.Match(tokens.first(3)).use_cached_logits,
          "evict least recently used");
}
}  // namespace
int main() {
  try {
    Disabled();
    WatermarkAndLogits();
    Pinning();
    Eviction();
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}

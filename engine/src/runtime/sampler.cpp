#include "runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace flint {
int32_t Sampler::Sample(std::span<const float> logits, float temperature, int32_t top_k,
                        float top_p, uint64_t& rng) {
  auto& c = candidates_;
  c.clear();
  auto higher = [](Candidate a, Candidate b) {
    return a.weight != b.weight ? a.weight > b.weight : a.id < b.id;
  };
  const float invalid = -std::numeric_limits<float>::infinity();
  float maximum = invalid;
  auto weight = [&](float logit) {
    // Promote the subtraction for extreme finite logits whose difference
    // overflows float (including when temperature is positive infinity).
    float delta = logit - maximum;
    if (!std::isfinite(delta))
      return std::exp(static_cast<float>((double(logit) - maximum) / temperature));
    return std::exp(delta / temperature);
  };
  double sum = 0;
  if (temperature > 0 && top_k > 0) {
    // Periodically trim to k. Once trimmed, almost all subsequent vocabulary
    // entries can be rejected by comparing to the current kth logit.
    const size_t k = top_k;
    float threshold = invalid;
    for (size_t i = 0; i < logits.size(); ++i) {
      float x = logits[i];
      if (x > threshold && std::isfinite(x)) {
        c.push_back({x, static_cast<int32_t>(i)});
        if (c.size() == 2 * k) {
          std::nth_element(c.begin(), c.begin() + k - 1, c.end(), higher);
          c.resize(k);
          threshold = c.back().weight;
        }
      }
    }
    if (c.empty()) throw std::invalid_argument("no finite logits");
    if (c.size() > k) {
      std::nth_element(c.begin(), c.begin() + k, c.end(), higher);
      c.resize(k);
    }
    std::sort(c.begin(), c.end(), higher);
    maximum = c.front().weight;
    for (auto& item : c) {
      item.weight = weight(item.weight);
      sum += item.weight;
    }
  } else {
    for (float x : logits)
      if (std::isfinite(x)) maximum = std::max(maximum, x);
    if (maximum == invalid) throw std::invalid_argument("no finite logits");
    if (temperature <= 0) return std::find(logits.begin(), logits.end(), maximum) - logits.begin();
    c.resize(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
      float probability = std::isfinite(logits[i]) ? weight(logits[i]) : 0.f;
      c[i] = {probability, static_cast<int32_t>(i)};
      sum += probability;
    }
  }
  if (top_p > 0 && top_p < 1) {
    auto end = std::remove_if(
        c.begin(), c.end(), [cutoff = sum * 1e-7](Candidate item) { return item.weight < cutoff; });
    // With the supported vocabulary the maximum always survives. Keep the
    // full distribution if a larger vocabulary makes the threshold too high.
    if (end != c.begin()) c.erase(end, c.end());
    if (top_k <= 0) std::sort(c.begin(), c.end(), higher);
    double target = top_p * sum;
    sum = 0;
    size_t count = 0;
    do {
      sum += c[count++].weight;
    } while (sum < target && count < c.size());
    c.resize(count);
  }
  uint64_t bits = (rng += 0x9e3779b97f4a7c15ULL);
  bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
  bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
  double draw = static_cast<double>((bits ^ (bits >> 31)) >> 11) * 0x1p-53 * sum;
  for (auto item : c) {
    draw -= item.weight;
    if (draw < 0) return item.id;
  }
  return c.back().id;
}
}  // namespace flint

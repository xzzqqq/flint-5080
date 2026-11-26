#include "runtime/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace flint {
namespace {
double Uniform(uint64_t& rng) {
  uint64_t bits = (rng += 0x9e3779b97f4a7c15ULL);
  bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
  bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<double>((bits ^ (bits >> 31)) >> 11) * 0x1p-53;
}
}  // namespace
double Sampler::Transform(std::span<const float> logits, float temperature, int32_t top_k,
                          float top_p) {
  if (!(temperature > 0)) throw std::invalid_argument("positive temperature required");
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
  if (top_k > 0) {
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
  return sum;
}

std::vector<double> Sampler::Probabilities(std::span<const float> logits, float temp, int32_t k,
                                           float p) {
  double sum = Transform(logits, temp, k, p);
  std::vector<double> prob(logits.size());
  for (auto item : candidates_) prob[item.id] = item.weight / sum;
  return prob;
}

int32_t Sampler::Sample(std::span<const float> logits, float temperature, int32_t top_k,
                        float top_p, uint64_t& rng) {
  if (temperature <= 0) {
    float best = -std::numeric_limits<float>::infinity();
    int id = -1;
    for (size_t i = 0; i < logits.size(); ++i)
      if (std::isfinite(logits[i]) && logits[i] > best) {
        best = logits[i];
        id = i;
      }
    if (id < 0) throw std::invalid_argument("no finite logits");
    return id;
  }
  auto sum = Transform(logits, temperature, top_k, top_p);
  const auto& c = candidates_;
  double draw = Uniform(rng) * sum;
  for (auto item : c) {
    draw -= item.weight;
    if (draw < 0) return item.id;
  }
  return c.back().id;
}
SpeculativeResult VerifySpeculative(std::span<const int32_t> ids, std::span<const int32_t> argmax,
                                    std::span<const float> draft, std::span<const float> target,
                                    int vocab, float temp, int32_t k, float p, uint64_t& rng) {
  if (temp == 0) {
    if (argmax.size() != ids.size() + 1) throw std::invalid_argument("target argmax size");
    size_t n = 0;
    while (n < ids.size() && ids[n] == argmax[n]) ++n;
    return {int(n), argmax[n]};
  }
  if (!(temp > 0) || vocab <= 0 || draft.size() != ids.size() * vocab ||
      target.size() != draft.size() + vocab)
    throw std::invalid_argument("invalid verification distributions");
  Sampler sampler;
  for (size_t i = 0; i < ids.size(); ++i) {
    int id = ids[i];
    auto q = sampler.Probabilities(draft.subspan(i * vocab, vocab), temp, k, p);
    auto prob = sampler.Probabilities(target.subspan(i * vocab, vocab), temp, k, p);
    if (id < 0 || id >= vocab || q[id] <= 0)
      throw std::invalid_argument("draft outside proposal support");
    if (Uniform(rng) * q[id] < prob[id]) continue;
    double sum = 0;
    for (int j = 0; j < vocab; ++j) sum += (prob[j] = std::max(0., prob[j] - q[j]));
    double draw = Uniform(rng) * sum;
    int last = -1;
    for (int j = 0; j < vocab; ++j)
      if (prob[j] > 0) {
        last = j;
        if ((draw -= prob[j]) < 0) break;
      }
    if (last < 0) throw std::logic_error("empty residual distribution");
    return {int(i), last};
  }
  return {int(ids.size()), sampler.Sample(target.last(vocab), temp, k, p, rng)};
}
}  // namespace flint

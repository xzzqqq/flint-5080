#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace flint {
class Sampler {
 public:
  int32_t Sample(std::span<const float> logits, float temperature, int32_t top_k, float top_p,
                 uint64_t& rng);

  // Normalized probabilities in vocabulary order, after the same filters as Sample.
  std::vector<double> Probabilities(std::span<const float>, float, int32_t, float);

 private:
  struct Candidate {
    float weight;
    int32_t id;
  };
  double Transform(std::span<const float>, float, int32_t, float);
  std::vector<Candidate> candidates_;
};
struct SpeculativeResult {
  int accepted_draft_count;
  int32_t bonus_token;
};
SpeculativeResult VerifySpeculative(std::span<const int32_t> draft_ids,
                                    std::span<const int32_t> target_argmax,
                                    std::span<const float> draft_logits,
                                    std::span<const float> target_logits, int vocab,
                                    float temperature, int32_t top_k, float top_p, uint64_t& rng);
}  // namespace flint

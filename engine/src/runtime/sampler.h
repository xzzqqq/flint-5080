#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace flint {
class Sampler {
 public:
  int32_t Sample(std::span<const float> logits, float temperature, int32_t top_k, float top_p,
                 uint64_t& rng);

 private:
  struct Candidate {
    float weight;
    int32_t id;
  };
  std::vector<Candidate> candidates_;
};
}  // namespace flint

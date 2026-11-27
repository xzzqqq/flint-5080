#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "runtime/sampler.h"

int main() {
  try {
    flint::Sampler sampler;
    std::vector<float> logits{NAN, 2, 2, 0, INFINITY, -INFINITY};
    uint64_t seed = 12345;
    auto sample = [&](float temp, int k, float p) {
      return sampler.Sample(logits, temp, k, p, seed);
    };
    auto require = [](bool ok, const char* why) {
      if (!ok) throw std::runtime_error(why);
    };
    require(sample(0, 0, 0) == 1, "greedy tie break");
    require(sample(1, 1, 0) == 1, "top k tie break");
    require(sample(1, 0, 0.01f) == 1, "nucleus retains best");
    logits = {-2, 0, 2};
    auto sequence = [&](uint64_t initial, float temp, int k, float p) {
      seed = initial;
      std::vector<int> result;
      for (int i = 0; i < 256; ++i) result.push_back(sample(temp, k, p));
      return result;
    };
    auto a = sequence(12345, 2, 0, 0);
    require(a == sequence(12345, 2, 0, 1), "zero defaults or reproducibility");
    require(a != sequence(54321, 2, 0, 0), "seed variation");
    for (auto id : sequence(12345, 0.01f, 0, 0)) require(id == 2, "temperature");
    for (auto id : sequence(12345, 2, 2, 0)) require(id == 1 || id == 2, "top k cutoff");
    for (auto id : sequence(12345, 2, 0, 0.8f)) require(id == 1 || id == 2, "nucleus cutoff");
    logits = {0, 1, 2, 3, 4, 4, 4, 4};
    for (auto id : sequence(12345, 1, 2, 0)) require(id == 4 || id == 5, "top k trimming or ties");
    logits = {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    auto uniform = sequence(12345, INFINITY, 0, 0);
    logits = {0, 0};
    require(uniform == sequence(12345, 1, 0, 0), "infinite temperature");
    std::vector<int32_t> proposals{1, 1}, targets{1, 0, 2};
    auto greedy = flint::VerifySpeculative(proposals, targets, {}, {}, {}, 3, 0, 0, 1, seed);
    require(greedy.accepted_draft_count == 1 && greedy.bonus_token == 0, "greedy first mismatch");
    // All selected proposals have ID 1. Target top-1 rejects at row 1.
    std::vector<int> candidates(32);
    for (int i = 0; i < 32; ++i) candidates[i] = i % 16;
    std::vector<float> q(32), p(3 * 17);
    q[1] = q[17] = 1;
    p[1] = 8;
    p[17 + 16] = 8;
    p[34 + 2] = 8;
    auto verify = [&]() {
      return flint::VerifySpeculative(proposals, {}, candidates, q, p, 17, 1, 1, 1, seed);
    };
    auto rejected = verify();
    require(rejected.accepted_draft_count == 1 && rejected.bonus_token == 16,
            "outside sparse support");
    p[17 + 16] = 0;
    p[17 + 1] = 8;
    auto accepted = verify();
    require(accepted.accepted_draft_count == 2 && accepted.bonus_token == 2, "all accepted bonus");
    int accepted_zero = 0, proposal = 0;
    q[0] = q[1] = 0.5f;
    p.assign(34, -INFINITY);
    p[0] = 0;
    p[1] = std::log(3.f);
    p[17] = 0;
    for (int i = 0; i < 4096; ++i) {
      auto r = flint::VerifySpeculative({&proposal, 1}, {}, std::span(candidates).first(16),
                                        std::span(q).first(16), p, 17, 1, 0, 1, seed);
      accepted_zero += r.accepted_draft_count;
      require(r.accepted_draft_count || r.bonus_token == 1, "residual support");
    }
    require(accepted_zero > 1900 && accepted_zero < 2200, "acceptance probability");
    logits = {NAN, INFINITY};
    try {
      sample(1, 0, 0);
    } catch (const std::invalid_argument&) {
      return 0;
    }
    throw std::runtime_error("all-nonfinite logits accepted");
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

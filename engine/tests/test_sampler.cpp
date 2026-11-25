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

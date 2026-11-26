#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "common/json_output.h"
#include "loader/kv_layout.h"
#include "model/forward_types.h"

namespace flint {
inline constexpr double kRtx5080PeakBandwidthGbs = 960.0;
inline constexpr double kRtx5080SolDivisor = kRtx5080PeakBandwidthGbs / 100.0;

struct StateSlice {
  size_t offset;
  KvSliceKind kind;
  KvSliceDType dtype;
  std::vector<std::byte> bytes;
};
struct PostState {
  unsigned blocks{};
  std::vector<uint16_t> carry;
  std::vector<StateSlice> state;
};

[[nodiscard]] Object CompareState(const PostState& reference, const PostState& actual);
[[nodiscard]] Object ComputeDiff(std::span<const float> a, std::span<const float> b);
// Sorts the samples in place.
[[nodiscard]] double Median(std::span<double> samples);
void CompareGolden(Object& reply, const std::filesystem::path& dir, const std::string& name,
                   std::span<const int> prompt_ids, std::span<const int> generated_ids,
                   const ForwardRunSpec& spec, std::span<const float> values, unsigned width,
                   int next_token);
}  // namespace flint

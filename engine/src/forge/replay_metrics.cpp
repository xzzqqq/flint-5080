#include "forge/replay_metrics.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>

#include "loader/safetensors.h"

namespace flint {
namespace fs = std::filesystem;
namespace {
// Safetensors payloads and byte slices need not be aligned for T.
template <class T>
T ValueAt(std::span<const std::byte> bytes, size_t index) {
  T value;
  std::memcpy(&value, bytes.data() + index * sizeof(T), sizeof(T));
  return value;
}
float FromBf16(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }

struct StateError {
  bool exact = true;
  size_t nonfinite = 0;
  double max_abs = 0;

  void Compare(std::span<const std::byte> a, std::span<const std::byte> b, KvSliceDType dtype) {
    const bool bf16 = dtype == KvSliceDType::kBF16;
    if (a.size() != b.size()) throw std::logic_error("incompatible state slices");
    exact &= std::equal(a.begin(), a.end(), b.begin());
    const size_t count = a.size() / (bf16 ? sizeof(uint16_t) : sizeof(float));
    for (size_t i = 0; i < count; ++i) {
      double x = bf16 ? FromBf16(ValueAt<uint16_t>(a, i)) : ValueAt<float>(a, i);
      double y = bf16 ? FromBf16(ValueAt<uint16_t>(b, i)) : ValueAt<float>(b, i);
      if (std::isfinite(x) && std::isfinite(y))
        max_abs = std::max(max_abs, std::abs(x - y));
      else
        ++nonfinite;
    }
  }
  Object Report() const {
    return {{"exact_match", exact},
            {"max_abs_err", nonfinite ? Json(nullptr) : Json(max_abs)},
            {"nonfinite", nonfinite}};
  }
};

Object TopLogits(const SafetensorsFile& file, const std::string& phase,
                 std::span<const float> values) {
  const auto key = phase + "top64_logits";
  auto* ids = file.Find(phase + "top64_ids");
  auto* logits = file.Find(key);
  if (!ids || !logits || ids->dtype != TensorDType::kI32 || logits->dtype != TensorDType::kF32 ||
      ids->shape.size() != 1 || ids->shape != logits->shape || !ids->nbytes)
    return {};
  std::vector<float> ref(logits->shape[0]), actual(ref.size());
  std::memcpy(ref.data(), file.Bytes(*logits).data(), logits->nbytes);
  auto indices = file.Bytes(*ids);
  for (size_t i = 0; i < ref.size(); ++i) {
    int id = ValueAt<int>(indices, i);
    if (id < 0 || size_t(id) >= values.size()) return {};
    actual[i] = values[id];
  }
  auto diff = ComputeDiff(ref, actual);
  diff["tensor"] = key;
  diff["top_k"] = ref.size();
  return diff;
}
}  // namespace

KldSummary KldAccumulator::Summary() const {
  auto sorted = position_kl;
  if (sorted.empty()) throw std::invalid_argument("no KLD positions");
  std::ranges::sort(sorted);
  auto n = sorted.size();
  return {n,
          top1_matches,
          std::accumulate(sorted.begin(), sorted.end(), 0.0) / n,
          sorted[n - 1 - n / 20],
          sorted.back(),
          max_abs_logit_diff};
}

void AccumulateKld(std::span<const float> ref_logits, std::span<const float> cand_logits,
                   size_t vocab, KldAccumulator& accumulator) {
  if (!vocab || ref_logits.size() != cand_logits.size() || ref_logits.size() % vocab)
    throw std::invalid_argument("incompatible KLD logits shapes");
  for (size_t pos = 0; pos < ref_logits.size(); pos += vocab) {
    auto ref = ref_logits.subspan(pos, vocab), cand = cand_logits.subspan(pos, vocab);
    auto r = std::ranges::max_element(ref), c = std::ranges::max_element(cand);
    const double rmax = *r, cmax = *c;
    double rsum = 0, csum = 0;
    for (size_t i = 0; i < vocab; ++i) {
      if (!std::isfinite(ref[i]) || !std::isfinite(cand[i]))
        throw std::invalid_argument("nonfinite KLD logit");
      accumulator.max_abs_logit_diff =
          std::max(accumulator.max_abs_logit_diff, std::abs(double(ref[i]) - cand[i]));
      rsum += std::exp(ref[i] - rmax);
      csum += std::exp(cand[i] - cmax);
    }
    const double rnorm = std::log(rsum), cnorm = std::log(csum);
    double kl = 0;
    for (size_t i = 0; i < vocab; ++i) {
      double log_p = (ref[i] - rmax) - rnorm, log_q = (cand[i] - cmax) - cnorm;
      kl += std::exp(log_p) * (log_p - log_q);
    }
    accumulator.position_kl.push_back(std::max(0.0, kl));
    accumulator.top1_matches += (r - ref.begin() == c - cand.begin());
  }
}

Object ComputeDiff(std::span<const float> a, std::span<const float> b) {
  if (a.empty() || a.size() != b.size()) throw std::runtime_error("incompatible comparison shapes");
  double mx = 0, sum = 0, sq = 0, aa = 0, bb = 0, ab = 0;
  size_t nonfinite = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    double x = a[i], y = b[i], d = std::abs(x - y);
    if (!std::isfinite(d)) ++nonfinite;
    mx = std::max(mx, d);
    sum += d;
    sq += d * d;
    aa += x * x;
    bb += y * y;
    ab += x * y;
  }
  return {{"exact_match", !std::memcmp(a.data(), b.data(), a.size_bytes())},
          {"nonfinite", nonfinite},
          {"max_abs_err", nonfinite ? Json(nullptr) : Json(mx)},
          {"mean_abs_err", sum / a.size()},
          {"rmse", std::sqrt(sq / a.size())},
          {"cosine_sim", aa && bb ? ab / std::sqrt(aa * bb) : double(aa == bb)}};
}
double Median(std::span<double> samples) {
  if (samples.empty()) throw std::invalid_argument("median requires at least one sample");
  std::ranges::sort(samples);
  return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}

Object CompareState(const PostState& reference, const PostState& actual) {
  StateError carry, kv, conv;
  carry.Compare(std::as_bytes(std::span(reference.carry)), std::as_bytes(std::span(actual.carry)),
                KvSliceDType::kBF16);
  if (reference.state.size() != actual.state.size())
    throw std::logic_error("incompatible state slice counts");
  for (size_t i = 0; i < reference.state.size(); ++i) {
    const auto& slice = reference.state[i];
    const auto& other = actual.state[i];
    if (slice.offset != other.offset || slice.kind != other.kind || slice.dtype != other.dtype)
      throw std::logic_error("incompatible state slice layout");
    auto& error = slice.kind == KvSliceKind::kKey || slice.kind == KvSliceKind::kValue ? kv : conv;
    error.Compare(slice.bytes, other.bytes, slice.dtype);
  }
  const bool blocks_match = reference.blocks == actual.blocks;
  return {{"exact_match", blocks_match && carry.exact && kv.exact && conv.exact},
          {"blocks_match", blocks_match},
          {"carry", carry.Report()},
          {"kv", kv.Report()},
          {"conv", conv.Report()}};
}

void CompareGolden(Object& reply, const std::filesystem::path& dir, const std::string& name,
                   std::span<const int> prompt_ids, std::span<const int> generated_ids,
                   const ForwardRunSpec& spec, std::span<const float> values, unsigned width,
                   int next_token) {
  if (dir.empty() || name.empty() ||
      name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.") !=
          std::string::npos)
    return;
  auto path = dir / ("e2e_" + name + ".safetensors");
  if (!fs::exists(path)) return;
  auto file = SafetensorsFile::Open(path);
  auto* prompt = file.Find("input_ids");
  if (!prompt || prompt->dtype != TensorDType::kI32 || prompt->nbytes != prompt_ids.size_bytes() ||
      std::memcmp(file.Bytes(*prompt).data(), prompt_ids.data(), prompt->nbytes))
    return;
  std::string phase = "prefill.";
  bool decode = !generated_ids.empty();
  auto* generated = file.Find("generated_ids");
  if (decode) {
    const size_t nbytes = generated_ids.size_bytes();
    if (!generated || generated->dtype != TensorDType::kI32 || generated->nbytes < nbytes ||
        std::memcmp(file.Bytes(*generated).data(), generated_ids.data(), nbytes))
      return;
    phase = "decode_" + std::to_string(generated_ids.size() - 1) + ".";
  }
  const bool head = spec.run_lm_head;
  auto key = phase + (head ? (decode ? "logits" : "last_logits")
                           : "step_" + std::to_string(spec.end_visit - 1));
  auto* t = file.Find(key);
  const std::array<int64_t, 2> shape{head ? static_cast<int64_t>(values.size()) : spec.seq_len,
                                     width};
  const auto expected_shape = std::span(shape).first(head ? 1 : 2);
  Object diff;
  if (t && std::ranges::equal(t->shape, expected_shape) &&
      (t->dtype == TensorDType::kF32 || t->dtype == TensorDType::kBF16)) {
    auto bytes = file.Bytes(*t);
    std::vector<float> ref(bytes.size() /
                           (t->dtype == TensorDType::kF32 ? sizeof(float) : sizeof(uint16_t)));
    if (t->dtype == TensorDType::kF32)
      std::memcpy(ref.data(), bytes.data(), bytes.size());
    else {
      for (size_t i = 0; i < ref.size(); ++i) ref[i] = FromBf16(ValueAt<uint16_t>(bytes, i));
    }
    diff = ComputeDiff(ref, values);
    diff["tensor"] = key;
  } else if (!t && head) {
    diff = TopLogits(file, phase, values);
  }
  if (head) {
    size_t offset = generated_ids.size_bytes();
    if (generated && generated->dtype == TensorDType::kI32 &&
        offset + sizeof(int) <= generated->nbytes) {
      int id;
      std::memcpy(&id, file.Bytes(*generated).data() + offset, sizeof(id));
      diff["expected_next_token_id"] = id;
      diff["next_token_match"] = id == next_token;
    }
  }
  if (!diff.empty()) reply["golden"] = std::move(diff);
}
}  // namespace flint

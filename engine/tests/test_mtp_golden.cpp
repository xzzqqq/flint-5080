#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

#include "common/cuda_check.h"
#include "loader/arenas.h"
#include "loader/kv_layout.h"
#include "loader/safetensors.h"
#include "loader/scratch_workspace.h"
#include "model/dspark_forward.h"
#include "model/inkling2_forward.h"
#include "runtime/sampler.h"

namespace {
using namespace flint;
void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
template <class T>
std::vector<T> Tensor(const SafetensorsFile& file, const std::string& name, TensorDType dtype) {
  auto* t = file.Find(name);
  Require(t && t->dtype == dtype, "invalid golden " + name);
  std::vector<T> v(t->nbytes / sizeof(T));
  std::memcpy(v.data(), file.Bytes(*t).data(), t->nbytes);
  return v;
}
template <class T>
std::vector<T> Read(const void* p, size_t count) {
  std::vector<T> v(count);
  CheckCuda(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
float Value(uint16_t bits) { return std::bit_cast<float>(uint32_t(bits) << 16); }
float Value(float x) { return x; }
template <class T>
bool Compare(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& key) {
  Require(actual.size() == expected.size(), "shape mismatch " + key);
  double mx = 0, ss = 0, norm = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    double y = Value(expected[i]), d = Value(actual[i]) - y;
    Require(std::isfinite(d), "nonfinite " + key);
    mx = std::max(mx, std::abs(d));
    ss += d * d;
    norm += y * y;
  }
  std::cout << key << " max=" << mx << " relative_l2=" << std::sqrt(ss / std::max(norm, 1e-30))
            << std::endl;
  return ss <= 1.e-4 * norm;
}
size_t Free() {
  size_t f, total;
  CheckCuda(cudaMemGetInfo(&f, &total));
  return f;
}
// Uniform attention has an analytical answer. Distinct values in the evicted
// prefix and the draft block exercise ring wrap, both visibility bounds and GQA.
void AttentionWindow() {
  constexpr int ctx = 1040, length = ctx + 8;
  auto arena = ScratchArena::Allocate(32ul << 20);
  ScratchWorkspace ws(arena.view());
  Gemm gemm(nullptr, ws.Take<char>(4ul << 20), 4ul << 20);
  auto data = ws.Take<uint16_t>(length * 1024);
  auto k = ws.Take<uint16_t>(length * 1024), v = ws.Take<uint16_t>(length * 1024);
  auto q = ws.Take<uint16_t>(8 * 3072), out = ws.Take<uint16_t>(8 * 3072);
  CheckCuda(cudaMemset(q, 0, 8 * 3072 * 2));
  std::vector<uint16_t> values(length * 1024);
  auto bf16 = [](float x) {
    uint32_t bits = std::bit_cast<uint32_t>(x);
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
  };
  for (int t = 0; t < length; ++t)
    for (int d = 0; d < 1024; ++d)
      values[t * 1024 + d] = bf16(float(d / 128 + (t < 16 ? 128 : t >= ctx ? 64 : 0)));
  CheckCuda(cudaMemcpy(data, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
  for (bool local : {false, true}) {
    DraftKv cache{k, v, local ? 1032 : length, local};
    CheckCuda(flint_dspark_kv_write(cache, data, data, 1024, 0, nullptr));
    auto tail = data + 1024 * 1024;
    CheckCuda(flint_dspark_kv_write(cache, tail, tail, length - 1024, 1024, nullptr));
    for (size_t bytes : {ws.rest.bytes, size_t(length) * sizeof(float)}) {
      CheckCuda(flint_dspark_block_attention(cache, q, out, static_cast<float*>(ws.rest.device_ptr),
                                             bytes, gemm.handle(), ctx, nullptr));
      auto attention = Read<uint16_t>(out, 8 * 3072);
      for (int i = 0; i < 8 * 3072; ++i) {
        int kv_head = (i % 3072) / 128 / 3;
        float mean = kv_head + (8 * 64.f + (local ? 0.f : 16 * 128.f)) / (local ? 1032 : length);
        Require(attention[i] == bf16(mean), "draft block visibility or GQA");
      }
    }
  }
}
void Case(const std::filesystem::path& model, const std::filesystem::path& dir, bool warmup) {
  auto file = SafetensorsFile::Open(dir / "mtp_golden.safetensors");
  auto ids = [&](const std::string& key) { return Tensor<int>(file, key, TensorDType::kI32); };
  auto prompt = ids("prompt_ids"), gold = ids("spec_generated_ids");
  auto ar = SafetensorsFile::Open(dir / "e2e_short_math.safetensors");
  Require(gold.size() == 49 && gold == Tensor<int>(ar, "generated_ids", TensorDType::kI32),
          "AR fixture mismatch");
  unsigned n = prompt.size();
  auto weights = WeightsArena::Load(model, nullptr, true);
  auto kv = KvStateArena::Allocate(weights.view().config,
                                   std::max<size_t>(1040, n + gold.size() + 8), true);
  auto scratch = ScratchArena::Allocate(), draft_ws = ScratchArena::Allocate(32ul << 20);
  InklingRunner target(weights.view(), kv.view(), scratch.view(), nullptr);
  DsparkRunner draft(weights.view(), kv.view(), draft_ws.view(), target.stream());
  auto io = ScratchArena::Allocate((16ul << 20) + size_t(n) * 2304 * 24);
  ScratchWorkspace ws(io.view());
  auto all_logits = ws.Take<float>(8 * 200058);
  auto input = ws.Take<int>(std::max(n, 8u)), next = ws.Take<int>(8);
  auto hidden = ws.Take<uint16_t>(std::max(n, 8u) * 2304),
       taps = ws.Take<uint16_t>(std::max(n, 8u) * 4 * 2304);
  auto upload = [&](void* dst, const void* src, size_t bytes) {
    CheckCuda(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, target.stream()));
  };
  upload(input, prompt.data(), n * sizeof(int));
  ForwardRunSpec spec{.end_visit = 48,
                      .run_embed = 1,
                      .run_lm_head = 1,
                      .reset_kv = 1,
                      .seq_len = n,
                      .d_input_ids = input,
                      .d_hidden_out = hidden,
                      .d_next_token = next,
                      .d_taps_out = taps};
  int anchor = target.Run(spec, {}).next_token_id;
  Require(anchor == ids("prefill_bonus_id")[0], "prefill bonus");
  DsparkCapture cap{};
  for (auto& mix : cap.mix) mix = ws.Take<uint16_t>(n * 2304);
  cap.embed_block = ws.Take<uint16_t>(8 * 2304);
  for (auto& layer : cap.layer_out) layer = ws.Take<uint16_t>(8 * 2304);
  cap.head_hidden = ws.Take<uint16_t>(8 * 2304);
  cap.f_rnn = ws.Take<uint16_t>(2304);
  cap.context = ws.Take<float>(7 * 768);
  cap.states = ws.Take<uint16_t>(7 * 256);
  cap.latents = ws.Take<uint16_t>(7 * 256);
  std::vector<int> top_ids(7 * 64);
  std::vector<float> top_logits(7 * 64);
  cap.h_top_ids = top_ids.data();
  cap.h_top_logits = top_logits.data();
  bool ok = true;
  auto bf = [&](const std::string& key, const void* p) {
    auto expected = Tensor<uint16_t>(file, key, TensorDType::kBF16);
    ok &= Compare(Read<uint16_t>(p, expected.size()), expected, key);
  };
  // Isolate draft module errors from target prefill parity.
  bf("prefill_taps", taps);
  auto fixture = Tensor<uint16_t>(file, "prefill_taps", TensorDType::kBF16);
  auto isolated = ws.Take<uint16_t>(fixture.size());
  upload(isolated, fixture.data(), fixture.size() * 2);
  draft.StoreContextKv(isolated, n, 0, &cap);
  CheckCuda(cudaDeviceSynchronize());
  for (int i = 0; i < 3; ++i) {
    auto layer = "layer_" + std::to_string(i);
    bf("mix_rmsnorm." + layer, cap.mix[i]);
    auto& v = kv.view();
    size_t plane = size_t(i == 1 ? v.max_full_seq_len : v.mtp_swa_window) * 1024;
    auto* k =
        static_cast<uint16_t*>(i == 1 ? v.mtp_full_kv : v.mtp_swa_kv) + (i == 2 ? 2 * plane : 0);
    bf("ctx_kv." + layer + ".k", k);
    bf("ctx_kv." + layer + ".v", k + plane);
  }
  bf("mix_rmsnorm.markov", cap.mix[3]);
  std::array<int, 8> block{anchor};
  draft.DraftGreedy(anchor, isolated + size_t(n - 1) * 4 * 2304, n, block.data() + 1, &cap);
  std::string prefix = "round_0.";
  bf(prefix + "embed_block", cap.embed_block);
  for (int i = 0; i < 3; ++i) bf(prefix + "layer_" + std::to_string(i) + "_out", cap.layer_out[i]);
  bf(prefix + "head_hidden", cap.head_hidden);
  bf(prefix + "f_rnn", cap.f_rnn);
  prefix += "markov.";
  ok &= Compare(Read<float>(cap.context, 7 * 768),
                Tensor<float>(file, prefix + "context", TensorDType::kF32), prefix + "context");
  bf(prefix + "states", cap.states);
  bf(prefix + "latents", cap.latents);
  auto expected_top = Tensor<uint16_t>(file, prefix + "top64_logits", TensorDType::kBF16);
  std::vector<float> top(expected_top.size());
  std::transform(expected_top.begin(), expected_top.end(), top.begin(),
                 [](uint16_t b) { return Value(b); });
  ok &= Compare(top_logits, top, prefix + "top64_logits");
  auto ref_top = ids(prefix + "top64_ids");
  int matches = 0;
  for (size_t i = 0; i < top_ids.size(); ++i) matches += top_ids[i] == ref_top[i];
  std::cout << "top64 exact ranks=" << matches << "/448" << std::endl;
  ok &= top_ids == ref_top;
  ok &= std::equal(block.begin() + 1, block.end(), ids(prefix + "draft_ids").begin());
  if (!warmup) {
    std::vector<float> greedy_logits(7 * 200058), sampled_logits(greedy_logits.size());
    std::array<int, 7> sampled{};
    auto greedy = [](std::span<const float> row) {
      return int(std::max_element(row.begin(), row.end()) - row.begin());
    };
    auto tap = isolated + size_t(n - 1) * 4 * 2304;
    draft.DraftWithLogits(anchor, tap, n, sampled.data(), greedy_logits.data(), greedy);
    Require(std::equal(sampled.begin(), sampled.end(), block.begin() + 1),
            "greedy sampler callback");
    int step = 0;
    draft.DraftWithLogits(anchor, tap, n, sampled.data(), sampled_logits.data(), [&](auto row) {
      int best = greedy(row);
      return step++ == 0 ? (best + 1) % 200058 : best;
    });
    Require(step == 7 &&
                std::equal(greedy_logits.begin(), greedy_logits.begin() + 200058,
                           sampled_logits.begin()) &&
                !std::equal(greedy_logits.begin() + 200058, greedy_logits.begin() + 2 * 200058,
                            sampled_logits.begin() + 200058),
            "sampled token feedback");
  }
  draft.StoreContextKv(taps, n, 0);
  auto last = taps + size_t(n - 1) * 4 * 2304;
  std::vector<int> generated{anchor};
  spec.seq_len = 8;
  spec.reset_kv = 0;
  spec.pos_offset = n;
  uint64_t rng = 0;
  for (int round = 0; generated.size() < gold.size(); ++round) {
    block[0] = anchor;
    draft.DraftGreedy(anchor, last, spec.pos_offset, block.data() + 1);
    auto key = "round_" + std::to_string(round) + ".";
    upload(input, block.data(), sizeof(block));
    spec.d_logits_out = all_logits;
    target.RunVerify(spec, {.return_logits = round == 0});
    auto verified = Read<int>(next, 8);
    if (round == 0) {
      auto logits = Read<float>(all_logits, 8 * 200058);
      for (int j = 0; j < 8; ++j) {
        auto row = std::span(logits).subspan(j * 200058, 200058);
        Require(std::max_element(row.begin(), row.end()) - row.begin() == verified[j],
                "verify logits row");
      }
    }
    auto result =
        VerifySpeculative(std::span(block).subspan(1), verified, {}, {}, 200058, 0, 0, 0, rng);
    int a = result.accepted_draft_count;
    std::cout << "round " << round << " accepted=" << a << " bonus=" << result.bonus_token
              << std::endl;
    Require(std::equal(block.begin() + 1, block.end(), ids(key + "draft_ids").begin()),
            key + "draft_ids");
    Require(
        a == ids(key + "accepted_count")[0] && result.bonus_token == ids(key + "next_bonus_id")[0],
        key + "verification");
    int count = std::min<size_t>(1 + a, gold.size() - generated.size());
    target.CommitVerify(spec.pos_offset, 8, count);

    draft.StoreContextKv(taps, count, spec.pos_offset);
    last = taps + size_t(count - 1) * 4 * 2304;
    spec.pos_offset += count;
    for (int i = 0; i < count; ++i) generated.push_back(i < a ? block[i + 1] : result.bonus_token);
    anchor = generated.back();
    if (warmup) break;
  }
  if (!warmup) {
    Require(generated == gold, "speculative sequence mismatch");
    std::cout << "short_math 49/49" << std::endl;
    Require(ok, "draft module parity");

    // Cross the SWA ring boundary with live, nonzero history. Compare the
    // committed fixed state against directly evaluating just the accepted prefix.
    spec.pos_offset = 1021;
    auto& v = kv.view();
    auto sizes = ComputeKvArenaSizes(weights.view().config);
    auto snapshot = [&]() {
      CheckCuda(cudaStreamSynchronize(target.stream()));
      std::vector<std::byte> fixed;
      for (auto [ptr, bytes] : FixedKvRegions(v, sizes))
        if (bytes) {
          auto data = Read<std::byte>(ptr, bytes);
          fixed.insert(fixed.end(), data.begin(), data.end());
        }
      return fixed;
    };
    auto seed = [&]() {
      CheckCuda(cudaMemsetAsync(v.arena_base, 0x39, v.arena_bytes, target.stream()));
    };
    for (unsigned keep : {0u, 3u, 8u}) {
      seed();
      if (keep) {
        spec.seq_len = keep;
        target.Run(spec, {});
      }
      auto expected = snapshot();
      seed();
      spec.seq_len = 8;
      target.RunVerify(spec);
      target.CommitVerify(spec.pos_offset, 8, keep);
      Require(snapshot() == expected, "wrapped SWA or SConv commit " + std::to_string(keep));
    }
    std::cout << "wraparound SWA and SConv rollback exact for 0, 3, 8" << std::endl;
  }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      AttentionWindow();
      return 0;
    }
    if (argc != 3) throw std::invalid_argument("usage: mtp_tests MODEL_DIR GOLDEN_DIR");
    Case(argv[1], argv[2], true);
    auto before = Free();
    Case(argv[1], argv[2], false);
    CheckCuda(cudaDeviceSynchronize());
    auto loss = int64_t(before) - int64_t(Free());
    std::cout << "teardown_delta_bytes=" << loss << std::endl;
    Require(loss == 0, "teardown leak");
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

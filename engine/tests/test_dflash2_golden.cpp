#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

#include "common/cuda_check.h"
#include "common/splitmix64.h"
#include "loader/arenas.h"
#include "loader/kv_layout.h"
#include "loader/safetensors.h"
#include "loader/scratch_workspace.h"
#include "model/dflash2_forward.h"
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
bool Compare(const std::vector<T>& actual, const std::vector<T>& expected, const std::string& key,
             double max_rel_l2 = 0.01) {
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
  return ss <= max_rel_l2 * max_rel_l2 * norm;
}
size_t Free() {
  size_t f, total;
  CheckCuda(cudaMemGetInfo(&f, &total));
  return f;
}
// Uniform attention has an analytical answer. Distinct values in the evicted
// prefix and the draft block exercise ring wrap, both visibility bounds and GQA.
void AttentionWindow() {
  constexpr int ctx = 2064, length = ctx + 8;
  auto arena = ScratchArena::Allocate(32ul << 20);
  ScratchWorkspace ws(arena.view());
  auto data = ws.Take<uint16_t>(length * 1024);
  auto k = ws.Take<uint16_t>(length * 1024), v = ws.Take<uint16_t>(length * 1024);
  auto q = ws.Take<uint16_t>(8 * 4096), out = ws.Take<uint16_t>(8 * 4096);
  CheckCuda(cudaMemset(q, 0, 8 * 4096 * 2));
  std::vector<uint16_t> values(length * 1024);
  auto bf16 = [](float x) {
    uint32_t bits = std::bit_cast<uint32_t>(x);
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
  };
  for (int t = 0; t < length; ++t)
    for (int d = 0; d < 1024; ++d)
      values[t * 1024 + d] = bf16(float(d / 128 + (t < 24 ? 128 : t >= ctx ? 64 : 0)));
  CheckCuda(cudaMemcpy(data, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
  {
    DraftKv cache{k, v, 2056};
    CheckCuda(flint_dflash2_kv_write(cache, data, data, 1024, 0, nullptr));
    auto tail = data + 1024 * 1024;
    CheckCuda(flint_dflash2_kv_write(cache, tail, tail, length - 1024, 1024, nullptr));
    for (size_t bytes : {ws.rest.bytes, size_t(length) * sizeof(float)}) {
      CheckCuda(flint_dflash2_block_attention(
          cache, q, out, static_cast<float*>(ws.rest.device_ptr), bytes, ctx, nullptr));
      auto attention = Read<uint16_t>(out, 8 * 4096);
      for (int i = 0; i < 8 * 4096; ++i) {
        int kv_head = (i % 4096) / 128 / 4;
        int row = i / 4096;
        int visible = 2048 + 7 - row;
        float mean = kv_head + ((7 - row) * 128 + 8 * 64.f) / visible;
        Require(attention[i] == bf16(mean), "draft block visibility or GQA");
      }
    }
  }
}
bool Case(const std::filesystem::path& model, const std::filesystem::path& draft_dir,
          const std::filesystem::path& dir, bool warmup) {
  auto file = SafetensorsFile::Open(dir / "dflash2_golden.safetensors");
  auto ids = [&](const std::string& key) { return Tensor<int>(file, key, TensorDType::kI32); };
  auto prompt = ids("prompt_ids"), gold = ids("spec_generated_ids");
  Require(!gold.empty() && gold == ids("ar_generated_ids"), "AR fixture mismatch");
  unsigned n = prompt.size();
  auto weights = WeightsArena::Load(model, nullptr, draft_dir);
  auto kv = KvStateArena::Allocate(weights.view().config,
                                   std::max<size_t>(1040, n + gold.size() + 8), true);
  auto scratch = ScratchArena::Allocate();
  auto target_ws = scratch.view();
  auto draft_ws = DFlash2Runner::ReserveScratch(target_ws);
  InklingRunner target(weights.view(), kv.view(), target_ws, nullptr);
  DFlash2Runner draft(weights.view(), kv.view(), draft_ws, target.stream());
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
  DFlash2Capture cap{};
  cap.embed_block = ws.Take<uint16_t>(8 * 2304);
  for (auto& layer : cap.layer_out) layer = ws.Take<uint16_t>(8 * 2304);
  cap.head_hidden = ws.Take<uint16_t>(8 * 2304);
  cap.unary = ws.Take<float>(7 * 16);
  cap.scores = ws.Take<float>(7 * 16 * 16);
  bool ok = true;
  auto bf = [&](const std::string& key, const void* p, double max_rel_l2 = 0.01) {
    auto expected = Tensor<uint16_t>(file, key, TensorDType::kBF16);
    ok &= Compare(Read<uint16_t>(p, expected.size()), expected, key, max_rel_l2);
  };
  // Isolate draft module errors from target prefill parity (48-visit looped MoE taps).
  bf("prefill_taps", taps, 0.05);
  auto fixture = Tensor<uint16_t>(file, "prefill_taps", TensorDType::kBF16);
  auto isolated = ws.Take<uint16_t>(fixture.size());
  upload(isolated, fixture.data(), fixture.size() * 2);
  draft.StoreContextKv(isolated, n, 0);
  CheckCuda(cudaDeviceSynchronize());
  for (int i = 0; i < 5; ++i) {
    auto layer = "layer_" + std::to_string(i);
    auto& v = kv.view();
    size_t plane = size_t(v.draft_swa_window) * 1024;
    auto k = static_cast<uint16_t*>(v.draft_swa_kv) + i * 2 * plane;
    bf("ctx_kv." + layer + ".k", k);
    bf("ctx_kv." + layer + ".v", k + plane);
  }
  std::array<int, 8> block{anchor};
  ForwardDraftResult proposal;
  draft.Draft({anchor, int(n), 0, 0, 0}, proposal, &cap);
  const std::string prefix = "round_0.";
  Require(std::ranges::equal(ids(prefix + "draft_ids"), proposal.draft_ids),
          "composed greedy path");
  bf(prefix + "embed_block", cap.embed_block);
  // FP64 attention alone moves the final layer by 4.13% and the head by
  // 1.70% versus eager FP32 on this fixture (2026-11-27 findings).
  for (int i = 0; i < 5; ++i)
    bf(prefix + "layer_" + std::to_string(i) + "_out", cap.layer_out[i], i == 4 ? 0.05 : 0.01);
  bf(prefix + "head_hidden", cap.head_hidden, 0.02);
  auto ranked_logits = Tensor<float>(file, prefix + "unary_logits", TensorDType::kF32);
  ok &= Compare(Read<float>(cap.unary, ranked_logits.size()), ranked_logits,
                "composed sorted unary logits");
  // Split-KV reduction rounds a few attention elements differently from the
  // dense FP32 oracle. Measure each layer on identical inputs before comparing
  // edge matrices, whose axes are the *ranked token IDs* of adjacent rows.
  std::array<std::vector<uint16_t>, 5> reference;
  for (int i = 0; i < 5; ++i) {
    reference[i] =
        Tensor<uint16_t>(file, prefix + "layer_" + std::to_string(i) + "_out", TensorDType::kBF16);
    auto device = ws.Take<uint16_t>(reference[i].size());
    upload(device, reference[i].data(), reference[i].size() * sizeof(uint16_t));
    cap.reference_layer_out[i] = device;
  }
  std::cout << "teacher-forced decoder and selector" << std::endl;
  draft.Draft({anchor, int(n), 0, 0, 0}, proposal, &cap);
  for (int i = 0; i < 5; ++i) bf(prefix + "layer_" + std::to_string(i) + "_out", cap.layer_out[i]);
  bf(prefix + "head_hidden", cap.head_hidden);
  Require(std::ranges::equal(ids(prefix + "cand_ids"), proposal.candidate_ids),
          "selector top-16 IDs");
  Require(std::ranges::equal(ids(prefix + "draft_ids"), proposal.draft_ids),
          "selector greedy path");
  for (auto [key, ptr] :
       {std::pair{"unary_logits", cap.unary}, std::pair{"selector_scores", cap.scores}}) {
    auto expected = Tensor<float>(file, prefix + key, TensorDType::kF32);
    ok &= Compare(Read<float>(ptr, expected.size()), expected, key);
  }
  // The walk below consumes the scores from the unforced eager block.
  std::ranges::fill(cap.reference_layer_out, nullptr);
  draft.Draft({anchor, int(n), 0, 0, 0}, proposal, &cap);
  for (float temperature : {0.f, 1.f}) {
    ForwardDraftResult eager, graph;
    ForwardDraftSpec settings{anchor, int(n), temperature, 12345, 0};
    draft.Draft(settings, eager);
    settings.use_cuda_graph = 1;
    draft.Draft(settings, graph);
    Require(!std::memcmp(&eager, &graph, sizeof(graph)), "graph/eager mismatch");
    if (temperature > 0) {
      auto scores = Read<float>(cap.scores, kDraftTokens * kSelectorTopK * kSelectorTopK);
      int predecessor = 0;
      uint64_t walk = settings.seed;
      for (int e = 0; e < kDraftTokens; ++e) {
        auto row = std::span(scores).subspan((e * kSelectorTopK + predecessor) * kSelectorTopK,
                                             kSelectorTopK);
        float peak = *std::ranges::max_element(row);
        double sum = 0;
        for (float value : row) sum += std::exp((value - peak) / temperature);
        double draw = flint::SplitMixUniform(walk);
        int chosen = -1;
        for (int c = 0; c < kSelectorTopK; ++c) {
          float q = std::exp((row[c] - peak) / temperature) / sum;
          Require(std::abs(graph.q_probs[e * kSelectorTopK + c] - q) < 1e-6,
                  "conditional draft probability");
          if (chosen < 0 && (draw -= q) < 0) chosen = c;
        }
        Require(
            chosen >= 0 && graph.draft_ids[e] == graph.candidate_ids[e * kSelectorTopK + chosen],
            "seeded draft path");
        predecessor = chosen;
      }
      std::vector<float> target_logits(8 * 200058, -INFINITY);
      // A disjoint target must reject immediately and return its only token.
      int outside = 0;
      auto candidates = std::span(graph.candidate_ids).first(16);
      while (std::ranges::find(candidates, outside) != candidates.end()) ++outside;
      target_logits[outside] = 0;
      uint64_t seed = 21;
      auto verified = VerifySpeculative(graph.draft_ids, {}, graph.candidate_ids, graph.q_probs,
                                        target_logits, 200058, temperature, 0, 1, seed);
      Require(!verified.accepted_draft_count && verified.bonus_token == outside,
              "sparse rejection");
    }
  }
  draft.StoreContextKv(taps, n, 0);
  std::vector<int> generated{anchor};
  spec.seq_len = 8;
  spec.reset_kv = 0;
  spec.pos_offset = n;
  uint64_t rng = 0;
  // The target and reference prefill taps have different BF16 rounding. Replay
  // the same proposal before comparing acceptance and the rejection bonus.
  const auto fixture_ids = ids("round_0.draft_ids");
  std::ranges::copy(fixture_ids, block.begin() + 1);
  upload(input, block.data(), sizeof(block));
  target.RunVerify(spec);
  auto oracle =
      VerifySpeculative(fixture_ids, Read<int>(next, 8), {}, {}, {}, 200058, 0, 0, 0, rng);
  Require(oracle.accepted_draft_count == ids("round_0.accepted_count")[0] &&
              oracle.bonus_token == ids("round_0.next_bonus_id")[0],
          "first verification on fixture proposal");
  target.CommitVerify(n, 8, 0);
  int accepted = 0;
  for (int round = 0; generated.size() < gold.size(); ++round) {
    block[0] = anchor;
    draft.Draft({anchor, int(spec.pos_offset), 0, 0, 1}, proposal);
    std::ranges::copy(proposal.draft_ids, block.begin() + 1);
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
        VerifySpeculative(std::span(block).subspan(1), verified, {}, {}, {}, 200058, 0, 0, 0, rng);
    int a = result.accepted_draft_count;
    std::cout << "round " << round << " accepted=" << a << " bonus=" << result.bonus_token
              << std::endl;
    auto expected =
        std::mismatch(block.begin() + 1, block.end(), verified.begin()).first - (block.begin() + 1);
    Require(a == expected, "accepted draft count");
    Require(result.bonus_token == verified[a], "verification bonus");
    int count = std::min<size_t>(1 + a, gold.size() - generated.size());
    accepted += std::min(a, count);
    target.CommitVerify(spec.pos_offset, 8, count);

    draft.StoreContextKv(taps, count, spec.pos_offset);
    spec.pos_offset += count;
    for (int i = 0; i < count; ++i) generated.push_back(i < a ? block[i + 1] : result.bonus_token);
    anchor = generated.back();
    if (warmup) break;
  }
  if (!warmup) {
    std::cout << "accepted=" << accepted << std::endl;
    Require(generated == gold, "speculative sequence mismatch");
    std::cout << "short_math matched " << gold.size() << " tokens" << std::endl;

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
      if (keep == 3) {
        auto different = block;
        std::fill(different.begin() + keep, different.end(), 777);
        upload(input, different.data(), sizeof(different));
        seed();
        target.RunVerify(spec);
        target.CommitVerify(spec.pos_offset, 8, keep);
        Require(snapshot() == expected, "rejected draft changed committed prefix");
        upload(input, block.data(), sizeof(block));
      }
    }
    std::cout << "wraparound SWA and SConv rollback exact for 0, 3, 8" << std::endl;
  }
  return ok;
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      AttentionWindow();
      return 0;
    }
    if (argc != 4)
      throw std::invalid_argument("usage: dflash2_tests MODEL_DIR DRAFT_DIR GOLDEN_DIR");
    Case(argv[1], argv[2], argv[3], true);
    auto before = Free();
    bool ok = Case(argv[1], argv[2], argv[3], false);
    CheckCuda(cudaDeviceSynchronize());
    auto loss = int64_t(before) - int64_t(Free());
    std::cout << "teardown_delta_bytes=" << loss << std::endl;
    Require(loss == 0, "teardown leak");
    Require(ok, "draft module parity");
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "common/cuda_check.h"
#include "common/splitmix64.h"
#include "loader/arenas.h"
#include "loader/scratch_workspace.h"
#include "ops/spec_verify_ops.h"
#include "runtime/sampler.h"

namespace {
using namespace flint;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Fixture {
  int len, vocab;
  SpecSampleMode mode;
  SpecVerifyInput in;
  ForwardDraftResult draft{};
  std::array<int, kDraftBlockSize> argmax{};
  std::vector<float> logits;
  size_t bytes;
  ScratchArena arena;
  ForwardDraftResult* d_draft;
  SpecVerifyInput* d_in;
  SpecVerifyOutput* d_out;
  int *d_ids, *d_argmax;
  float* d_logits;
  void* scratch;
  cudaStream_t stream{};
  cudaGraphExec_t graph{};

  Fixture(int width, int v, SpecSampleMode m)
      : len(width),
        vocab(v),
        mode(m),
        in{9, 42, uint32_t(len), m == SPEC_SAMPLE_GREEDY ? 0.f : 1.f, 0.f, 0, 5080},
        logits(size_t(len) * vocab),
        bytes(flint_spec_verify_scratch_bytes(len, vocab, mode)),
        arena(ScratchArena::Allocate(bytes + logits.size() * sizeof(float) + 8192)) {
    ScratchWorkspace ws(arena.view());
    d_draft = ws.Take<ForwardDraftResult>(1);
    d_in = ws.Take<SpecVerifyInput>(1);
    d_out = ws.Take<SpecVerifyOutput>(1);
    d_ids = ws.Take<int>(len);
    d_argmax = ws.Take<int>(len);
    d_logits = ws.Take<float>(logits.size());
    scratch = bytes ? ws.rest.device_ptr : nullptr;
    Require((mode == SPEC_SAMPLE_GREEDY) == (bytes == 0), "scratch query");
    CheckCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CheckCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CheckCuda(flint_spec_prepare_verify_ids(&d_in->anchor, d_draft, d_ids, len, stream));
    CheckCuda(flint_spec_verify_accept(d_draft, mode == SPEC_SAMPLE_GREEDY ? d_argmax : nullptr,
                                       mode == SPEC_SAMPLE_GREEDY ? nullptr : d_logits, d_in, d_out,
                                       scratch, bytes, len, vocab, mode, stream));
    cudaGraph_t captured{};
    CheckCuda(cudaStreamEndCapture(stream, &captured));
    CheckCuda(cudaGraphInstantiate(&graph, captured, 0));
    CheckCuda(cudaGraphDestroy(captured));
    for (int r = 0; r < len - 1; ++r) {
      draft.draft_ids[r] = r;
      for (int j = 0; j < kSelectorTopK; ++j)
        draft.candidate_ids[r * kSelectorTopK + j] = (r + j) % vocab;
      draft.q_probs[r * kSelectorTopK] = 1.f;
    }
  }
  ~Fixture() {
    cudaGraphExecDestroy(graph);
    cudaStreamDestroy(stream);
  }
  void Upload() {
    auto put = [&](void* dst, const void* src, size_t size) {
      CheckCuda(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    };
    put(d_draft, &draft, sizeof(draft));
    put(d_argmax, argmax.data(), len * sizeof(int));
    put(d_logits, logits.data(), logits.size() * sizeof(float));
  }
  SpecVerifyOutput Run() {
    CheckCuda(cudaMemcpyAsync(d_in, &in, sizeof(in), cudaMemcpyHostToDevice, stream));
    CheckCuda(cudaGraphLaunch(graph, stream));
    SpecVerifyOutput result;
    CheckCuda(cudaMemcpyAsync(&result, d_out, sizeof(result), cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaStreamSynchronize(stream));
    return result;
  }
  int Compare() {
    auto out = Run();
    uint64_t rng = in.seed;
    auto ref =
        VerifySpeculative(std::span(draft.draft_ids).first(len - 1), std::span(argmax).first(len),
                          std::span(draft.candidate_ids).first((len - 1) * kSelectorTopK),
                          std::span(draft.q_probs).first((len - 1) * kSelectorTopK), logits, vocab,
                          in.temperature, in.top_k, in.top_p, rng);
    int a = ref.accepted_draft_count;
    int draws = mode == SPEC_SAMPLE_GREEDY ? 0 : std::min(a + 1, len - 1) + 1;
    Require(out.next_seed == in.seed + draws * kSplitMixIncrement, "SplitMix draw count");
    Require(out.error_code == SPEC_VERIFY_OK && out.accepted_draft_count == a &&
                std::equal(out.tokens, out.tokens + a, draft.draft_ids) &&
                out.tokens[a] == ref.bonus_token && out.next_seed == rng,
            "CPU verify parity");
    return a;
  }
};

void Greedy(int len) {
  Fixture f(len, 23, SPEC_SAMPLE_GREEDY);
  for (int n : {0, len / 2, len - 1}) {
    f.argmax.fill(20);
    std::copy_n(f.draft.draft_ids, n, f.argmax.begin());
    f.Upload();
    Require(f.Compare() == n, "greedy accepted count");
    std::array<int, kDraftBlockSize> ids{};
    CheckCuda(cudaMemcpy(ids.data(), f.d_ids, len * sizeof(int), cudaMemcpyDeviceToHost));
    Require(
        ids[0] == f.in.anchor && std::equal(ids.begin() + 1, ids.begin() + len, f.draft.draft_ids),
        "verify ID handoff");
  }
}

void Full() {
  Fixture f(2, 32, SPEC_SAMPLE_STOCHASTIC_FULL);
  std::ranges::fill(f.logits, -INFINITY);
  f.logits[0] = f.logits[31] = f.logits[33] = 0;
  f.Upload();
  int accepted = 0;
  for (f.in.seed = 0; f.in.seed < 4096; ++f.in.seed) accepted += f.Compare();
  Require(std::abs(accepted - 2048) < 128, "half-mass proposal acceptance");
  for (float temp : {.01f, 1.f, float(INFINITY)}) {
    f.in.temperature = temp;
    f.logits[0] = -std::numeric_limits<float>::max();
    f.logits[1] = std::numeric_limits<float>::max();
    f.logits[31] = NAN;
    f.Upload();
    for (int i = 0; i < 16; ++i, ++f.in.seed) f.Compare();
  }
  f.logits[0] = NAN;
  f.logits[1] = INFINITY;
  f.Upload();
  Require(f.Run().error_code == SPEC_VERIFY_NO_FINITE_LOGITS, "all nonfinite row");
}

void Filtered() {
  Fixture f(4, 1025, SPEC_SAMPLE_STOCHASTIC_FILTERED);
  for (int r = 0; r < f.len; ++r) {
    // Equal logits straddle the k cutoff. Several distinct logits round to the same weight.
    auto* row = f.logits.data() + r * f.vocab;
    std::fill_n(row, f.vocab, -17.f);
    row[0] = -2e-8f;
    row[1] = -1e-8f;
    row[2] = row[4] = row[6] = 0;
    row[9] = NAN;
  }
  for (int k : {0, 2, 5}) {
    f.in.top_k = k;
    for (float p : {0.f, .39f, .8f, std::nextafter(1.f, 0.f)}) {
      if (!k && !p) continue;
      f.in.top_p = p;
      for (bool force_bonus : {false, true}) {
        auto logits = f.logits;
        if (force_bonus) {
          for (int r = 0; r < f.len - 1; ++r) f.logits[r * f.vocab + r] = 100.f;
        }
        f.Upload();
        for (int i = 0; i < 32; ++i, ++f.in.seed) {
          int n = f.Compare();
          if (force_bonus) Require(n == f.len - 1, "filtered bonus");
        }
        f.logits = std::move(logits);
      }
    }
  }
  f.draft.draft_ids[0] = 17;
  f.Upload();
  Require(f.Run().error_code == SPEC_VERIFY_DRAFT_OUTSIDE_SUPPORT, "outside support");
  f.draft.draft_ids[0] = 0;
  f.draft.candidate_ids[15] = f.vocab;
  f.Upload();
  Require(f.Run().error_code == SPEC_VERIFY_INVALID_CANDIDATE, "invalid candidate");
  // A malformed proposal with NaN total forces rejection and a zero residual on its support.
  f.draft.candidate_ids[15] = 15;
  f.draft.q_probs[15] = NAN;
  f.Upload();
  Require(f.Run().error_code == SPEC_VERIFY_EMPTY_RESIDUAL, "empty residual");
  --f.in.verify_len;
  Require(f.Run().error_code == SPEC_VERIFY_INVALID_PARAM, "shape mismatch");
}

void NucleusTail() {
  Fixture f(1, 1025, SPEC_SAMPLE_STOCHASTIC_FILTERED);
  f.in.top_p = std::nextafter(1.f, 0.f);
  // This SplitMix state produces the largest representable uniform on the bonus draw.
  f.in.seed = 0x31628af67b2131abULL;
  std::ranges::fill(f.logits, -17.f);
  f.logits[512] = 0.f;
  f.Upload();
  f.Compare();
  Require(f.Run().tokens[0] == 512, "nucleus guard excludes low-mass tail");
  CheckCuda(flint_spec_verify_accept(f.d_draft, nullptr, f.d_logits, f.d_in, f.d_out, f.scratch,
                                     f.bytes - 1, f.len, f.vocab, f.mode, f.stream));
  SpecVerifyOutput out;
  CheckCuda(cudaMemcpyAsync(&out, f.d_out, sizeof(out), cudaMemcpyDeviceToHost, f.stream));
  CheckCuda(cudaStreamSynchronize(f.stream));
  Require(out.error_code == SPEC_VERIFY_INVALID_PARAM, "undersized sort scratch");
}

void RealVocab(int width, SpecSampleMode mode) {
  Fixture f(width, 200058, mode);
  std::mt19937 rng(5080);
  std::normal_distribution<float> normal(0.f, 2.f);
  std::ranges::generate(f.logits, [&] { return normal(rng); });
  for (int r = 0; r < width - 1; ++r) {
    for (int j = 0; j < kSelectorTopK; ++j)
      f.draft.q_probs[r * kSelectorTopK + j] = float(kSelectorTopK - j);
    f.logits[r * f.vocab + r] = 24.f;
  }
  if (mode == SPEC_SAMPLE_STOCHASTIC_FILTERED) {
    f.in.top_k = 100;
    f.in.top_p = .94f;
  }
  f.Upload();
  Require(f.Compare() == width - 1, "real vocab bonus");
  f.logits[(width - 2) * f.vocab + width - 2] = -1000.f;
  f.Upload();
  Require(f.Compare() == width - 2 && f.Compare() == width - 2, "real vocab residual replay");
}
}  // namespace

int main() {
  try {
    Full();
    Filtered();
    NucleusTail();
    for (int width : {4, 6, 8}) {
      Greedy(width);
      RealVocab(width, SPEC_SAMPLE_STOCHASTIC_FULL);
      RealVocab(width, SPEC_SAMPLE_STOCHASTIC_FILTERED);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>

#include "common/cuda_check.h"
#include "common/json_output.h"
#include "engine_api.h"
#include "loader/arenas.h"
#include "loader/model_constants.h"
#include "loader/safetensors.h"
#include "loader/scratch_workspace.h"
#include "model/inkling2_forward.h"
#include "runtime/config.h"

namespace {
using namespace flint;
using namespace std::chrono_literals;
void Require(bool ok, const std::string& why) {
  if (!ok) throw std::runtime_error(why);
}
using Engine = std::unique_ptr<EngineHandle, decltype(&engine_destroy)>;
Engine Create(const std::string& config) {
  char* message = nullptr;
  Engine e(engine_create(config.c_str(), &message), engine_destroy);
  std::unique_ptr<char, decltype(&engine_string_free)> diagnostic(message, engine_string_free);
  Require(bool(e), message ? message : "engine_create");
  return e;
}
struct Result {
  std::vector<int32_t> ids;
  std::vector<size_t> batches;
  EngineTelemetry t{};
  EngineStatus status{};
  EngineStopReason stop_reason{};
  std::string error;
};
struct Capture {
  std::promise<Result> done;
  Result result;
  EngineHandle* engine;
  unsigned cancel_at;
  static void Callback(const EngineEvent* event, void* data) {
    auto& c = *static_cast<Capture*>(data);
    if (event->kind == ENGINE_EVENT_TOKENS) {
      c.result.batches.push_back(event->n_tokens);
      c.result.ids.insert(c.result.ids.end(), event->token_ids, event->token_ids + event->n_tokens);
      if (c.result.ids.size() == c.cancel_at) engine_cancel(c.engine, event->req_id);
    } else {
      c.result.t = event->telemetry;
      c.result.status = event->status;
      c.result.stop_reason = event->stop_reason;
      if (event->error_message) c.result.error = event->error_message;
      // Transfer all callback-owned state before waking the submitting thread.
      auto done = std::move(c.done);
      done.set_value(std::move(c.result));
    }
  }
};
Result Run(EngineHandle* e, const std::vector<int32_t>& prompt, RequestParams p,
           unsigned cancel_at = 0, std::chrono::milliseconds cancel_after = 0ms) {
  Capture c{.engine = e, .cancel_at = cancel_at};
  auto result = c.done.get_future();
  auto id = engine_submit(e, &p, prompt.data(), prompt.size(), nullptr, 0, Capture::Callback, &c);
  Require(id, "submission failed");
  if (cancel_after.count()) {
    std::this_thread::sleep_for(cancel_after);
    engine_cancel(e, id);
    auto timely = result.wait_for(5s) == std::future_status::ready;
    result.wait();
    Require(timely, "cancellation timeout");
  }
  return result.get();
}
void Success(const Result& r) { Require(r.status == ENGINE_OK, r.error); }
template <class T = uint64_t>
T Counter(EngineHandle* e, const char* key);
void ConfigSmoke() {
  auto defaults = RuntimeConfig::Parse("{}");
  Require(defaults.model_dir.empty() && !defaults.max_context &&
              defaults.prefill_chunk_size == 256 && defaults.spec_verify_len == 8 &&
              defaults.use_speculative && defaults.use_cuda_graph &&
              defaults.prefix_cache_slots == 3 && defaults.scratch_bytes == (450ull << 20) &&
              defaults.vram_reserve_bytes == (512ull << 20),
          "config defaults");
  for (const char* bad :
       {"", "null", "[]", "{}x", "{\"model_dir\":true}", "{\"max_context\":-1}",
        "{\"max_context\":4294967296}", "{\"prefix_cache_slots\":1.2}",
        "{\"vram_reserve_bytes\":\"0\"}", "{\"use_cuda_graph\":1}", "{\"prefill_chunk_size\":0}",
        "{\"spec_verify_len\":0}", "{\"spec_verify_len\":1}", "{\"spec_verify_len\":9}",
        "{\"prefill_chunk_size\":2049}", "{\"scratch_bytes\":67108863}"}) {
    char* error = nullptr;
    Engine unexpected(engine_create(bad, &error), engine_destroy);
    std::unique_ptr<char, decltype(&engine_string_free)> diagnostic(error, engine_string_free);
    Require(!unexpected && error && *error, "invalid config accepted");
  }
  for (unsigned width : {4, 6, 8})
    Require(
        RuntimeConfig::Parse(Encode(Object{{"spec_verify_len", width}}).c_str()).spec_verify_len ==
            width,
        "valid verify width");
  auto e = Create("{}");
  auto r = Run(e.get(), {1}, {.max_tokens = 1});
  Success(r);
  Require(r.ids == std::vector<int32_t>{42}, "smoke token");
  Require(
      Counter<bool>(e.get(), "use_speculative") && !Counter<bool>(e.get(), "speculative_active"),
      "smoke AR fallback");
  for (const char* key : {"prefix_cache_slots", "prefix_cache_used_slots", "prefix_cache_hits",
                          "prefix_cache_misses", "prefix_cache_saved_tokens"})
    Require(Counter(e.get(), key) == 0, key);
  for (auto p : {RequestParams{.temperature = NAN}, RequestParams{.temperature = -1},
                 RequestParams{.top_p = NAN}, RequestParams{.top_p = 1.1f},
                 RequestParams{.top_k = -1}, RequestParams{.n_stop_token_ids = 1}}) {
    Require(Run(e.get(), {1}, p).status == ENGINE_ERR_INVALID_INPUT,
            "invalid sampling parameters accepted");
  }
}
std::vector<int32_t> Ids(const SafetensorsFile& f, const char* key) {
  auto* entry = f.Find(key);
  Require(entry && entry->dtype == TensorDType::kI32 && entry->nbytes % 4 == 0, key);
  std::vector<int32_t> ids(entry->nbytes / 4);
  std::memcpy(ids.data(), f.Bytes(*entry).data(), entry->nbytes);
  return ids;
}
size_t FreeBytes() {
  size_t available, total;
  CheckCuda(cudaDeviceSynchronize());
  CheckCuda(cudaMemGetInfo(&available, &total));
  return available;
}
template <class T>
T Counter(EngineHandle* e, const char* key) {
  std::unique_ptr<char, decltype(&engine_string_free)> stats(engine_stats_json(e),
                                                             engine_string_free);
  Require(bool(stats), "stats unavailable");
  auto root = json::Parser(stats.get()).Parse();
  if constexpr (std::is_same_v<T, bool>)
    return root.at(key).as<bool>();
  else
    return root.at(key).number<T>();
}
void SpecBoundaries(EngineHandle* e, const std::vector<int32_t>& prompt, const Result& reference,
                    int32_t eos_id) {
  const auto& gold = reference.ids;
  // Exercise limits on both sides of the checkpoint's actual first round.
  size_t first = reference.batches.at(1);
  for (unsigned limit : {1u, 2u, static_cast<unsigned>(1 + first)}) {
    auto r = Run(e, prompt, {.max_tokens = limit, .ignore_eos = 1});
    Success(r);
    Require(r.ids == std::vector<int32_t>(gold.begin(), gold.begin() + limit) &&
                r.t.decode_tokens == limit && r.t.spec_accepted == (limit > 2 ? first - 1 : 0) &&
                r.t.decode_rounds == unsigned(limit > 1) && r.stop_reason == ENGINE_STOP_MAX_TOKENS,
            "budget within speculative batch");
  }
  RequestParams p{.max_tokens = uint32_t(gold.size()),
                  .stop_token_ids = &gold[2],
                  .n_stop_token_ids = 1,
                  .ignore_eos = 1};
  auto stopped = Run(e, prompt, p);
  Success(stopped);
  Require(stopped.ids == std::vector<int32_t>(gold.begin(), gold.begin() + 2) &&
              stopped.t.spec_accepted == unsigned(first > 1) &&
              stopped.t.decode_rounds == (first > 1 ? 1u : 2u) &&
              stopped.stop_reason == ENGINE_STOP_STOP_TOKEN,
          "stop inside accepted drafts");
  p.stop_token_ids = &gold[1];
  auto empty_round = Run(e, prompt, p);
  Success(empty_round);
  Require(empty_round.ids.size() == 1 && empty_round.t.decode_rounds == 1 &&
              empty_round.t.spec_accepted == 0 && empty_round.t.spec_accept_hist[0] == 1,
          "first verify prediction stopped");
  p.stop_token_ids = nullptr;
  p.n_stop_token_ids = 0;
  p.ignore_eos = 0;
  auto eos = Run(e, prompt, p);
  Success(eos);
  Require(eos.ids == std::vector<int32_t>(gold.begin(), std::ranges::find(gold, eos_id)) &&
              eos.stop_reason == ENGINE_STOP_EOS,
          "spec EOS");
  p.ignore_eos = 1;
  auto cancelled = Run(e, prompt, p, 1 + first);
  Require(cancelled.status == ENGINE_ERR_CANCELLED &&
              cancelled.batches == std::vector<size_t>{1, first} &&
              cancelled.t.spec_accepted == first - 1 && cancelled.t.decode_rounds == 1,
          "cancel on batched callback");
}
void PrefixTests(const std::filesystem::path& model_dir, const std::filesystem::path& draft_dir,
                 const std::vector<int32_t>& math, const std::vector<int32_t>& gold) {
  using Tokens = std::vector<int32_t>;
  auto create = [&](unsigned slots) {
    return Create(Encode(Object{{"model_dir", model_dir.string()},
                                {"draft_model_dir", draft_dir.string()},
                                {"max_context", 2048},
                                {"prefill_chunk_size", 8},
                                {"prefix_cache_slots", slots}}));
  };
  RequestParams greedy{.max_tokens = 16, .ignore_eos = 1};
  RequestParams full{.max_tokens = uint32_t(gold.size()), .ignore_eos = 1};
  RequestParams stochastic{.max_tokens = 16,
                           .temperature = 0.7f,
                           .top_p = 0.9f,
                           .top_k = 40,
                           .seed = 12345,
                           .ignore_eos = 1};
  Tokens stripped = math;
  stripped.insert(stripped.end(), {42, 89, 17});
  Require(stripped[math.size()] != gold.front(), "suffix must diverge from decoded reasoning");
  Tokens shared(35, 67);
  auto branch = [&](int id) {
    Tokens prompt = shared;
    prompt.insert(prompt.end(), {id, 31, 42});
    return prompt;
  };
  auto extend = [&](const Tokens& decoded) {
    Tokens prompt = stripped;
    prompt.insert(prompt.end(), decoded.begin(), decoded.end() - 1);
    prompt.insert(prompt.end(), {78, 91});
    return prompt;
  };
  Result cold_math, cold_stripped, cold_tail, cold_branch, cold_seed1, cold_seed2;
  // Warm the cache-disabled executor too. Its sampled and tail-prefill
  // kernels are loaded lazily on the first pass through these shapes.
  {
    auto warm = create(0);
    Success(Run(warm.get(), stripped, stochastic));
    Success(Run(warm.get(), branch(73), greedy));
  }
  size_t before = FreeBytes();
  {
    auto cold = create(0);
    auto run = [&](const Tokens& prompt, RequestParams params) {
      auto r = Run(cold.get(), prompt, params);
      Success(r);
      Require(!r.t.cached_prefix_tokens && r.t.prefill_tokens == prompt.size() &&
                  r.ids.size() == params.max_tokens,
              "disabled cache telemetry");
      return r;
    };
    run(math, greedy);  // Warm the model and graph before timing the cold reference.
    cold_math = run(math, full);
    cold_stripped = run(stripped, greedy);
    cold_tail = run(extend(cold_stripped.ids), greedy);
    cold_branch = run(branch(73), greedy);
    cold_seed1 = run(math, stochastic);
    ++stochastic.seed;
    cold_seed2 = run(math, stochastic);
    --stochastic.seed;
    Require(cold_math.ids == gold && cold_seed1.ids != cold_seed2.ids, "cold references");
  }
  {
    auto cache = create(3);
    auto* e = cache.get();
    Require(Counter<bool>(e, "speculative_active") && Counter<bool>(e, "use_speculative"),
            "DFlash 2 active");
    uint64_t total_prompt = 0, total_decode = 0, total_rounds = 0, total_accepted = 0;
    uint64_t hits = 0, misses = 0, saved = 0;
    auto run = [&](const Tokens& prompt, RequestParams params, size_t expected) {
      auto r = Run(e, prompt, params);
      Success(r);
      Require(r.t.prefill_tokens == prompt.size() && r.t.cached_prefix_tokens == expected &&
                  r.ids.size() == params.max_tokens,
              "prefix telemetry or output length");
      if (expected)
        ++hits;
      else
        ++misses;
      saved += expected;
      total_prompt += r.t.prefill_tokens;
      total_decode += r.t.decode_tokens;
      total_rounds += r.t.decode_rounds;
      total_accepted += r.t.spec_accepted;
      return r;
    };
    auto parity = [&](const char* name, const Result& r, const Result& reference) {
      Require(r.ids == reference.ids, std::string(name) + " cold token mismatch");
      std::cout << name << " cached_prefix_tokens=" << r.t.cached_prefix_tokens
                << " parity=" << r.ids.size() << "/" << reference.ids.size() << std::endl;
    };
    Require(math.size() == 27 && gold.size() == 49, "short_math fixture shape");
    parity("first", run(math, full, 0), cold_math);
    size_t resident = FreeBytes();
    auto chunks = Counter(e, "total_prefill_chunks");
    auto repeat = run(math, full, math.size());
    Require(chunks == (math.size() + 7) / 8 && Counter(e, "total_prefill_chunks") == chunks,
            "exact hit skips every target prefill chunk");
    parity("exact", repeat, cold_math);
    Require(repeat.t.spec_accepted > 0 && repeat.t.spec_accepted == cold_math.t.spec_accepted &&
                repeat.t.decode_rounds == cold_math.t.decode_rounds &&
                repeat.t.decode_rounds < repeat.t.decode_tokens,
            "exact speculative rounds");
    std::cout << "DFlash 2 short_math parity=" << repeat.ids.size() << "/" << gold.size()
              << " accepted=" << repeat.t.spec_accepted << " rounds=" << repeat.t.decode_rounds
              << " tokens=" << repeat.t.decode_tokens << std::endl;
    std::cout << "short_math cold prefill_ms=" << cold_math.t.prefill_ms
              << " exact prefill_ms=" << repeat.t.prefill_ms << std::endl;
    Require(repeat.t.prefill_ms < cold_math.t.prefill_ms / 2, "exact prefill speedup");
    parity("stripped L2", run(stripped, greedy, math.size()), cold_stripped);
    auto tail = extend(cold_stripped.ids);
    parity("live L1", run(tail, greedy, stripped.size() + greedy.max_tokens - 1), cold_tail);
    parity("seed1", run(math, stochastic, math.size()), cold_seed1);
    ++stochastic.seed;
    parity("seed2", run(math, stochastic, math.size()), cold_seed2);
    run(branch(71), greedy, 0);
    run(branch(72), greedy, 0);
    Require(Counter(e, "prefix_cache_used_slots") == 2, "pin and end checkpoints");
    parity("adaptive LCP", run(branch(73), greedy, shared.size()), cold_branch);
    Require(Counter(e, "prefix_cache_slots") == 3 && Counter(e, "prefix_cache_hits") == hits &&
                Counter(e, "prefix_cache_misses") == misses &&
                Counter(e, "prefix_cache_saved_tokens") == saved &&
                Counter(e, "prefix_cache_used_slots") == 2,
            "prefix cache stats");
    std::cout << "prefix hits=" << hits << " misses=" << misses << " saved_tokens=" << saved
              << " used_slots=2" << std::endl;
    Require(Counter(e, "total_prefill_tokens") == total_prompt &&
                Counter(e, "total_cached_prefix_tokens") == saved &&
                Counter(e, "total_decode_tokens") == total_decode &&
                Counter(e, "total_decode_rounds") == total_rounds &&
                Counter(e, "total_spec_accepted") == total_accepted,
            "lifetime telemetry totals");
    SpecBoundaries(e, math, cold_math, LoadModelConfig(model_dir / "config.json").eos_token_id);
    auto after = FreeBytes();
    std::cout << "resident VRAM delta=" << int64_t(resident) - int64_t(after) << " B" << std::endl;
    Require(resident == after, "resident prefix cache VRAM drift");
  }
  auto after = FreeBytes();
  std::cout << "prefix destroy VRAM delta=" << int64_t(before) - int64_t(after) << " B"
            << std::endl;
  Require(before == after, "prefix cache VRAM drift after destroy");
}
void LongDraftPrefill(const std::filesystem::path& model_dir,
                      const std::filesystem::path& draft_dir, const std::vector<int32_t>& math) {
  constexpr unsigned length = 2304, pin = 2176, chunk = 128;
  auto create = [&](bool speculative) {
    return Create(Encode(Object{{"model_dir", model_dir.string()},
                                {"draft_model_dir", draft_dir.string()},
                                {"use_speculative", speculative},
                                {"prefix_cache_slots", speculative ? 3 : 0},
                                {"max_context", length + 32},
                                {"prefill_chunk_size", chunk}}));
  };
  std::vector<int32_t> prompt(pin + math.size());
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = math[i % math.size()];
  std::copy(math.begin(), math.end(), prompt.end() - math.size());
  RequestParams greedy{.max_tokens = 16, .ignore_eos = 1};
  auto ar = create(false);
  const auto reference = Run(ar.get(), prompt, greedy);
  Success(reference);
  ar.reset();
  auto spec = create(true);
  auto cold = Run(spec.get(), prompt, greedy);
  Success(cold);
  Require(cold.ids == reference.ids && !cold.t.cached_prefix_tokens,
          "long draft prefill AR parity");
  // The longer branch pins the LCP. Restoring it for the shorter suffix needs KV
  // older than the window retained at the end of the longer branch.
  auto branch = prompt;
  branch.resize(length, 63);
  ++branch[pin];
  std::copy(math.begin(), math.end(), branch.end() - math.size());
  auto pinned = Run(spec.get(), branch, greedy);
  Success(pinned);
  Require(!pinned.t.cached_prefix_tokens, "long branch should pin a new intermediate snapshot");
  auto hit = Run(spec.get(), prompt, greedy);
  Success(hit);
  Require(
      hit.t.cached_prefix_tokens == pin && hit.ids == reference.ids && hit.batches == cold.batches,
      "long pin restore changed target or draft output");

  // Start at a saved, nonzero boundary and cancel after some skipped projections.
  auto prefix = std::vector<int32_t>(prompt.begin(), prompt.begin() + chunk);
  ++prefix.front();
  Success(Run(spec.get(), prefix, {.max_tokens = 1}));
  std::copy(prefix.begin(), prefix.end(), branch.begin());
  auto delay = std::chrono::milliseconds(std::max<int>(1, cold.t.prefill_ms / 5));
  auto cancelled = Run(spec.get(), branch, greedy, 0, delay);
  Require(cancelled.status == ENGINE_ERR_CANCELLED && cancelled.ids.empty() &&
              cancelled.t.cached_prefix_tokens == chunk && cancelled.t.prefill_tokens > chunk &&
              cancelled.t.prefill_tokens < length,
          "cancel within an incomplete draft window");
  auto retry = Run(spec.get(), branch, greedy);
  Success(retry);
  Require(retry.t.cached_prefix_tokens == chunk, "interrupted tail reused");
  spec.reset();
  ar = create(false);
  auto retry_ar = Run(ar.get(), branch, greedy);
  Success(retry_ar);
  Require(retry.ids == retry_ar.ids, "post-cancel AR parity");
}
// Before the verify fix short_sky[35] favored 149881 at T=1 (+0.33836365)
// and 149910 at T=8 (+0.13246155). On short_code[72] the T=8 margin was zero.
// Teacher-force the emitted history through ordinary eight-row forward. This
// oracle neither drafts proposals nor commits a speculative prefix.
void VerifyBatches(const ModelWeightsView& weights, uint32_t max_context, const char* name,
                   const std::vector<int>& prompt, const Result& ar, const Result& spec) {
  constexpr unsigned width = 8, vocab = model::kUnpaddedVocabSize;
  auto kv = KvStateArena::Allocate(weights.config, max_context + width - 1, true);
  auto scratch = ScratchArena::Allocate();
  InklingRunner target(weights, kv.view(), scratch.view(), nullptr);
  auto io = ScratchArena::Allocate(width * vocab * sizeof(float) + (1 << 20));
  ScratchWorkspace ws(io.view());
  auto input = ws.Take<int>(width), next = ws.Take<int>(width);
  auto hidden = ws.Take<uint16_t>(width * model::kHiddenSize);
  auto logits = ws.Take<float>(width * vocab);
  ForwardRunSpec s{.end_visit = model::kNumLogicalVisits,
                   .run_embed = 1,
                   .run_lm_head = 1,
                   .d_input_ids = input,
                   .d_hidden_out = hidden,
                   .d_logits_out = logits,
                   .d_next_token = next};
  auto upload = [&](const int* src, unsigned count) {
    CheckCuda(
        cudaMemcpyAsync(input, src, count * sizeof(int), cudaMemcpyHostToDevice, target.stream()));
  };
  auto prefill = [&] {
    int anchor = -1;
    for (s.pos_offset = 0; s.pos_offset < prompt.size(); s.pos_offset += s.seq_len) {
      s.seq_len = std::min<size_t>(width, prompt.size() - s.pos_offset);
      s.reset_kv = s.pos_offset == 0;
      upload(prompt.data() + s.pos_offset, s.seq_len);
      anchor = target.Run(s, {}).next_token_id;
    }
    s.reset_kv = 0;
    return anchor;
  };
  size_t mismatch = std::ranges::mismatch(spec.ids, ar.ids).in1 - spec.ids.begin();
  std::cout << name << " DFlash 2 vs AR prefix=" << mismatch << "/" << ar.ids.size() << std::endl;
  auto report = [&](const char* mode, unsigned row) {
    std::vector<float> scores(vocab);
    CheckCuda(cudaMemcpy(scores.data(), logits + row * vocab, vocab * sizeof(float),
                         cudaMemcpyDeviceToHost));
    auto best = std::ranges::max_element(scores);
    auto token = best - scores.begin();
    float score = *best;
    *best = -INFINITY;
    auto second = std::ranges::max_element(scores);
    std::cout << name << " index=" << mismatch << " " << mode << " pos=" << s.pos_offset
              << " row=" << row << " top2=" << token << ":" << std::setprecision(10) << score << ","
              << second - scores.begin() << ":" << *second << " margin=" << score - *second
              << std::endl;
  };
  if (mismatch < ar.ids.size()) {
    Require(prefill() == ar.ids.front() && mismatch > 0, "AR prefill");
    s.is_decode = 1;
    s.seq_len = 1;
    for (size_t j = 1; j <= mismatch; ++j) {
      s.pos_offset = prompt.size() + j - 1;
      upload(&ar.ids[j - 1], 1);
      Require(target.Run(s, {.return_logits = uint8_t(j == mismatch)}).next_token_id == ar.ids[j],
              "AR replay");
    }
    report("T=1", 0);
    s.is_decode = 0;
  }
  Require(prefill() == spec.ids.front(), "speculative prefill");
  s.seq_len = width;
  s.all_token_logits = 1;
  for (size_t j = 1; j < spec.ids.size(); j += width) {
    s.pos_offset = prompt.size() + j - 1;
    size_t rows = std::min<size_t>(width, spec.ids.size() - j);
    std::array<int, width> block{}, verified{};
    std::copy_n(spec.ids.begin() + j - 1, rows, block.begin());
    upload(block.data(), width);
    bool diverged = j <= mismatch && mismatch < j + rows;
    bool compare = diverged || j <= 1 + width;
    std::vector<uint32_t> graph;
    if (compare) {
      target.RunVerify(s, {.use_cuda_graph = 1, .return_logits = 1});
      graph.resize(width * vocab);
      CheckCuda(cudaMemcpy(graph.data(), logits, graph.size() * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));
      if (diverged) report("T=8 graph", mismatch - j);
      target.CommitVerify(s.pos_offset, width, 0);
    }
    target.Run(s, {.return_logits = uint8_t(compare)});
    if (compare) {
      std::vector<uint32_t> eager(graph.size());
      CheckCuda(cudaMemcpy(eager.data(), logits, eager.size() * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));
      Require(eager == graph, "T=8 graph verify vs ordinary forward after rollback");
      if (diverged) report("T=8 eager", mismatch - j);
    }
    CheckCuda(cudaMemcpy(verified.data(), next, sizeof(verified), cudaMemcpyDeviceToHost));
    for (size_t k = 0; k < rows; ++k)
      Require(
          verified[k] == spec.ids[j + k],
          std::string(name) + " speculative vs teacher-forced target at " + std::to_string(j + k));
  }
  std::cout << name << " teacher-forced T=8=" << spec.ids.size() << "/" << spec.ids.size()
            << std::endl;
}
void ModelTests(const std::filesystem::path& model_dir, const std::filesystem::path& draft_dir,
                const std::filesystem::path& golden_dir) {
  constexpr unsigned kMaxContext = 521;
  auto config = Encode(Object{{"model_dir", model_dir.string()},
                              {"draft_model_dir", draft_dir.string()},
                              {"max_context", kMaxContext},
                              {"prefill_chunk_size", 8},
                              {"use_speculative", false}});
  auto file = [&](std::string name) {
    return SafetensorsFile::Open(golden_dir / ("e2e_short_" + name + ".safetensors"));
  };
  auto prompt = Ids(file("math"), "input_ids");
  RequestParams p{.max_tokens = 4, .ignore_eos = 1};
  for (bool speculative : {true, false}) {
    p.temperature = 0;
    auto warm = Create(Encode(Object{{"model_dir", model_dir.string()},
                                     {"draft_model_dir", speculative ? draft_dir.string() : ""},
                                     {"prefill_chunk_size", 8}}));
    Require(
        Counter<bool>(warm.get(), "use_speculative") &&
            Counter<bool>(warm.get(), "speculative_active") == (speculative && !draft_dir.empty()),
        "automatic draft selection");
    Require(Counter(warm.get(), "max_context") >= prompt.size(), "auto-sized context");
    Success(Run(warm.get(), prompt, p));
    p.temperature = 0.8f;
    Success(Run(warm.get(), prompt, p));
  }
  size_t before = FreeBytes();
  auto engine = Create(config);
  auto* e = engine.get();
  Require(!Counter<bool>(e, "speculative_active") && !Counter<bool>(e, "use_speculative"),
          "explicit AR");
  std::vector<std::pair<std::vector<int32_t>, Result>> ar_runs;
  unsigned total = 0;
  p.temperature = 0;
  for (auto name : {"math", "sky", "code"}) {
    auto f = file(name);
    auto gold = Ids(f, "generated_ids");
    p.max_tokens = gold.size();
    auto r = Run(e, Ids(f, "input_ids"), p);
    Success(r);
    auto hits =
        std::mismatch(gold.begin(), gold.end(), r.ids.begin(), r.ids.end()).first - gold.begin();
    std::cout << name << " chunk=8 greedy=" << hits << "/" << gold.size() << std::endl;
    Require(r.ids == gold && r.stop_reason == ENGINE_STOP_MAX_TOKENS, "greedy mismatch");
    total += hits;
    ar_runs.emplace_back(Ids(f, "input_ids"), std::move(r));
  }
  auto graphs = Counter(e, "graph_captures"), launches = Counter(e, "graph_launches");
  std::cout << "greedy=" << total << "/288 graph_captures=" << graphs
            << " graph_launches=" << launches << std::endl;
  Require(total == 288 && graphs == 1 && launches == 285, "graph reuse");
  p.max_tokens = 16;
  p.temperature = 0.8f;
  p.top_k = 40;
  p.top_p = 0.9f;
  p.seed = 12345;
  auto a = Run(e, prompt, p), b = Run(e, prompt, p);
  ++p.seed;
  auto c = Run(e, prompt, p);
  Success(a);
  Success(b);
  Success(c);
  Require(a.ids.size() == 16 && a.ids == b.ids && a.ids != c.ids, "stochastic reproducibility");
  auto gold = Ids(file("math"), "generated_ids");
  p = {.max_tokens = uint32_t(gold.size()),
       .stop_token_ids = &gold.front(),
       .n_stop_token_ids = 1,
       .ignore_eos = 1};
  auto stop = Run(e, prompt, p);
  Success(stop);
  Require(
      stop.ids.empty() && stop.t.decode_rounds == 0 && stop.stop_reason == ENGINE_STOP_STOP_TOKEN,
      "stop on first token");
  auto eos = LoadModelConfig(model_dir / "config.json").eos_token_id;
  auto at = std::find(gold.begin(), gold.end(), int(eos));
  Require(at != gold.end(), "missing EOS fixture");
  p.stop_token_ids = nullptr;
  p.n_stop_token_ids = 0;
  p.ignore_eos = 0;
  auto stopped = Run(e, prompt, p);
  Success(stopped);
  Require(stopped.ids == std::vector<int32_t>(gold.begin(), at) &&
              stopped.stop_reason == ENGINE_STOP_EOS,
          "model EOS stop");
  p.stop_token_ids = &*at;
  p.n_stop_token_ids = 1;
  auto both = Run(e, prompt, p);
  Success(both);
  Require(both.ids == stopped.ids && both.stop_reason == ENGINE_STOP_EOS,
          "model EOS takes priority over caller stop");
  p.ignore_eos = 1;
  auto explicit_stop = Run(e, prompt, p);
  Success(explicit_stop);
  Require(explicit_stop.ids == stopped.ids && explicit_stop.stop_reason == ENGINE_STOP_STOP_TOKEN,
          "caller stop remains active when model EOS is ignored");
  std::vector<int32_t> long_prompt(kMaxContext - 1, 42);
  p = {.max_tokens = 1, .ignore_eos = 1};
  auto long_run = Run(e, long_prompt, p);
  Success(long_run);
  Require(long_run.ids.size() == 1 && long_run.t.prefill_tokens == 520, "long chunked prefill");
  std::cout << "long prefill tokens=" << long_run.t.prefill_tokens
            << " ms=" << long_run.t.prefill_ms << std::endl;
  std::vector<int32_t> prefix(48, 43);
  Success(Run(e, prefix, p));
  std::copy(prefix.begin(), prefix.end(), long_prompt.begin());
  auto cancelled = Run(e, long_prompt, p, 0,
                       std::chrono::milliseconds(std::max<int>(1, long_run.t.prefill_ms / 4)));
  Require(cancelled.status == ENGINE_ERR_CANCELLED && cancelled.stop_reason == ENGINE_STOP_NONE &&
              cancelled.t.cached_prefix_tokens == prefix.size() &&
              cancelled.t.prefill_tokens > prefix.size() &&
              cancelled.t.prefill_tokens < long_prompt.size() && cancelled.ids.empty(),
          "mid-prefill cancellation");
  auto resumed = Run(e, long_prompt, p);
  Success(resumed);
  Require(resumed.t.cached_prefix_tokens == cancelled.t.prefill_tokens &&
              resumed.t.prefill_tokens == long_prompt.size(),
          "resume cancelled prefill at live tail");
  p.max_tokens = 8;
  auto limited = Run(e, long_prompt, p);
  Success(limited);
  Require(limited.ids.size() == 2 && limited.stop_reason == ENGINE_STOP_CONTEXT_LIMIT,
          "context limit after one decode step");
  p.max_tokens = 2;
  auto exact_limit = Run(e, long_prompt, p);
  Success(exact_limit);
  Require(exact_limit.ids == limited.ids && exact_limit.stop_reason == ENGINE_STOP_MAX_TOKENS,
          "max tokens takes priority at context limit");
  p.max_tokens = 8;
  auto interrupted = Run(e, prompt, p, 2);
  Require(interrupted.status == ENGINE_ERR_CANCELLED && interrupted.ids.size() == 2 &&
              interrupted.t.decode_rounds == 1,
          "decode cancellation");
  Require(Run(e, {-1}, p).status == ENGINE_ERR_INVALID_INPUT &&
              Run(e, {201024}, p).status == ENGINE_ERR_INVALID_INPUT,
          "invalid prompt token");
  auto overflow = Run(e, std::vector<int32_t>(kMaxContext + 1), p);
  Require(
      overflow.status == ENGINE_ERR_CONTEXT_OVERFLOW && overflow.stop_reason == ENGINE_STOP_NONE,
      "context overflow");
  auto healthy = Run(e, prompt, p);
  Success(healthy);
  Require(healthy.ids == std::vector<int32_t>(gold.begin(), gold.begin() + 8),
          "post-cancel KV reset");
  engine.reset();
  size_t after = FreeBytes();
  std::cout << "VRAM delta=" << int64_t(before) - int64_t(after) << " B" << std::endl;
  Require(before == after, "VRAM drift after destroy");
  if (draft_dir.empty()) return;
  std::vector<Result> spec_runs;
  {
    auto spec = Create(Encode(Object{{"model_dir", model_dir.string()},
                                     {"draft_model_dir", draft_dir.string()},
                                     {"max_context", kMaxContext},
                                     {"prefill_chunk_size", 8},
                                     {"use_speculative", true}}));
    for (const auto& [ids, ar] : ar_runs) {
      auto r = Run(spec.get(), ids, {.max_tokens = ar.t.decode_tokens, .ignore_eos = 1});
      Success(r);
      Require(r.ids.size() == ar.ids.size() && r.stop_reason == ENGINE_STOP_MAX_TOKENS &&
                  r.t.spec_accepted > 0 && r.t.decode_rounds < r.t.decode_tokens,
              "greedy speculative telemetry");
      spec_runs.push_back(std::move(r));
    }
  }
  for (unsigned width : {4, 6}) {
    auto spec = Create(Encode(Object{{"model_dir", model_dir.string()},
                                     {"draft_model_dir", draft_dir.string()},
                                     {"max_context", kMaxContext},
                                     {"prefill_chunk_size", 8},
                                     {"spec_verify_len", width}}));
    auto r = Run(spec.get(), prompt, {.max_tokens = uint32_t(gold.size()), .ignore_eos = 1});
    Success(r);
    Require(r.ids == ar_runs.front().second.ids && Counter(spec.get(), "spec_verify_len") == width,
            "configured verify width short_math AR parity");
    const auto hist = std::span(r.t.spec_accept_hist);
    uint32_t accepted = 0;
    for (unsigned k = 0; k < hist.size(); ++k) {
      accepted += k * hist[k];
      Require(k < width || !hist[k], "accepted beyond verify width");
    }
    // With ignore_eos and sufficient context, only a final one-token budget uses AR.
    const bool final_ar = r.batches.back() == 1;
    Require(accepted > 0 && accepted == r.t.spec_accepted &&
                std::accumulate(hist.begin(), hist.end(), 0u) + final_ar == r.t.decode_rounds,
            "speculative acceptance histogram totals");
  }
  LongDraftPrefill(model_dir, draft_dir, prompt);
  PrefixTests(model_dir, draft_dir, prompt, gold);
  for (unsigned room : {1, 5}) {
    auto edge = Create(Encode(Object{{"model_dir", model_dir.string()},
                                     {"draft_model_dir", draft_dir.string()},
                                     {"max_context", prompt.size() + room},
                                     {"prefill_chunk_size", 3}}));
    auto r = Run(edge.get(), prompt, {.max_tokens = 16, .ignore_eos = 1});
    Success(r);
    Require(r.ids == std::vector<int32_t>(gold.begin(), gold.begin() + room + 1) &&
                r.stop_reason == ENGINE_STOP_CONTEXT_LIMIT,
            "bounded verify at context limit");
  }
  {
    auto weights = WeightsArena::Load(model_dir, nullptr, draft_dir);
    const char* names[] = {"short_math", "short_sky", "short_code"};
    for (size_t i = 0; i < ar_runs.size(); ++i)
      VerifyBatches(weights.view(), kMaxContext, names[i], ar_runs[i].first, ar_runs[i].second,
                    spec_runs[i]);
  }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 1)
      ConfigSmoke();
    else {
      Require((argc == 5 || (argc == 7 && std::string_view(argv[5]) == "--draft-dir")) &&
                  std::string_view(argv[1]) == "--model-dir" &&
                  std::string_view(argv[3]) == "--golden-dir",
              "expected --model-dir MODEL --golden-dir GOLDEN");
      ModelTests(argv[2], argc == 7 ? argv[6] : "", argv[4]);
    }
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}

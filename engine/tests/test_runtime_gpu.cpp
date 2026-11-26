#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <thread>

#include "common/cuda_check.h"
#include "common/json_output.h"
#include "engine_api.h"
#include "loader/safetensors.h"
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
  Require(
      defaults.model_dir.empty() && !defaults.max_context && defaults.prefill_chunk_size == 256 &&
          defaults.use_speculative && defaults.use_cuda_graph && defaults.prefix_cache_slots == 3 &&
          defaults.scratch_bytes == (450ull << 20) && defaults.vram_reserve_bytes == (512ull << 20),
      "config defaults");
  for (const char* bad :
       {"", "null", "[]", "{}x", "{\"model_dir\":true}", "{\"max_context\":-1}",
        "{\"max_context\":4294967296}", "{\"prefix_cache_slots\":1.2}",
        "{\"vram_reserve_bytes\":\"0\"}", "{\"use_cuda_graph\":1}", "{\"prefill_chunk_size\":0}",
        "{\"prefill_chunk_size\":2049}", "{\"scratch_bytes\":67108863}"}) {
    char* error = nullptr;
    Engine unexpected(engine_create(bad, &error), engine_destroy);
    std::unique_ptr<char, decltype(&engine_string_free)> diagnostic(error, engine_string_free);
    Require(!unexpected && error && *error, "invalid config accepted");
  }
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
void SpecBoundaries(EngineHandle* e, const std::vector<int32_t>& prompt,
                    const std::vector<int32_t>& gold, int32_t eos_id) {
  // short_math starts with three accepted drafts followed by the target bonus.
  for (unsigned limit : {1, 3, 5}) {
    auto r = Run(e, prompt, {.max_tokens = limit, .ignore_eos = 1});
    Success(r);
    Require(r.ids == std::vector<int32_t>(gold.begin(), gold.begin() + limit) &&
                r.t.decode_tokens == limit && r.t.spec_accepted == std::min(limit - 1, 3u) &&
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
              stopped.t.spec_accepted == 1 && stopped.t.decode_rounds == 1 &&
              stopped.stop_reason == ENGINE_STOP_STOP_TOKEN,
          "stop inside accepted drafts");
  p.stop_token_ids = nullptr;
  p.n_stop_token_ids = 0;
  p.ignore_eos = 0;
  auto eos = Run(e, prompt, p);
  Success(eos);
  Require(eos.ids == std::vector<int32_t>(gold.begin(), std::ranges::find(gold, eos_id)) &&
              eos.stop_reason == ENGINE_STOP_EOS,
          "spec EOS");
  auto cancelled = Run(e, prompt, p, 5);
  Require(cancelled.status == ENGINE_ERR_CANCELLED &&
              cancelled.batches == std::vector<size_t>{1, 4} && cancelled.t.spec_accepted == 3 &&
              cancelled.t.decode_rounds == 1,
          "cancel on batched callback");
}
void PrefixTests(const std::filesystem::path& model_dir, const std::vector<int32_t>& math,
                 const std::vector<int32_t>& gold) {
  using Tokens = std::vector<int32_t>;
  auto create = [&](unsigned slots) {
    return Create(Encode(Object{{"model_dir", model_dir.string()},
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
            "DSpark active");
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
    std::cout << "DSpark short_math parity=" << repeat.ids.size() << "/" << gold.size()
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
    SpecBoundaries(e, math, gold, LoadModelConfig(model_dir / "config.json").eos_token_id);
    auto after = FreeBytes();
    std::cout << "resident VRAM delta=" << int64_t(resident) - int64_t(after) << " B" << std::endl;
    Require(resident == after, "resident prefix cache VRAM drift");
  }
  auto after = FreeBytes();
  std::cout << "prefix destroy VRAM delta=" << int64_t(before) - int64_t(after) << " B"
            << std::endl;
  Require(before == after, "prefix cache VRAM drift after destroy");
}
Engine CreateWithoutMtpFile(const std::filesystem::path& model_dir) {
  namespace fs = std::filesystem;
  auto path = (fs::temp_directory_path() / "flint-runtime-XXXXXX").string();
  Require(mkdtemp(path.data()) != nullptr, "create AR checkpoint directory");
  struct Cleanup {
    fs::path dir;
    ~Cleanup() {
      std::error_code ec;
      fs::remove_all(dir, ec);
    }
  } tmp{path};
  for (auto name : {"config.json", "model.safetensors"})
    fs::create_symlink(fs::absolute(model_dir / name), tmp.dir / name);
  return Create(Encode(Object{{"model_dir", path}, {"prefill_chunk_size", 8}}));
}
void ModelTests(const std::filesystem::path& model_dir, const std::filesystem::path& golden_dir) {
  constexpr unsigned kMaxContext = 521;
  auto config = Encode(Object{{"model_dir", model_dir.string()},
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
    auto warm =
        speculative
            ? Create(Encode(Object{{"model_dir", model_dir.string()}, {"prefill_chunk_size", 8}}))
            : CreateWithoutMtpFile(model_dir);
    Require(Counter<bool>(warm.get(), "use_speculative") &&
                Counter<bool>(warm.get(), "speculative_active") == speculative,
            "automatic MTP selection");
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
  {
    auto spec = Create(Encode(Object{{"model_dir", model_dir.string()},
                                     {"max_context", kMaxContext},
                                     {"prefill_chunk_size", 8},
                                     {"use_speculative", true}}));
    for (const auto& [ids, ar] : ar_runs) {
      auto r = Run(spec.get(), ids, {.max_tokens = ar.t.decode_tokens, .ignore_eos = 1});
      Success(r);
      Require(r.ids == ar.ids && r.t.spec_accepted > 0 && r.t.decode_rounds < r.t.decode_tokens,
              "greedy speculative vs AR");
      std::cout << "DSpark vs AR " << r.ids.size() << "/" << ar.ids.size()
                << " rounds=" << r.t.decode_rounds << " accepted=" << r.t.spec_accepted
                << std::endl;
    }
  }
  PrefixTests(model_dir, prompt, gold);
  for (unsigned room : {1, 5}) {
    auto edge = Create(Encode(Object{{"model_dir", model_dir.string()},
                                     {"max_context", prompt.size() + room},
                                     {"prefill_chunk_size", 3}}));
    auto r = Run(edge.get(), prompt, {.max_tokens = 16, .ignore_eos = 1});
    Success(r);
    Require(r.ids == std::vector<int32_t>(gold.begin(), gold.begin() + room + 1) &&
                r.stop_reason == ENGINE_STOP_CONTEXT_LIMIT,
            "bounded verify at context limit");
  }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 1)
      ConfigSmoke();
    else {
      Require(argc == 5 && std::string_view(argv[1]) == "--model-dir" &&
                  std::string_view(argv[3]) == "--golden-dir",
              "expected --model-dir MODEL --golden-dir GOLDEN");
      ModelTests(argv[2], argv[4]);
    }
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}

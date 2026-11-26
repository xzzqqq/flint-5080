#include "runtime/engine_runtime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/cuda_check.h"
#include "common/json_output.h"
#include "loader/arenas.h"
#include "loader/kv_layout.h"
#include "loader/scratch_workspace.h"
#include "model/inkling2_forward.h"
#include "ops/stub_ops.h"
#include "runtime/config.h"
#include "runtime/prefix_cache.h"
#include "runtime/sampler.h"

namespace flint {
namespace {
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
uint64_t NewSeed() {
  static std::atomic<uint64_t> sequence{(uint64_t(std::random_device{}()) << 32) ^
                                        uint64_t(Clock::now().time_since_epoch().count())};
  uint64_t seed;
  do {
    seed = ++sequence;
  } while (!seed);
  return seed;
}
struct Request {
  uint64_t id{};
  RequestParams params{};
  std::vector<int32_t> prompt, stops;
  std::optional<int32_t> eos_token_id;
  TokenCallback callback;
  void* data;
  std::atomic<bool> cancelled{};
  EngineTelemetry telemetry{};
  EngineStatus status = ENGINE_OK;
  EngineStopReason stop_reason = ENGINE_STOP_NONE;
  const char* error{};
  void Fail(EngineStatus code, const char* reason) {
    status = code;
    error = reason;
  }
  bool Emit(int32_t token) {
    if (cancelled) return false;
    if (token == eos_token_id) {
      stop_reason = ENGINE_STOP_EOS;
      return false;
    }
    if (std::ranges::find(stops, token) != stops.end()) {
      stop_reason = ENGINE_STOP_STOP_TOKEN;
      return false;
    }
    ++telemetry.decode_tokens;
    EngineEvent event{.req_id = id,
                      .kind = ENGINE_EVENT_TOKENS,
                      .stop_reason = ENGINE_STOP_NONE,
                      .token_ids = &token,
                      .n_tokens = 1,
                      .telemetry = telemetry};
    callback(&event, data);
    return true;
  }
};
struct Stream {
  cudaStream_t handle{};
  ~Stream() {
    if (handle) cudaStreamDestroy(handle);
  }
};
struct FreePinned {
  void operator()(void* ptr) const noexcept { cudaFreeHost(ptr); }
};

template <class T>
std::unique_ptr<T, FreePinned> AllocatePinned(size_t count) {
  T* ptr;
  CheckCuda(cudaHostAlloc(&ptr, count * sizeof(T), cudaHostAllocDefault));
  return std::unique_ptr<T, FreePinned>(ptr);
}

PrefixCache AllocatePrefixCache(RuntimeConfig& cfg, const ModelConfig& c) {
  size_t free, total;
  CheckCuda(cudaMemGetInfo(&free, &total));
  size_t budget = free - std::min(free, cfg.vram_reserve_bytes);
  size_t fixed = ComputeKvArenaSizes(c).Fixed();
  if (cfg.prefix_cache_slots > budget / fixed)
    throw std::invalid_argument("prefix_cache_slots exceed free VRAM budget");
  auto capacity = KvStateArena::MaxFullSeqLenForBudget(c, budget - fixed * cfg.prefix_cache_slots);
  if (!capacity || cfg.max_context > capacity)
    throw std::invalid_argument("max_context exceeds free KV VRAM budget");
  if (!cfg.max_context) cfg.max_context = capacity;
  return PrefixCache(cfg.prefix_cache_slots, c, cfg.max_context);
}
struct DecodeState {
  uint32_t position;
  int32_t token;
};

class ModelExecutor {
 public:
  ModelExecutor(RuntimeConfig& cfg, cudaStream_t stream)
      : config_(cfg),
        stream_(stream),
        weights_(WeightsArena::Load(cfg.model_dir, stream)),
        scratch_(ScratchArena::Allocate(cfg.scratch_bytes)),
        io_(ScratchArena::Allocate(size_t(cfg.prefill_chunk_size) *
                                       (model::kHiddenSize * sizeof(uint16_t) + sizeof(int32_t)) +
                                   model::kUnpaddedVocabSize * sizeof(float) + 1024)),
        cache_(AllocatePrefixCache(cfg, model_config())),
        host_ids_(AllocatePinned<int32_t>(cfg.prefill_chunk_size)),
        host_logits_(AllocatePinned<float>(model::kUnpaddedVocabSize)),
        kv_(KvStateArena::Allocate(model_config(), cfg.max_context)),
        runner_(weights_.view(), kv_.view(), scratch_.view(), stream) {
    ScratchWorkspace ws(io_.view());
    ids_ = ws.Take<int32_t>(cfg.prefill_chunk_size);
    hidden_ = ws.Take<uint16_t>(size_t(cfg.prefill_chunk_size) * model::kHiddenSize);
    logits_ = ws.Take<float>(model::kUnpaddedVocabSize);
    next_ = ws.Take<int32_t>(1);
  }

  [[nodiscard]] const ModelConfig& model_config() const { return weights_.view().config; }
  [[nodiscard]] PrefixCacheStats cache_stats() const { return cache_.stats(); }
  void InvalidateCache() { cache_.InvalidateAll(); }
  [[nodiscard]] ForwardGraphStats graph_stats() const {
    std::lock_guard lock(graph_mutex_);
    return graphs_;
  }
  [[nodiscard]] DecodeState Prefill(Request& r);
  void Decode(Request& r, DecodeState state);

 private:
  enum class Phase { kPrefill, kDecode };
  int32_t Forward(RequestParams&, std::span<const int32_t> input, uint32_t position, Phase,
                  bool head);
  int32_t Sample(RequestParams& params, std::span<const float> logits) {
    return sampler_.Sample(logits, params.temperature, params.top_k, params.top_p, params.seed);
  }
  std::span<const float> HostLogits() const {
    return {host_logits_.get(), model_config().unpadded_vocab_size};
  }

  const RuntimeConfig& config_;
  cudaStream_t stream_;
  WeightsArena weights_;
  ScratchArena scratch_, io_;
  PrefixCache cache_;
  std::unique_ptr<int32_t, FreePinned> host_ids_;
  std::unique_ptr<float, FreePinned> host_logits_;
  KvStateArena kv_;
  InklingRunner runner_;
  Sampler sampler_;
  int32_t *ids_, *next_;
  uint16_t* hidden_;
  float* logits_;
  mutable std::mutex graph_mutex_;
  ForwardGraphStats graphs_{};
};

int32_t ModelExecutor::Forward(RequestParams& params, std::span<const int32_t> input,
                               uint32_t position, Phase phase, bool head) {
  std::ranges::copy(input, host_ids_.get());
  CheckCuda(
      cudaMemcpyAsync(ids_, host_ids_.get(), input.size_bytes(), cudaMemcpyHostToDevice, stream_));
  const bool decode = phase == Phase::kDecode;
  const bool stochastic = params.temperature > 0;
  const ForwardRunSpec spec{.end_visit = model::kNumLogicalVisits,
                            .run_embed = 1,
                            .run_lm_head = head,
                            .is_decode = decode,
                            .reset_kv = position == 0,
                            .seq_len = static_cast<uint32_t>(input.size()),
                            .pos_offset = position,
                            .d_input_ids = ids_,
                            .d_hidden_out = hidden_,
                            .d_logits_out = logits_,
                            .d_next_token = next_};
  const ForwardOptions opts{.use_cuda_graph = decode && config_.use_cuda_graph,
                            .return_logits = head && (stochastic || (!decode && cache_.enabled()))};
  const auto out = runner_.Run(spec, opts);
  if (opts.use_cuda_graph) {
    std::lock_guard lock(graph_mutex_);
    graphs_ = runner_.graph_stats();
  }
  if (!opts.return_logits) return out.next_token_id;
  CheckCuda(cudaMemcpyAsync(host_logits_.get(), logits_, HostLogits().size_bytes(),
                            cudaMemcpyDeviceToHost, stream_));
  CheckCuda(cudaStreamSynchronize(stream_));
  return stochastic ? Sample(params, HostLogits()) : out.next_token_id;
}
DecodeState ModelExecutor::Prefill(Request& r) {
  auto& t = r.telemetry;
  auto start_prefill = Clock::now();
  const std::span<const int32_t> prompt(r.prompt);
  auto plan = cache_.Match(prompt);
  uint32_t pos = plan.matched_tokens;
  t.cached_prefix_tokens = t.prefill_tokens = pos;
  bool copied_state = plan.restore_slot >= 0;
  if (copied_state) cache_.Restore(plan.restore_slot, kv_.view(), stream_);
  int32_t token = 0;
  if (plan.use_cached_logits) {
    token = Sample(r.params, cache_.SlotLogits(plan.logits_slot));
  } else {
    cache_.OnKvWrite(pos);
    while (pos < prompt.size() && !r.cancelled) {
      uint32_t chunk = std::min(config_.prefill_chunk_size, uint32_t(prompt.size()) - pos);
      if (plan.pin_pos > pos) chunk = std::min(chunk, plan.pin_pos - pos);
      auto input = prompt.subspan(pos, chunk);
      bool last = pos + chunk == prompt.size();
      token = Forward(r.params, input, pos, Phase::kPrefill, last);
      pos += chunk;
      cache_.AppendLive(input);
      t.prefill_tokens = pos;
      if (cache_.enabled() && (pos == plan.pin_pos || last) && !r.cancelled) {
        std::span<const float> logits;
        if (last) logits = HostLogits();
        cache_.Save(prompt.first(pos), kv_.view(), logits, stream_);
        copied_state = true;
      }
    }
  }
  if (copied_state) CheckCuda(cudaStreamSynchronize(stream_));
  t.prefill_ms = Milliseconds(start_prefill);
  return {.position = pos, .token = token};
}

void ModelExecutor::Decode(Request& r, DecodeState state) {
  auto [pos, token] = state;
  auto& t = r.telemetry;
  while (r.Emit(token)) {
    if (t.decode_tokens >= r.params.max_tokens) {
      r.stop_reason = ENGINE_STOP_MAX_TOKENS;
      return;
    }
    if (pos >= config_.max_context) {
      r.stop_reason = ENGINE_STOP_CONTEXT_LIMIT;
      return;
    }
    if (r.cancelled) return;
    auto start = Clock::now();
    auto fed_token = token;
    cache_.OnKvWrite(pos);
    token = Forward(r.params, {&fed_token, 1}, pos, Phase::kDecode, true);
    cache_.AppendLive({&fed_token, 1});
    ++pos;
    ++t.decode_rounds;
    t.decode_ms += Milliseconds(start);
  }
}
void RunSmoke(ScratchArenaView scratch, cudaStream_t stream, Request& r) {
  auto p = static_cast<float*>(scratch.device_ptr);
  auto start = Clock::now();
  CheckCuda(cudaMemsetAsync(p, 0, sizeof(float), stream));
  CheckCuda(cudaStreamSynchronize(stream));
  r.telemetry.prefill_ms = Milliseconds(start);
  r.telemetry.prefill_tokens = r.prompt.size();
  start = Clock::now();
  SmokeAddParams add{.input = p, .output = p + 1, .n = 1, .bias = 42.f};
  CheckCuda(flint_launch_smoke_add_f32(&add, stream));
  float value;
  CheckCuda(cudaMemcpyAsync(&value, p + 1, sizeof(value), cudaMemcpyDeviceToHost, stream));
  CheckCuda(cudaStreamSynchronize(stream));
  if (value != 42.f) throw std::runtime_error("smoke_add returned unexpected value");
  r.telemetry.decode_ms = Milliseconds(start);
  r.telemetry.decode_rounds = 1;
  if (!r.Emit(static_cast<int>(value))) {
    r.telemetry.decode_rounds = 0;
    return;
  }
  r.stop_reason =
      r.telemetry.decode_tokens >= r.params.max_tokens ? ENGINE_STOP_MAX_TOKENS : ENGINE_STOP_EOS;
}
}  // namespace

struct EngineRuntime::Impl {
  RuntimeConfig config;
  cudaDeviceProp gpu{};
  Stream stream;
  std::optional<ModelExecutor> model;
  std::optional<ScratchArena> smoke;
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::unique_ptr<Request>> queue;
  std::unordered_map<uint64_t, Request*> pending;
  bool stopping = false;
  uint64_t submitted{}, completed{}, cancellations{};
  PrefixCacheStats cache_stats{};  // Published with completed_requests under mutex.
  std::thread worker;

  explicit Impl(const char* json);
  ~Impl();
  void CopyRequest(Request& r, const RequestParams* params, const int32_t* ids, size_t count,
                   size_t media_count) const;
  void Execute(Request& r);
  void Finish(Request& r);
  void Work();
};
void EngineRuntime::Impl::Execute(Request& r) {
  if (r.status != ENGINE_OK || r.cancelled) return;
  if (!r.params.max_tokens) {
    r.telemetry.prefill_tokens = r.prompt.size();
    r.stop_reason = ENGINE_STOP_MAX_TOKENS;
    return;
  }
  if (model) {
    auto state = model->Prefill(r);
    model->Decode(r, state);
  } else {
    RunSmoke(smoke->view(), stream.handle, r);
  }
}
void EngineRuntime::Impl::Finish(Request& r) {
  {
    std::lock_guard lock(mutex);
    pending.erase(r.id);
    ++completed;
    if (model) cache_stats = model->cache_stats();
    if (r.cancelled) r.Fail(ENGINE_ERR_CANCELLED, "request cancelled");
  }
  EngineEvent terminal{.req_id = r.id,
                       .kind = r.status == ENGINE_OK ? ENGINE_EVENT_DONE : ENGINE_EVENT_ERROR,
                       .status = r.status,
                       .stop_reason = r.status == ENGINE_OK ? r.stop_reason : ENGINE_STOP_NONE,
                       .telemetry = r.telemetry,
                       .error_message = r.error};
  r.callback(&terminal, r.data);
}
void EngineRuntime::Impl::Work() {
  auto device_status = cudaSetDevice(0);
  for (;;) {
    std::unique_ptr<Request> r;
    {
      std::unique_lock lock(mutex);
      ready.wait(lock, [&] { return stopping || !queue.empty(); });
      if (queue.empty()) return;
      r = std::move(queue.front());
      queue.pop_front();
    }
    try {
      CheckCuda(device_status);
      Execute(*r);
    } catch (const std::exception& err) {
      if (model) model->InvalidateCache();
      cudaStreamSynchronize(stream.handle);
      r->Fail(ENGINE_ERR_INTERNAL, err.what());
      Finish(*r);
      continue;
    }
    Finish(*r);
  }
}
void EngineRuntime::Impl::CopyRequest(Request& r, const RequestParams* p, const int32_t* ids,
                                      size_t n, size_t media) const {
  if (!p || !ids || !n || media || (p->n_stop_token_ids && !p->stop_token_ids) ||
      std::isnan(p->temperature) || p->temperature < 0 || std::isnan(p->top_p) || p->top_p > 1 ||
      p->top_k < 0)
    return r.Fail(ENGINE_ERR_INVALID_INPUT, "invalid text request or sampling parameters");
  if (n > UINT32_MAX || (model && n > config.max_context))
    return r.Fail(ENGINE_ERR_CONTEXT_OVERFLOW, "prompt exceeds context capacity");
  if (model && std::any_of(ids, ids + n, [&](int32_t id) {
        return id < 0 || uint32_t(id) >= model->model_config().vocab_size;
      }))
    return r.Fail(ENGINE_ERR_INVALID_INPUT, "prompt token outside vocabulary");
  r.prompt.assign(ids, ids + n);
  if (p->n_stop_token_ids)
    r.stops.assign(p->stop_token_ids, p->stop_token_ids + p->n_stop_token_ids);
  if (model && !p->ignore_eos) r.eos_token_id = model->model_config().eos_token_id;
  r.params = *p;
  r.params.stop_token_ids = r.stops.data();
  if (!r.params.seed) r.params.seed = NewSeed();
}
EngineRuntime::Impl::Impl(const char* json) : config(RuntimeConfig::Parse(json)) {
  CheckCuda(cudaSetDevice(0));
  CheckCuda(cudaGetDeviceProperties(&gpu, 0));
  CheckCuda(cudaStreamCreateWithFlags(&stream.handle, cudaStreamNonBlocking));
  if (config.model_dir.empty())
    smoke.emplace(ScratchArena::Allocate(2 * sizeof(float)));
  else
    model.emplace(config, stream.handle);
  worker = std::thread(&Impl::Work, this);
}

EngineRuntime::Impl::~Impl() {
  {
    std::lock_guard lock(mutex);
    stopping = true;
    for (auto [id, r] : pending) r->cancelled = true;
  }
  ready.notify_one();
  worker.join();
}

EngineRuntime::EngineRuntime(const char* json) : impl_(std::make_unique<Impl>(json)) {}
EngineRuntime::~EngineRuntime() = default;

uint64_t EngineRuntime::Submit(const RequestParams* params, const int32_t* ids, size_t n,
                               size_t media, TokenCallback cb, void* data) {
  auto r = std::make_unique<Request>();
  r->callback = cb;
  r->data = data;
  impl_->CopyRequest(*r, params, ids, n, media);
  std::lock_guard lock(impl_->mutex);
  if (impl_->stopping) return 0;
  const uint64_t id = impl_->submitted + 1;
  r->id = id;
  impl_->pending.emplace(id, r.get());
  try {
    impl_->queue.push_back(std::move(r));
  } catch (...) {
    impl_->pending.erase(id);
    throw;
  }
  impl_->submitted = id;
  impl_->ready.notify_one();
  return id;
}
void EngineRuntime::Cancel(uint64_t id) {
  std::lock_guard lock(impl_->mutex);
  if (!id || id > impl_->submitted) return;
  ++impl_->cancellations;
  if (auto it = impl_->pending.find(id); it != impl_->pending.end()) it->second->cancelled = true;
}
std::string EngineRuntime::StatsJson() const {
  size_t free, total;
  CheckCuda(cudaSetDevice(0));
  CheckCuda(cudaMemGetInfo(&free, &total));
  auto& e = *impl_;
  std::lock_guard lock(e.mutex);
  const auto graphs = e.model ? e.model->graph_stats() : ForwardGraphStats{};
  return Encode(Object{{"abi_version", FLINT_ENGINE_ABI_VERSION},
                       {"mode", e.model ? "model" : "smoke"},
                       {"gpu_name", e.gpu.name},
                       {"sm_major", e.gpu.major},
                       {"sm_minor", e.gpu.minor},
                       {"vram_total_bytes", total},
                       {"vram_free_bytes", free},
                       {"max_context", e.config.max_context},
                       {"prefill_chunk_size", e.config.prefill_chunk_size},
                       {"use_cuda_graph", e.config.use_cuda_graph},
                       {"prefix_cache_slots", e.model ? e.config.prefix_cache_slots : 0},
                       {"prefix_cache_used_slots", e.cache_stats.used_slots},
                       {"prefix_cache_hits", e.cache_stats.hits},
                       {"prefix_cache_misses", e.cache_stats.misses},
                       {"prefix_cache_saved_tokens", e.cache_stats.saved_tokens},
                       {"submitted_requests", e.submitted},
                       {"completed_requests", e.completed},
                       {"cancelled_requests", e.cancellations},
                       {"queued_requests", e.queue.size()},
                       {"active_requests", e.pending.size() - e.queue.size()},
                       {"graph_captures", graphs.captures},
                       {"graph_launches", graphs.launches}});
}
}  // namespace flint

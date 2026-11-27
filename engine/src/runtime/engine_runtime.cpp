#include "runtime/engine_runtime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
#include "model/dspark_forward.h"
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
  uint32_t prefill_chunks{};
  EngineStatus status = ENGINE_OK;
  EngineStopReason stop_reason = ENGINE_STOP_NONE;
  const char* error{};
  void Fail(EngineStatus code, const char* reason) {
    status = code;
    error = reason;
  }
  // Compute the visible prefix before publishing or committing a verify block.
  std::span<const int32_t> LimitBatch(std::span<const int32_t> tokens) {
    auto count = std::min<size_t>(tokens.size(), params.max_tokens - telemetry.decode_tokens);
    for (size_t i = 0; i < count; ++i) {
      if (tokens[i] == eos_token_id)
        stop_reason = ENGINE_STOP_EOS;
      else if (std::ranges::find(stops, tokens[i]) != stops.end())
        stop_reason = ENGINE_STOP_STOP_TOKEN;
      if (stop_reason != ENGINE_STOP_NONE) return tokens.first(i);
    }
    if (telemetry.decode_tokens + count == params.max_tokens) stop_reason = ENGINE_STOP_MAX_TOKENS;
    return tokens.first(count);
  }
  void EmitBatch(std::span<const int32_t> visible, uint32_t accepted = 0) {
    if (cancelled || visible.empty()) return;
    telemetry.decode_tokens += visible.size();
    telemetry.spec_accepted += std::min<size_t>(accepted, visible.size());
    EngineEvent event{.req_id = id,
                      .kind = ENGINE_EVENT_TOKENS,
                      .stop_reason = ENGINE_STOP_NONE,
                      .token_ids = visible.data(),
                      .n_tokens = visible.size(),
                      .telemetry = telemetry};
    callback(&event, data);
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

constexpr uint32_t kBlock = model::kMtpConfig.block_size;
constexpr size_t kTap = 4 * model::kHiddenSize;
constexpr size_t kVocab = model::kUnpaddedVocabSize;
PrefixCache AllocatePrefixCache(RuntimeConfig& cfg, const ModelConfig& c, bool with_mtp) {
  size_t free, total;
  CheckCuda(cudaMemGetInfo(&free, &total));
  size_t budget = free - std::min(free, cfg.vram_reserve_bytes);
  size_t fixed = ComputeKvArenaSizes(c, with_mtp).Fixed();
  if (cfg.prefix_cache_slots > budget / fixed)
    throw std::invalid_argument("prefix_cache_slots exceed free VRAM budget");
  auto capacity =
      KvStateArena::MaxFullSeqLenForBudget(c, budget - fixed * cfg.prefix_cache_slots, with_mtp);
  // Draft attention writes a whole block, even at the logical context boundary.
  if (with_mtp) capacity -= std::min(capacity, kBlock - 1);
  if (!capacity || cfg.max_context > capacity)
    throw std::invalid_argument("max_context exceeds free KV VRAM budget");
  if (!cfg.max_context) cfg.max_context = capacity;
  return PrefixCache(cfg.prefix_cache_slots, c, cfg.max_context, with_mtp);
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
        weights_(WeightsArena::Load(cfg.model_dir, stream, cfg.use_speculative)),
        active_(weights_.view().has_mtp),
        rows_(active_ ? std::max(kBlock, cfg.prefill_chunk_size) : cfg.prefill_chunk_size),
        // The minimum 64 MiB target configuration previously had a separate 32 MiB draft arena.
        scratch_(ScratchArena::Allocate(active_ ? std::max(cfg.scratch_bytes, size_t{96} << 20)
                                                : cfg.scratch_bytes)),
        target_scratch_(scratch_.view()),
        draft_scratch_(active_ ? DsparkRunner::ReserveScratch(target_scratch_)
                               : ScratchArenaView{}),
        io_(ScratchArena::Allocate(rows_ *
                                       (model::kHiddenSize * sizeof(uint16_t) + sizeof(int32_t)) +
                                   kVocab * sizeof(float) * (active_ ? kBlock : 1) + 2048 +
                                   (active_ ? (rows_ + 1) * kTap * sizeof(uint16_t) : 0))),
        cache_(AllocatePrefixCache(cfg, model_config(), active_)),
        host_ids_(AllocatePinned<int32_t>(rows_)),
        host_logits_(AllocatePinned<float>(kVocab * (active_ ? kBlock : 1))),
        kv_(KvStateArena::Allocate(model_config(), cfg.max_context + (active_ ? kBlock - 1 : 0),
                                   active_)),
        runner_(weights_.view(), kv_.view(), target_scratch_, stream) {
    ScratchWorkspace ws(io_.view());
    ids_ = ws.Take<int32_t>(rows_);
    hidden_ = ws.Take<uint16_t>(rows_ * model::kHiddenSize);
    logits_ = ws.Take<float>(kVocab * (active_ ? kBlock : 1));
    next_ = ws.Take<int32_t>(active_ ? kBlock : 1);
    if (active_) {
      taps_ = ws.Take<uint16_t>(rows_ * kTap);
      last_tap_ = ws.Take<uint16_t>(kTap);
      host_tap_ = AllocatePinned<uint16_t>(kTap);
      host_next_ = AllocatePinned<int32_t>(kBlock);
      draft_logits_ = AllocatePinned<float>((kBlock - 1) * kVocab);
      dspark_.emplace(weights_.view(), kv_.view(), draft_scratch_, stream);
    }
  }

  [[nodiscard]] const ModelConfig& model_config() const { return weights_.view().config; }
  bool speculative_active() const { return active_; }
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

  void StoreContext(uint32_t rows, uint32_t pos) {
    dspark_->StoreContextKv(taps_, rows, pos);
    CheckCuda(cudaMemcpyAsync(last_tap_, taps_ + (rows - 1) * kTap, kTap * sizeof(uint16_t),
                              cudaMemcpyDeviceToDevice, stream_));
  }
  void SpecRound(Request&, uint32_t& pos, int32_t& anchor);
  const RuntimeConfig& config_;
  cudaStream_t stream_;
  WeightsArena weights_;
  const bool active_;
  const size_t rows_;
  ScratchArena scratch_;
  ScratchArenaView target_scratch_, draft_scratch_;
  ScratchArena io_;
  PrefixCache cache_;
  std::unique_ptr<int32_t, FreePinned> host_ids_;
  std::unique_ptr<float, FreePinned> host_logits_;
  KvStateArena kv_;
  InklingRunner runner_;
  std::optional<DsparkRunner> dspark_;
  std::unique_ptr<int32_t, FreePinned> host_next_;
  std::unique_ptr<uint16_t, FreePinned> host_tap_;
  std::unique_ptr<float, FreePinned> draft_logits_;
  uint16_t *taps_{}, *last_tap_{};
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
                            .d_next_token = next_,
                            .d_taps_out = taps_};
  const ForwardOptions opts{.use_cuda_graph = decode && config_.use_cuda_graph,
                            .return_logits = head && (stochastic || (!decode && cache_.enabled()))};
  const auto out = runner_.Run(spec, opts);
  if (active_) StoreContext(input.size(), position);
  if (opts.use_cuda_graph) {
    std::lock_guard lock(graph_mutex_);
    graphs_ = runner_.graph_stats();
  }
  if (!opts.return_logits) return out.next_token_id;
  if (active_ && !decode && cache_.enabled())
    CheckCuda(cudaMemcpyAsync(host_tap_.get(), last_tap_, kTap * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost, stream_));
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
    if (active_) {
      auto tap = cache_.SlotLastTap(plan.logits_slot);
      CheckCuda(cudaMemcpyAsync(last_tap_, tap.data(), tap.size_bytes(), cudaMemcpyHostToDevice,
                                stream_));
    }
    token = Sample(r.params, cache_.SlotLogits(plan.logits_slot));
  } else {
    cache_.OnKvWrite(pos);
    while (pos < prompt.size() && !r.cancelled) {
      uint32_t chunk = std::min(config_.prefill_chunk_size, uint32_t(prompt.size()) - pos);
      if (plan.pin_pos > pos) chunk = std::min(chunk, plan.pin_pos - pos);
      auto input = prompt.subspan(pos, chunk);
      bool last = pos + chunk == prompt.size();
      token = Forward(r.params, input, pos, Phase::kPrefill, last);
      ++r.prefill_chunks;
      pos += chunk;
      cache_.AppendLive(input);
      t.prefill_tokens = pos;
      if (cache_.enabled() && (pos == plan.pin_pos || last) && !r.cancelled) {
        std::span<const float> logits;
        if (last) logits = HostLogits();
        std::span<const uint16_t> tap;
        if (last && active_) tap = {host_tap_.get(), kTap};
        cache_.Save(prompt.first(pos), kv_.view(), logits, stream_, tap);
        copied_state = true;
      }
    }
  }
  if (copied_state || active_) CheckCuda(cudaStreamSynchronize(stream_));
  t.prefill_ms = Milliseconds(start_prefill);
  return {.position = pos, .token = token};
}

void ModelExecutor::SpecRound(Request& r, uint32_t& pos, int32_t& anchor) {
  auto start = Clock::now();
  auto& p = r.params;
  const bool stochastic = p.temperature > 0;
  const uint32_t len = std::min(kBlock, config_.max_context - pos);
  std::span<int32_t, kBlock> block(host_ids_.get(), kBlock);
  block.front() = anchor;
  if (stochastic)
    dspark_->DraftWithLogits(anchor, last_tap_, pos, block.data() + 1, draft_logits_.get(),
                             [&](auto row) { return Sample(p, row); });
  else
    dspark_->DraftGreedy(anchor, last_tap_, pos, block.data() + 1, nullptr, config_.use_cuda_graph);
  auto input = block.first(len);
  CheckCuda(
      cudaMemcpyAsync(ids_, input.data(), input.size_bytes(), cudaMemcpyHostToDevice, stream_));
  runner_.RunVerify({.seq_len = len,
                     .pos_offset = pos,
                     .d_input_ids = ids_,
                     .d_hidden_out = hidden_,
                     .d_logits_out = logits_,
                     .d_next_token = next_,
                     .d_taps_out = taps_},
                    {.use_cuda_graph = config_.use_cuda_graph, .return_logits = stochastic});
  if (config_.use_cuda_graph) {
    std::lock_guard lock(graph_mutex_);
    graphs_ = runner_.graph_stats();
  }
  SpeculativeResult result;
  try {
    std::span<int32_t> argmax(host_next_.get(), len);
    std::span<const float> draft, target;
    if (stochastic) {
      draft = {draft_logits_.get(), (len - 1) * kVocab};
      target = {host_logits_.get(), len * kVocab};
      CheckCuda(cudaMemcpyAsync(host_logits_.get(), logits_, target.size_bytes(),
                                cudaMemcpyDeviceToHost, stream_));
    } else {
      CheckCuda(cudaMemcpyAsync(argmax.data(), next_, argmax.size_bytes(), cudaMemcpyDeviceToHost,
                                stream_));
    }
    CheckCuda(cudaStreamSynchronize(stream_));
    result = VerifySpeculative(input.subspan(1), argmax, draft, target, kVocab, p.temperature,
                               p.top_k, p.top_p, p.seed);
  } catch (...) {
    runner_.CommitVerify(pos, len, 0);
    throw;
  }
  const uint32_t accepted = result.accepted_draft_count;
  std::array<int32_t, kBlock> output;
  std::ranges::copy(input.subspan(1, accepted), output.begin());
  output[accepted] = result.bonus_token;
  auto visible = r.LimitBatch(std::span(output).first(accepted + 1));
  // For an EOS/stop prediction, its predecessor was consumed. At an output
  // limit the final emitted token is still the next (unconsumed) anchor.
  uint32_t committed = visible.size() + (r.stop_reason == ENGINE_STOP_EOS ||
                                         r.stop_reason == ENGINE_STOP_STOP_TOKEN);
  runner_.CommitVerify(pos, len, committed);
  StoreContext(committed, pos);
  CheckCuda(cudaStreamSynchronize(stream_));
  cache_.OnKvWrite(pos);
  cache_.AppendLive(input.first(committed));
  pos += committed;
  anchor = result.bonus_token;
  ++r.telemetry.decode_rounds;
  r.telemetry.decode_ms += Milliseconds(start);
  r.EmitBatch(visible, accepted);
}

void ModelExecutor::Decode(Request& r, DecodeState state) {
  auto [pos, token] = state;
  auto& t = r.telemetry;
  r.EmitBatch(r.LimitBatch({&token, 1}));
  while (!r.cancelled && r.stop_reason == ENGINE_STOP_NONE) {
    if (pos >= config_.max_context) {
      r.stop_reason = ENGINE_STOP_CONTEXT_LIMIT;
      return;
    }
    if (active_ && r.params.max_tokens - t.decode_tokens > 1 && config_.max_context - pos > 1) {
      SpecRound(r, pos, token);
      continue;
    }
    auto start = Clock::now();
    auto fed_token = token;
    token = Forward(r.params, {&fed_token, 1}, pos, Phase::kDecode, true);
    if (active_) CheckCuda(cudaStreamSynchronize(stream_));
    cache_.OnKvWrite(pos);
    cache_.AppendLive({&fed_token, 1});
    ++pos;
    ++t.decode_rounds;
    t.decode_ms += Milliseconds(start);
    r.EmitBatch(r.LimitBatch({&token, 1}));
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
  int32_t token = static_cast<int32_t>(value);
  r.EmitBatch(r.LimitBatch({&token, 1}));
  if (!r.telemetry.decode_tokens) {
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
  uint64_t total_prefill_chunks{}, total_prefill_tokens{}, total_cached_prefix_tokens{},
      total_decode_tokens{}, total_decode_rounds{}, total_spec_accepted{};
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
    total_prefill_chunks += r.prefill_chunks;
    total_prefill_tokens += r.telemetry.prefill_tokens;
    total_cached_prefix_tokens += r.telemetry.cached_prefix_tokens;
    total_decode_tokens += r.telemetry.decode_tokens;
    total_decode_rounds += r.telemetry.decode_rounds;
    total_spec_accepted += r.telemetry.spec_accepted;
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
                       {"use_speculative", e.config.use_speculative},
                       {"speculative_active", e.model && e.model->speculative_active()},
                       {"total_prefill_chunks", e.total_prefill_chunks},
                       {"total_prefill_tokens", e.total_prefill_tokens},
                       {"total_cached_prefix_tokens", e.total_cached_prefix_tokens},
                       {"total_decode_tokens", e.total_decode_tokens},
                       {"total_decode_rounds", e.total_decode_rounds},
                       {"total_spec_accepted", e.total_spec_accepted},
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

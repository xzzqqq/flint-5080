#include "engine_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>

#include "common/cuda_check.h"
#include "common/json_output.h"
#include "loader/arenas.h"
#include "loader/kv_layout.h"
#include "model/inkling2_forward.h"
#include "ops/scratch_workspace.h"
#include "ops/stub_ops.h"
#include "runtime/config.h"
#include "runtime/sampler.h"

namespace {
using namespace flint;
using Clock = std::chrono::steady_clock;
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
char* CopyString(const char* s) {
  size_t bytes = std::strlen(s) + 1;
  char* p = static_cast<char*>(std::malloc(bytes));
  if (p) std::memcpy(p, s, bytes);
  return p;
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
  TokenCallback callback;
  void* data;
  std::atomic<bool> cancelled{};
  EngineTelemetry telemetry{};
  EngineStatus status = ENGINE_OK;
  const char* error{};
  void Fail(EngineStatus code, const char* reason) {
    status = code;
    error = reason;
  }
  bool Emit(int32_t token) {
    if (cancelled || std::ranges::find(stops, token) != stops.end()) return false;
    ++telemetry.decode_tokens;
    EngineEvent event{.req_id = id,
                      .kind = ENGINE_EVENT_TOKENS,
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
struct PinnedIO {
  int32_t* ids{};
  float* logits{};
  ~PinnedIO() {
    if (ids) cudaFreeHost(ids);
    if (logits) cudaFreeHost(logits);
  }
};
struct Model {
  WeightsArena weights;
  ScratchArena scratch;
  ScratchArena io;
  PinnedIO host;
  int32_t *ids, *next;
  uint16_t* hidden;
  float* logits;
  std::optional<KvStateArena> kv;
  std::optional<InklingRunner> runner;
  Sampler sampler;
  Model(RuntimeConfig& cfg, cudaStream_t stream)
      : weights(WeightsArena::Load(cfg.model_dir, stream)),
        scratch(ScratchArena::Allocate(cfg.scratch_bytes)),
        io(ScratchArena::Allocate(size_t(cfg.prefill_chunk_size) *
                                      (model::kHiddenSize * sizeof(uint16_t) + sizeof(int32_t)) +
                                  model::kUnpaddedVocabSize * sizeof(float) + 1024)) {
    ops_detail::ScratchWorkspace ws(io.view());
    ids = ws.Take<int32_t>(cfg.prefill_chunk_size);
    hidden = ws.Take<uint16_t>(size_t(cfg.prefill_chunk_size) * model::kHiddenSize);
    logits = ws.Take<float>(model::kUnpaddedVocabSize);
    next = ws.Take<int32_t>(1);
    CheckCuda(cudaMallocHost(&host.ids, cfg.prefill_chunk_size * sizeof(int32_t)));
    CheckCuda(cudaMallocHost(&host.logits, model::kUnpaddedVocabSize * sizeof(float)));
    const auto& c = weights.view().config;
    size_t free, total;
    CheckCuda(cudaMemGetInfo(&free, &total));
    size_t budget = free - std::min(free, cfg.vram_reserve_bytes);
    size_t fixed = ComputeKvArenaSizes(c).Fixed();
    if (cfg.prefix_cache_slots > budget / fixed)
      throw std::invalid_argument("prefix_cache_slots exceed free VRAM budget");
    auto capacity =
        KvStateArena::MaxFullSeqLenForBudget(c, budget - fixed * cfg.prefix_cache_slots);
    if (!capacity || cfg.max_context > capacity)
      throw std::invalid_argument("max_context exceeds free KV VRAM budget");
    if (!cfg.max_context) cfg.max_context = capacity;
    kv.emplace(KvStateArena::Allocate(c, cfg.max_context));
    runner.emplace(weights.view(), kv->view(), scratch.view(), stream);
  }
};
}  // namespace
struct EngineHandle {
  RuntimeConfig config;
  cudaDeviceProp gpu{};
  Stream stream;
  std::optional<Model> model;
  std::optional<ScratchArena> smoke;
  std::mutex mutex;
  std::condition_variable ready;
  std::deque<std::unique_ptr<Request>> queue;
  std::unordered_map<uint64_t, Request*> pending;
  bool stopping = false;
  uint64_t submitted{}, completed{}, cancellations{};
  GraphStats graphs{};
  std::thread worker;
};
namespace {
void RunSmoke(EngineHandle& e, Request& r) {
  auto stream = e.stream.handle;
  auto p = static_cast<float*>(e.smoke->view().device_ptr);
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
  if (!r.Emit(static_cast<int>(value))) r.telemetry.decode_rounds = 0;
}
void RunModel(EngineHandle& e, Request& r) {
  auto& m = *e.model;
  auto stream = e.stream.handle;
  auto& t = r.telemetry;
  const bool stochastic = r.params.temperature > 0;
  auto forward = [&](const int32_t* input, uint32_t count, uint32_t pos, bool decode, bool head) {
    std::memcpy(m.host.ids, input, count * sizeof(int32_t));
    CheckCuda(cudaMemcpyAsync(m.ids, m.host.ids, count * sizeof(int32_t), cudaMemcpyHostToDevice,
                              stream));
    ForwardBatch batch{.end_visit = model::kNumLogicalVisits,
                       .run_embed = 1,
                       .run_lm_head = head,
                       .is_decode = decode,
                       .reset_kv = pos == 0,
                       .seq_len = count,
                       .pos_offset = pos,
                       .d_input_ids = m.ids,
                       .d_hidden_out = m.hidden,
                       .d_logits_out = m.logits,
                       .d_next_token = m.next};
    ForwardRunOptions opts{.use_cuda_graph = decode && e.config.use_cuda_graph,
                           .return_logits = head && stochastic};
    auto out = m.runner->Run(batch, opts);
    if (opts.use_cuda_graph) {
      std::lock_guard lock(e.mutex);
      e.graphs = m.runner->graph_stats();
    }
    if (!opts.return_logits) return out.next_token_id;
    auto vocab = m.weights.view().config.unpadded_vocab_size;
    CheckCuda(cudaMemcpyAsync(m.host.logits, m.logits, vocab * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaStreamSynchronize(stream));
    return m.sampler.Sample({m.host.logits, vocab}, r.params.temperature, r.params.top_k,
                            r.params.top_p, r.params.seed);
  };
  uint32_t pos = 0;
  int32_t token = 0;
  while (pos < r.prompt.size() && !r.cancelled) {
    auto start = Clock::now();
    uint32_t chunk =
        std::min(e.config.prefill_chunk_size, static_cast<uint32_t>(r.prompt.size()) - pos);
    token = forward(r.prompt.data() + pos, chunk, pos, false, pos + chunk == r.prompt.size());
    pos += chunk;
    t.prefill_tokens = pos;
    t.prefill_ms += Milliseconds(start);
  }
  if (!r.Emit(token)) return;
  while (t.decode_tokens < r.params.max_tokens && pos < e.config.max_context && !r.cancelled) {
    auto start = Clock::now();
    token = forward(&token, 1, pos, true, true);
    ++pos;
    ++t.decode_rounds;
    t.decode_ms += Milliseconds(start);
    if (!r.Emit(token)) break;
  }
}
void Execute(EngineHandle& e, Request& r) {
  if (r.status != ENGINE_OK || r.cancelled) return;
  if (!r.params.max_tokens) {
    r.telemetry.prefill_tokens = r.prompt.size();
    return;
  }
  if (e.model)
    RunModel(e, r);
  else
    RunSmoke(e, r);
}
void Finish(EngineHandle& e, Request& r) {
  {
    std::lock_guard lock(e.mutex);
    e.pending.erase(r.id);
    ++e.completed;
    if (r.cancelled) r.Fail(ENGINE_ERR_CANCELLED, "request cancelled");
  }
  EngineEvent terminal{.req_id = r.id,
                       .kind = r.status == ENGINE_OK ? ENGINE_EVENT_DONE : ENGINE_EVENT_ERROR,
                       .status = r.status,
                       .telemetry = r.telemetry,
                       .error_message = r.error};
  r.callback(&terminal, r.data);
}
void Work(EngineHandle& e) {
  auto device_status = cudaSetDevice(0);
  for (;;) {
    std::unique_ptr<Request> r;
    {
      std::unique_lock lock(e.mutex);
      e.ready.wait(lock, [&] { return e.stopping || !e.queue.empty(); });
      if (e.queue.empty()) return;
      r = std::move(e.queue.front());
      e.queue.pop_front();
    }
    try {
      CheckCuda(device_status);
      Execute(e, *r);
    } catch (const std::exception& err) {
      cudaStreamSynchronize(e.stream.handle);
      r->Fail(ENGINE_ERR_INTERNAL, err.what());
      Finish(e, *r);
      continue;
    }
    Finish(e, *r);
  }
}
void CopyRequest(const EngineHandle& e, Request& r, const RequestParams* p, const int32_t* ids,
                 size_t n, size_t media) {
  if (!p || !ids || !n || media || (p->n_stop_token_ids && !p->stop_token_ids) ||
      std::isnan(p->temperature) || p->temperature < 0 || std::isnan(p->top_p) || p->top_p > 1 ||
      p->top_k < 0)
    return r.Fail(ENGINE_ERR_INVALID_INPUT, "invalid text request or sampling parameters");
  if (n > UINT32_MAX || (e.model && n > e.config.max_context))
    return r.Fail(ENGINE_ERR_CONTEXT_OVERFLOW, "prompt exceeds context capacity");
  if (e.model && std::any_of(ids, ids + n, [&](int32_t id) {
        return id < 0 || uint32_t(id) >= e.model->weights.view().config.vocab_size;
      }))
    return r.Fail(ENGINE_ERR_INVALID_INPUT, "prompt token outside vocabulary");
  r.prompt.assign(ids, ids + n);
  if (p->n_stop_token_ids)
    r.stops.assign(p->stop_token_ids, p->stop_token_ids + p->n_stop_token_ids);
  if (e.model && !p->ignore_eos) r.stops.push_back(e.model->weights.view().config.eos_token_id);
  r.params = *p;
  r.params.stop_token_ids = r.stops.data();
  if (!r.params.seed) r.params.seed = NewSeed();
}
}  // namespace
uint32_t engine_abi_version() { return FLINT_ENGINE_ABI_VERSION; }
EngineHandle* engine_create(const char* json, char** err_out) {
  if (err_out) *err_out = nullptr;
  try {
    auto e = std::make_unique<EngineHandle>();
    e->config = RuntimeConfig::Parse(json);
    CheckCuda(cudaSetDevice(0));
    CheckCuda(cudaGetDeviceProperties(&e->gpu, 0));
    CheckCuda(cudaStreamCreateWithFlags(&e->stream.handle, cudaStreamNonBlocking));
    if (e->config.model_dir.empty())
      e->smoke.emplace(ScratchArena::Allocate(2 * sizeof(float)));
    else
      e->model.emplace(e->config, e->stream.handle);
    e->worker = std::thread(Work, std::ref(*e));
    return e.release();
  } catch (const std::exception& err) {
    if (err_out) *err_out = CopyString(err.what());
    return nullptr;
  }
}
void engine_destroy(EngineHandle* e) {
  if (!e) return;
  {
    std::lock_guard lock(e->mutex);
    e->stopping = true;
    for (auto [id, r] : e->pending) r->cancelled = true;
  }
  e->ready.notify_one();
  e->worker.join();
  delete e;
}
void engine_string_free(char* s) { std::free(s); }
uint64_t engine_submit(EngineHandle* e, const RequestParams* params, const int32_t* ids, size_t n,
                       const MediaSlice*, size_t media, TokenCallback cb, void* data) {
  if (!e || !cb) return 0;
  try {
    auto r = std::make_unique<Request>();
    r->callback = cb;
    r->data = data;
    CopyRequest(*e, *r, params, ids, n, media);
    std::lock_guard lock(e->mutex);
    if (e->stopping) return 0;
    const uint64_t id = e->submitted + 1;
    r->id = id;
    e->pending.emplace(id, r.get());
    try {
      e->queue.push_back(std::move(r));
    } catch (...) {
      e->pending.erase(id);
      throw;
    }
    e->submitted = id;
    e->ready.notify_one();
    return id;
  } catch (const std::exception&) {
    return 0;
  }
}
void engine_cancel(EngineHandle* e, uint64_t id) {
  if (!e) return;
  std::lock_guard lock(e->mutex);
  if (!id || id > e->submitted) return;
  ++e->cancellations;
  if (auto it = e->pending.find(id); it != e->pending.end()) it->second->cancelled = true;
}
char* engine_stats_json(EngineHandle* e) {
  size_t free, total;
  if (!e || cudaSetDevice(0) != cudaSuccess || cudaMemGetInfo(&free, &total) != cudaSuccess)
    return nullptr;
  try {
    std::lock_guard lock(e->mutex);
    return CopyString(Encode(Object{{"abi_version", FLINT_ENGINE_ABI_VERSION},
                                    {"mode", e->model ? "model" : "smoke"},
                                    {"gpu_name", e->gpu.name},
                                    {"sm_major", e->gpu.major},
                                    {"sm_minor", e->gpu.minor},
                                    {"vram_total_bytes", total},
                                    {"vram_free_bytes", free},
                                    {"max_context", e->config.max_context},
                                    {"prefill_chunk_size", e->config.prefill_chunk_size},
                                    {"use_cuda_graph", e->config.use_cuda_graph},
                                    {"submitted_requests", e->submitted},
                                    {"completed_requests", e->completed},
                                    {"cancelled_requests", e->cancellations},
                                    {"queued_requests", e->queue.size()},
                                    {"active_requests", e->pending.size() - e->queue.size()},
                                    {"graph_captures", e->graphs.captures},
                                    {"graph_launches", e->graphs.launches}})
                          .c_str());
  } catch (const std::exception&) {
    return nullptr;
  }
}

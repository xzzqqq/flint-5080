#include "forge/replay.h"

#include <unistd.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>

#include "common/cuda_check.h"
#include "forge/safetensors_writer.h"
#include "loader/kv_layout.h"

namespace flint {
namespace fs = std::filesystem;
namespace {

size_t FreeBytes() {
  size_t free, total;
  CheckCuda(cudaMemGetInfo(&free, &total));
  return free;
}
template <class T>
std::vector<T> Read(const T* device, size_t count) {
  std::vector<T> data(count);
  CheckCuda(cudaMemcpy(data.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
  return data;
}
template <class T>
void Upload(cudaStream_t stream, T* device, const std::vector<T>& data) {
  CheckCuda(cudaMemcpyAsync(device, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice,
                            stream));
}
unsigned Count(const Json& q, const char* name, unsigned fallback, unsigned lo, unsigned hi) {
  auto n = q.get<unsigned>(name, fallback);
  if (n < lo || n > hi) throw std::runtime_error(std::string(name) + " out of range");
  return n;
}
}  // namespace

std::optional<ForwardModule> Replay::Load(const fs::path& path) {
  return std::make_optional<ForwardModule>(path, config_.state_dir / "loaded",
                                           weights_ ? &weights_->view() : nullptr, kv_->view(),
                                           scratch_->view(), stream_);
}

Replay::Replay(DaemonConfig cfg) : config_(std::move(cfg)) {
  CheckCuda(cudaSetDevice(0));
  model_config_ = model::MakeDefaultInklingConfig();
  if (!config_.model_dir.empty()) {
    auto start = std::chrono::steady_clock::now();
    weights_.emplace(WeightsArena::Load(config_.model_dir));
    weights_load_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    model_config_ = weights_->view().config;
  }
  plan_ = BuildVisitPlan(model_config_);
  width_ = model_config_.hidden_size;
  vocab_ = model_config_.unpadded_vocab_size;
  kv_.emplace(KvStateArena::Allocate(model_config_, config_.kv_seq_len));
  scratch_.emplace(ScratchArena::Allocate(size_t(config_.scratch_mb) << 20));
  capacity_ = std::min<size_t>({config_.kv_seq_len, model::kSlidingWindow,
                                scratch_->view().bytes / (width_ * sizeof(float))});
  if (!capacity_) throw std::runtime_error("scratch too small for one token");
  size_t elements = size_t(capacity_) * width_;
  auto hidden_bytes = elements * sizeof(uint16_t);
  io_.emplace(ScratchArena::Allocate((2 + model::kCarryPlanes) * hidden_bytes +
                                     vocab_ * sizeof(float) + (capacity_ + 1) * sizeof(int)));
  hidden_ = static_cast<uint16_t*>(io_->view().device_ptr);
  out_ = hidden_ + elements;
  carry_ = out_ + elements;
  logits_ = reinterpret_cast<float*>(carry_ + model::kCarryPlanes * elements);
  ids_ = reinterpret_cast<int*>(logits_ + vocab_);
  next_ = ids_ + capacity_;
  for (auto dir : {"candidates", "loaded"}) fs::create_directories(config_.state_dir / dir);
  auto baseline = config_.state_dir / "candidates/baseline.so";
  fs::copy_file(config_.baseline, baseline.string() + ".tmp", fs::copy_options::overwrite_existing);
  fs::rename(baseline.string() + ".tmp", baseline);
  config_.baseline = baseline;
  CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  try {
    // The driver retains graph launch storage at its high-water mark for
    // the process lifetime. Warm the same two live executables as interleaved
    // A/B, then destroy both before sampling steady-state unload losses.
    if (weights_) {
      CheckCuda(cudaMemsetAsync(ids_, 0, sizeof(int), stream_));
      const auto free_before = FreeBytes();
      auto spec = Spec(0, model::kNumLogicalVisits, 1, 0, true, true);
      spec.is_decode = 1;
      spec.reset_kv = 1;
      {
        auto a = Load(config_.baseline), b = Load(config_.baseline);
        ForwardOptions opts{.use_cuda_graph = 1, .return_logits = 1};
        a->Run(spec, opts);
        b->Run(spec, opts);
      }
      warmup_vram_bytes_ = int64_t(free_before) - int64_t(FreeBytes());
    }
  } catch (...) {
    cudaStreamDestroy(stream_);
    throw;
  }
}
Replay::~Replay() { cudaStreamDestroy(stream_); }
Object Replay::Status() const {
  cudaDeviceProp gpu{};
  CheckCuda(cudaGetDeviceProperties(&gpu, 0));
  size_t free, total;
  CheckCuda(cudaMemGetInfo(&free, &total));
  return {{"pid", getpid()},
          {"gpu", gpu.name},
          {"model_dir", config_.model_dir.string()},
          {"weights_load_ms", weights_load_ms_},
          {"weights_h2d_ms", weights_ ? weights_->transfer_ms() : 0},
          {"warmup_vram_bytes", warmup_vram_bytes_},
          {"weights_bytes", weights_ ? weights_->view().arena_bytes : 0},
          {"kv_bytes", kv_->view().arena_bytes},
          {"kv_seq_len", config_.kv_seq_len},
          {"scratch_bytes", scratch_->view().bytes},
          {"io_bytes", io_->view().bytes},
          {"free_vram_bytes", free},
          {"total_vram_bytes", total},
          {"baseline", config_.baseline.string()},
          {"lazy_cache_entries", cache_.size()}};
}
ForwardRunSpec Replay::Spec(unsigned begin, unsigned end, unsigned length, unsigned pos, bool embed,
                            bool head, unsigned blocks) {
  return {.begin_visit = begin,
          .end_visit = end,
          .run_embed = embed,
          .run_lm_head = head,
          .is_decode = pos != 0,
          .seq_len = length,
          .pos_offset = pos,
          .d_input_ids = ids_,
          .d_hidden_in = hidden_,
          .d_block_in = carry_,
          .num_block_in = blocks,
          .d_hidden_out = out_,
          .d_block_out = carry_,
          .d_logits_out = logits_,
          .d_next_token = next_};
}
Recording Replay::Record(const std::vector<int>& ids, unsigned begin, unsigned end, unsigned step) {
  Recording r;
  r.ids = ids;
  auto baseline = Load(config_.baseline);
  auto& state = kv_->view();
  kv_->Reset(stream_);
  Upload(stream_, ids_, ids);
  for (unsigned i = 0; i < step; ++i) {
    auto run =
        baseline->Run(Spec(0, model::kNumLogicalVisits, r.ids.size(), r.position, true, true));
    r.position += r.ids.size();
    r.ids = {run.next_token_id};
    r.generated.push_back(run.next_token_id);
    Upload(stream_, ids_, r.ids);
  }
  r.blocks = baseline->Run(Spec(0, begin, r.ids.size(), r.position, true, false)).num_block_out;
  auto count = r.ids.size() * width_;
  r.hidden = ReadHidden(count);
  r.carry = Read(carry_, model::kCarryPlanes * count);
  for (unsigned v = begin; v < end; ++v) {
    for (auto [offset, bytes, kind, dtype] :
         VisitSnapshotRanges(state, plan_[v], r.position + r.ids.size()))
      r.state.push_back({offset, kind, dtype,
                         Read(static_cast<const std::byte*>(state.arena_base) + offset, bytes)});
  }
  UploadHidden(r.hidden);
  bool head = begin == 0 && end == model::kNumLogicalVisits;
  auto result = baseline->Run(Spec(begin, end, r.ids.size(), r.position, head, head, r.blocks),
                              {.return_logits = head});
  r.post = ReadPostState(r, result.num_block_out);
  r.reference = ReadHidden(count);
  r.next = result.next_token_id;
  if (head) r.logits = Read(logits_, vocab_);
  return r;
}
std::vector<float> Replay::ReadHidden(size_t count) const {
  auto bits = Read(out_, count);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) out[i] = std::bit_cast<float>(uint32_t(bits[i]) << 16);
  return out;
}
void Replay::UploadHidden(const std::vector<float>& data) {
  std::vector<uint16_t> bits(data.size());
  for (size_t i = 0; i < data.size(); ++i) bits[i] = std::bit_cast<uint32_t>(data[i]) >> 16;
  Upload(stream_, hidden_, bits);
  CheckCuda(cudaStreamSynchronize(stream_));
}
PostState Replay::ReadPostState(const Recording& r, unsigned blocks) const {
  PostState post{blocks, Read(carry_, r.carry.size()), {}};
  const auto* base = static_cast<const std::byte*>(kv_->view().arena_base);
  post.state.reserve(r.state.size());
  for (const auto& slice : r.state)
    post.state.push_back(
        {slice.offset, slice.kind, slice.dtype, Read(base + slice.offset, slice.bytes.size())});
  return post;
}
void Replay::Restore(const Recording& r) {
  Upload(stream_, carry_, r.carry);
  for (const auto& slice : r.state)
    Upload(stream_, static_cast<std::byte*>(kv_->view().arena_base) + slice.offset, slice.bytes);
}
Object Replay::Request(const Json& q) {
  const auto cmd = q.at("command").as<std::string>();
  if (cmd == "status") return Status();
  if (cmd != "run" && cmd != "profile" && cmd != "dump" && cmd != "ab")
    throw std::runtime_error("unknown command");
  const auto before = FreeBytes();
  Object reply;
  try {
    reply = Measure(q, cmd);
  } catch (const std::exception& e) {
    reply["error"] = e.what();
  }
  CheckCuda(cudaDeviceSynchronize());
  const auto after = FreeBytes();
  const size_t loss = before > after ? before - after : 0;
  reply["vram_loss_bytes"] = loss;
  reply["vram_leak_warn"] = loss > (4u << 20);
  if (loss > (4u << 20)) {
    std::string warning = "WARN: forward unload lost " + std::to_string(loss) + " VRAM bytes";
    std::cerr << warning << '\n';
    reply["warning"] = warning;
  }
  return reply;
}
Replay::Measurement Replay::ParseMeasurement(const Json& q, const std::string& cmd) const {
  auto begin = Count(q, "begin_visit", 0, 0, model::kNumLogicalVisits - 1),
       end = Count(q, "end_visit", model::kNumLogicalVisits, begin + 1, model::kNumLogicalVisits);
  const auto name = q.get<std::string>("case", "short_math");
  std::vector<int> ids;
  for (auto& id : q.at("input_ids").as<Array>()) {
    int token = id.number<int>();
    if (token < 0 || static_cast<unsigned>(token) >= model_config_.vocab_size)
      throw std::runtime_error("invalid token id");
    ids.push_back(token);
  }
  if (ids.empty() || ids.size() > capacity_)
    throw std::runtime_error("input exceeds resident I/O capacity");
  auto step = Count(q, "step", 0, 0, config_.kv_seq_len - ids.size());
  auto warmup = Count(q, "warmup", 1, 0, 10000), iters = Count(q, "iters", 5, 1, 10000);
  auto rounds = cmd == "ab" ? Count(q, "rounds", 5, 1, 10000) : 1;
  if (size_t(iters) * rounds > 100000) throw std::runtime_error("too many timed iterations");
  return {begin, end, step, warmup, iters, rounds, name, std::move(ids)};
}
const Recording& Replay::ResolveRecording(const Measurement& m, Object& reply,
                                          Recording& uncached) {
  if (m.e2e()) {
    uncached = Record(m.ids, m.begin, m.end, m.step);
    return uncached;
  }
  Array ids;
  for (auto id : m.ids) ids.emplace_back(id);
  auto key = Encode(Array{m.name, std::move(ids), m.begin, m.end, m.step});
  auto it = cache_.find(key);
  reply["lazy_cache_hit"] = it != cache_.end();
  if (it == cache_.end()) it = cache_.emplace(key, Record(m.ids, m.begin, m.end, m.step)).first;
  return it->second;
}
ForwardRunResult Replay::RunRestored(ForwardModule& module, const ForwardRunSpec& spec,
                                     const ForwardOptions& opts, const Recording& r) {
  Restore(r);
  return module.Run(spec, opts);
}
ForwardRunResult Replay::RunSingle(ForwardModule& a, const ForwardRunSpec& spec,
                                   const ForwardOptions& opts, const Recording& r, unsigned iters,
                                   Object& reply) {
  std::vector<double> times_a;
  times_a.reserve(iters);
  ForwardRunResult last{};
  for (unsigned i = 0; i < iters; ++i) {
    last = RunRestored(a, spec, opts, r);
    times_a.push_back(last.elapsed_ms);
  }
  const double mean = std::accumulate(times_a.begin(), times_a.end(), 0.) / times_a.size();
  double median = Median(times_a), bw = last.logical_bytes / (1e6 * median);
  reply["elapsed_ms"] =
      Object{{"median", median}, {"min", times_a.front()}, {"max", times_a.back()}, {"mean", mean}};
  reply["logical_bytes"] = last.logical_bytes;
  reply["effective_bw_gbs"] = bw;
  reply["sol_pct"] = bw / kRtx5080SolDivisor;
  reply["traffic_basis"] = "model.md 6.1 single-token logical weight budget, mean expert bitrate";
  return last;
}
ForwardRunResult Replay::RunPair(ForwardModule& a, ForwardModule& b, const ForwardRunSpec& spec,
                                 const ForwardOptions& opts, const Recording& r,
                                 const Measurement& m, Object& reply) {
  ForwardRunResult last{};
  Array pairs;
  std::vector<double> round_a, round_b;
  std::vector<float> a_output, a_logits;
  PostState a_state;
  int a_next = -1;
  round_a.reserve(m.rounds);
  round_b.reserve(m.rounds);
  pairs.reserve(m.rounds);
  std::vector<double> group_a(m.iters), group_b(m.iters);
  for (unsigned round = 0; round < m.rounds; ++round) {
    for (unsigned i = 0; i < m.iters; ++i) {
      last = RunRestored(a, spec, opts, r);
      group_a[i] = last.elapsed_ms;
      if (round + 1 == m.rounds && i + 1 == m.iters) {
        a_state = ReadPostState(r, last.num_block_out);
        a_output = ReadHidden(r.hidden.size());
        if (m.e2e()) a_logits = Read(logits_, vocab_);
        a_next = last.next_token_id;
      }
      last = RunRestored(b, spec, opts, r);
      group_b[i] = last.elapsed_ms;
    }
    round_a.push_back(Median(group_a));
    round_b.push_back(Median(group_b));
    pairs.emplace_back(Object{{"a_ms", round_a.back()}, {"b_ms", round_b.back()}});
  }
  double am = Median(round_a), bm = Median(round_b);
  reply["rounds"] = std::move(pairs);
  reply["a_median_ms"] = am;
  reply["b_median_ms"] = bm;
  reply["delta_ms"] = am - bm;
  reply["speedup"] = am / bm;
  reply["state_error"] = CompareState(a_state, ReadPostState(r, last.num_block_out));
  reply["numerical_error"] = ComputeDiff(a_output, ReadHidden(r.hidden.size()));
  if (m.e2e()) {
    reply["logits_error"] = ComputeDiff(a_logits, Read(logits_, vocab_));
    reply["next_token_match"] = a_next == last.next_token_id;
  }
  return last;
}
void Replay::FormatStages(const ForwardRunResult& last, Object& reply) {
  if (last.num_stages) {
    reply["profile_elapsed_ms"] = last.elapsed_ms;
    Array stages;
    for (unsigned i = 0; i < last.num_stages; ++i) {
      auto& t = last.stages[i];
      double bw = t.traffic_bytes / (1e6 * t.elapsed_ms);
      stages.emplace_back(Object{{"name", std::string(t.name, strnlen(t.name, sizeof(t.name)))},
                                 {"elapsed_ms", t.elapsed_ms},
                                 {"pct_total", 100. * t.elapsed_ms / last.elapsed_ms},
                                 {"traffic_bytes", t.traffic_bytes},
                                 {"bw_gbs", bw},
                                 {"sol_pct", bw / kRtx5080SolDivisor}});
    }
    reply["stages"] = std::move(stages);
  }
}
void Replay::Dump(const Json& q, const Measurement& m, const Recording& r,
                  const std::vector<float>& output, Object& reply) {
  Array shape{r.ids.size(), width_};
  std::vector<DumpTensor> tensors{
      {"input_ids", "I32", {r.ids.size()}, std::as_bytes(std::span(r.ids))},
      {"hidden_in", "F32", shape, std::as_bytes(std::span(r.hidden))},
      {"ref_hidden_out", "F32", shape, std::as_bytes(std::span(r.reference))},
      {"cand_hidden_out", "F32", shape, std::as_bytes(std::span(output))}};
  const std::array<uint32_t, 5> layout{m.begin, m.end, r.position, r.blocks,
                                       kv_->view().max_full_seq_len};
  if (q.get("with_state", false)) {
    // Absolute byte offsets below use this full-KV capacity and the loader arena layout.
    const char* names[] = {"begin_visit", "end_visit", "pos_offset", "num_block_in",
                           "max_full_seq_len"};
    for (size_t i = 0; i < layout.size(); ++i)
      tensors.push_back({names[i], "U32", {}, std::as_bytes(std::span(layout).subspan(i, 1))});
    tensors.push_back({"carry",
                       "BF16",
                       {int(model::kCarryPlanes), r.ids.size(), width_},
                       std::as_bytes(std::span(r.carry))});
    for (const auto& slice : r.state)
      tensors.push_back({"state_offset_" + std::to_string(slice.offset),
                         "U8",
                         {slice.bytes.size()},
                         slice.bytes});
  }
  auto dump = WriteDump(q.at("out").as<std::string>(), tensors);
  reply.merge(dump);
}
Object Replay::Measure(const Json& q, const std::string& cmd) {
  auto m = ParseMeasurement(q, cmd);
  bool e2e = m.e2e();
  Object reply;
  using Clock = std::chrono::steady_clock;
  auto elapsed_ms = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  auto record_start = Clock::now();
  Recording uncached;
  const auto& r = ResolveRecording(m, reply, uncached);
  reply["recording_ms"] = elapsed_ms(record_start);
  auto spec = Spec(m.begin, m.end, r.ids.size(), r.position, e2e, e2e, r.blocks);
  UploadHidden(r.hidden);
  Upload(stream_, ids_, r.ids);
  ForwardOptions options{.use_cuda_graph = q.get("use_cuda_graph", true),
                         .use_speculative = q.get("use_speculative", true),
                         .return_logits = e2e,
                         .enable_profiling = cmd == "profile" || q.get("enable_profiling", false)};
  auto load_start = Clock::now();
  auto a = Load(q.at(cmd == "ab" ? "so_a" : "so_path").as<std::string>());
  auto b = cmd == "ab" ? Load(q.at("so_b").as<std::string>()) : std::nullopt;
  reply["module_load_ms"] = elapsed_ms(load_start);
  for (unsigned i = 0; i < m.warmup; ++i) {
    RunRestored(*a, spec, options, r);
    if (b) RunRestored(*b, spec, options, r);
  }
  options.profiler_range = q.get("profiler_range", false);
  auto last = b ? RunPair(*a, *b, spec, options, r, m, reply)
                : RunSingle(*a, spec, options, r, m.iters, reply);
  auto output = ReadHidden(r.hidden.size());
  auto logits = e2e ? Read(logits_, vocab_) : std::vector<float>{};
  if (!b) {
    reply["state_error"] = CompareState(r.post, ReadPostState(r, last.num_block_out));
    reply["numerical_error"] = ComputeDiff(r.reference, output);
    if (e2e) {
      reply["logits_error"] = ComputeDiff(r.logits, logits);
      reply["next_token_id"] = last.next_token_id;
      reply["next_token_match"] = last.next_token_id == r.next;
    }
  }
  FormatStages(last, reply);
  a->ReportGraphStats(reply, b ? "graph_a" : "graph");
  if (b) b->ReportGraphStats(reply, "graph_b");
  auto unload_start = Clock::now();
  b.reset();
  a.reset();
  reply["module_unload_ms"] = elapsed_ms(unload_start);
  CompareGolden(reply, config_.golden_dir, m.name, m.ids, r.generated, spec, e2e ? logits : output,
                width_, last.next_token_id);
  if (cmd == "dump") Dump(q, m, r, output, reply);
  return reply;
}
}  // namespace flint

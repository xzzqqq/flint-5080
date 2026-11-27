#include "forge/replay.h"

#include <unistd.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>

#include "common/cuda_check.h"
#include "common/cuda_profiler.h"
#include "forge/safetensors_writer.h"
#include "loader/kv_layout.h"
#include "loader/safetensors.h"
#include "runtime/config.h"
#include "runtime/sampler.h"

namespace flint {
namespace fs = std::filesystem;
namespace {

constexpr unsigned kLogitRows = 8;

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
    weights_.emplace(WeightsArena::Load(config_.model_dir, nullptr, config_.draft_dir));
    weights_load_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    model_config_ = weights_->view().config;
  }
  plan_ = BuildVisitPlan(model_config_);
  width_ = model_config_.hidden_size;
  vocab_ = model_config_.unpadded_vocab_size;
  bool draft = weights_ && weights_->view().has_draft;
  scratch_.emplace(ScratchArena::Allocate(size_t(config_.scratch_mb) << 20));
  auto chunk =
      std::min<size_t>(model::kSlidingWindow, scratch_->view().bytes / (width_ * sizeof(float)));
  auto io_bytes = [&](size_t rows) {
    return (2 + model::kCarryPlanes + (draft ? 4 : 0)) * rows * width_ * sizeof(uint16_t) +
           kLogitRows * vocab_ * sizeof(float) + (rows + kLogitRows) * sizeof(int);
  };
  if (!config_.kv_seq_len) {
    const size_t free = FreeBytes();
    const size_t overhead = io_bytes(chunk) + RuntimeConfig{}.vram_reserve_bytes;
    config_.kv_seq_len =
        KvStateArena::MaxFullSeqLenForBudget(model_config_, free - std::min(free, overhead), draft);
    if (!config_.kv_seq_len)
      throw std::runtime_error(
          "insufficient VRAM for Forge KV after weights, scratch and workspace reserve");
  }
  capacity_ = std::min<size_t>(chunk, config_.kv_seq_len);
  if (!capacity_) throw std::runtime_error("scratch too small for one token");
  kv_.emplace(KvStateArena::Allocate(model_config_, config_.kv_seq_len, draft));
  size_t elements = size_t(capacity_) * width_;
  io_.emplace(ScratchArena::Allocate(io_bytes(capacity_)));
  hidden_ = static_cast<uint16_t*>(io_->view().device_ptr);
  out_ = hidden_ + elements;
  carry_ = out_ + elements;
  logits_ = reinterpret_cast<float*>(carry_ + model::kCarryPlanes * elements);
  taps_ = reinterpret_cast<uint16_t*>(logits_ + kLogitRows * vocab_);
  ids_ = reinterpret_cast<int*>(taps_ + (draft ? 4 * elements : 0));
  next_ = ids_ + capacity_;
  for (auto dir : {"candidates", "loaded"}) fs::create_directories(config_.state_dir / dir);
  auto baseline = config_.state_dir / "candidates/baseline.so";
  baseline_preserved_ = fs::exists(baseline) && !config_.reset_baseline;
  if (!baseline_preserved_) {
    fs::copy_file(config_.baseline, baseline.string() + ".tmp",
                  fs::copy_options::overwrite_existing);
    fs::rename(baseline.string() + ".tmp", baseline);
  }
  config_.baseline = baseline;
  CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  try {
    // The driver retains graph launch storage at its high-water mark for
    // the process lifetime. Warm both replay paths on two live A/B modules
    // before sampling steady-state unload losses.
    if (weights_) {
      unsigned batch_rows = std::min(capacity_, kLogitRows);
      CheckCuda(cudaMemsetAsync(ids_, 0, batch_rows * sizeof(int), stream_));
      const auto free_before = FreeBytes();
      auto spec = Spec(0, model::kNumLogicalVisits, 1, 0, true, true);
      spec.is_decode = 1;
      spec.reset_kv = 1;
      {
        auto a = Load(config_.baseline), b = Load(config_.baseline);
        ForwardOptions opts{.use_cuda_graph = 1, .return_logits = 1};
        a->Run(spec, opts);
        b->Run(spec, opts);
        // KLD also exercises the batched target kernels when no draft is loaded.
        auto batch = Spec(0, model::kNumLogicalVisits, batch_rows, 0, true, true);
        batch.all_token_logits = batch.reset_kv = 1;
        a->Run(batch, {.return_logits = 1});
        if (draft && capacity_ >= 3) {
          spec.d_taps_out = taps_;
          unsigned len = std::min(capacity_ - 1, kLogitRows);
          for (auto* module : {&*a, &*b}) {
            int anchor = module->Run(spec).next_token_id;
            module->StoreContext(taps_, 1, 0);
            auto fixed = FreezeProposal(*module, anchor, 1, len);
            RunSpecRound(*module, anchor, 1, len, fixed, false);
            anchor = module->Run(spec).next_token_id;
            module->StoreContext(taps_, 1, 0);
            RunSpecRound(*module, anchor, 1, len, {}, false);
          }
        }
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
          {"draft_model_dir", config_.draft_dir.string()},
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
          {"baseline_preserved", baseline_preserved_},
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
  ResetKv();
  Upload(stream_, ids_, ids);
  for (unsigned i = 0; i < step; ++i) {
    auto run =
        baseline->Run(Spec(0, model::kNumLogicalVisits, r.ids.size(), r.position, true, true));
    r.position += r.ids.size();
    r.ids = {run.next_token_id};
    r.generated.push_back(run.next_token_id);
    Upload(stream_, ids_, r.ids);
  }
  RecordTail(r, *baseline, begin, end, false);
  return r;
}

void Replay::RecordTail(Recording& r, ForwardModule& baseline, unsigned begin, unsigned end,
                        bool resident_prefix) {
  const auto& state = kv_->view();
  r.blocks = baseline.Run(Spec(0, begin, r.ids.size(), r.position, true, false)).num_block_out;
  auto count = r.ids.size() * width_;
  r.hidden = ReadHidden(count);
  r.carry = Read(carry_, model::kCarryPlanes * count);
  for (unsigned v = begin; v < end; ++v) {
    for (auto [offset, bytes, kind, dtype] :
         VisitSnapshotRanges(state, plan_[v], r.position + r.ids.size())) {
      if (resident_prefix && (kind == KvSliceKind::kKey || kind == KvSliceKind::kValue)) {
        // The real model appends one row. The synthetic forward updates slot zero.
        auto capacity = plan_[v].is_full_attention ? state.max_full_seq_len : state.swa_window;
        bytes = model::kKvDim * sizeof(uint16_t);
        if (weights_) offset += (r.position % capacity) * bytes;
      }
      r.state.push_back({offset, kind, dtype,
                         Read(static_cast<const std::byte*>(state.arena_base) + offset, bytes)});
    }
  }
  UploadHidden(r.hidden);
  bool head = begin == 0 && end == model::kNumLogicalVisits;
  auto result = baseline.Run(Spec(begin, end, r.ids.size(), r.position, head, head, r.blocks),
                             {.return_logits = head});
  r.post = ReadPostState(r, result.num_block_out);
  r.reference = ReadHidden(count);
  r.next = result.next_token_id;
  if (head) r.logits = Read(logits_, vocab_);
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
  if (cmd == "clear_cache") {
    cache_.clear();
    prefix_.reset();
    return {};
  }
  if (cmd != "run" && cmd != "profile" && cmd != "dump" && cmd != "ab" && cmd != "kld")
    throw std::runtime_error("unknown command");
  const auto before = FreeBytes();
  Object reply;
  try {
    reply = cmd == "kld" ? Kld(q) : Measure(q, cmd);
  } catch (const std::exception& e) {
    prefix_.reset();
    reply["error"] = e.what();
  }
  CheckCuda(cudaDeviceSynchronize());
  const auto after = FreeBytes();
  const size_t loss = before > after ? before - after : 0;
  if (cmd == "kld") reply["unload_loss_bytes"] = loss;
  reply["vram_loss_bytes"] = loss;
  reply["vram_leak_warn"] = loss > (4u << 20);
  if (loss > (4u << 20)) {
    std::string warning = "WARN: forward unload lost " + std::to_string(loss) + " VRAM bytes";
    std::cerr << warning << '\n';
    reply["warning"] = warning;
  }
  return reply;
}
Object Replay::Kld(const Json& q) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  if (!weights_) throw std::runtime_error("KLD requires a loaded model");
  const unsigned limit = std::min(capacity_, kLogitRows);
  const auto chunk = Count(q, "chunk_size", limit, 1, limit);
  ForwardOptions options{.use_cuda_graph = q.get("use_cuda_graph", false), .return_logits = 1};
  struct Case {
    std::string name;
    std::vector<int> ids;
    unsigned eval_start;
    std::vector<float> reference;
    KldAccumulator metrics;
  };
  std::vector<Case> cases;
  for (const auto& item : q.at("cases").as<Array>()) {
    Case c{item.at("name").as<std::string>(), {}, item.at("eval_start").number<unsigned>(), {}, {}};
    for (const auto& id : item.at("token_ids").as<Array>()) {
      int token = id.number<int>();
      if (token < 0 || unsigned(token) >= model_config_.vocab_size)
        throw std::invalid_argument("invalid KLD token id");
      c.ids.push_back(token);
    }
    if (c.ids.size() > config_.kv_seq_len || c.eval_start >= c.ids.size())
      throw std::invalid_argument("invalid KLD sequence length or eval_start");
    cases.push_back(std::move(c));
  }
  if (cases.empty()) throw std::invalid_argument("KLD requires cases");
  auto ref = q.at("so_ref").as<std::string>(), cand = q.at("so_cand").as<std::string>();
  // Load each DSO once. Only evaluated reference rows survive the module switch.
  for (bool reference : {true, false}) {
    auto module = Load(reference ? ref : cand);
    for (auto& c : cases) {
      if (reference) c.reference.resize((c.ids.size() - c.eval_start) * vocab_);
      ResetKv();
      for (unsigned pos = 0; pos < c.ids.size();) {
        unsigned rows = std::min<size_t>(chunk, c.ids.size() - pos);
        CheckCuda(cudaMemcpyAsync(ids_, c.ids.data() + pos, rows * sizeof(int),
                                  cudaMemcpyHostToDevice, stream_));
        auto spec = Spec(0, model::kNumLogicalVisits, rows, pos, true, true);
        spec.all_token_logits = 1;
        module->Run(spec, options);
        unsigned first = std::max(pos, c.eval_start);
        if (first < pos + rows) {
          auto values = std::span(c.reference)
                            .subspan(size_t(first - c.eval_start) * vocab_,
                                     size_t(pos + rows - first) * vocab_);
          const float* src = logits_ + size_t(first - pos) * vocab_;
          if (reference)
            CheckCuda(cudaMemcpy(values.data(), src, values.size_bytes(), cudaMemcpyDeviceToHost));
          else
            AccumulateKld(values, Read(src, values.size()), vocab_, c.metrics);
        }
        pos += rows;
      }
      if (!reference) std::vector<float>().swap(c.reference);
    }
  }
  auto report = [](const KldSummary& s) -> Object {
    return {{"num_positions", s.num_positions},
            {"mean_kl", s.mean_kl},
            {"p95_kl", s.p95_kl},
            {"max_kl", s.max_kl},
            {"top1_match_rate", double(s.top1_matches) / s.num_positions},
            {"max_abs_logit_diff", s.max_abs_logit_diff}};
  };
  Array per_case;
  KldAccumulator total;
  for (const auto& c : cases) {
    auto row = report(c.metrics.Summary());
    row["name"] = c.name;
    row["eval_start"] = c.eval_start;
    per_case.emplace_back(std::move(row));
    total.position_kl.insert(total.position_kl.end(), c.metrics.position_kl.begin(),
                             c.metrics.position_kl.end());
    total.top1_matches += c.metrics.top1_matches;
    total.max_abs_logit_diff = std::max(total.max_abs_logit_diff, c.metrics.max_abs_logit_diff);
  }
  auto reply = report(total.Summary());
  reply["ref"] = ref;
  reply["cand"] = cand;
  reply["chunk_size"] = chunk;
  reply["cases"] = std::move(per_case);
  reply["elapsed_ms"] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
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
  auto context_len = q.get<unsigned>("context_len", 0);
  if (ids.empty() || (!context_len && ids.size() > capacity_))
    throw std::runtime_error("input exceeds resident I/O capacity");
  auto step = q.get<unsigned>("step", 0);
  if (context_len) step = std::max(step, 1u);
  if (!context_len && ids.size() + size_t(step) > kv_->view().max_full_seq_len)
    throw std::invalid_argument("decode exceeds KV capacity; increase --kv-seq-len");
  auto warmup = Count(q, "warmup", 1, 0, 10000), iters = Count(q, "iters", 5, 1, 10000);
  auto rounds = cmd == "ab" ? Count(q, "rounds", 5, 1, 10000) : 1;
  if (size_t(iters) * rounds > 100000) throw std::runtime_error("too many timed iterations");
  return {begin, end,           step,        warmup,
          iters, rounds,        context_len, Count(q, "verify_len", 8, 2, 8),
          name,  std::move(ids)};
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
  if (last.num_stages > FLINT_MAX_STAGE_TIMINGS) throw std::runtime_error("invalid stage count");
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
std::vector<int> Replay::FreezeProposal(ForwardModule& baseline, int anchor, unsigned pos,
                                        unsigned len) {
  const auto proposal = baseline.Draft({anchor, int(pos), 0, 0, 0});
  std::vector<int> block(len);
  block.front() = anchor;
  std::copy_n(proposal.draft_ids, len - 1, block.begin() + 1);
  return block;
}
Replay::SpecRound Replay::RunSpecRound(ForwardModule& candidate, int anchor, unsigned pos,
                                       unsigned len, const std::vector<int>& fixed_block,
                                       bool profiler_range, bool enable_profiling) {
  using Clock = std::chrono::steady_clock;
  SpecRound round{};
  if (!fixed_block.empty()) Upload(stream_, ids_, fixed_block);
  CheckCuda(cudaStreamSynchronize(stream_));
  if (profiler_range) CheckCuda(cudaProfilerStart());
  try {
    auto start = Clock::now();
    if (fixed_block.empty()) {
      const auto result = candidate.SpecRound({.input = {anchor, int(pos), len},
                                               .d_input_ids = ids_,
                                               .d_hidden_out = out_,
                                               .d_logits_out = logits_,
                                               .d_next_token = next_,
                                               .d_taps_out = taps_,
                                               .cache_graph = 1});
      auto count = 1 + result.output.accepted_draft_count;
      round.emitted.assign(result.output.tokens, result.output.tokens + count);
      round.body_ms = result.elapsed_ms;
    } else {
      auto spec = Spec(0, model::kNumLogicalVisits, len, pos, true, true);
      spec.d_taps_out = taps_;
      auto opts = candidate.speculative_options;
      opts.enable_profiling = enable_profiling;
      round.verify_result = candidate.Verify(spec, opts);
      round.verified = Read(next_, len);
      uint64_t seed = 0;
      auto result = VerifySpeculative(std::span(fixed_block).subspan(1), round.verified, {}, {}, {},
                                      vocab_, 0, 0, 0, seed);
      round.emitted.assign(fixed_block.begin() + 1,
                           fixed_block.begin() + 1 + result.accepted_draft_count);
      round.emitted.push_back(result.bonus_token);
      round.body_ms = round.verify_result.elapsed_ms;
    }
    auto verified = Clock::now();
    candidate.Commit(pos, len, round.emitted.size());
    candidate.StoreContext(taps_, round.emitted.size(), pos);
    CheckCuda(cudaStreamSynchronize(stream_));
    auto stop = Clock::now();
    round.elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    round.commit_ms = std::chrono::duration<double, std::milli>(stop - verified).count();
  } catch (...) {
    if (profiler_range) cudaProfilerStop();
    throw;
  }
  if (profiler_range) CheckCuda(cudaProfilerStop());
  return round;
}
void Replay::ResetKv() {
  prefix_.reset();
  kv_->Reset(stream_);
}

void Replay::PreparePrefix(const Measurement& m, bool speculative, Object& reply) {
  const unsigned length = m.context_len ? m.context_len : m.ids.size();
  const unsigned tail = speculative ? m.verify_len : 1;
  if (capacity_ < tail || uint64_t(length) + m.step - 1 + tail > kv_->view().max_full_seq_len)
    throw std::invalid_argument(
        "prefix and decode exceed available capacity; increase --kv-seq-len");
  Array ids;
  for (auto id : m.ids) ids.emplace_back(id);
  auto key = Encode(Array{fs::last_write_time(config_.baseline).time_since_epoch().count(),
                          std::move(ids), m.context_len, m.step, m.begin, m.end, speculative});
  bool hit = m.context_len && prefix_ && prefix_->key == key;
  reply["prefix_cache_hit"] = hit;
  reply["context_len"] = m.context_len;
  if (hit) return;
  ResetKv();
  auto baseline = Load(config_.baseline);
  auto& prefix = prefix_.emplace();
  prefix.key = std::move(key);
  const auto context_start = model::DraftContextStart(length + m.step - 1);
  unsigned rows = 0;
  auto advance = [&](const std::vector<int>& tokens, bool head) {
    rows = tokens.size();
    Upload(stream_, ids_, tokens);
    auto spec = Spec(0, model::kNumLogicalVisits, rows, prefix.position, true, head);
    const auto lo = std::max(prefix.position, context_start);
    const auto end = prefix.position + rows;
    if (speculative && lo < end) spec.d_taps_out = taps_;
    auto result = baseline->Run(spec);
    if (spec.d_taps_out)
      baseline->StoreContext(taps_ + size_t(lo - prefix.position) * 4 * width_, end - lo, lo);
    prefix.position += rows;
    if (head) prefix.generated.push_back(result.next_token_id);
  };
  unsigned chunk = m.context_len ? std::min(512u, capacity_) : length;
  while (prefix.position < length) {
    std::vector<int> tokens(std::min(chunk, length - prefix.position));
    for (size_t i = 0; i < tokens.size(); ++i)
      tokens[i] = m.ids[(prefix.position + i) % m.ids.size()];
    advance(tokens, prefix.position + tokens.size() == length);
  }
  for (unsigned i = 1; i < m.step; ++i) advance({prefix.generated.back()}, true);
  if (speculative) {
    CheckCuda(cudaStreamSynchronize(stream_));
    SaveSpecState(prefix);
  } else {
    auto& r = prefix.recording;
    r.ids = {prefix.generated.back()};
    r.position = prefix.position;
    Upload(stream_, ids_, r.ids);
    RecordTail(r, *baseline, m.begin, m.end, true);
  }
}

void Replay::SaveSpecState(Prefix& prefix) {
  const unsigned pos = prefix.position;
  const auto& kv = kv_->view();
  auto sizes = ComputeKvArenaSizes(model_config_, true);
  auto save = [&](void* ptr, size_t bytes) {
    auto* address = static_cast<std::byte*>(ptr);
    prefix.mutable_state.emplace_back(address, Read(address, bytes));
  };
  save(kv.kv_conv_state, sizes.kv_conv);
  save(kv.sublayer_conv_state, sizes.sublayer_conv);
  auto save_ring = [&](void* base, unsigned window, unsigned planes) {
    if (pos + 8 <= window) return;
    size_t stride = model::kKvDim * sizeof(uint16_t);
    unsigned first = pos % window, count = std::min(8u, window - first);
    for (unsigned i = 0; i < planes; ++i) {
      auto plane = static_cast<std::byte*>(base) + i * window * stride;
      save(plane + first * stride, count * stride);
      if (count < 8) save(plane, (8 - count) * stride);
    }
  };
  save_ring(kv.swa_kv, kv.swa_window, 2 * kv.num_swa_visits);
  save_ring(kv.draft_swa_kv, kv.draft_swa_window, 10);
}
Object Replay::MeasureSpeculative(const Json& q, const std::string& cmd, const Measurement& m) {
  Object reply;
  PreparePrefix(m, true, reply);
  const auto& prefix = *prefix_;
  const bool enable_profiling =
      (cmd == "profile" && !q.get("profiler_range", false)) || q.get("enable_profiling", false);
  const bool fixed = cmd == "dump" || enable_profiling || q.get("fixed_proposal", false);
  std::vector<int> block;
  if (fixed) {
    for (auto& [ptr, bytes] : prefix.mutable_state) Upload(stream_, ptr, bytes);
    auto baseline = Load(config_.baseline);
    block = FreezeProposal(*baseline, prefix.generated.back(), prefix.position, m.verify_len);
  }
  auto a = Load(q.at(cmd == "ab" ? "so_a" : "so_path").as<std::string>());
  auto b = cmd == "ab" ? Load(q.at("so_b").as<std::string>()) : std::nullopt;
  a->speculative_options.use_cuda_graph = q.get("use_cuda_graph", true);
  if (b) b->speculative_options = a->speculative_options;
  SpecRound result;
  bool profiler_range = false;
  auto round = [&](ForwardModule& candidate) {
    for (auto& [ptr, bytes] : prefix.mutable_state) Upload(stream_, ptr, bytes);
    result = RunSpecRound(candidate, prefix.generated.back(), prefix.position, m.verify_len, block,
                          profiler_range, enable_profiling);
    return result.elapsed_ms;
  };
  for (unsigned i = 0; i < m.warmup; ++i) {
    round(*a);
    if (b) round(*b);
  }
  profiler_range = q.get("profiler_range", false);
  std::vector<double> times, paired, body_times, commit_times;
  Array pairs;
  for (unsigned j = 0; j < m.rounds; ++j) {
    std::vector<double> group_a, group_b;
    for (unsigned i = 0; i < m.iters; ++i) {
      group_a.push_back(round(*a));
      body_times.push_back(result.body_ms);
      commit_times.push_back(result.commit_ms);
      if (b) {
        auto first = result.emitted;
        group_b.push_back(round(*b));
        RecordSpeculativePair(reply, j, i, first, result.emitted);
      }
    }
    times.insert(times.end(), group_a.begin(), group_a.end());
    if (b) {
      paired.insert(paired.end(), group_b.begin(), group_b.end());
      pairs.emplace_back(Object{{"a_ms", Median(group_a)}, {"b_ms", Median(group_b)}});
    }
  }
  auto median = Median(times);
  reply["elapsed_ms"] =
      Object{{"median", median},
             {"min", times.front()},
             {"max", times.back()},
             {"mean", std::accumulate(times.begin(), times.end(), 0.) / times.size()}};
  if (b) {
    auto second = Median(paired);
    reply["rounds"] = std::move(pairs);
    reply["a_median_ms"] = median;
    reply["b_median_ms"] = second;
    reply["speedup"] = median / second;
    reply["delta_ms"] = median - second;
  }
  a->ReportGraphStats(reply, b ? "graph_a" : "graph");
  if (b) b->ReportGraphStats(reply, "graph_b");
  auto& emitted = result.emitted;
  reply["spec_accepted"] = emitted.size() - 1;
  reply["tokens_emitted"] = emitted.size();
  reply["verify_len"] = m.verify_len;
  reply["spec_mode"] = fixed ? "fixed_proposal" : "merged";
  reply[fixed ? "verify_ms" : "round_ms"] = Median(body_times);
  reply["commit_and_store_ms"] = Median(commit_times);
  if (enable_profiling) {
    FormatStages(result.verify_result, reply);
    reply["verify_logical_bytes"] = result.verify_result.logical_bytes;
  }
  Array ids;
  for (int token : emitted) ids.emplace_back(token);
  reply["emitted_ids"] = std::move(ids);
  if (!m.context_len && !config_.golden_dir.empty() &&
      m.name.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.") ==
          std::string::npos) {
    auto path = config_.golden_dir / ("e2e_" + m.name + ".safetensors");
    if (fs::exists(path)) {
      auto file = SafetensorsFile::Open(path);
      auto prompt = file.Find("input_ids"), expected = file.Find("generated_ids");
      if (prompt && expected && prompt->dtype == TensorDType::kI32 &&
          expected->dtype == TensorDType::kI32 && prompt->nbytes == m.ids.size() * sizeof(int) &&
          !std::memcmp(file.Bytes(*prompt).data(), m.ids.data(), prompt->nbytes)) {
        reply["golden"] =
            CompareSpeculativeGolden(file.Bytes(*expected), prefix.generated, emitted);
      }
    }
  }
  if (cmd == "dump") {
    auto dump =
        WriteDump(q.at("out").as<std::string>(),
                  {{"verify_input_ids", "I32", {m.verify_len}, std::as_bytes(std::span(block))},
                   {"target_ids", "I32", {m.verify_len}, std::as_bytes(std::span(result.verified))},
                   {"emitted_ids", "I32", {emitted.size()}, std::as_bytes(std::span(emitted))}});
    reply.merge(dump);
  }
  return reply;
}
Object Replay::Measure(const Json& q, const std::string& cmd) {
  auto m = ParseMeasurement(q, cmd);
  if (!m.context_len) prefix_.reset();
  bool e2e = m.e2e();
  if (e2e && m.step && weights_ && weights_->view().has_draft && q.get("use_speculative", true))
    return MeasureSpeculative(q, cmd, m);
  Object reply{{"context_len", m.context_len}, {"prefix_cache_hit", false}};
  using Clock = std::chrono::steady_clock;
  auto elapsed_ms = [](Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  };
  auto record_start = Clock::now();
  Recording uncached;
  if (m.context_len) PreparePrefix(m, false, reply);
  const auto& r = m.context_len ? prefix_->recording : ResolveRecording(m, reply, uncached);
  reply["recording_ms"] = elapsed_ms(record_start);
  auto spec = Spec(m.begin, m.end, r.ids.size(), r.position, e2e, e2e, r.blocks);
  UploadHidden(r.hidden);
  Upload(stream_, ids_, r.ids);
  ForwardOptions options{
      .use_cuda_graph = q.get("use_cuda_graph", true),
      .use_speculative = q.get("use_speculative", true),
      .return_logits = e2e,
      .enable_profiling = (cmd == "profile" && !q.get("profiler_range", false)) ||
                          q.get("enable_profiling", false)};
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
  if (!m.context_len)
    CompareGolden(reply, config_.golden_dir, m.name, m.ids, r.generated, spec,
                  e2e ? logits : output, width_, last.next_token_id);
  if (cmd == "dump") Dump(q, m, r, output, reply);
  return reply;
}
}  // namespace flint

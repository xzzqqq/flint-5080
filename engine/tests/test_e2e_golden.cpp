#include <dlfcn.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "forge/forward_abi.h"
#include "loader/arenas.h"
#include "loader/safetensors.h"
#include "ops/scratch_workspace.h"

namespace {
using namespace flint;
using ops_detail::CheckCuda;
void Require(bool ok, const std::string& reason) {
  if (!ok) throw std::runtime_error(reason);
}
struct Module {
  void* handle;
  explicit Module(const std::string& path) : handle(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)) {
    if (!handle) throw std::runtime_error(dlerror());
  }
  ~Module() { dlclose(handle); }
  template <class F>
  F symbol(const char* name) {
    auto p = dlsym(handle, name);
    if (!p) throw std::runtime_error(dlerror());
    return reinterpret_cast<F>(p);
  }
};
struct Runner {
  Module module;
  decltype(&flint_forward_run_layers) run;
  decltype(&flint_forward_run_layers_capture) inspect;
  std::unique_ptr<ForwardContext, decltype(&flint_forward_destroy)> ctx;
  Runner(const std::string& path, const ModelWeightsView* w, const KvStateArenaView* kv,
         const ScratchArenaView& scratch)
      : module(path),
        run(module.symbol<decltype(run)>("flint_forward_run_layers")),
        inspect(module.symbol<decltype(inspect)>("flint_forward_run_layers_capture")),
        ctx(nullptr, module.symbol<decltype(&flint_forward_destroy)>("flint_forward_destroy")) {
    Require(module.symbol<decltype(&flint_forward_abi_version)>("flint_forward_abi_version")() == 2,
            "ABI v2 required");
    ctx.reset(module.symbol<decltype(&flint_forward_create)>("flint_forward_create")(
        w, kv, &scratch, nullptr));
    Require(bool(ctx), "create context");
  }
  ForwardRunResult operator()(const ForwardRunSpec& s, const ForwardCapture* cap = nullptr) {
    ForwardOptions opts{.use_cuda_graph = 1, .return_logits = s.run_lm_head};
    ForwardRunResult r{};
    CheckCuda(cap ? inspect(ctx.get(), &s, &opts, &r, cap) : run(ctx.get(), &s, &opts, &r));
    CheckCuda(r.status);
    return r;
  }
};
template <class T>
std::vector<T> Read(const void* p, size_t count) {
  std::vector<T> v(count);
  CheckCuda(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
template <class T>
std::vector<T> Tensor(const SafetensorsFile& f, const std::string& name, TensorDType dtype) {
  auto e = f.Find(name);
  Require(e && e->dtype == dtype, "invalid golden " + name);
  std::vector<T> v(e->nbytes / sizeof(T));
  std::memcpy(v.data(), f.Bytes(*e).data(), e->nbytes);
  return v;
}
std::vector<int> Top5(const std::vector<float>& x) {
  std::vector<int> ids(x.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(),
                    [&](int a, int b) { return x[a] != x[b] ? x[a] > x[b] : a < b; });
  ids.resize(5);
  return ids;
}
bool Compare(const SafetensorsFile& f, const std::string& key, const void* ptr, size_t count,
             bool logits = false) {
  std::vector<float> x, y;
  if (logits) {
    x = Read<float>(ptr, count);
    y = Tensor<float>(f, key, TensorDType::kF32);
  } else {
    auto to_float = [](const std::vector<uint16_t>& v) {
      std::vector<float> a(v.size());
      std::transform(v.begin(), v.end(), a.begin(),
                     [](uint16_t h) { return std::bit_cast<float>(uint32_t(h) << 16); });
      return a;
    };
    x = to_float(Read<uint16_t>(ptr, count));
    y = to_float(Tensor<uint16_t>(f, key, TensorDType::kBF16));
  }
  Require(x.size() == y.size(), "shape " + key);
  double mx = 0, ss = 0, xx = 0, yy = 0, xy = 0;
  for (size_t i = 0; i < count; ++i) {
    double a = x[i], b = y[i], d = a - b;
    Require(std::isfinite(d), "nonfinite " + key);
    mx = std::max(mx, std::abs(d));
    ss += d * d;
    xx += a * a;
    yy += b * b;
    xy += a * b;
  }
  double cos = xy / std::sqrt(xx * yy);
  std::cout << key << " max=" << mx << " rmse=" << std::sqrt(ss / count) << " cosine=" << cos;
  // Allow normal BF16 rounding while catching both direction and scale errors.
  bool ok = cos >= 0.9999 && ss <= 1e-4 * yy;
  if (logits) {
    auto a = Top5(x), b = Top5(y);
    int common = 0;
    for (int id : a) common += std::count(b.begin(), b.end(), id);
    std::cout << " top1=" << (a.front() == b.front()) << " top5=" << common << "/5";
    ok &= a.front() == b.front();
  }
  std::cout << std::endl;
  return ok;
}
bool Case(Runner& run, const ModelConfig& c, const KvStateArenaView& kv,
          const SafetensorsFile& file, const std::string& name) {
  auto ids = Tensor<int>(file, "input_ids", TensorDType::kI32);
  auto gold = Tensor<int>(file, "generated_ids", TensorDType::kI32);
  auto t = unsigned(ids.size()), vocab = c.unpadded_vocab_size;
  size_t n = size_t(t) * c.hidden_size;
  auto io = ScratchArena::Allocate(2 * n * (1 + 8 + 53) + vocab * 4 + t * 4 + 32768);
  ops_detail::ScratchWorkspace ws(io.view());
  auto hidden = ws.Take<uint16_t>(n), carry = ws.Take<uint16_t>(8 * n);
  ForwardCapture cap{};
  cap.embed = ws.Take<uint16_t>(n);
  for (auto& step : cap.steps) step = ws.Take<uint16_t>(n);
  for (auto& tap : cap.taps) tap = ws.Take<uint16_t>(n);
  auto logits = ws.Take<float>(vocab);
  auto input = ws.Take<int>(t), next = ws.Take<int>(1);
  unsigned blocks = 0;
  auto upload = [&](const int* p, size_t len) {
    CheckCuda(cudaMemcpy(input, p, len * sizeof(int), cudaMemcpyHostToDevice));
  };
  upload(ids.data(), t);
  ForwardRunSpec pre{.end_visit = 48,
                     .run_embed = 1,
                     .run_lm_head = 1,
                     .reset_kv = 1,
                     .seq_len = t,
                     .d_input_ids = input,
                     .d_hidden_in = hidden,
                     .d_block_in = carry,
                     .d_hidden_out = hidden,
                     .d_block_out = carry,
                     .num_block_out = &blocks,
                     .d_logits_out = logits,
                     .d_next_token = next};
  std::cout << "case " << name << std::endl;
  auto first = run(pre, &cap);
  bool ok = Compare(file, "prefill.embed", cap.embed, n);
  for (int v = 0; v < 48; ++v)
    ok &= Compare(file, "prefill.step_" + std::to_string(v), cap.steps[v], n);
  int i = 0;
  for (int v : {5, 23, 41, 48})
    ok &= Compare(file, "prefill.mtp_tap_" + std::to_string(v), cap.taps[i++], n);
  ok &= Compare(file, "prefill.last_logits", logits, vocab, true);
  if (name == "short_math") {
    const auto full_logits = Read<uint32_t>(logits, vocab);
    const auto full_state = Read<std::byte>(kv.arena_base, kv.arena_bytes);
    const auto full_carry = Read<uint16_t>(carry, FLINT_CARRY_PLANES * n);
    auto replay = [&](std::span<const unsigned> ends) {
      auto staged = pre;
      staged.run_lm_head = 0;
      for (unsigned end : ends) {
        staged.end_visit = end;
        run(staged);
        Require(Read<uint16_t>(hidden, n) == Read<uint16_t>(cap.steps[end - 1], n),
                "staged hidden mismatch at " + std::to_string(end));
        if (ends.size() == 5)
          std::cout << "staged " << staged.begin_visit << ".." << end << " exact" << std::endl;
        staged.run_embed = staged.reset_kv = 0;
        staged.begin_visit = end;
        staged.num_block_in = blocks;
      }
      staged.run_lm_head = 1;
      run(staged);
      Require(Read<uint32_t>(logits, vocab) == full_logits &&
                  Read<uint16_t>(carry, FLINT_CARRY_PLANES * n) == full_carry &&
                  Read<std::byte>(kv.arena_base, kv.arena_bytes) == full_state,
              "staged final logits/carry/KV mismatch");
    };
    replay(std::array<unsigned, 5>{6, 18, 30, 42, 48});
    std::array<unsigned, 48> singles;
    std::iota(singles.begin(), singles.end(), 1);
    replay(singles);
    std::cout << "48 single-visit ranges exact; staged carry and KV exact" << std::endl;
  }
  auto decode = pre;
  decode.run_embed = 1;
  decode.reset_kv = 0;
  decode.seq_len = 1;
  decode.is_decode = 1;
  std::vector<int> actual{first.next_token_id};
  double elapsed = 0;
  auto step = [&](int token, size_t idx, bool capture) {
    upload(&token, 1);
    decode.pos_offset = t + idx;
    return run(decode, capture ? &cap : nullptr);
  };
  for (size_t j = 1; j < gold.size(); ++j) {
    auto r = step(actual.back(), j - 1, j == 1);
    actual.push_back(r.next_token_id);
    elapsed += r.elapsed_ms;
    if (j == 1) {
      for (int v = 0; v < 48; ++v)
        ok &= Compare(file, "decode_0.step_" + std::to_string(v), cap.steps[v], c.hidden_size);
      ok &= Compare(file, "decode_0.logits", logits, vocab, true);
    }
  }
  size_t hits = std::inner_product(actual.begin(), actual.end(), gold.begin(), size_t(0),
                                   std::plus<>(), std::equal_to<>());
  std::cout << "greedy " << hits << "/" << gold.size() << " prefill_ms=" << first.elapsed_ms
            << " decode_ms=" << elapsed / (gold.size() - 1) << " ids=";
  for (auto id : actual) std::cout << id << ' ';
  std::cout << std::endl;
  if (hits != gold.size()) {
    upload(ids.data(), t);
    size_t forced = run(pre).next_token_id == gold[0];
    for (size_t j = 1; j < gold.size(); ++j) {
      auto r = step(gold[j - 1], j - 1, false);
      if (r.next_token_id == gold[j])
        ++forced;
      else {
        auto rank = Tensor<float>(file, "decode_" + std::to_string(j - 1) + ".top64_logits",
                                  TensorDType::kF32);
        std::cout << "forced mismatch position=" << j << " expected=" << gold[j]
                  << " actual=" << r.next_token_id << " golden_margin=" << rank[0] - rank[1]
                  << std::endl;
      }
    }
    std::cout << "teacher_forced " << forced << "/" << gold.size() << std::endl;
    ok = false;
  }
  return ok;
}
size_t Free() {
  size_t avail, total;
  CheckCuda(cudaMemGetInfo(&avail, &total));
  return avail;
}
void Smoke(const std::string& path, bool exercise_gpu) {
  ModelConfig c{.num_hidden_layers = 24,
                .n_layers_per_attn_res_block = 6,
                .loop_core_start = 6,
                .loop_core_end = 18,
                .loop_num_passes = 3};
  for (int i = 5; i < 24; i += 6) c.is_full_layer[i] = 1;
  auto plan = BuildVisitPlan(c);
  int full = 0, swa = 0;
  for (unsigned v = 0; v < 48; ++v) {
    auto p = plan[v];
    int layer = v < 18 ? v : v < 42 ? 6 + (v - 18) % 12 : v - 24;
    Require(p.visit_idx == v && p.physical_layer == layer &&
                p.pass_idx == (v < 6 || v >= 42 ? 0 : (v - 6) / 12) &&
                p.has_messages == (v >= 18 && v < 42) && p.is_block_boundary == (layer % 6 == 0) &&
                p.is_full_attention == (layer % 6 == 5) &&
                p.kv_slot_idx == (p.is_full_attention ? full++ : swa++),
            "visit plan");
  }
  if (!exercise_gpu) {
    // Fast ABI probe, including the result contract for an uninitialized host.
    // forge_tests exercises the full --no-model graph and eager execution in CTest.
    Module module(path);
    Require(module.symbol<decltype(&flint_forward_abi_version)>("flint_forward_abi_version")() == 2,
            "forward ABI");
    ForwardRunSpec spec{.end_visit = 48, .run_embed = 1, .seq_len = 1};
    ForwardOptions opts{};
    ForwardRunResult result{};
    auto run = module.symbol<decltype(&flint_forward_run_layers)>("flint_forward_run_layers");
    Require(run(nullptr, &spec, &opts, &result) == cudaErrorInvalidValue &&
                result.status == cudaErrorInvalidValue && result.next_token_id == -1 &&
                result.num_stages == 0,
            "uninitialized forward context");
    module.symbol<decltype(&flint_forward_destroy)>("flint_forward_destroy")(nullptr);
    return;
  }
  auto scratch = ScratchArena::Allocate(32768);
  auto io = ScratchArena::Allocate(16384);
  auto out = static_cast<float*>(io.view().device_ptr);
  auto id = reinterpret_cast<int*>(out + 2304);
  CheckCuda(cudaMemset(id, 0, sizeof(int)));
  Runner run(path, nullptr, nullptr, scratch.view());
  run({.end_visit = 48, .run_embed = 1, .seq_len = 1, .d_input_ids = id, .d_hidden_out = out});
  Require(Read<float>(out, 1)[0] == 1176.f / 1024, "stub hidden");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    std::string path = "./forward.so", model, golden;
    for (int i = 1; i < argc; i += 2) {
      Require(i + 1 < argc, "missing argument");
      std::string key = argv[i];
      if (key == "--forward")
        path = argv[i + 1];
      else if (key == "--model-dir")
        model = argv[i + 1];
      else if (key == "--golden-dir")
        golden = argv[i + 1];
      else
        throw std::invalid_argument(key);
    }
    Smoke(path, !model.empty());
    if (model.empty()) return 0;
    Require(!golden.empty(), "missing golden directory");
    std::cout << std::setprecision(9);
    // Lazy CUDA/cuBLAS kernel loading retains driver memory until process
    // exit. Exercise one warm visit before measuring a fresh arena/DSO lifetime.
    const auto cold = Free();
    {
      auto w = WeightsArena::Load(model);
      auto kv = KvStateArena::Allocate(w.view().config, 1);
      auto scratch = ScratchArena::Allocate();
      auto io = ScratchArena::Allocate(8192);
      auto id = static_cast<int*>(io.view().device_ptr);
      CheckCuda(cudaMemset(id, 0, sizeof(int)));
      Runner warm(path, &w.view(), &kv.view(), scratch.view());
      warm({.end_visit = 1,
            .run_embed = 1,
            .reset_kv = 1,
            .seq_len = 1,
            .d_input_ids = id,
            .d_hidden_out = id + 1});
    }
    const auto before = Free();
    std::cout << "cold_warmup_delta_bytes=" << int64_t(cold) - int64_t(before) << std::endl;
    bool ok = true;
    {
      auto w = WeightsArena::Load(model);
      auto kv = KvStateArena::Allocate(w.view().config, 1024);
      auto scratch = ScratchArena::Allocate();
      Runner run(path, &w.view(), &kv.view(), scratch.view());
      for (auto name : {"short_math", "short_sky", "short_code"})
        ok &= Case(run, w.view().config, kv.view(),
                   SafetensorsFile::Open(std::filesystem::path(golden) /
                                         ("e2e_" + std::string(name) + ".safetensors")),
                   name);
    }
    auto after = Free();
    size_t loss = before > after ? before - after : 0;
    std::cout << "teardown_loss_bytes=" << loss << std::endl;
    return ok && loss == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

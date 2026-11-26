#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

#include "common/json_output.h"
#include "forge/replay.h"
#include "forge/safetensors_writer.h"
#include "loader/safetensors.h"
#include "model/visit_plan.h"

namespace {
using namespace flint;
namespace fs = std::filesystem;
void Require(bool value, const char* what) {
  if (!value) throw std::runtime_error(what);
}

struct Fixture {
  fs::path root, socket;
  pid_t pid{-1};
  std::string response;  // Owns numeric tokens in the most recent parsed response.
  Fixture() {
    char temp[] = "/tmp/flint-forge-XXXXXX";
    Require(mkdtemp(temp) != nullptr, "mkdtemp");
    root = temp;
    socket = root / "host.sock";
  }
  ~Fixture() {
    if (pid > 0) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
    fs::remove_all(root);
  }
  void Shutdown() {
    Call({{"command", "shutdown"}});
    int code;
    Require(waitpid(pid, &code, 0) == pid, "waitpid");
    pid = -1;
    Require(WIFEXITED(code) && !WEXITSTATUS(code) && !fs::exists(socket), "clean shutdown");
  }
  void Launch(const char* host, const char* forward, const char* model = nullptr) {
    pid = fork();
    Require(pid >= 0, "fork");
    if (!pid) {
      if (model) {
        execl(host, host, "--daemon", "--socket", socket.c_str(), "--state-dir", root.c_str(),
              "--baseline", forward, "--model-dir", model, "--kv-seq-len", "32", nullptr);
        _exit(127);
      }
      execl(host, host, "--daemon", "--socket", socket.c_str(), "--state-dir", root.c_str(),
            "--baseline", forward, "--no-model", "--golden-dir", root.c_str(), nullptr);
      _exit(127);
    }
  }
  Json Call(Object q) {
    struct Client {
      int fd;
      ~Client() { close(fd); }
    } client{::socket(AF_UNIX, SOCK_STREAM, 0)};
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strcpy(addr.sun_path, socket.c_str());
    for (unsigned i = 0; connect(client.fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
         ++i) {
      Require(i < 9000, "socket startup timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::string line = Encode(q) + "\n";
    Require(send(client.fd, line.data(), line.size(), MSG_NOSIGNAL) == (ssize_t)line.size(),
            "send");
    response.clear();
    char buf[8192];
    while (response.empty() || response.back() != '\n') {
      auto n = recv(client.fd, buf, sizeof(buf), 0);
      Require(n > 0, "truncated response");
      response.append(buf, n);
    }
    auto parsed = json::Parser(response).Parse();
    if (!parsed.at("ok").as<bool>()) throw std::runtime_error(parsed.at("error").as<std::string>());
    if (parsed.as<Object>().contains("vram_loss_bytes")) {
      Require(parsed.at("vram_loss_bytes").number<size_t>() <= (4u << 20), "VRAM leak");
      Require(!parsed.at("vram_leak_warn").as<bool>(), "VRAM warning");
      if (q.at("command").as<std::string>() != "kld")
        Require(parsed.at("numerical_error").at("exact_match").as<bool>(), "replay mismatch");
    }
    return parsed;
  }
};

void TestKld() {
  KldAccumulator metrics;
  std::array<float, 4> logits{3, 1, 0, 0};
  AccumulateKld(logits, logits, 2, metrics);
  auto s = metrics.Summary();
  Require(s.num_positions == 2 && s.top1_matches == 2 && s.mean_kl == 0 && s.p95_kl == 0 &&
              s.max_kl == 0 && s.max_abs_logit_diff == 0,
          "identical KLD");

  // Uniform reference, candidate probabilities proportional to [1, e^2].
  std::array<float, 2> ref{5000, 5000}, cand{-3000, -2998};
  AccumulateKld(ref, cand, 2, metrics);
  s = metrics.Summary();
  double expected = std::log(std::cosh(1.0));
  Require(std::abs(s.mean_kl - expected / 3) < 1e-12 && std::abs(s.p95_kl - expected) < 1e-12 &&
              s.p95_kl == s.max_kl && s.top1_matches == 2 && s.max_abs_logit_diff == 8000,
          "shifted asymmetric KLD");

  float extreme = std::numeric_limits<float>::max();
  ref = {extreme, -extreme};
  cand = {-extreme, extreme};
  AccumulateKld(ref, cand, 2, metrics);
  Require(metrics.Summary().max_kl == 2.0 * extreme, "extreme finite KLD");
}

void TestSpeculativeComparisons() {
  const std::array tokens{10, 20, 30, 40};
  const auto expected = std::as_bytes(std::span(tokens));
  const auto prefix = std::span(tokens).first(2);
  const auto emitted = std::span(tokens).last(2);
  auto check = [](std::span<const std::byte> reference, std::span<const int> history,
                  std::span<const int> output, size_t count, bool match) {
    auto report = CompareSpeculativeGolden(reference, history, output);
    Require(report.at("matched_tokens_checked").as<uint64_t>() == count, "golden coverage");
    Require(report.contains("next_token_match") == (count != 0), "golden verdict availability");
    if (count) Require(report.at("next_token_match").as<bool>() == match, "golden token match");
  };
  check(expected, prefix, emitted, 2, true);
  check(expected.first(3 * sizeof(int)), prefix, emitted, 1, true);
  check(expected, prefix, std::array{30, 99}, 2, false);
  check(expected.first(prefix.size_bytes()), prefix, emitted, 0, false);
  check(expected.first(sizeof(int)), prefix, emitted, 0, false);
  check(expected, std::array{10, 99}, emitted, 0, false);

  Object ab;
  RecordSpeculativePair(ab, 0, 0, emitted, emitted);
  Require(ab.at("next_token_match").as<bool>(), "matching A/B pair");
  RecordSpeculativePair(ab, 0, 1, emitted, std::array{30, 99});
  RecordSpeculativePair(ab, 1, 0, emitted, emitted);
  Require(!ab.at("next_token_match").as<bool>(), "early A/B mismatch survives a matching round");
  auto& first = ab.at("first_token_mismatch");
  Require(first.at("round").as<uint64_t>() == 0 && first.at("iteration").as<uint64_t>() == 1 &&
              first.at("token_index").as<int64_t>() == 1,
          "first A/B mismatch position");
  Object shorter;
  RecordSpeculativePair(shorter, 0, 0, emitted.first(1), emitted);
  Require(!shorter.at("next_token_match").as<bool>(), "different A/B output lengths");
}

void TestStateError() {
  PostState ref{2, {0x3f80}, {}};
  ref.state = {{0, KvSliceKind::kKvConv, KvSliceDType::kF32, std::vector<std::byte>(4)},
               {4, KvSliceKind::kSublayerConv, KvSliceDType::kF32, std::vector<std::byte>(4)},
               {8, KvSliceKind::kKey, KvSliceDType::kBF16, std::vector<std::byte>(2)},
               {10, KvSliceKind::kValue, KvSliceDType::kBF16, std::vector<std::byte>(2)}};
  auto changed = ref;
  changed.blocks = 1;
  changed.carry[0] = 0x4000;  // BF16 1 -> 2
  uint16_t v = 0x4040;        // BF16 0 -> 3
  std::memcpy(changed.state[3].bytes.data(), &v, sizeof(v));
  float conv = -4.5f;
  std::memcpy(changed.state[1].bytes.data(), &conv, sizeof(conv));
  auto error = CompareState(ref, changed);
  Require(!error.at("exact_match").as<bool>() && !error.at("blocks_match").as<bool>(),
          "block count");
  for (auto [key, expected] : {std::pair{"carry", 1.0}, {"kv", 3.0}, {"conv", 4.5}}) {
    const auto& group = error.at(key);
    Require(!group.at("exact_match").as<bool>() &&
                group.at("max_abs_err").as<double>() == expected &&
                group.at("nonfinite").as<uint64_t>() == 0,
            "state writeback delta");
  }
  changed = ref;
  conv = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(changed.state[0].bytes.data(), &conv, sizeof(conv));
  error = CompareState(ref, changed);
  Require(!error.at("exact_match").as<bool>() && error.at("blocks_match").as<bool>() &&
              error.at("conv").at("nonfinite").as<uint64_t>() == 1,
          "nonfinite conv writeback");
}

void TestDecodeGolden(Fixture& f, Object q, int first_token) {
  q["step"] = 1;
  int second_token = f.Call(q).at("next_token_id").number<int>();
  q["step"] = 2;
  int next_token = f.Call(q).at("next_token_id").number<int>();
  std::array<int, 3> prompt{1, 42, 123}, generated{first_token, second_token, next_token};
  std::array<int, 64> ids;
  std::array<float, 64> logits;
  for (int i = 0; i < 64; ++i) {
    // The stub head scores tokens by distance from its greedy token.
    ids[i] = next_token + (i % 2 ? -(i + 1) / 2 : i / 2);
    logits[i] = -std::abs(ids[i] - next_token) / 1024.f;
  }
  q["case"] = "stub";
  auto path = f.root / "e2e_stub.safetensors";
  std::vector<DumpTensor> tensors{
      {"input_ids", "I32", {3}, std::as_bytes(std::span(prompt))},
      {"generated_ids", "I32", {3}, std::as_bytes(std::span(generated))}};
  WriteDump(path, tensors);
  auto token = f.Call(q).at("golden");
  Require(!token.as<Object>().contains("tensor") && token.at("next_token_match").as<bool>(),
          "standalone golden token");
  tensors.push_back({"decode_1.top64_ids", "I32", {64}, std::as_bytes(std::span(ids))});
  tensors.push_back({"decode_1.top64_logits", "F32", {64}, std::as_bytes(std::span(logits))});
  WriteDump(path, tensors);
  auto golden = f.Call(q).at("golden");
  Require(golden.at("tensor").as<std::string>() == "decode_1.top64_logits" &&
              golden.at("exact_match").as<bool>() && golden.at("top_k").number<int>() == 64 &&
              golden.at("expected_next_token_id").number<int>() == next_token &&
              golden.at("next_token_match").as<bool>(),
          "decode top64 golden");
}

void TestKldRpc(const char* host, const char* forward, const char* model) {
  Fixture f;
  f.Launch(host, forward, model);
  Array cases{Object{{"name", "tail"}, {"token_ids", Array{1, 2, 3, 4, 5}}, {"eval_start", 1}},
              Object{{"name", "prompt"}, {"token_ids", Array{42, 43}}, {"eval_start", 0}}};
  auto reply = f.Call({{"command", "kld"},
                       {"so_ref", forward},
                       {"so_cand", forward},
                       {"chunk_size", 2},
                       {"cases", std::move(cases)}});
  const auto& rows = reply.at("cases").as<Array>();
  Require(rows.size() == 2 && rows[0].at("num_positions").number<int>() == 4 &&
              rows[1].at("num_positions").number<int>() == 2,
          "KLD chunk boundaries");
  Require(reply.at("num_positions").number<int>() == 6 &&
              reply.at("top1_match_rate").number<double>() == 1,
          "KLD position agreement");
  for (auto key : {"mean_kl", "p95_kl", "max_kl", "max_abs_logit_diff", "unload_loss_bytes"})
    Require(reply.at(key).number<double>() == 0, key);
  Require(fs::is_empty(f.root / "loaded"), "leftover KLD DSO");
  f.Shutdown();
}
void Test(const char* host, const char* forward) {
  Fixture f;
  f.Launch(host, forward);
  auto status = f.Call({{"command", "status"}});
  Require(status.at("pid").number<int>() == f.pid && status.at("weights_bytes").number<int>() == 0,
          "status");
  Object q{{"command", "run"},
           {"so_path", forward},
           {"input_ids", Array{1, 42, 123}},
           {"warmup", 1},
           {"iters", 2}};
  auto e2e = f.Call(q);
  Require(
      e2e.at("next_token_match").as<bool>() && e2e.at("logits_error").at("exact_match").as<bool>(),
      "e2e logits");
  Require(e2e.at("logical_bytes").number<size_t>() == 3407910336ull, "logical roofline budget");
  Require(e2e.at("state_error").at("exact_match").as<bool>(), "e2e writeback");
  TestDecodeGolden(f, q, e2e.at("next_token_id").number<int>());
  auto probe = q;
  probe["command"] = "dump";
  probe["input_ids"] = Array{0};
  probe["out"] = (f.root / "stub_probe.safetensors").string();
  f.Call(probe);
  auto probe_file = SafetensorsFile::Open(f.root / "stub_probe.safetensors");
  float actual;
  auto values = probe_file.Bytes(*probe_file.Find("cand_hidden_out"));
  std::memcpy(&actual, values.data(), sizeof(actual));
  float expected = 0;
  for (unsigned v = 0; v < model::kNumLogicalVisits; ++v) {
    auto bits = std::bit_cast<uint32_t>(expected + (v + 1) / 1024.f);
    expected = std::bit_cast<float>((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u);
  }
  Require(actual == expected, "stub BF16 hidden");
  q["end_visit"] = 4;
  Require(!f.Call(q).at("lazy_cache_hit").as<bool>(), "first partial hit");
  q["use_cuda_graph"] = false;
  Require(f.Call(q).at("lazy_cache_hit").as<bool>(), "partial miss");
  q["command"] = "ab";
  q["so_a"] = forward;
  q["so_b"] = forward;
  q["rounds"] = 3;
  auto ab = f.Call(q);
  Require(ab.at("speedup").number<double>() > 0, "speedup");
  Require(ab.at("state_error").at("exact_match").as<bool>(), "A/B writeback");
  const auto& pairs = ab.at("rounds").as<Array>();
  Require(pairs.size() == 3, "round count");
  for (auto& p : pairs)
    Require(p.at("a_ms").number<double>() > 0 && p.at("b_ms").number<double>() > 0,
            "interleaved timing");
  q["command"] = "profile";
  q["use_cuda_graph"] = true;
  auto profile = f.Call(q);
  Require(profile.at("stages").as<Array>().size() == 4, "stage count");
  for (auto& stage : profile.at("stages").as<Array>())
    Require(stage.at("elapsed_ms").number<double>() > 0, "event time");
  q["begin_visit"] = 2;
  q["command"] = "dump";
  q["step"] = 2;
  q["with_state"] = true;
  auto path = f.root / "slice.safetensors";
  q["out"] = path.string();
  f.Call(q);
  auto file = SafetensorsFile::Open(path);
  for (auto name : {"input_ids", "hidden_in", "ref_hidden_out", "cand_hidden_out", "begin_visit",
                    "end_visit", "pos_offset", "num_block_in", "max_full_seq_len", "carry"})
    Require(file.Find(name) != nullptr, "missing tensor");
  auto ref = file.Bytes(*file.Find("ref_hidden_out")),
       cand = file.Bytes(*file.Find("cand_hidden_out"));
  Require(ref.size() == 2304 * sizeof(float) && !std::memcmp(ref.data(), cand.data(), ref.size()),
          "dump mismatch");
  auto config = model::MakeDefaultInklingConfig();
  auto plan = BuildVisitPlan(config);
  size_t offset = size_t(plan[2].kv_slot_idx) * 2 * config.sliding_window_size * model::kKvDim *
                  sizeof(uint16_t);
  auto* slice = file.Find("state_offset_" + std::to_string(offset));
  Require(slice && slice->dtype == TensorDType::kU8, "missing visit K slice");
  auto kv = file.Bytes(*slice);
  Require(std::any_of(kv.begin(), kv.end(), [](auto v) { return v != std::byte{}; }),
          "empty decode state");
  Require(fs::is_empty(f.root / "loaded"), "leftover DSO");
  f.Shutdown();
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 4) {
      TestKldRpc(argv[1], argv[2], argv[3]);
      return 0;
    }
    Require(argc == 3, "expected host and forward paths");
    TestKld();
    TestStateError();
    TestSpeculativeComparisons();
    Test(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

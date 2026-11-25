#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include "forge/json_output.h"
#include "loader/safetensors.h"

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
  void Launch(const char* host, const char* forward) {
    pid = fork();
    Require(pid >= 0, "fork");
    if (!pid) {
      execl(host, host, "--daemon", "--socket", socket.c_str(), "--state-dir", root.c_str(),
            "--baseline", forward, "--no-model", nullptr);
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
      Require(i < 500, "socket startup timed out");
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
      Require(parsed.at("numerical_error").at("exact_match").as<bool>(), "replay mismatch");
    }
    return parsed;
  }
};

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
  for (auto name :
       {"input_ids", "hidden_in", "ref_hidden_out", "cand_hidden_out", "kv_state", "conv_state"})
    Require(file.Find(name) != nullptr, "missing tensor");
  auto ref = file.Bytes(*file.Find("ref_hidden_out")),
       cand = file.Bytes(*file.Find("cand_hidden_out"));
  Require(ref.size() == 2304 * sizeof(float) && !std::memcmp(ref.data(), cand.data(), ref.size()),
          "dump mismatch");
  auto kv = file.Bytes(*file.Find("kv_state"));
  Require(std::any_of(kv.begin(), kv.end(), [](auto v) { return v != std::byte{}; }),
          "empty decode state");
  Require(fs::is_empty(f.root / "loaded"), "leftover DSO");
  f.Call({{"command", "shutdown"}});
  int code;
  Require(waitpid(f.pid, &code, 0) == f.pid, "waitpid");
  f.pid = -1;
  Require(WIFEXITED(code) && !WEXITSTATUS(code) && !fs::exists(f.socket), "clean shutdown");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    Require(argc == 3, "expected host and forward paths");
    Test(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

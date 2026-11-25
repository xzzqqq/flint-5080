#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <charconv>
#include <chrono>
#include <cstring>
#include <system_error>

#include "forge/replay.h"

namespace flint {
namespace {
struct Fd {
  int value;
  ~Fd() {
    if (value >= 0) close(value);
  }
};
void Sys(bool ok, const char* what) {
  if (!ok) throw std::system_error(errno, std::generic_category(), what);
}
unsigned Positive(std::string_view text) {
  unsigned n{};
  auto end = text.data() + text.size();
  auto [p, ec] = std::from_chars(text.data(), end, n);
  if (ec != std::errc{} || p != end || !n) throw std::runtime_error("expected a positive integer");
  return n;
}
bool Respond(int fd, Replay& replay, std::string_view line) {
  bool stop = false;
  Object reply;
  try {
    auto q = json::Parser(line).Parse();
    stop = q.at("command").as<std::string>() == "shutdown";
    if (!stop) {
      auto start = std::chrono::steady_clock::now();
      reply = replay.Request(q);
      reply["daemon_ms"] =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
    }
    reply["ok"] = !reply.contains("error");
  } catch (const std::exception& e) {
    reply = {{"ok", false}, {"error", e.what()}};
  }
  auto data = Encode(reply) + "\n";
  std::string_view remaining = data;
  while (!remaining.empty()) {
    auto n = send(fd, remaining.data(), remaining.size(), MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    remaining.remove_prefix(n);
  }
  return stop;
}
bool Serve(int fd, Replay& replay) {
  std::string pending;
  char buf[4096];
  while (true) {
    const auto n = recv(fd, buf, sizeof(buf), 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    pending.append(buf, n);
    size_t cut;
    while ((cut = pending.find('\n')) != pending.npos) {
      if (Respond(fd, replay, std::string_view(pending).substr(0, cut))) return true;
      pending.erase(0, cut + 1);
    }
    if (pending.size() > 1024 * 1024) {
      constexpr std::string_view error = "{\"ok\":false,\"error\":\"request too large\"}\n";
      send(fd, error.data(), error.size(), MSG_NOSIGNAL);
      return false;
    }
  }
}
}  // namespace
int RunDaemon(std::span<char*> args) {
  DaemonConfig cfg;
  bool synthetic = false;
  for (size_t i = 0; i < args.size(); ++i) {
    std::string_view key = args[i];
    if (key == "--no-model") {
      synthetic = true;
      continue;
    }
    if (++i == args.size()) throw std::runtime_error("missing daemon option value");
    auto v = args[i];
    if (key == "--socket")
      cfg.socket = v;
    else if (key == "--baseline")
      cfg.baseline = v;
    else if (key == "--state-dir")
      cfg.state_dir = v;
    else if (key == "--model-dir")
      cfg.model_dir = v;
    else if (key == "--golden-dir")
      cfg.golden_dir = v;
    else if (key == "--kv-seq-len")
      cfg.kv_seq_len = Positive(v);
    else if (key == "--scratch-mb")
      cfg.scratch_mb = Positive(v);
    else
      throw std::runtime_error("unknown daemon option " + std::string(key));
  }
  if (cfg.socket.empty() || cfg.baseline.empty() || cfg.state_dir.empty() ||
      synthetic != cfg.model_dir.empty())
    throw std::runtime_error(
        "expected --socket PATH --baseline PATH --state-dir PATH and --model-dir PATH or "
        "--no-model");
  if (!cfg.kv_seq_len) cfg.kv_seq_len = synthetic ? 256 : 65536;
  if (!cfg.scratch_mb) cfg.scratch_mb = synthetic ? 16 : 450;
  std::filesystem::create_directories(cfg.state_dir);
  Fd server{socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)};
  Sys(server.value >= 0, "socket");
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (cfg.socket.string().size() >= sizeof(addr.sun_path))
    throw std::runtime_error("socket path too long");
  std::strcpy(addr.sun_path, cfg.socket.c_str());
  Sys(bind(server.value, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "bind");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { unlink(path.c_str()); }
  } cleanup{cfg.socket};
  Sys(chmod(cfg.socket.c_str(), 0600) == 0, "chmod");
  Replay replay(cfg);
  Sys(listen(server.value, 8) == 0, "listen");
  while (true) {
    Fd client{accept4(server.value, nullptr, nullptr, SOCK_CLOEXEC)};
    if (client.value < 0 && errno == EINTR) continue;
    Sys(client.value >= 0, "accept");
    timeval timeout{30, 0};
    setsockopt(client.value, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    if (Serve(client.value, replay)) return 0;
  }
}
}  // namespace flint

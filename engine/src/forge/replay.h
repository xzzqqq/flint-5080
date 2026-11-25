#pragma once
#include <optional>

#include "forge/forge_daemon.h"
#include "forge/forward_abi.h"
#include "loader/arenas.h"

namespace flint {
struct Recording {
  std::vector<int> ids, generated;
  unsigned position{};
  std::vector<float> hidden, reference, logits, kv, conv;
  int next{-1};
};

class Replay {
 public:
  explicit Replay(DaemonConfig config);
  ~Replay();
  Object Status() const;
  Object Request(const Json& request);

 private:
  class Module;
  Object Measure(const Json& request, const std::string& command);
  std::unique_ptr<Module> Load(const std::filesystem::path& path);
  Recording Record(const std::vector<int>& ids, unsigned begin, unsigned end, unsigned step);
  ForwardRunSpec Spec(unsigned begin, unsigned end, unsigned length, unsigned position, bool embed,
                      bool head) const;
  void Restore(const Recording& recording, unsigned begin);
  DaemonConfig config_;
  std::optional<WeightsArena> weights_;
  std::optional<KvStateArena> kv_;
  std::optional<ScratchArena> scratch_, io_;
  double weights_load_ms_{};
  cudaStream_t stream_{};
  unsigned width_{2304}, vocab_{201024}, capacity_{}, serial_{};
  int *ids_{}, *next_{};
  float *hidden_{}, *out_{}, *logits_{};
  std::map<std::string, Recording> cache_;
};
}  // namespace flint

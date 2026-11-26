#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "forge/forge_daemon.h"
#include "forge/forward_module.h"
#include "forge/replay_metrics.h"
#include "loader/arenas.h"
#include "model/visit_plan.h"

namespace flint {
struct Recording {
  PostState post;
  std::vector<StateSlice> state;
  std::vector<uint16_t> carry;
  unsigned blocks{};
  std::vector<int> ids, generated;
  unsigned position{};
  std::vector<float> hidden, reference, logits;
  int next{-1};
};

class Replay {
 public:
  explicit Replay(DaemonConfig config);
  ~Replay();
  Object Status() const;
  Object Request(const Json& request);

 private:
  struct Measurement {
    unsigned begin, end, step, warmup, iters, rounds;
    std::string name;
    std::vector<int> ids;
    bool e2e() const { return begin == 0 && end == model::kNumLogicalVisits; }
  };
  Measurement ParseMeasurement(const Json&, const std::string&) const;
  const Recording& ResolveRecording(const Measurement&, Object&, Recording&);
  ForwardRunResult RunSingle(ForwardModule&, const ForwardRunSpec&, const ForwardOptions&,
                             const Recording&, unsigned, Object&);
  ForwardRunResult RunPair(ForwardModule&, ForwardModule&, const ForwardRunSpec&,
                           const ForwardOptions&, const Recording&, const Measurement&, Object&);
  ForwardRunResult RunRestored(ForwardModule&, const ForwardRunSpec&, const ForwardOptions&,
                               const Recording&);
  static void FormatStages(const ForwardRunResult&, Object&);
  void Dump(const Json&, const Measurement&, const Recording&, const std::vector<float>&, Object&);

  Object Measure(const Json& request, const std::string& command);
  [[nodiscard]] std::optional<ForwardModule> Load(const std::filesystem::path& path);
  Recording Record(const std::vector<int>& ids, unsigned begin, unsigned end, unsigned step);
  ForwardRunSpec Spec(unsigned begin, unsigned end, unsigned length, unsigned position, bool embed,
                      bool head, unsigned blocks = 0);
  std::vector<float> ReadHidden(size_t count) const;
  void UploadHidden(const std::vector<float>& data);
  PostState ReadPostState(const Recording& recording, unsigned blocks) const;
  void Restore(const Recording& recording);
  DaemonConfig config_;
  std::optional<WeightsArena> weights_;
  std::optional<KvStateArena> kv_;
  std::optional<ScratchArena> scratch_, io_;
  double weights_load_ms_{};
  int64_t warmup_vram_bytes_{};
  cudaStream_t stream_{};
  ModelConfig model_config_;
  VisitPlan plan_;
  unsigned width_, vocab_, capacity_{};
  uint16_t* carry_{};
  int *ids_{}, *next_{};
  uint16_t *hidden_{}, *out_{};
  float* logits_{};
  std::map<std::string, Recording> cache_;
};
}  // namespace flint

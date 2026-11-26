#pragma once

#include <cuda_runtime_api.h>

#include <filesystem>
#include <string>

#include "common/json_output.h"
#include "forge/forward_abi.h"

namespace flint {
class ForwardModule {
 public:
  ForwardModule(const std::filesystem::path& path, const std::filesystem::path& load_dir,
                const ModelWeightsView* weights, const KvStateArenaView& kv,
                const ScratchArenaView& scratch, cudaStream_t stream);
  ~ForwardModule();
  ForwardModule(const ForwardModule&) = delete;
  ForwardModule& operator=(const ForwardModule&) = delete;

  ForwardRunResult Run(const ForwardRunSpec& spec, const ForwardOptions& opts = {});
  void ReportGraphStats(Object& reply, const char* key) const;

 private:
  void Close() noexcept;

  std::string copy_path_;
  void* handle_{};
  ForwardContext* context_{};
  decltype(&flint_forward_destroy) destroy_{};
  decltype(&flint_forward_graph_stats) stats_{};
  decltype(&flint_forward_run_layers) run_{};
};
}  // namespace flint

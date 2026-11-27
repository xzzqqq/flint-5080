#pragma once
#include <filesystem>
#include <span>

#include "common/json_output.h"

namespace flint {
struct DaemonConfig {
  std::filesystem::path socket, baseline, state_dir, model_dir, draft_dir, golden_dir;
  bool reset_baseline = false;
  unsigned kv_seq_len{}, scratch_mb{};
};
int RunDaemon(std::span<char*> args);
}  // namespace flint

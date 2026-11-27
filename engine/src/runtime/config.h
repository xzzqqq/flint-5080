#pragma once
#include "common/json.h"
#include "loader/model_constants.h"

namespace flint {
struct RuntimeConfig {
  std::string model_dir, draft_model_dir;
  uint32_t max_context = 0, prefill_chunk_size = 256;
  size_t scratch_bytes = 450ull << 20, vram_reserve_bytes = 512ull << 20;
  uint32_t prefix_cache_slots = 3;
  uint32_t spec_verify_len = 8;
  bool use_cuda_graph = true;
  bool use_speculative = true;

  static RuntimeConfig Parse(const char* input) {
    if (!input) throw std::invalid_argument("config_json is null");
    auto root = json::Parser(input).Parse();
    if (!std::holds_alternative<json::Json::Object>(root.value))
      throw std::invalid_argument("config_json must be an object");
    RuntimeConfig c;
    auto read = [&](const char* key, auto& value) {
      try {
        value = root.get(key, value);
      } catch (const std::exception& e) {
        throw std::invalid_argument(std::string(key) + ": " + e.what());
      }
    };
    read("model_dir", c.model_dir);
    read("draft_model_dir", c.draft_model_dir);
    read("max_context", c.max_context);
    read("prefill_chunk_size", c.prefill_chunk_size);
    read("scratch_bytes", c.scratch_bytes);
    read("vram_reserve_bytes", c.vram_reserve_bytes);
    read("prefix_cache_slots", c.prefix_cache_slots);
    read("spec_verify_len", c.spec_verify_len);
    read("use_cuda_graph", c.use_cuda_graph);
    read("use_speculative", c.use_speculative);
    if (c.spec_verify_len < 2 || c.spec_verify_len > model::kDraftConfig.block_size)
      throw std::invalid_argument("spec_verify_len must be in [2, 8]");
    if (!c.prefill_chunk_size || c.prefill_chunk_size > 2048)
      throw std::invalid_argument("prefill_chunk_size must be in [1, 2048]");
    if (c.scratch_bytes < (64ull << 20))
      throw std::invalid_argument("scratch_bytes must be >= 64 MiB");
    return c;
  }
};
}  // namespace flint

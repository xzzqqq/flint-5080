#pragma once
#include <cuda_runtime_api.h>

#include <memory>

#include "loader/model_constants.h"
namespace flint {
struct ForwardBatch {
  uint32_t begin_visit, end_visit;
  uint8_t run_embed, run_lm_head, is_decode, reset_kv;
  uint32_t seq_len, pos_offset;
  const int32_t* d_input_ids;
  const void* d_hidden_in;
  const void* d_block_in;
  uint32_t num_block_in;
  void* d_hidden_out;
  void* d_block_out;
  float* d_logits_out;
  int32_t* d_next_token;
  bool operator==(const ForwardBatch&) const = default;
};
struct ForwardRunOptions {
  uint8_t use_cuda_graph;
  uint8_t use_speculative;
  uint8_t return_logits;
  uint8_t enable_profiling;
  uint8_t profiler_range;
  bool operator==(const ForwardRunOptions&) const = default;
};
struct StageTiming {
  char name[32];
  float elapsed_ms;
  uint64_t traffic_bytes;
};
struct ForwardOutput {
  float elapsed_ms;
  uint64_t logical_bytes;
  int32_t next_token_id;
  uint32_t num_block_out;
  uint32_t num_stages;
  StageTiming stages[64];
};
struct CaptureBuffers {
  void* embed;
  void* steps[model::kNumLogicalVisits];
  void* taps[4]; /* visit inputs 5,23,41 and final norm */
};
struct GraphStats {
  uint64_t captures, launches;
};
class InklingRunner {
 public:
  InklingRunner(const ModelWeightsView&, const KvStateArenaView&, ScratchArenaView, cudaStream_t);
  ~InklingRunner();
  ForwardOutput Run(const ForwardBatch&, const ForwardRunOptions&, const CaptureBuffers* = nullptr);
  GraphStats graph_stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace flint

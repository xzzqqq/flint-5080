#ifndef FLINT_FORWARD_TYPES_H
#define FLINT_FORWARD_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "loader/model_views.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ForwardOptions {
  uint8_t use_cuda_graph;
  uint8_t use_speculative;
  uint8_t return_logits;
  uint8_t enable_profiling;
  uint8_t profiler_range;
} ForwardOptions;

#define FLINT_MAX_STAGE_TIMINGS 64
typedef struct ForwardStageTiming {
  char name[32];
  float elapsed_ms;
  uint64_t traffic_bytes;
} ForwardStageTiming;

typedef struct ForwardCapture {
  void* embed;
  void* steps[FLINT_NUM_LOGICAL_VISITS];
  void* taps[4]; /* visit inputs 5,23,41 and final norm */
} ForwardCapture;

/* Hidden and carry planes are BF16 [seq_len,hidden_size]. Carry slots 0..3 are active block states,
   slots 4/5 are the prologue hidden/block, slots 6/7 the previous pass hidden/block.
   num_block_in/out count active blocks.
   Snapshot validity is implicit in the visit boundary. */
typedef struct ForwardRunSpec {
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
} ForwardRunSpec;

typedef struct ForwardRunResult {
  int32_t status;
  float elapsed_ms;
  uint64_t logical_bytes;
  int32_t next_token_id;
  uint32_t num_block_out;
  uint32_t num_stages;
  ForwardStageTiming stages[FLINT_MAX_STAGE_TIMINGS];
} ForwardRunResult;

typedef struct ForwardGraphStats {
  uint64_t captures, launches;
} ForwardGraphStats;

#ifdef __cplusplus
}
#endif
#endif

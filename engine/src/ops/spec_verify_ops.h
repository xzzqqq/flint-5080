#pragma once
#include <stddef.h>

#include "ops/draft_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SpecSampleMode {
  SPEC_SAMPLE_GREEDY,
  SPEC_SAMPLE_STOCHASTIC_FULL,
  SPEC_SAMPLE_STOCHASTIC_FILTERED
} SpecSampleMode;
typedef enum SpecVerifyError {
  SPEC_VERIFY_OK,
  SPEC_VERIFY_INVALID_PARAM,
  SPEC_VERIFY_NO_FINITE_LOGITS,
  SPEC_VERIFY_DRAFT_OUTSIDE_SUPPORT,
  SPEC_VERIFY_INVALID_CANDIDATE,
  SPEC_VERIFY_EMPTY_RESIDUAL
} SpecVerifyError;
typedef struct SpecVerifyInput {
  int32_t anchor, pos;
  uint32_t verify_len;
  float temperature, top_p;
  int32_t top_k;
  uint64_t seed;
} SpecVerifyInput;
typedef struct SpecVerifyOutput {
  int32_t accepted_draft_count;
  int32_t tokens[kDraftBlockSize];
  uint64_t next_seed;
  int32_t error_code;
} SpecVerifyOutput;

int flint_spec_prepare_verify_ids(const int32_t* anchor, const ForwardDraftResult* draft,
                                  int32_t* ids, int verify_len, void* stream);
// Scratch must be 256-byte aligned. Greedy verification requires no scratch.
size_t flint_spec_verify_scratch_bytes(int verify_len, int vocab, SpecSampleMode mode);
int flint_spec_verify_accept(const ForwardDraftResult* draft, const int32_t* argmax,
                             const float* logits, const SpecVerifyInput* input,
                             SpecVerifyOutput* output, void* scratch, size_t scratch_bytes,
                             int verify_len, int vocab, SpecSampleMode mode, void* stream);
#ifdef __cplusplus
}
#endif

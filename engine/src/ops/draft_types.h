#pragma once
#include <stdint.h>

enum { kDraftBlockSize = 8, kDraftTokens = 7, kSelectorTopK = 16 };
typedef struct ForwardDraftSpec {
  int32_t anchor;
  int32_t pos;
  float temperature;
  uint64_t seed;
  uint8_t use_cuda_graph;
} ForwardDraftSpec;
typedef struct ForwardDraftResult {
  int32_t draft_ids[kDraftTokens];
  int32_t candidate_ids[kDraftTokens * kSelectorTopK];
  float q_probs[kDraftTokens * kSelectorTopK];
} ForwardDraftResult;

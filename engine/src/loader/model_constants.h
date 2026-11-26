#pragma once
#include "loader/model_views.h"
namespace flint::model {
inline constexpr int kHiddenSize = 2304;
inline constexpr int kNumPhysicalLayers = FLINT_NUM_PHYSICAL_LAYERS;
inline constexpr int kVocabSize = 201024;
inline constexpr int kUnpaddedVocabSize = 200058;
inline constexpr int kNumQHeads = 24;
inline constexpr int kNumKvHeads = 8;
inline constexpr int kHeadDim = 128;
inline constexpr int kSlidingWindow = 1024;
inline constexpr int kRelDim = 16;
inline constexpr int kRelExtent = 1024;
inline constexpr int kSconvKernelSize = 4;
inline constexpr int kLayersPerResBlock = 6;
inline constexpr int kNumExperts = FLINT_NUM_EXPERTS;
inline constexpr int kTopKExperts = 8;
inline constexpr int kIntermediateSize = 1152;
inline constexpr int kLoopCoreStart = 6;
inline constexpr int kLoopCoreEnd = 18;
inline constexpr int kLoopNumPasses = 3;
inline constexpr int kLoopGateRank = 64;
inline constexpr int kNumLogicalVisits = FLINT_NUM_LOGICAL_VISITS;
inline constexpr int kNumSwaVisits = FLINT_NUM_SWA_VISITS;
inline constexpr int kNumFullVisits = FLINT_NUM_FULL_VISITS;
inline constexpr int kQDim = kNumQHeads * kHeadDim;
inline constexpr int kKvDim = kNumKvHeads * kHeadDim;
inline constexpr int kRelProjDim = kNumQHeads * kRelDim;
inline constexpr int kConvStateLen = kSconvKernelSize - 1;
inline constexpr int kDynConvProjDim = kNumKvHeads * kSconvKernelSize;
inline constexpr int kGateUpSize = 2 * kIntermediateSize;
inline constexpr int kMessageGateInDim = 2 * kHiddenSize;
inline constexpr int kAttentionChunkTokens = 32;
inline constexpr int kCarryPlanes = FLINT_CARRY_PLANES;
inline constexpr MtpConfig kMtpConfig{.hidden_size = kHiddenSize,
                                      .num_hidden_layers = 3,
                                      .num_attention_heads = kNumQHeads,
                                      .num_key_value_heads = kNumKvHeads,
                                      .head_dim = kHeadDim,
                                      .intermediate_size = 9216,
                                      .max_position_embeddings = 1048576,
                                      .sliding_window_size = kSlidingWindow,
                                      .block_size = 8,
                                      .markov_rank = 256,
                                      .rms_norm_eps = 1e-6f,
                                      .rope_theta = 1e6f,
                                      .is_local_layer = {1, 0, 1},
                                      .target_layer_ids = {5, 23, 41, 48}};
inline constexpr ModelConfig MakeDefaultInklingConfig() {
  ModelConfig c{};
  c.hidden_size = kHiddenSize;
  c.num_hidden_layers = kNumPhysicalLayers;
  c.vocab_size = kVocabSize;
  c.unpadded_vocab_size = kUnpaddedVocabSize;
  c.num_attention_heads = kNumQHeads;
  c.num_key_value_heads = kNumKvHeads;
  c.head_dim = kHeadDim;
  c.sliding_window_size = kSlidingWindow;
  c.d_rel = kRelDim;
  c.rel_extent = kRelExtent;
  c.log_scaling_n_floor = 8192u;
  c.sconv_kernel_size = kSconvKernelSize;
  c.n_layers_per_attn_res_block = kLayersPerResBlock;
  c.n_routed_experts = kNumExperts;
  c.num_experts_per_tok = kTopKExperts;
  c.intermediate_size = kIntermediateSize;
  c.log_scaling_alpha = 0.1f;
  c.rms_norm_eps = 1e-6f;
  c.qk_l2_clip_threshold = 22.627416997969522f;
  c.rel_l2_clip_threshold = 8.f;
  c.route_scale = 8.f;
  c.mlp_gate_softcap = 4.f;
  c.mlp_up_softcap = 25.f;
  c.final_logit_softcapping = 30.f;
  c.loop_core_start = kLoopCoreStart;
  c.loop_core_end = kLoopCoreEnd;
  c.loop_num_passes = kLoopNumPasses;
  c.loop_gate_rank = kLoopGateRank;
  c.eos_token_id = 200006u;
  c.pad_token_id = 199999;
  for (unsigned i = kLayersPerResBlock - 1; i < c.num_hidden_layers; i += kLayersPerResBlock)
    c.is_full_layer[i] = 1;
  return c;
}
}  // namespace flint::model

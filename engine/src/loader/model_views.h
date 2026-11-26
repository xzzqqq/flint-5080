#ifndef FLINT_MODEL_VIEWS_H
#define FLINT_MODEL_VIEWS_H

#include <stddef.h>
#include <stdint.h>

enum {
  FLINT_NUM_PHYSICAL_LAYERS = 24,
  FLINT_NUM_LOGICAL_VISITS = 48,
  FLINT_NUM_SWA_VISITS = 40,
  FLINT_NUM_FULL_VISITS = 8,
  FLINT_NUM_EXPERTS = 128,
  FLINT_NUM_CONTROLLERS = 4,
  FLINT_CARRY_PLANES = 8
};
#define FLINT_EXL3_MUL1_MAGIC 0x83DCD12Du

typedef struct MtpConfig {
  uint32_t hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim;
  uint32_t intermediate_size, max_position_embeddings, sliding_window_size, block_size, markov_rank;
  float rms_norm_eps, rope_theta;
  uint8_t is_local_layer[3];
  uint32_t target_layer_ids[4];
} MtpConfig;

typedef struct ModelConfig {
  uint32_t hidden_size, num_hidden_layers, vocab_size, unpadded_vocab_size;
  uint32_t num_attention_heads, num_key_value_heads, head_dim, sliding_window_size;
  uint32_t d_rel, rel_extent, log_scaling_n_floor;
  float log_scaling_alpha, rms_norm_eps, qk_l2_clip_threshold, rel_l2_clip_threshold;
  uint32_t sconv_kernel_size, n_layers_per_attn_res_block, n_routed_experts, num_experts_per_tok;
  uint32_t intermediate_size;
  float route_scale, mlp_gate_softcap, mlp_up_softcap, final_logit_softcapping;
  uint32_t loop_core_start, loop_core_end, loop_num_passes, loop_gate_rank;
  uint32_t eos_token_id, pad_token_id;
  uint8_t is_full_layer[FLINT_NUM_PHYSICAL_LAYERS];
  uint8_t has_mtp;
  MtpConfig mtp;
} ModelConfig;

typedef struct Exl3MatrixView {
  const int16_t* trellis;
  const uint16_t *suh, *svh;
  uint32_t k, n;
  uint8_t k2;
} Exl3MatrixView;

typedef struct Exl3ExpertDesc {
  const int16_t* trellis;
  const void *suh, *svh;
  int32_t k2;
} Exl3ExpertDesc;

typedef struct ControllerWeightsView {
  const void *down_weight, *forget_weight, *gain_weight;
} ControllerWeightsView;

typedef struct LayerWeightsView {
  uint8_t is_full_attention, is_loop_layer;
  const void *attn_norm, *q_input_norm, *k_input_norm, *v_input_norm, *mlp_norm;
  const void *attn_sconv, *mlp_sconv;
  Exl3MatrixView q_proj, k_proj, v_proj, o_gate_proj, o_proj;
  const void *k_sconv, *v_sconv, *k_static_w, *k_weight_proj, *v_static_w, *v_weight_proj;
  const void *wr_du, *rel_logits_proj, *q_l2_clip_scale, *rel_l2_clip_scale, *attn_sink_proj;
  const float *attn_query, *mlp_query;
  const void* router_weight;
  const float* router_bias;
  Exl3MatrixView gate_up_proj[FLINT_NUM_EXPERTS], down_proj[FLINT_NUM_EXPERTS];
  /* Controller order: attention, key, value, MLP. */
  ControllerWeightsView controllers[FLINT_NUM_CONTROLLERS];
} LayerWeightsView;

typedef struct MtpLayerWeightsView {
  const void *dflare_norm, *input_layernorm, *post_attention_layernorm;
  const void *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm;
  const void *k_proj_ctx, *v_proj_ctx, *k_norm_ctx, *gate_proj, *up_proj, *down_proj;
} MtpLayerWeightsView;
typedef struct MtpWeightsView {
  const void *fuse_weight, *mask_embed, *norm, *dflare_norm, *hidden_out, *hidden_bias;
  const void *markov_w1, *rnn_proj, *rnn_bias;
  const float* fuse_alpha;
  MtpLayerWeightsView layers[3];
} MtpWeightsView;

typedef struct ModelWeightsView {
  ModelConfig config;
  const void *embed_weight, *embed_norm_weight, *final_norm_weight, *unembed_weight;
  const float *output_query, *cross_pass_queries, *cross_pass_kv_queries;
  LayerWeightsView layers[FLINT_NUM_PHYSICAL_LAYERS];
  void* arena_base;
  size_t arena_bytes;
  uint32_t mapped_tensor_count, verified_mul1_count;
  /* Device table [physical layer][gate_up, down][expert]. */
  const Exl3ExpertDesc* expert_table;
  uint8_t has_mtp;
  MtpWeightsView mtp;
} ModelWeightsView;

typedef struct VisitPlanEntry {
  uint8_t visit_idx, physical_layer, pass_idx, is_full_attention;
  uint8_t is_block_boundary, has_messages, kv_slot_idx;
  uint8_t is_loop_pass_start, is_prologue_end;
  int8_t mtp_tap_idx;
} VisitPlanEntry;

typedef struct KvStateArenaView {
  void* arena_base;
  size_t arena_bytes;
  uint32_t max_full_seq_len, swa_window, num_swa_visits, num_full_visits, num_total_visits;
  uint32_t num_kv_heads, head_dim, conv_state_len;
  uint32_t hidden_size;
  void *swa_kv, *full_kv;
  float *kv_conv_state, *sublayer_conv_state;
  uint8_t has_mtp_kv;
  uint32_t mtp_swa_window;
  void *mtp_swa_kv, *mtp_full_kv;
} KvStateArenaView;

typedef struct ScratchArenaView {
  void* device_ptr;
  size_t bytes;
} ScratchArenaView;
#endif

#ifndef FLINT_OP_PARAMS_H
#define FLINT_OP_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SmokeAddParams {
  const float* input;
  float* output;
  int n;
  float bias;
} SmokeAddParams;

/* The stub reserves one F32 recurrent value per visit in each state arena. */
typedef struct StubParams {
  const int* ids;
  const float* input;
  float *output, *kv, *conv, *logits;
  int* next;
  const unsigned short* weight;
  unsigned width, count, vocab, position, visit;
} StubParams;
int flint_stub_embed(StubParams params, void* stream);
int flint_stub_visit(StubParams params, void* stream);
int flint_stub_head(StubParams params, void* stream);

/* Launchers borrow device pointers and enqueue on cuda_stream. Matrices are
   contiguous, row-major; weights and activations are BF16 unless specified. */
typedef enum FlintType { FLINT_F32, FLINT_BF16, FLINT_F16 } FlintType;
/* EXL3 trellis is [k/16,n/16,8*k2] I16; inner, suh and svh are F16. */
typedef struct {
  const short* trellis;
  void* inner;
  int k, n, k2;
} DecodeParams;
typedef struct {
  const void *x, *scale;
  void* y;
  int rows, width;
  FlintType input, output;
} WhtParams;
typedef struct {
  const void* x;
  void* y;
  int count;
  FlintType input, output;
} CastParams;
typedef struct {
  const void *x, *weight;
  void* y;
  const int* ids;
  int rows, width;
  FlintType input, output;
} NormParams;
typedef struct {
  const void* x;
  float* logits;
  int* ids;
  int rows, vocab, stride;
  FlintType input;
} HeadParams;
/* Static weight [channels,4] is oldest first. Dynamic weight [4,channels]
   and delta [tokens,8,4] are newest first. State [channels,3] is oldest first. */
typedef struct {
  const void *x, *weight, *delta;
  float* state;
  void* y;
  int tokens, channels;
  FlintType input, output;
} ConvParams;
typedef struct {
  const void *x, *scale;
  void* y;
  int tokens, position, log_scale;
} ClipParams;
/* Caller selects one visit's cache planes. For ring writes, only the last capacity slots survive.
 */
typedef struct {
  const void *k, *v;
  void *cache_k, *cache_v;
  int tokens, position, capacity, sliding;
} CacheParams;
/* scores: tokens * 24 * capacity floats. Cache must retain every visible key. */
typedef struct {
  const void *q, *cache_k, *cache_v, *rel, *sink;
  void* y;
  float* scores;
  int tokens, position, capacity, sliding;
} AttentionParams;
typedef struct {
  const void *x, *gate;
  void* y;
  int count;
} GateMulParams;
/* Slots follow CUDA unsorted top-k order: strict winners by ID, then cutoff ties. */
typedef struct {
  const float *logits, *bias;
  int* ids;
  float* weights;
  int tokens;
} RouterParams;
typedef struct {
  const void* gate_up;
  void* y;
  int rows, width;
} SwigluParams;
/* expert_outputs: [tokens,8,width], in the same slot order as weights. */
typedef struct {
  const void* expert_outputs;
  const float* weights;
  void* y;
  int tokens, width;
  const int* ids; /* Optional. Accumulate in ascending expert order when provided. */
} AccumParams;
/* blocks: contiguous [num_blocks,tokens,width]; partial: [tokens,width].
   scratch: [tokens,queries,num_blocks+1,2] FP32 (inverse RMS and score). */
typedef struct {
  const void *blocks, *partial;
  const float* query;
  float *y, *scratch;
  int num_blocks, tokens, width, queries;
  FlintType block_type, partial_type;
} ResidualParams;
typedef struct {
  const float *x, *message;
  void* y;
  int tokens, width;
} PrepParams;
typedef struct {
  const float *x, *message, *forget, *gain;
  float* y;
  int count;
} MixParams;
typedef struct {
  const float *qm, *kv;
  float* y;
  int layer, width;
} MergeParams;
int flint_exl3_decode_inner_f16(DecodeParams p, void* cuda_stream);
int flint_wht128_scale_in(WhtParams p, void* cuda_stream);
int flint_wht128_scale_out(WhtParams p, void* cuda_stream);
int flint_add_residual(const float* partial, const void* bf16, void* out, int count,
                       FlintType output, void* stream);
int flint_cast(CastParams p, void* cuda_stream);
int flint_embed_rmsnorm(NormParams p, void* cuda_stream);
int flint_rmsnorm(NormParams p, void* cuda_stream);
int flint_unweighted_rmsnorm(NormParams p, void* cuda_stream);
int flint_logit_softcap_argmax(HeadParams p, void* cuda_stream);
int flint_static_sconv(ConvParams p, void* cuda_stream);
int flint_dynamic_sconv(ConvParams p, void* cuda_stream);
int flint_q_norm_clip(ClipParams p, void* cuda_stream);
int flint_k_l2_clip(ClipParams p, void* cuda_stream);
int flint_rel_norm_clip(ClipParams p, void* cuda_stream);
int flint_kv_cache_write(CacheParams p, void* cuda_stream);
int flint_attention_forward(AttentionParams p, void* cuda_stream);
int flint_gated_mul_sigmoid(GateMulParams p, void* cuda_stream);
int flint_moe_router_top8(RouterParams p, void* cuda_stream);
int flint_softcapped_swiglu(SwigluParams p, void* cuda_stream);
int flint_moe_weighted_accum(AccumParams p, void* cuda_stream);
int flint_residual_norm_states(ResidualParams p, float* states, void* stream);
int flint_residual_softmax(float* scores, int rows, int states, float scale, void* stream);
int flint_block_attn_residual(ResidualParams p, void* cuda_stream);
int flint_message_gate_prep(PrepParams p, void* cuda_stream);
int flint_message_gate_mix(MixParams p, void* cuda_stream);
int flint_merge_cross_pass_queries(MergeParams p, void* cuda_stream);

/* Returns the CUDA launch status (zero on success). */
int flint_launch_smoke_add_f32(const SmokeAddParams* params, void* cuda_stream);

#ifdef __cplusplus
}
#endif
#endif

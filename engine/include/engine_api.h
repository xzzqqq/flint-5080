#ifndef FLINT_ENGINE_API_H
#define FLINT_ENGINE_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLINT_ENGINE_ABI_VERSION 2u

/* All CUDA and request state is private to the engine. */
typedef struct EngineHandle EngineHandle;

typedef enum EngineStatus {
  ENGINE_OK = 0,
  ENGINE_ERR_INVALID_INPUT = 1,
  ENGINE_ERR_CONTEXT_OVERFLOW = 2,
  ENGINE_ERR_CANCELLED = 3,
  ENGINE_ERR_INTERNAL = 4
} EngineStatus;

typedef enum EngineEventKind {
  ENGINE_EVENT_TOKENS = 0,
  ENGINE_EVENT_DONE = 1,
  ENGINE_EVENT_ERROR = 2
} EngineEventKind;

typedef enum EngineStopReason {
  ENGINE_STOP_NONE = 0,
  ENGINE_STOP_EOS = 1,
  ENGINE_STOP_STOP_TOKEN = 2,
  ENGINE_STOP_MAX_TOKENS = 3,
  ENGINE_STOP_CONTEXT_LIMIT = 4
} EngineStopReason;

/* Counts are per request. Times are elapsed milliseconds. */
typedef struct EngineTelemetry {
  uint32_t prefill_tokens;
  uint32_t cached_prefix_tokens;
  uint32_t decode_tokens;
  double prefill_ms;
  double decode_ms;
  uint32_t decode_rounds;
  uint32_t spec_accepted;
} EngineTelemetry;

/* All pointers in an event are borrowed for the duration of the callback.
 * TOKENS has n_tokens IDs, DONE carries final telemetry and the stop reason,
 * and ERROR carries the failure status and a NUL-terminated diagnostic.
 * stop_reason is ENGINE_STOP_NONE for TOKENS and ERROR. */
typedef struct EngineEvent {
  uint64_t req_id;
  EngineEventKind kind;
  EngineStatus status;
  EngineStopReason stop_reason;
  const int32_t* token_ids;
  size_t n_tokens;
  EngineTelemetry telemetry;
  const char* error_message;
} EngineEvent;

typedef struct RequestParams {
  uint32_t max_tokens;
  float temperature;
  float top_p;
  int32_t top_k;
  uint64_t seed;
  const int32_t* stop_token_ids;
  size_t n_stop_token_ids;
  uint8_t ignore_eos;
} RequestParams;

/* Reserved for vision input. In Phase 0 pass NULL and n_media = 0. */
typedef struct MediaSlice {
  uint32_t kind;
  const uint8_t* data;
  size_t size;
  uint32_t token_offset;
} MediaSlice;

typedef void (*TokenCallback)(const EngineEvent* event, void* user_data);

uint32_t engine_abi_version(void);

/* Create on CUDA device 0. If creation fails, return NULL and optionally
 * place an owned diagnostic in *err_out. Free it with engine_string_free. */
EngineHandle* engine_create(const char* config_json, char** err_out);

/* Finish external calls on the handle before destroying it. Cancels active and
 * queued requests, drains their callbacks, and joins the worker before releasing
 * resources. Do not call from the worker callback. Accepts NULL. */
void engine_destroy(EngineHandle* engine);

/* Deep-copies params (including stop_token_ids) and token_ids before returning
 * a nonzero request ID. The worker invokes callback asynchronously until exactly
 * one terminal DONE or ERROR. Keep user_data valid through that callback.
 * Missing handle or callback, or failure to enqueue, returns zero.
 * Callbacks may cancel their request and must return normally. */
uint64_t engine_submit(EngineHandle* engine, const RequestParams* params, const int32_t* token_ids,
                       size_t n_tokens, const MediaSlice* media, size_t n_media,
                       TokenCallback callback, void* user_data);

/* Request cancellation before the terminal event. Calls for issued IDs are
 * counted even if the request has already finished. */
void engine_cancel(EngineHandle* engine, uint64_t req_id);

/* Owned NUL-terminated JSON snapshot, or NULL on failure. */
char* engine_stats_json(EngineHandle* engine);

/* Releases strings allocated by the engine. Accepts NULL. */
void engine_string_free(char* string);

#ifdef __cplusplus
}
#endif

#endif

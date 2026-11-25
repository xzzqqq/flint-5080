#include "engine_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <span>

#include "common/json_output.h"
#include "ops/stub_ops.h"

struct EngineHandle {
  EngineHandle() = default;
  EngineHandle(const EngineHandle&) = delete;
  EngineHandle& operator=(const EngineHandle&) = delete;

  ~EngineHandle() {
    if (scratch != nullptr) {
      cudaFree(scratch);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  cudaDeviceProp gpu{};
  cudaStream_t stream{};
  float* scratch{};
  std::mutex scratch_mutex;

  std::mutex requests_mutex;
  std::map<uint64_t, bool> cancelled_by_id;
  std::atomic<uint64_t> submitted{};
  std::atomic<uint64_t> cancellations{};
};

namespace {

// Holds an entry only while the request can be cancelled. Never hold the
// registry lock across a user callback: callbacks may call engine_cancel.
class RegisteredRequest {
 public:
  RegisteredRequest(EngineHandle& engine, uint64_t id) : engine_(engine) {
    std::scoped_lock lock(engine_.requests_mutex);
    entry_ = engine_.cancelled_by_id.emplace(id, false).first;
  }

  RegisteredRequest(const RegisteredRequest&) = delete;
  RegisteredRequest& operator=(const RegisteredRequest&) = delete;

  ~RegisteredRequest() {
    std::scoped_lock lock(engine_.requests_mutex);
    engine_.cancelled_by_id.erase(entry_);
  }

  [[nodiscard]] bool IsCancelled() const {
    std::scoped_lock lock(engine_.requests_mutex);
    return entry_->second;
  }

 private:
  EngineHandle& engine_;
  std::map<uint64_t, bool>::iterator entry_;
};

[[nodiscard]] char* CopyString(const char* source) {
  const size_t bytes = std::strlen(source) + 1;
  auto* owned = static_cast<char*>(std::malloc(bytes));
  if (owned != nullptr) {
    std::memcpy(owned, source, bytes);
  }
  return owned;
}

[[nodiscard]] cudaError_t InitializeDevice(EngineHandle& engine) {
  int device_count = 0;
  if (const auto status = cudaGetDeviceCount(&device_count); status != cudaSuccess) {
    return status;
  }
  if (device_count == 0) {
    return cudaErrorNoDevice;
  }
  if (const auto status = cudaSetDevice(0); status != cudaSuccess) {
    return status;
  }
  if (const auto status = cudaGetDeviceProperties(&engine.gpu, 0); status != cudaSuccess) {
    return status;
  }
  if (const auto status = cudaStreamCreateWithFlags(&engine.stream, cudaStreamNonBlocking);
      status != cudaSuccess) {
    return status;
  }
  return cudaMalloc(&engine.scratch, 2 * sizeof(float));
}

using Clock = std::chrono::steady_clock;

double ElapsedMilliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

[[nodiscard]] cudaError_t RunSmokeToken(EngineHandle& engine, int32_t& token,
                                        EngineTelemetry& telemetry) {
  std::scoped_lock lock(engine.scratch_mutex);
  const auto prefill_start = Clock::now();
  if (const auto status = cudaSetDevice(0); status != cudaSuccess) {
    return status;
  }
  if (const auto status = cudaMemsetAsync(engine.scratch, 0, sizeof(float), engine.stream);
      status != cudaSuccess) {
    return status;
  }
  if (const auto status = cudaStreamSynchronize(engine.stream); status != cudaSuccess) {
    return status;
  }
  telemetry.prefill_ms = ElapsedMilliseconds(prefill_start);

  const auto decode_start = Clock::now();
  const SmokeAddParams params{
      .input = engine.scratch, .output = engine.scratch + 1, .n = 1, .bias = 42.0f};
  if (const int status = flint_launch_smoke_add_f32(&params, engine.stream);
      status != cudaSuccess) {
    return static_cast<cudaError_t>(status);
  }
  float value;
  if (const auto status = cudaMemcpyAsync(&value, params.output, sizeof(value),
                                          cudaMemcpyDeviceToHost, engine.stream);
      status != cudaSuccess) {
    return status;
  }
  if (const auto status = cudaStreamSynchronize(engine.stream); status != cudaSuccess) {
    return status;
  }
  if (value != params.bias) {
    return cudaErrorUnknown;
  }
  token = static_cast<int32_t>(value);
  telemetry.decode_ms = ElapsedMilliseconds(decode_start);
  return cudaSuccess;
}

EngineEvent ErrorEvent(uint64_t id, EngineStatus status, const char* message) {
  return {.req_id = id, .kind = ENGINE_EVENT_ERROR, .status = status, .error_message = message};
}

EngineEvent RunRequest(EngineHandle& engine, uint64_t id, const RequestParams& params,
                       uint32_t prompt_size, TokenCallback callback, void* user_data) {
  RegisteredRequest request(engine, id);
  EngineTelemetry telemetry{.prefill_tokens = prompt_size};
  int32_t token;
  EngineEvent terminal{.req_id = id, .kind = ENGINE_EVENT_DONE};
  if (const auto status = RunSmokeToken(engine, token, telemetry); status != cudaSuccess) {
    terminal = ErrorEvent(id, ENGINE_ERR_INTERNAL, cudaGetErrorString(status));
  } else {
    const std::span stop_tokens(params.stop_token_ids, params.n_stop_token_ids);
    if (params.max_tokens > 0 && !request.IsCancelled() &&
        std::ranges::find(stop_tokens, token) == stop_tokens.end()) {
      telemetry.decode_tokens = 1;
      telemetry.decode_rounds = 1;
      const EngineEvent tokens{.req_id = id,
                               .kind = ENGINE_EVENT_TOKENS,
                               .token_ids = &token,
                               .n_tokens = 1,
                               .telemetry = telemetry};
      callback(&tokens, user_data);
    }
  }
  if (request.IsCancelled()) {
    terminal = ErrorEvent(id, ENGINE_ERR_CANCELLED, "request cancelled");
  }
  terminal.telemetry = telemetry;
  return terminal;
}
}  // namespace

uint32_t engine_abi_version() { return FLINT_ENGINE_ABI_VERSION; }

EngineHandle* engine_create(const char* config_json, char** err_out) {
  if (err_out != nullptr) {
    *err_out = nullptr;
  }
  const auto fail = [err_out](const char* message) -> EngineHandle* {
    if (err_out != nullptr) {
      *err_out = CopyString(message);
    }
    return nullptr;
  };
  if (!config_json || !*config_json) {
    return fail("config_json must be nonempty");
  }

  std::unique_ptr<EngineHandle> engine(new (std::nothrow) EngineHandle);
  if (!engine) {
    return fail("engine allocation failed");
  }
  if (const auto status = InitializeDevice(*engine); status != cudaSuccess) {
    return fail(cudaGetErrorString(status));
  }
  return engine.release();
}

void engine_destroy(EngineHandle* engine) { delete engine; }

void engine_string_free(char* string) { std::free(string); }

uint64_t engine_submit(EngineHandle* engine, const RequestParams* params, const int32_t* token_ids,
                       size_t n_tokens, const MediaSlice*, size_t n_media, TokenCallback callback,
                       void* user_data) {
  if (!engine || !callback) {
    return 0;
  }
  const uint64_t id = ++engine->submitted;
  EngineEvent terminal;
  if (!params || !token_ids || n_tokens == 0 || n_media != 0 ||
      (params->n_stop_token_ids != 0 && !params->stop_token_ids)) {
    terminal = ErrorEvent(id, ENGINE_ERR_INVALID_INPUT,
                          "expected a text request with tokens and valid parameters");
  } else if (n_tokens > UINT32_MAX) {
    terminal =
        ErrorEvent(id, ENGINE_ERR_CONTEXT_OVERFLOW, "prompt token count exceeds telemetry range");
  } else {
    try {
      terminal =
          RunRequest(*engine, id, *params, static_cast<uint32_t>(n_tokens), callback, user_data);
    } catch (const std::bad_alloc&) {
      terminal = ErrorEvent(id, ENGINE_ERR_INTERNAL, "request allocation failed");
    }
  }
  callback(&terminal, user_data);
  return id;
}

void engine_cancel(EngineHandle* engine, uint64_t id) {
  if (!engine || id == 0 || id > engine->submitted.load()) {
    return;
  }
  std::scoped_lock lock(engine->requests_mutex);
  ++engine->cancellations;
  if (auto entry = engine->cancelled_by_id.find(id); entry != engine->cancelled_by_id.end()) {
    entry->second = true;
  }
}

char* engine_stats_json(EngineHandle* engine) {
  if (!engine || cudaSetDevice(0) != cudaSuccess) {
    return nullptr;
  }
  size_t available_bytes, total_bytes;
  if (cudaMemGetInfo(&available_bytes, &total_bytes) != cudaSuccess) {
    return nullptr;
  }
  try {
    auto json = flint::Encode(flint::Object{{"abi_version", FLINT_ENGINE_ABI_VERSION},
                                            {"gpu_name", engine->gpu.name},
                                            {"sm_major", engine->gpu.major},
                                            {"sm_minor", engine->gpu.minor},
                                            {"vram_total_bytes", total_bytes},
                                            {"vram_free_bytes", available_bytes},
                                            {"submitted_requests", engine->submitted.load()},
                                            {"cancelled_requests", engine->cancellations.load()}});
    return CopyString(json.c_str());
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

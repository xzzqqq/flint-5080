#include "engine_api.h"

#include <cstdlib>
#include <cstring>
#include <exception>

#include "runtime/engine_runtime.h"

struct EngineHandle {
  explicit EngineHandle(const char* json) : runtime(json) {}
  flint::EngineRuntime runtime;
};

namespace {
char* CopyString(const char* text) {
  const size_t bytes = std::strlen(text) + 1;
  auto* copy = static_cast<char*>(std::malloc(bytes));
  if (copy) std::memcpy(copy, text, bytes);
  return copy;
}
}  // namespace

uint32_t engine_abi_version() { return FLINT_ENGINE_ABI_VERSION; }

EngineHandle* engine_create(const char* json, char** err_out) {
  if (err_out) *err_out = nullptr;
  try {
    return new EngineHandle(json);
  } catch (const std::exception& err) {
    if (err_out) *err_out = CopyString(err.what());
    return nullptr;
  }
}

void engine_destroy(EngineHandle* e) { delete e; }

uint64_t engine_submit(EngineHandle* e, const RequestParams* params, const int32_t* ids, size_t n,
                       const MediaSlice*, size_t media, TokenCallback cb, void* data) {
  if (!e || !cb) return 0;
  try {
    return e->runtime.Submit(params, ids, n, media, cb, data);
  } catch (const std::exception&) {
    return 0;
  }
}

void engine_cancel(EngineHandle* e, uint64_t id) {
  if (e) e->runtime.Cancel(id);
}

char* engine_stats_json(EngineHandle* e) {
  if (!e) return nullptr;
  try {
    return CopyString(e->runtime.StatsJson().c_str());
  } catch (const std::exception&) {
    return nullptr;
  }
}

void engine_string_free(char* s) { std::free(s); }

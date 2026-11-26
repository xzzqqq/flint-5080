#pragma once

#include <memory>
#include <string>

#include "engine_api.h"

namespace flint {

class EngineRuntime {
 public:
  explicit EngineRuntime(const char* config_json);
  ~EngineRuntime();

  [[nodiscard]] uint64_t Submit(const RequestParams* params, const int32_t* ids, size_t count,
                                size_t media_count, TokenCallback callback, void* user_data);
  void Cancel(uint64_t id);
  [[nodiscard]] std::string StatsJson() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace flint

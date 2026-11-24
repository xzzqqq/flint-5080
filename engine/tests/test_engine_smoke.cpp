#include <cmath>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "engine_api.h"
#include "forge/forge_smoke.h"

namespace {
void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Capture {
  EngineHandle* engine;
  bool cancel_on_tokens = false;
  uint64_t request_id{};
  std::vector<EngineEventKind> kinds;
  std::vector<int32_t> tokens;
  EngineTelemetry telemetry{};
  EngineStatus status{};
  bool valid = true;
};

void Collect(const EngineEvent* event, void* user_data) {
  auto& capture = *static_cast<Capture*>(user_data);
  if (event->req_id == 0 || (capture.request_id != 0 && capture.request_id != event->req_id)) {
    capture.valid = false;
  }
  capture.request_id = event->req_id;
  capture.kinds.push_back(event->kind);
  capture.status = event->status;
  capture.telemetry = event->telemetry;

  if (event->kind == ENGINE_EVENT_TOKENS && event->token_ids != nullptr) {
    const std::span tokens(event->token_ids, event->n_tokens);
    capture.tokens.insert(capture.tokens.end(), tokens.begin(), tokens.end());
    if (capture.cancel_on_tokens) {
      engine_cancel(capture.engine, event->req_id);
    }
  }
  if (event->kind == ENGINE_EVENT_ERROR && (!event->error_message || !*event->error_message)) {
    capture.valid = false;
  }
}

void CheckRequests(EngineHandle* engine) {
  constexpr int32_t kPrompt[] = {1, 2, 3, 4};
  RequestParams params{.max_tokens = 8};
  uint64_t last_id = 0;
  const auto submit = [&](std::span<const int32_t> prompt, std::span<const MediaSlice> media = {},
                          bool cancel_on_tokens = false) {
    Capture capture{.engine = engine, .cancel_on_tokens = cancel_on_tokens};
    const auto id = engine_submit(engine, &params, prompt.data(), prompt.size(), media.data(),
                                  media.size(), Collect, &capture);
    Require(capture.valid && id > last_id && capture.request_id == id, "invalid request/event ID");
    last_id = id;
    return capture;
  };

  const auto success = submit(kPrompt);
  Require(success.kinds == std::vector{ENGINE_EVENT_TOKENS, ENGINE_EVENT_DONE} &&
              success.tokens == std::vector<int32_t>{42} && success.status == ENGINE_OK,
          "token stream");
  const auto& telemetry = success.telemetry;
  Require(telemetry.prefill_tokens == 4 && telemetry.decode_tokens == 1 &&
              telemetry.decode_rounds == 1 && telemetry.cached_prefix_tokens == 0 &&
              telemetry.spec_accepted == 0 && std::isfinite(telemetry.prefill_ms) &&
              telemetry.prefill_ms >= 0 && std::isfinite(telemetry.decode_ms) &&
              telemetry.decode_ms >= 0,
          "done telemetry");

  const MediaSlice media[] = {{}};
  for (const auto& invalid : {submit({}), submit(kPrompt, media)}) {
    Require(invalid.kinds == std::vector{ENGINE_EVENT_ERROR} &&
                invalid.status == ENGINE_ERR_INVALID_INPUT,
            "invalid request");
  }

  const auto cancelled = submit(kPrompt, {}, true);
  Require(cancelled.kinds == std::vector{ENGINE_EVENT_TOKENS, ENGINE_EVENT_ERROR} &&
              cancelled.status == ENGINE_ERR_CANCELLED,
          "callback cancellation");

  params.max_tokens = 0;
  const auto empty = submit(kPrompt);
  Require(empty.kinds == std::vector{ENGINE_EVENT_DONE} && empty.telemetry.decode_tokens == 0,
          "zero token limit");

  std::unique_ptr<char, decltype(&engine_string_free)> stats(engine_stats_json(engine),
                                                             engine_string_free);
  Require(stats != nullptr, "missing engine stats");
  const std::string_view json(stats.get());
  Require(json.find("\"submitted_requests\":5") != json.npos &&
              json.find("\"cancelled_requests\":1") != json.npos,
          "engine stats");
}

void TestEngine() {
  Require(engine_abi_version() == FLINT_ENGINE_ABI_VERSION, "engine ABI mismatch");
  char* error = nullptr;
  Require(!engine_create("", &error) && error, "missing config diagnostic");
  engine_string_free(error);

  std::unique_ptr<EngineHandle, decltype(&engine_destroy)> engine(engine_create("{}", &error),
                                                                  engine_destroy);
  std::unique_ptr<char, decltype(&engine_string_free)> diagnostic(error, engine_string_free);
  Require(engine && !diagnostic, error ? error : "engine_create failed");
  CheckRequests(engine.get());
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Require(argc == 3 && std::string_view(argv[1]) == "--forward", "expected --forward PATH");
    TestEngine();
    flint::RunForgeSmoke(argv[2]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

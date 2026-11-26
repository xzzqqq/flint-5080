#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
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
  EngineStopReason stop_reason{};
  bool valid = true;
  std::thread::id caller = std::this_thread::get_id();
  std::promise<void>* done{};
  std::promise<void>* entered{};
  std::shared_future<void> release{};
  std::vector<uint64_t>* order{};
};

void Collect(const EngineEvent* event, void* user_data) {
  auto& capture = *static_cast<Capture*>(user_data);
  if (event->req_id == 0 || (capture.request_id != 0 && capture.request_id != event->req_id)) {
    capture.valid = false;
  }
  capture.valid &= capture.caller != std::this_thread::get_id();
  capture.request_id = event->req_id;
  capture.kinds.push_back(event->kind);
  capture.status = event->status;
  capture.stop_reason = event->stop_reason;
  capture.valid &= event->kind == ENGINE_EVENT_DONE || event->stop_reason == ENGINE_STOP_NONE;
  capture.telemetry = event->telemetry;

  if (event->kind == ENGINE_EVENT_TOKENS && event->token_ids != nullptr) {
    const std::span tokens(event->token_ids, event->n_tokens);
    capture.tokens.insert(capture.tokens.end(), tokens.begin(), tokens.end());
    if (capture.entered) {
      capture.entered->set_value();
      capture.release.wait();
    }
    if (capture.cancel_on_tokens) {
      engine_cancel(capture.engine, event->req_id);
    }
  }
  if (event->kind == ENGINE_EVENT_ERROR && (!event->error_message || !*event->error_message)) {
    capture.valid = false;
  }
  if (event->kind != ENGINE_EVENT_TOKENS) {
    if (capture.order) capture.order->push_back(event->req_id);
    if (capture.done) capture.done->set_value();
  }
}

void CheckRequests(EngineHandle* engine) {
  constexpr int32_t kPrompt[] = {1, 2, 3, 4};
  RequestParams params{.max_tokens = 8};
  uint64_t last_id = 0;
  const auto submit = [&](std::span<const int32_t> prompt, std::span<const MediaSlice> media = {},
                          bool cancel_on_tokens = false) {
    Capture capture{.engine = engine, .cancel_on_tokens = cancel_on_tokens};
    std::promise<void> done;
    auto terminal = done.get_future();
    capture.done = &done;
    const auto id = engine_submit(engine, &params, prompt.data(), prompt.size(), media.data(),
                                  media.size(), Collect, &capture);
    Require(id, "failed to enqueue");
    terminal.get();
    capture.done = nullptr;
    Require(capture.valid && id > last_id && capture.request_id == id, "invalid request/event ID");
    last_id = id;
    return capture;
  };

  const auto success = submit(kPrompt);
  Require(success.kinds == std::vector{ENGINE_EVENT_TOKENS, ENGINE_EVENT_DONE} &&
              success.tokens == std::vector<int32_t>{42} && success.status == ENGINE_OK &&
              success.stop_reason == ENGINE_STOP_EOS,
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
  Require(empty.kinds == std::vector{ENGINE_EVENT_DONE} && empty.telemetry.decode_tokens == 0 &&
              empty.stop_reason == ENGINE_STOP_MAX_TOKENS,
          "zero token limit");

  std::unique_ptr<char, decltype(&engine_string_free)> stats(engine_stats_json(engine),
                                                             engine_string_free);
  Require(stats != nullptr, "missing engine stats");
  const std::string_view json(stats.get());
  Require(json.find("\"submitted_requests\":5") != json.npos &&
              json.find("\"cancelled_requests\":1") != json.npos,
          "engine stats");
}

void CheckQueue(bool destroy) {
  std::unique_ptr<EngineHandle, decltype(&engine_destroy)> owner(engine_create("{}", nullptr),
                                                                 engine_destroy);
  auto* e = owner.get();
  Require(e, "queue engine_create");
  std::promise<void> entered, release;
  auto started = entered.get_future();
  std::vector<uint64_t> order;
  Capture first{
      .engine = e, .entered = &entered, .release = release.get_future().share(), .order = &order};
  Capture second{.engine = e, .order = &order}, third{.engine = e, .order = &order};
  std::promise<void> done;
  auto terminal = done.get_future();
  third.done = &done;
  RequestParams p{.max_tokens = 1};
  int32_t prompt = 1, stop = 42;
  auto submit = [&](Capture& c) {
    return engine_submit(e, &p, &prompt, 1, nullptr, 0, Collect, &c);
  };
  auto a = submit(first);
  started.get();
  auto b = submit(second);
  p.stop_token_ids = &stop;
  p.n_stop_token_ids = 1;
  auto c = submit(third);
  stop = 12;
  p.max_tokens = 0;
  engine_cancel(e, b);
  std::unique_ptr<char, decltype(&engine_string_free)> snapshot(engine_stats_json(e),
                                                                engine_string_free);
  bool queue_stats =
      snapshot &&
      std::string_view(snapshot.get()).find("\"queued_requests\":2") != std::string_view::npos &&
      std::string_view(snapshot.get()).find("\"active_requests\":1") != std::string_view::npos;
  if (destroy) {
    std::thread unblock([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      release.set_value();
    });
    owner.reset();
    unblock.join();
  } else {
    release.set_value();
    terminal.get();
    owner.reset();
  }
  Require(queue_stats, "queue and active counters");
  Require(order == std::vector<uint64_t>{a, b, c}, "FIFO terminal order");
  Require(first.valid && second.valid && third.valid && second.status == ENGINE_ERR_CANCELLED &&
              third.tokens.empty() &&
              first.stop_reason == (destroy ? ENGINE_STOP_NONE : ENGINE_STOP_MAX_TOKENS) &&
              third.stop_reason == (destroy ? ENGINE_STOP_NONE : ENGINE_STOP_STOP_TOKEN) &&
              third.status == (destroy ? ENGINE_ERR_CANCELLED : ENGINE_OK) &&
              first.status == (destroy ? ENGINE_ERR_CANCELLED : ENGINE_OK),
          "pending request outcomes");
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
  CheckQueue(false);
  CheckQueue(true);
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

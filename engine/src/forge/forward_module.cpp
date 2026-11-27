#include "forge/forward_module.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <system_error>

#include "common/cuda_check.h"

namespace flint {
namespace fs = std::filesystem;
namespace {
template <typename Function>
Function Symbol(void* handle, const char* name) {
  dlerror();
  auto* address = dlsym(handle, name);
  if (const char* error = dlerror()) throw std::runtime_error(error);
  return reinterpret_cast<Function>(address);
}
}  // namespace

ForwardModule::ForwardModule(const fs::path& path, const fs::path& load_dir,
                             const ModelWeightsView* weights, const KvStateArenaView& kv,
                             const ScratchArenaView& scratch, cudaStream_t stream)
    : copy_path_((load_dir / (path.stem().string() + "-XXXXXX.so")).string()) {
  // dlopen identifies modules by path. Each load needs its own file, even for A/B of one DSO.
  const int fd = mkstemps(copy_path_.data(), sizeof(".so") - 1);
  if (fd < 0)
    throw std::system_error(errno, std::generic_category(), "create temporary forward module");
  try {
    if (close(fd) < 0)
      throw std::system_error(errno, std::generic_category(), "close temporary forward module");
    fs::copy_file(path, copy_path_, fs::copy_options::overwrite_existing);
    handle_ = dlopen(copy_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw std::runtime_error(dlerror());
    auto version =
        Symbol<decltype(&flint_forward_abi_version)>(handle_, "flint_forward_abi_version");
    if (version() != FLINT_FORWARD_ABI_VERSION)
      throw std::runtime_error("forward ABI version mismatch");
    destroy_ = Symbol<decltype(destroy_)>(handle_, "flint_forward_destroy");
    run_ = Symbol<decltype(run_)>(handle_, "flint_forward_run_layers");
    stats_ = reinterpret_cast<decltype(stats_)>(dlsym(handle_, "flint_forward_graph_stats"));
    store_context_ = Symbol<decltype(store_context_)>(handle_, "flint_forward_store_context");
    draft_ = Symbol<decltype(draft_)>(handle_, "flint_forward_draft");
    verify_ = Symbol<decltype(verify_)>(handle_, "flint_forward_verify");
    spec_round_ = Symbol<decltype(spec_round_)>(handle_, "flint_forward_spec_round");
    commit_ = Symbol<decltype(commit_)>(handle_, "flint_forward_commit");
    auto create = Symbol<decltype(&flint_forward_create)>(handle_, "flint_forward_create");
    context_ = create(weights, &kv, &scratch, stream);
    if (!context_) throw std::runtime_error("forward context creation failed");
  } catch (...) {
    Close();
    throw;
  }
}

ForwardModule::~ForwardModule() { Close(); }

void ForwardModule::Close() noexcept {
  if (context_) {
    // A candidate may report an error after enqueueing work on its stream.
    cudaDeviceSynchronize();
    destroy_(context_);
  }
  if (handle_) dlclose(handle_);
  unlink(copy_path_.c_str());
}

ForwardRunResult ForwardModule::Run(const ForwardRunSpec& spec, const ForwardOptions& opts) {
  ForwardRunResult result{};
  int code = run_(context_, &spec, &opts, &result);
  CheckCuda(code ? code : result.status, "run_layers");
  if (result.num_stages > FLINT_MAX_STAGE_TIMINGS || !std::isfinite(result.elapsed_ms) ||
      result.elapsed_ms < 0)
    throw std::runtime_error("invalid forward timing result");
  return result;
}

void ForwardModule::StoreContext(const void* taps, uint32_t n, uint32_t pos) {
  CheckCuda(store_context_(context_, taps, n, pos), "store draft context");
}
ForwardDraftResult ForwardModule::Draft(const ForwardDraftSpec& spec) {
  ForwardDraftResult result;
  CheckCuda(draft_(context_, &spec, &result), "draft block");
  return result;
}
ForwardRunResult ForwardModule::Verify(const ForwardRunSpec& spec, const ForwardOptions& o) {
  ForwardRunResult r{};
  CheckCuda(verify_(context_, &spec, &o, &r), "verify block");
  return r;
}
ForwardSpecRoundResult ForwardModule::SpecRound(const ForwardSpecRoundSpec& spec) {
  ForwardSpecRoundResult r{};
  CheckCuda(spec_round_(context_, &spec, &speculative_options, &r), "speculative round");
  if (r.output.error_code || r.output.accepted_draft_count < 0 ||
      uint32_t(r.output.accepted_draft_count) >= spec.input.verify_len) {
    Commit(spec.input.pos, spec.input.verify_len, 0);
    throw std::runtime_error("invalid speculative output: " + std::to_string(r.output.error_code));
  }
  return r;
}
void ForwardModule::Commit(uint32_t pos, uint32_t len, uint32_t accepted) {
  CheckCuda(commit_(context_, pos, len, accepted), "commit verified prefix");
}
void ForwardModule::ReportGraphStats(Object& reply, const char* key) const {
  if (stats_) {
    auto stats = stats_(context_);
    reply[key] = Object{{"captures", stats.captures}, {"launches", stats.launches}};
  }
}
}  // namespace flint

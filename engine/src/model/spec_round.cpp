#include "model/spec_round.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <tuple>

#include "common/cuda_check.h"
#include "common/splitmix64.h"
#include "loader/scratch_workspace.h"

namespace flint {
namespace {
struct Inputs {
  ForwardDraftSpec draft;
  SpecVerifyInput verify;
};
struct Storage {
  Inputs input;
  SpecVerifyOutput output;
};
auto GraphKey(const ForwardSpecRoundSpec& s) {
  return std::make_tuple(s.input.verify_len, s.d_input_ids, s.d_hidden_out, s.d_logits_out,
                         s.d_next_token, s.d_taps_out);
}
struct Slot {
  cudaGraphExec_t graph{};
  decltype(GraphKey({})) key{};
  ~Slot() {
    if (graph) cudaGraphExecDestroy(graph);
  }
};
}  // namespace
struct SpeculativeRunner::Impl {
  InklingRunner& target;
  Storage* device;
  DFlash2Runner draft;
  const int vocab;
  std::array<Slot, 3> slots{};
  Storage* host{};
  cudaEvent_t start{}, stop{};
  ForwardGraphStats stats{};

  Impl(const ModelWeightsView& w, const KvStateArenaView& kv, ScratchWorkspace ws,
       InklingRunner& target)
      : target(target),
        device(ws.Take<Storage>(1)),
        draft(w, kv, ws.rest, target.stream()),
        vocab(w.config.unpadded_vocab_size) {}
  ~Impl() {
    if (host) cudaFreeHost(host);
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
  }
  void Enqueue(const ForwardSpecRoundSpec& s, SpecSampleMode mode, bool& backed_up) {
    auto stream = target.stream();
    auto& input = device->input;
    auto proposal = draft.device_result();
    auto len = s.input.verify_len;
    draft.EnqueueBlockAt(&input.draft);
    CheckCuda(
        flint_spec_prepare_verify_ids(&input.draft.anchor, proposal, s.d_input_ids, len, stream));
    target.EnqueueVerifyBody({.seq_len = len,
                              .d_input_ids = s.d_input_ids,
                              .d_hidden_out = s.d_hidden_out,
                              .d_logits_out = s.d_logits_out,
                              .d_next_token = s.d_next_token,
                              .d_taps_out = s.d_taps_out},
                             {.return_logits = s.d_logits_out != nullptr}, &input.draft.pos);
    backed_up = true;
    auto ws = target.scratch_tail();
    CheckCuda(flint_spec_verify_accept(proposal, s.d_next_token, s.d_logits_out, &input.verify,
                                       &device->output, ws.device_ptr, ws.bytes, len, vocab, mode,
                                       stream));
  }
  cudaGraphExec_t Prepare(const ForwardSpecRoundSpec& s, SpecSampleMode mode) {
    auto& slot = slots[mode];
    auto key = GraphKey(s);
    if (slot.graph && slot.key == key) return slot.graph;
    auto stream = target.stream();
    cudaGraph_t source{};
    CheckCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      bool unused{};
      Enqueue(s, mode, unused);
    } catch (...) {
      cudaStreamEndCapture(stream, &source);
      if (source) cudaGraphDestroy(source);
      throw;
    }
    CheckCuda(cudaStreamEndCapture(stream, &source));
    cudaGraphExec_t graph{};
    auto status = cudaGraphInstantiate(&graph, source, 0);
    cudaGraphDestroy(source);
    CheckCuda(status);
    if (slot.graph) cudaGraphExecDestroy(slot.graph);
    slot.graph = graph;
    slot.key = key;
    ++stats.captures;
    return graph;
  }
};
SpeculativeRunner::SpeculativeRunner(const ModelWeightsView& w, const KvStateArenaView& kv,
                                     ScratchArenaView scratch, InklingRunner& target)
    : impl_(std::make_unique<Impl>(w, kv, ScratchWorkspace(scratch), target)) {
  CheckCuda(cudaHostAlloc(&impl_->host, sizeof(Storage), cudaHostAllocDefault));
  CheckCuda(cudaEventCreate(&impl_->start));
  CheckCuda(cudaEventCreate(&impl_->stop));
}
SpeculativeRunner::~SpeculativeRunner() = default;
DFlash2Runner& SpeculativeRunner::draft() { return impl_->draft; }
ForwardGraphStats SpeculativeRunner::graph_stats() const { return impl_->stats; }

ForwardSpecRoundResult SpeculativeRunner::Run(const ForwardSpecRoundSpec& request,
                                              const ForwardOptions& opts) {
  auto& c = *impl_;
  auto s = request;
  auto in = s.input;
  bool stochastic = in.temperature > 0;
  const bool return_logits = stochastic || opts.return_logits;
  if (!return_logits) s.d_logits_out = nullptr;
  if (in.anchor < 0 || in.anchor >= c.vocab || in.pos < 0 || in.verify_len < 2 ||
      !std::isfinite(in.temperature) || in.temperature < 0 || !std::isfinite(in.top_p) ||
      in.top_p < 0 || in.top_p > 1 || in.top_k < 0 || !s.d_input_ids || !s.d_hidden_out ||
      !s.d_next_token || !s.d_taps_out || (return_logits && !s.d_logits_out))
    throw std::invalid_argument("invalid speculative round");
  auto mode = !stochastic                                  ? SPEC_SAMPLE_GREEDY
              : in.top_k || (in.top_p > 0 && in.top_p < 1) ? SPEC_SAMPLE_STOCHASTIC_FILTERED
                                                           : SPEC_SAMPLE_STOCHASTIC_FULL;
  auto stream = c.target.stream();
  c.target.BeginExternalVerify(in.pos, in.verify_len);
  bool backed_up = false;
  try {
    auto graph = opts.use_cuda_graph && s.cache_graph ? c.Prepare(s, mode) : nullptr;
    c.host->input.draft = {in.anchor, in.pos, in.temperature, in.seed};
    if (stochastic) in.seed += kDraftTokens * kSplitMixIncrement;
    c.host->input.verify = in;
    CheckCuda(cudaMemcpyAsync(&c.device->input, &c.host->input, sizeof(Inputs),
                              cudaMemcpyHostToDevice, stream));
    CheckCuda(cudaEventRecord(c.start, stream));
    if (graph) {
      CheckCuda(cudaGraphLaunch(graph, stream));
      backed_up = true;
      ++c.stats.launches;
    } else {
      c.Enqueue(s, mode, backed_up);
    }
    CheckCuda(cudaEventRecord(c.stop, stream));
    CheckCuda(cudaMemcpyAsync(&c.host->output, &c.device->output, sizeof(SpecVerifyOutput),
                              cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaStreamSynchronize(stream));
    ForwardSpecRoundResult result{c.host->output};
    CheckCuda(cudaEventElapsedTime(&result.elapsed_ms, c.start, c.stop));
    return result;
  } catch (...) {
    if (backed_up)
      c.target.CommitVerify(in.pos, in.verify_len, 0);
    else
      c.target.CancelExternalVerify();
    // Finish any queued input upload before the pinned packet can be reused.
    CheckCuda(cudaStreamSynchronize(stream));
    throw;
  }
}
}  // namespace flint

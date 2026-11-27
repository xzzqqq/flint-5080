#include <cuda_runtime.h>

#include <climits>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda/functional>

#include "common/splitmix64.h"
#include "ops/spec_verify_ops.h"

namespace {
constexpr int kTile = 256;
using Reduce = cub::BlockReduce<double, kTile>;
struct Row {
  float maximum;
  double sum;
  int keep;
};
struct Decision {
  double uniform, q_total;
};

// The query and launch share this layout. The final region belongs to CUB.
struct Layout {
  char* base;
  size_t used{};
  template <class T>
  T* Take(size_t count) {
    used = (used + 255) & ~size_t(255);
    auto* ptr = base ? reinterpret_cast<T*>(base + used) : nullptr;
    used += count * sizeof(T);
    return ptr;
  }
};
struct Workspace {
  Row* rows;
  Decision* decision;
  double *partials, *mass;
  float* weights;
  uint32_t *keys{}, *sorted_keys{};
  int *ids{}, *ranked{}, *offsets{};
  void* temp{};
  size_t temp_bytes{};
};
int Tiles(int vocab) { return 1 + (vocab - 1) / kTile; }
bool Shape(int len, int vocab) {
  return len > 0 && len <= kDraftBlockSize && vocab > 0 && vocab <= INT_MAX / len;
}
cudaError_t Sort(const Workspace& w, int len, int vocab, size_t& bytes, cudaStream_t stream) {
  return cub::DeviceSegmentedRadixSort::SortPairsDescending(
      w.temp, bytes, static_cast<const uint32_t*>(w.keys), w.sorted_keys,
      static_cast<const int*>(w.ids), w.ranked, len * vocab, len, w.offsets,
      w.offsets ? w.offsets + 1 : nullptr, 0, 32, stream);
}
cudaError_t MakeWorkspace(Layout& layout, Workspace& w, int len, int vocab, SpecSampleMode mode,
                          cudaStream_t stream) {
  const size_t n = size_t(len) * vocab;
  w.rows = layout.Take<Row>(len);
  w.decision = layout.Take<Decision>(1);
  w.partials = layout.Take<double>(size_t(len) * Tiles(vocab));
  w.mass = layout.Take<double>(vocab);
  w.weights = layout.Take<float>(n);
  if (mode == SPEC_SAMPLE_STOCHASTIC_FILTERED) {
    w.keys = layout.Take<uint32_t>(n);
    w.sorted_keys = layout.Take<uint32_t>(n);
    w.ids = layout.Take<int>(n);
    w.ranked = layout.Take<int>(n);
    w.offsets = layout.Take<int>(len + 1);
    auto err = Sort(w, len, vocab, w.temp_bytes, stream);
    if (err != cudaSuccess) return err;
    w.temp = layout.Take<char>(w.temp_bytes);
  }
  return cudaSuccess;
}
__global__ void PrepareIds(const int* anchor, const ForwardDraftResult* draft, int* ids) {
  int i = threadIdx.x;
  ids[i] = i ? draft->draft_ids[i - 1] : *anchor;
}
__global__ void Invalid(SpecVerifyOutput* out) { out->error_code = SPEC_VERIFY_INVALID_PARAM; }

__global__ void Greedy(const ForwardDraftResult* draft, const int* target,
                       const SpecVerifyInput* in, SpecVerifyOutput* out, int len) {
  *out = {};
  out->next_seed = in->seed;
  if (in->temperature != 0 || in->verify_len != len) {
    out->error_code = SPEC_VERIFY_INVALID_PARAM;
    return;
  }
  int r = 0;
  while (r + 1 < len && draft->draft_ids[r] == target[r]) {
    out->tokens[r] = target[r];
    ++r;
  }
  out->accepted_draft_count = r;
  out->tokens[r] = target[r];
}

__device__ float Weight(float x, float maximum, float temperature) {
  if (!isfinite(x)) return 0;
  float d = x - maximum;
  return expf(isfinite(d) ? d / temperature : float((double(x) - maximum) / temperature));
}
__global__ void TileMax(const float* logits, Workspace w, int vocab) {
  int j = blockIdx.x * kTile + threadIdx.x;
  float x = j < vocab ? logits[blockIdx.y * vocab + j] : -INFINITY;
  __shared__ Reduce::TempStorage tmp;
  double maximum = Reduce(tmp).Reduce(isfinite(x) ? x : -INFINITY, cuda::maximum<>{});
  if (!threadIdx.x) w.partials[blockIdx.y * gridDim.x + blockIdx.x] = maximum;
}
__global__ void RowMax(Workspace w, int tiles) {
  double m = -INFINITY;
  for (int b = threadIdx.x; b < tiles; b += kTile) m = fmax(m, w.partials[blockIdx.x * tiles + b]);
  __shared__ Reduce::TempStorage tmp;
  double maximum = Reduce(tmp).Reduce(m, cuda::maximum<>{});
  if (!threadIdx.x) w.rows[blockIdx.x].maximum = maximum;
}
__global__ void SortKeys(const float* logits, const SpecVerifyInput* in, Workspace w, int vocab) {
  int j = blockIdx.x * kTile + threadIdx.x;
  int r = blockIdx.y;
  if (j == 0) {
    w.offsets[r] = r * vocab;
    if (r + 1 == gridDim.y) w.offsets[r + 1] = (r + 1) * vocab;
  }
  if (j >= vocab) return;
  int i = r * vocab + j;
  float x = logits[i];
  // top-k ranks logits; nucleus alone ranks rounded FP32 weights, including ties.
  if (in->top_k == 0) x = Weight(x, w.rows[r].maximum, in->temperature);
  uint32_t bits = __float_as_uint(x == 0 ? 0.f : x);
  w.keys[i] = isfinite(x) ? (bits ^ ((bits >> 31) ? UINT32_MAX : 0x80000000u)) : 0;
  w.ids[i] = j;
}
__global__ void TileWeights(const float* logits, const SpecVerifyInput* in, Workspace w,
                            int vocab) {
  int r = blockIdx.y, j = blockIdx.x * kTile + threadIdx.x;
  float weight = 0;
  if (j < vocab) {
    int i = r * vocab + (w.ranked ? w.ranked[r * vocab + j] : j);
    if (in->top_k <= 0 || j < in->top_k)
      weight = Weight(logits[i], w.rows[r].maximum, in->temperature);
    w.weights[i] = weight;
  }
  __shared__ Reduce::TempStorage tmp;
  double sum = Reduce(tmp).Sum(double(weight));
  if (!threadIdx.x) w.partials[r * gridDim.x + blockIdx.x] = sum;
}
__global__ void RowSum(const SpecVerifyInput* in, Workspace w, int tiles, int vocab) {
  int r = blockIdx.x;
  const double* partials = w.partials + r * tiles;
  double sum = 0;
  for (int b = 0; b < tiles; ++b) sum += partials[b];
  int keep = vocab;
  if (w.ranked && in->top_p > 0 && in->top_p < 1 && sum > 0) {
    const int* ranked = w.ranked + r * vocab;
    const float* weights = w.weights + r * vocab;
    const double target = sum * in->top_p;
    const double cutoff = weights[ranked[0]] >= sum * 1e-7 ? sum * 1e-7 : 0;
    sum = 0;
    // Skip whole tiles before the first threshold crossing, then walk its prefix.
    for (int b = 0; b < tiles; ++b) {
      int first = b * kTile, end = first + min(kTile, vocab - first);
      if (sum + partials[b] < target && weights[ranked[end - 1]] >= cutoff) {
        sum += partials[b];
        continue;
      }
      keep = first;
      while (keep < end && weights[ranked[keep]] >= cutoff && sum < target)
        sum += weights[ranked[keep++]];
      if (keep < end || sum >= target) break;
    }
  }
  w.rows[r].sum = sum;
  w.rows[r].keep = keep;
}
__global__ void TrimNucleus(Workspace w, int vocab) {
  int r = blockIdx.y, j = blockIdx.x * kTile + threadIdx.x;
  if (j >= w.rows[r].keep && j < vocab) w.weights[r * vocab + w.ranked[r * vocab + j]] = 0;
}

__global__ void Accept(const ForwardDraftResult* draft, const SpecVerifyInput* in,
                       SpecVerifyOutput* out, Workspace w, int len, int vocab) {
  *out = {};
  uint64_t rng = in->seed;
  bool filtered = in->top_k > 0 || (in->top_p > 0 && in->top_p < 1);
  if (in->verify_len != len || !(in->temperature > 0) || in->top_k < 0 ||
      !(in->top_p >= 0 && in->top_p <= 1) || filtered != bool(w.ranked)) {
    out->error_code = SPEC_VERIFY_INVALID_PARAM;
    return;
  }
  for (int r = 0; r < len; ++r) {
    if (!isfinite(w.rows[r].maximum)) {
      out->error_code = SPEC_VERIFY_NO_FINITE_LOGITS;
      return;
    }
    out->accepted_draft_count = r;
    if (r == len - 1) break;
    int id = draft->draft_ids[r];
    if (id < 0 || id >= vocab) {
      out->error_code = SPEC_VERIFY_DRAFT_OUTSIDE_SUPPORT;
      return;
    }
    const int* support = draft->candidate_ids + r * kSelectorTopK;
    const float* q = draft->q_probs + r * kSelectorTopK;
    int at = -1;
    double total = 0;
    for (int j = 0; j < kSelectorTopK; ++j) {
      if (support[j] < 0 || support[j] >= vocab) {
        out->error_code = SPEC_VERIFY_INVALID_CANDIDATE;
        return;
      }
      if (support[j] == id && at < 0) at = j;
      total += q[j];
    }
    if (at < 0 || !(q[at] > 0)) {
      out->error_code = SPEC_VERIFY_DRAFT_OUTSIDE_SUPPORT;
      return;
    }
    double p = w.weights[r * vocab + id] / w.rows[r].sum;
    if (!(flint::SplitMixUniform(rng) * (q[at] / total) < p)) {
      w.decision->q_total = total;
      break;
    }
    out->tokens[r] = id;
  }
  w.decision->uniform = flint::SplitMixUniform(rng);
  out->next_seed = rng;
}
__global__ void WinningMass(const ForwardDraftResult* draft, const SpecVerifyOutput* out,
                            Workspace w, int len, int vocab) {
  if (out->error_code) return;
  int r = out->accepted_draft_count, j = blockIdx.x * kTile + threadIdx.x;
  double mass = 0;
  if (j < vocab) {
    bool bonus = r + 1 == len;
    // Sampler::Sample traverses retained candidates in ranked order on a filtered bonus.
    int id = bonus && w.ranked ? w.ranked[r * vocab + j] : j;
    mass = w.weights[r * vocab + id];
    if (!bonus) {
      mass /= w.rows[r].sum;
      for (int k = 0; k < kSelectorTopK; ++k) {
        int at = r * kSelectorTopK + k;
        if (draft->candidate_ids[at] == id)
          mass = fmax(0., mass - draft->q_probs[at] / w.decision->q_total);
      }
    }
    w.mass[j] = mass;
  }
  __shared__ Reduce::TempStorage tmp;
  double sum = Reduce(tmp).Sum(mass);
  if (!threadIdx.x) w.partials[blockIdx.x] = sum;
}
__global__ void Sample(SpecVerifyOutput* out, Workspace w, int len, int vocab, int tiles) {
  if (out->error_code) return;
  double sum = 0;
  int last = -1;
  for (int b = 0; b < tiles; ++b) {
    sum += w.partials[b];
    if (w.partials[b] > 0) last = b;
  }
  if (last < 0) {
    out->error_code = SPEC_VERIFY_EMPTY_RESIDUAL;
    return;
  }
  double draw = w.decision->uniform * sum;
  int b = 0;
  while (b < last && draw >= w.partials[b]) draw -= w.partials[b++];
  int r = out->accepted_draft_count;
  int first = b * kTile, end = first + min(kTile, vocab - first);
  for (int j = first; j < end; ++j) {
    if (w.mass[j] <= 0) continue;
    out->tokens[r] = r + 1 == len && w.ranked ? w.ranked[r * vocab + j] : j;
    if ((draw -= w.mass[j]) < 0) break;
  }
}
}  // namespace

int flint_spec_prepare_verify_ids(const int* anchor, const ForwardDraftResult* draft, int* ids,
                                  int len, void* stream) {
  if (!anchor || !draft || !ids || !Shape(len, 1)) return cudaErrorInvalidValue;
  PrepareIds<<<1, len, 0, static_cast<cudaStream_t>(stream)>>>(anchor, draft, ids);
  return cudaGetLastError();
}

size_t flint_spec_verify_scratch_bytes(int len, int vocab, SpecSampleMode mode) {
  if (!Shape(len, vocab) ||
      (mode != SPEC_SAMPLE_STOCHASTIC_FULL && mode != SPEC_SAMPLE_STOCHASTIC_FILTERED))
    return 0;
  Layout layout{};
  Workspace w{};
  return MakeWorkspace(layout, w, len, vocab, mode, nullptr) == cudaSuccess ? layout.used : 0;
}

int flint_spec_verify_accept(const ForwardDraftResult* draft, const int* argmax,
                             const float* logits, const SpecVerifyInput* in, SpecVerifyOutput* out,
                             void* scratch, size_t bytes, int len, int vocab, SpecSampleMode mode,
                             void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (!out) return cudaErrorInvalidValue;
  auto invalid = [&] {
    Invalid<<<1, 1, 0, s>>>(out);
    return cudaGetLastError();
  };
  if (!Shape(len, vocab) || !draft || !in) return invalid();
  if (mode == SPEC_SAMPLE_GREEDY) {
    if (!argmax) return invalid();
    Greedy<<<1, 1, 0, s>>>(draft, argmax, in, out, len);
    return cudaGetLastError();
  }
  if ((mode != SPEC_SAMPLE_STOCHASTIC_FULL && mode != SPEC_SAMPLE_STOCHASTIC_FILTERED) || !logits ||
      !scratch || reinterpret_cast<uintptr_t>(scratch) % 256)
    return invalid();
  Layout layout{static_cast<char*>(scratch)};
  Workspace w{};
  auto err = MakeWorkspace(layout, w, len, vocab, mode, s);
  if (err != cudaSuccess) return err;
  if (bytes < layout.used) return invalid();
  int tiles = Tiles(vocab);
  dim3 grid(tiles, len);
  TileMax<<<grid, kTile, 0, s>>>(logits, w, vocab);
  RowMax<<<len, kTile, 0, s>>>(w, tiles);
  if (w.ranked) {
    SortKeys<<<grid, kTile, 0, s>>>(logits, in, w, vocab);
    err = Sort(w, len, vocab, w.temp_bytes, s);
    if (err != cudaSuccess) return err;
  }
  TileWeights<<<grid, kTile, 0, s>>>(logits, in, w, vocab);
  RowSum<<<len, 1, 0, s>>>(in, w, tiles, vocab);
  if (w.ranked) TrimNucleus<<<grid, kTile, 0, s>>>(w, vocab);
  Accept<<<1, 1, 0, s>>>(draft, in, out, w, len, vocab);
  WinningMass<<<tiles, kTile, 0, s>>>(draft, out, w, len, vocab);
  Sample<<<1, 1, 0, s>>>(out, w, len, vocab, tiles);
  return cudaGetLastError();
}

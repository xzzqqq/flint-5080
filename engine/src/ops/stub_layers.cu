#include <cuda_runtime.h>

#include "op_params.h"

namespace {
constexpr unsigned kThreads = 256;

__global__ void Embed(StubParams p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  unsigned col = i % p.width, token = static_cast<unsigned>(p.ids[i / p.width]) % p.vocab;
  float weight = p.weight ? p.weight[size_t(token) * p.width + col] / 65536.f : 0.f;
  p.output[i] = weight + ((token + col + p.position + i / p.width) % 1024) / 1024.f;
}

__global__ void Visit(StubParams p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  float value = p.input[i] + (p.visit + 1) / 1024.f;
  if (p.weight) value += p.weight[i % p.width] / 1048576.f;
  if (i == 0 && p.kv && p.conv) {
    value += (p.kv[p.visit] + p.conv[p.visit]) / 128.f;
    p.kv[p.visit] = value;
    p.conv[p.visit] = value / 2.f;
  }
  p.output[i] = value;
}

__global__ void Head(StubParams p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.vocab) return;
  int token = static_cast<unsigned>(fabsf(p.input[p.count - p.width]) * 4096.f) % p.vocab;
  if (i == 0 && p.next) *p.next = token;
  if (p.logits) p.logits[i] = -abs(int(i) - token) / 1024.f;
}
}  // namespace
int flint_stub_embed(StubParams p, void* stream) {
  Embed<<<(p.count + kThreads - 1) / kThreads, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(p);
  return cudaGetLastError();
}
int flint_stub_visit(StubParams p, void* stream) {
  Visit<<<(p.count + kThreads - 1) / kThreads, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(p);
  return cudaGetLastError();
}
int flint_stub_head(StubParams p, void* stream) {
  Head<<<(p.vocab + kThreads - 1) / kThreads, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(p);
  return cudaGetLastError();
}

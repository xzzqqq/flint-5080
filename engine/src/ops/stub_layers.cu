#include <cuda_bf16.h>

#include "stub_ops.h"

namespace {
constexpr unsigned kThreads = 256;

__global__ void Embed(StubParams p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  unsigned col = i % p.width, token = static_cast<unsigned>(p.ids[i / p.width]) % p.vocab;
  p.output[i] = __bfloat16_as_ushort(
      __float2bfloat16_rn(((token + col + p.position + i / p.width) % 1024) / 1024.f));
}

__global__ void Visit(StubParams p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count) return;
  float value = __bfloat162float(__ushort_as_bfloat16(p.input[i])) + (p.visit + 1) / 1024.f;
  if (i == 0) {
    value += (__bfloat162float(__ushort_as_bfloat16(*p.k)) + *p.conv) / 128.f;
    *p.k = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    *p.v = *p.k;
    *p.conv = value / 2.f;
  }
  p.output[i] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void Head(StubParams p) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.vocab) return;
  int token =
      static_cast<unsigned>(
          fabsf(__bfloat162float(__ushort_as_bfloat16(p.input[p.count - p.width]))) * 4096.f) %
      p.vocab;
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

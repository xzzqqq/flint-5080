#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "op_params.h"

namespace {
__device__ float Read(const void* x, int i, FlintType type) {
  if (type == FLINT_F32) return static_cast<const float*>(x)[i];
  if (type == FLINT_F16) return __half2float(static_cast<const __half*>(x)[i]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(x)[i]);
}
__device__ void Write(void* y, int i, float v, FlintType type) {
  if (type == FLINT_F32)
    static_cast<float*>(y)[i] = v;
  else if (type == FLINT_F16)
    static_cast<__half*>(y)[i] = __float2half_rn(v);
  else
    static_cast<__nv_bfloat16*>(y)[i] = __float2bfloat16_rn(v);
}

template <int K2>
__global__ void Decode(DecodeParams p) {
  int tile = blockIdx.x, step = threadIdx.x;
  const unsigned* words = reinterpret_cast<const unsigned*>(p.trellis) + tile * (4 * K2);
  int start = ((step + 1) * K2 / 2 - 16 + 128 * K2) % (128 * K2);
  unsigned window =
      __funnelshift_l(words[(start / 32 + 1) % (4 * K2)], words[start / 32], start % 32);
  unsigned product = (window >> 16) * 0x83DCD12Du;
  int sum = __dp4a(product, 0x01010101u, 1024u);
  int lane = step / 8, j = step % 8;
  int r = (tile / (p.n / 16)) * 16 + 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1);
  int c = (tile % (p.n / 16)) * 16 + lane / 4 + 8 * (j / 4);
  static_cast<__half*>(p.inner)[r * p.n + c] = __float2half_rn(
      sum * __half2float(__ushort_as_half(0x1eee)) + __half2float(__ushort_as_half(0xc931)));
}
__global__ void Wht(WhtParams p, bool scale_in) {
  __shared__ float values[128];
  int i = blockIdx.x * 128 + threadIdx.x;
  float scale = Read(p.scale, i % p.width, FLINT_F16);
  values[threadIdx.x] = Read(p.x, i, p.input) * (scale_in ? scale : 1.f);
  __syncthreads();
  // The reference's padded FP32 GEMM splits K into four 32-element reductions.
  float sum = 0;
  for (int begin = 0; begin < 128; begin += 32) {
    float part = 0;
    for (int j = begin; j < begin + 32; ++j) {
      float h = (__popc(j & threadIdx.x) & 1) ? -0.0883883461356163f : 0.0883883461356163f;
      part = fmaf(values[j], h, part);
    }
    sum = __fadd_rn(sum, part);
  }
  Write(p.y, i, scale_in ? sum : sum * scale, p.output);
}
int Rotate(WhtParams p, void* s, bool in) {
  if (p.rows <= 0 || p.width <= 0 || p.width % 128) return cudaErrorInvalidValue;
  Wht<<<p.rows*(p.width / 128), 128, 0, static_cast<cudaStream_t>(s)>>>(p, in);
  return cudaGetLastError();
}
__global__ void Cast(CastParams p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < p.count) Write(p.y, i, Read(p.x, i, p.input), p.output);
}
}  // namespace
int flint_cast(CastParams p, void* s) {
  Cast<<<(p.count + 255) / 256, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
  return cudaGetLastError();
}
int flint_exl3_decode_inner_f16(DecodeParams p, void* s) {
  if (p.k <= 0 || p.n <= 0 || p.k % 16 || p.n % 16) return cudaErrorInvalidValue;
  int tiles = p.k / 16 * (p.n / 16);
  switch (p.k2) {
    case 6:
      Decode<6><<<tiles, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
      break;
    case 7:
      Decode<7><<<tiles, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
      break;
    case 12:
      Decode<12><<<tiles, 256, 0, static_cast<cudaStream_t>(s)>>>(p);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
int flint_wht128_scale_in(WhtParams p, void* s) { return Rotate(p, s, true); }
int flint_wht128_scale_out(WhtParams p, void* s) { return Rotate(p, s, false); }

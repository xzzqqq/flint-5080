#include "ops/gemm_cublas.h"

#include <stdexcept>

#include "ops/scratch_workspace.h"
namespace flint {
namespace {
void Check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasGetStatusString(s));
}
cudaDataType_t Type(FlintType t) {
  switch (t) {
    case FLINT_F32:
      return CUDA_R_32F;
    case FLINT_BF16:
      return CUDA_R_16BF;
    case FLINT_F16:
      return CUDA_R_16F;
  }
  throw std::invalid_argument("unsupported GEMM dtype");
}
}  // namespace
Gemm::Gemm(cudaStream_t stream, void* workspace, size_t bytes) : stream_(stream) {
  Check(cublasCreate(&handle_));
  try {
    Check(cublasSetStream(handle_, stream));
    Check(cublasSetWorkspace(handle_, workspace, bytes));
  } catch (...) {
    cublasDestroy(handle_);
    throw;
  }
}
Gemm::~Gemm() { cublasDestroy(handle_); }
void Gemm::Multiply(const void* a, const void* b, void* c, int m, int n, int k, FlintType operand,
                    FlintType output, bool transpose_b) const {
  float one = 1, zero = 0;
  Check(cublasGemmEx(handle_, transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
                     b, Type(operand), transpose_b ? k : n, a, Type(operand), k, &zero, c,
                     Type(output), n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
// Mirror row_matmul in the reference: cuBLAS kernel choice (and hence BF16
// rounding) depends on M. Pad the tail and reuse one pair of 32-row tiles.
void RunRowMatmul(const Gemm& g, ScratchArenaView scratch, const void* a, const void* b, void* c,
                  int m, int n, int k, FlintType input, FlintType output, bool transpose) {
  ops_detail::ScratchWorkspace ws(scratch);
  size_t stride_a = size_t(k) * (input == FLINT_F32 ? 4 : 2),
         stride_c = size_t(n) * (output == FLINT_F32 ? 4 : 2);
  auto x = static_cast<const char*>(a);
  auto y = static_cast<char*>(c);
  while (m >= 32) {
    g.Multiply(x, b, y, 32, n, k, input, output, transpose);
    x += 32 * stride_a;
    y += 32 * stride_c;
    m -= 32;
  }
  if (!m) return;
  auto padded = ws.Take<char>(32 * stride_a), out = ws.Take<char>(32 * stride_c);
  ops_detail::CheckCuda(
      cudaMemcpyAsync(padded, x, m * stride_a, cudaMemcpyDeviceToDevice, g.stream()));
  ops_detail::CheckCuda(cudaMemsetAsync(padded + m * stride_a, 0, (32 - m) * stride_a, g.stream()));
  g.Multiply(padded, b, out, 32, n, k, input, output, transpose);
  ops_detail::CheckCuda(
      cudaMemcpyAsync(y, out, m * stride_c, cudaMemcpyDeviceToDevice, g.stream()));
}
void RunFp32Linear(const Gemm& g, ScratchArenaView scratch, const void* x, FlintType input,
                   const void* weight, FlintType weight_type, float* out, int rows, int n, int k) {
  ops_detail::ScratchWorkspace ws(scratch);
  auto promote = [&](const void* src, FlintType type, int count) -> const void* {
    if (type == FLINT_F32) return src;
    auto dst = ws.Take<float>(count);
    ops_detail::CheckCuda(flint_cast({src, dst, count, type, FLINT_F32}, g.stream()));
    return dst;
  };
  auto a = promote(x, input, rows * k);
  auto b = promote(weight, weight_type, n * k);
  RunRowMatmul(g, ws.rest, a, b, out, rows, n, k, FLINT_F32, FLINT_F32, true);
}
}  // namespace flint

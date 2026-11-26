#include "ops/gemm_cublas.h"

#include <stdexcept>

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
void Gemm::LinearStrided(const float* x, const float* w, float* y, int m, int n, int k,
                         int weight_stride, bool accumulate) const {
  float one = 1, beta = accumulate ? 1.f : 0.f;
  Check(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w, weight_stride, x, k, &beta,
                    y, n));
}
}  // namespace flint

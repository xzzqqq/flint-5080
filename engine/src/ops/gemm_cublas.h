#pragma once
#include <cublas_v2.h>

#include "ops/op_types.h"
namespace flint {
// One handle per execution stream. Workspace is caller-owned for the handle lifetime.
class Gemm {
 public:
  Gemm(cudaStream_t stream, void* workspace, size_t bytes);
  ~Gemm();
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;
  cublasHandle_t handle() const { return handle_; }
  cudaStream_t stream() const { return stream_; }
  // A[m,k], B[n,k] when transposed or B[k,n] otherwise, C[m,n].
  void Multiply(const void* a, const void* b, void* c, int m, int n, int k, FlintType operand_type,
                FlintType output_type, bool transpose_b) const;

  void LinearStrided(const float* x, const float* w, float* y, int m, int n, int k,
                     int weight_stride, bool accumulate) const;

 private:
  cublasHandle_t handle_{};
  cudaStream_t stream_;
};
}  // namespace flint

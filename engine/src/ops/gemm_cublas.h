#pragma once
#include <cublas_v2.h>

#include "loader/model_views.h"
#include "ops/op_params.h"
namespace flint {
// One handle per execution stream. Workspace is caller-owned for the handle lifetime.
class Gemm {
 public:
  Gemm(cudaStream_t stream, void* workspace, size_t bytes);
  ~Gemm();
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;
  cudaStream_t stream() const { return stream_; }
  // A[m,k], B[n,k] when transposed or B[k,n] otherwise, C[m,n].
  void Multiply(const void* a, const void* b, void* c, int m, int n, int k, FlintType operand_type,
                FlintType output_type, bool transpose_b) const;

 private:
  cublasHandle_t handle_{};
  cudaStream_t stream_;
};
// Promote mixed input/weight types to FP32 in scratch for C = A @ B^T.
void RunFp32Linear(const Gemm&, ScratchArenaView, const void* x, FlintType input,
                   const void* weight, FlintType weight_type, float* out, int rows, int n, int k);
// Matches the reference's padded 32-row reduction for either orientation of B.
void RunRowMatmul(const Gemm&, ScratchArenaView, const void* a, const void* b, void* c, int rows,
                  int n, int k, FlintType input, FlintType output, bool transpose);
}  // namespace flint

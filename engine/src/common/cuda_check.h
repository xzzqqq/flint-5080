#pragma once
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
namespace flint {
class CudaError : public std::runtime_error {
 public:
  CudaError(cudaError_t code, const char* context)
      : std::runtime_error((context ? std::string(context) + ": " : "") + cudaGetErrorString(code)),
        code_(code) {}
  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};
inline void CheckCuda(cudaError_t err, const char* context = nullptr) {
  if (err != cudaSuccess) throw CudaError(err, context);
}
inline void CheckCuda(int code, const char* context = nullptr) {
  CheckCuda(static_cast<cudaError_t>(code), context);
}
}  // namespace flint

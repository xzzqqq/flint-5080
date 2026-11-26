#pragma once

#include <cuda_runtime_api.h>

#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
// CUDA runtime-only toolkit installations export both APIs without the profiler header.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif

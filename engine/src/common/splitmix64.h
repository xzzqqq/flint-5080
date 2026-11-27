#pragma once
#include <stdint.h>
namespace flint {
inline constexpr uint64_t kSplitMixIncrement = 0x9e3779b97f4a7c15ULL;
#ifdef __CUDACC__
__host__ __device__
#endif
    inline double SplitMixUniform(uint64_t& rng) {
  uint64_t bits = (rng += kSplitMixIncrement);
  bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
  bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<double>((bits ^ (bits >> 31)) >> 11) * 0x1p-53;
}
}  // namespace flint

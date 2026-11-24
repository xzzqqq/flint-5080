#ifndef FLINT_FORGE_SMOKE_H_
#define FLINT_FORGE_SMOKE_H_

#include <filesystem>

namespace flint {
// Throws with a diagnostic on loader, CUDA, numerical or leak-check failure.
void RunForgeSmoke(const std::filesystem::path& forward_path);
}  // namespace flint

#endif  // FLINT_FORGE_SMOKE_H_

#pragma once
#include <filesystem>
#include <span>

#include "common/json_output.h"

namespace flint {
struct DumpTensor {
  std::string name, dtype;
  Array shape;
  std::span<const std::byte> data;
};
Object WriteDump(const std::filesystem::path& path, const std::vector<DumpTensor>& tensors);
}  // namespace flint

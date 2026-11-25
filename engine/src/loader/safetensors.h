#ifndef FLINT_SAFETENSORS_H
#define FLINT_SAFETENSORS_H

#include <cstddef>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "loader/model_views.h"

namespace flint {
enum class TensorDType { kBF16, kF16, kF32, kF64, kI16, kI32, kI64, kU8, kU16, kU32, kBool };
size_t TensorDTypeBytes(TensorDType dtype);

struct TensorEntry {
  std::string name;
  TensorDType dtype;
  std::vector<int64_t> shape;
  size_t data_offset;  // Relative to the payload.
  size_t nbytes;
};

class SafetensorsFile {
 public:
  static SafetensorsFile Open(const std::filesystem::path& path);
  const TensorEntry* Find(std::string_view name) const;
  std::span<const std::byte> Bytes(const TensorEntry& tensor) const;
  const auto& tensors() const { return tensors_; }
  const auto& metadata() const { return metadata_; }
  size_t file_size() const { return mapping_.get_deleter().size; }
  size_t payload_bytes() const { return file_size() - payload_offset_; }

 private:
  SafetensorsFile() = default;
  struct Unmap {
    size_t size;
    void operator()(const std::byte* mapping) const;
  };
  std::unique_ptr<const std::byte, Unmap> mapping_{nullptr, Unmap{0}};
  size_t payload_offset_ = 0;
  std::vector<TensorEntry> tensors_;
  std::map<std::string, size_t, std::less<>> index_;
  std::map<std::string, std::string> metadata_;
};

ModelConfig LoadModelConfig(const std::filesystem::path& path);
}  // namespace flint
#endif

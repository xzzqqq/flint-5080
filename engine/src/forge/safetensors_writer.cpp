#include "forge/safetensors_writer.h"

#include <unistd.h>

#include <bit>
#include <fstream>

#include "loader/safetensors.h"

namespace flint {
Object WriteDump(const std::filesystem::path& path, const std::vector<DumpTensor>& tensors) {
  static_assert(std::endian::native == std::endian::little);
  Object header;
  size_t pos = 0;
  for (auto& t : tensors) {
    header[t.name] = Object{
        {"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", Array{pos, pos + t.data.size()}}};
    pos += t.data.size();
  }
  auto text = Encode(header);
  text.resize((text.size() + 7) / 8 * 8, ' ');
  uint64_t size = text.size();
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  auto tmp = path.string() + ".tmp." + std::to_string(getpid());
  try {
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(tmp, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(text.data(), size);
    for (auto& t : tensors) out.write(reinterpret_cast<const char*>(t.data.data()), t.data.size());
    out.close();
    auto file = SafetensorsFile::Open(tmp);
    std::filesystem::rename(tmp, path);
    return {{"path", path.string()}, {"file_bytes", file.file_size()}, {"tensors", header}};
  } catch (...) {
    std::filesystem::remove(tmp);
    throw;
  }
}
}  // namespace flint

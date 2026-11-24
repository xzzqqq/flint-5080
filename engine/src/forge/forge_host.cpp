#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>

#include "forge_smoke.h"

namespace {
std::filesystem::path ParseSmokeCommand(std::span<char*> arguments) {
  std::filesystem::path forward_path;
  bool smoke = false;
  for (size_t i = 0; i < arguments.size(); ++i) {
    const std::string_view argument(arguments[i]);
    if (argument == "--smoke") {
      smoke = true;
    } else if (argument == "--forward" && i + 1 < arguments.size()) {
      forward_path = arguments[++i];
    } else {
      throw std::runtime_error("usage: forge-host --smoke [--forward PATH]");
    }
  }
  if (!smoke) {
    throw std::runtime_error("expected --smoke");
  }
  if (forward_path.empty()) {
    return std::filesystem::read_symlink("/proc/self/exe").parent_path() / "forward.so";
  }
  return forward_path;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const std::span arguments(argv, static_cast<size_t>(argc));
    flint::RunForgeSmoke(ParseSmokeCommand(arguments.subspan(1)));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "forge-host: " << error.what() << '\n';
    return 1;
  }
}

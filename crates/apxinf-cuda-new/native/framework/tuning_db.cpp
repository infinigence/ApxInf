// Best-effort persistent storage for exact-key recipes.
//
// The file name is an FNV-1a hash of the caller-provided exact key. The file
// stores the full key on its first line and the opaque serialized recipe on its
// second, so reads still reject hash collisions and truncated content. Writes
// use a process-specific temporary file followed by rename. Empty directories,
// missing or malformed files, and I/O failures behave as cache misses; callers
// own recipe serialization, parsing, and implementation validation.
#include "runtime_internal.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace apxinf::framework {
namespace {

std::string path(const std::string& directory, const std::string& key) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char character : key) {
    hash ^= character;
    hash *= 1099511628211ULL;
  }
  std::ostringstream output;
  output << directory << '/' << std::hex << hash << ".recipe";
  return output.str();
}

}  // namespace

std::string read_recipe(const std::string& directory, const std::string& key) {
  if (directory.empty()) return {};
  std::ifstream input(path(directory, key));
  std::string stored_key;
  std::string recipe;
  if (!std::getline(input, stored_key) || stored_key != key ||
      !std::getline(input, recipe)) {
    return {};
  }
  return recipe;
}

void write_recipe(const std::string& directory, const std::string& key,
                  const std::string& recipe) {
  if (directory.empty()) return;
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) return;
  const auto destination = path(directory, key);
  const auto temporary = destination + "." + std::to_string(getpid()) + ".tmp";
  {
    std::ofstream output(temporary);
    output << key << '\n' << recipe << '\n';
    if (!output) return;
  }
  std::filesystem::rename(temporary, destination, error);
}

}  // namespace apxinf::framework

#pragma once
#include <sstream>
#include <stdexcept>
namespace apxinf::efficient_compat {
template<class... Args> [[noreturn]] inline void fail(Args&&... args) {
  std::ostringstream message;
  (message << ... << args);
  throw std::invalid_argument(message.str());
}
}
#define TORCH_CHECK(condition, ...) do { \
  if (!(condition)) ::apxinf::efficient_compat::fail(__VA_ARGS__); \
} while (false)

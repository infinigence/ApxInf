#pragma once

#include "runtime_internal.h"

#include <cstring>
#include <limits>
#include <vector>

namespace apxinf::framework {

// Prepare-local allocations become borrowed runner storage only after tuning.
// Slots point into stable provider state; destruction never dereferences them.
class Scratch {
  struct Slice {
    void* slot;
    void* owned;
    size_t bytes;
  };
  std::vector<Slice> slices_;
  size_t bytes_ = 0;

  void add(void* slot, void* owned, size_t bytes) {
    if (bytes_ > std::numeric_limits<size_t>::max() - 255 ||
        bytes > std::numeric_limits<size_t>::max() - bytes_ - 255) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "scratch size overflow");
    }
    const size_t aligned = (bytes + 255) & ~size_t(255);
    slices_.push_back({slot, owned, aligned});
    bytes_ += aligned;
  }

 public:
  Scratch() = default;
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    for (auto& slice : slices_) {
      if (slice.owned) cudaFree(slice.owned);
    }
  }

  template <class T>
  void allocate(T** slot, size_t bytes) {
    if (!bytes) return;
    add(slot, nullptr, bytes);
    check_cuda(cudaMalloc(&slices_.back().owned, bytes));
    std::memcpy(slot, &slices_.back().owned, sizeof(void*));
  }

  template <class T>
  void adopt(T** slot, size_t bytes) {
    if (bytes) add(slot, *slot, bytes);
  }

  size_t bytes() const noexcept { return bytes_; }

  void bind(void* buffer, size_t capacity) {
    if (capacity < bytes_ || (!buffer && bytes_) ||
        reinterpret_cast<uintptr_t>(buffer) % 256) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid runner scratch binding");
    }
    size_t offset = 0;
    for (auto& slice : slices_) {
      if (slice.owned) {
        check_cuda(cudaFree(slice.owned));
        slice.owned = nullptr;
      }
      void* pointer = static_cast<char*>(buffer) + offset;
      std::memcpy(slice.slot, &pointer, sizeof(void*));
      offset += slice.bytes;
    }
  }
};

}  // namespace apxinf::framework

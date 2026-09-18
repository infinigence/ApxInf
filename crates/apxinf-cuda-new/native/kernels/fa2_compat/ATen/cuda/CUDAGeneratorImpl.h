#pragma once

// ApxInf builds FlashAttention without libtorch. The upstream FA2 headers only
// need an object that can carry a Philox seed and offset into device code; the
// inference build disables dropout, so no generator implementation is needed.

#include <cstdint>

namespace at {

struct PhiloxCudaState {
  std::uint64_t seed_value = 0;
  std::uint64_t offset_value = 0;
  const std::uint64_t* graph_seed = nullptr;
  const std::uint64_t* graph_offset = nullptr;
  std::uint32_t graph_increment = 0;
  bool uses_graph_state = false;
};

}  // namespace at

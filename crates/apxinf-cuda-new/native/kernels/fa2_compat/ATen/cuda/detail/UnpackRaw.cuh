#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cstdint>
#include <tuple>

namespace at::cuda::philox {

__device__ inline std::tuple<std::uint64_t, std::uint64_t> unpack(
    const PhiloxCudaState& state) {
  if (!state.uses_graph_state) {
    return {state.seed_value, state.offset_value};
  }
  return {*state.graph_seed,
          *state.graph_offset + static_cast<std::uint64_t>(state.graph_increment)};
}

}  // namespace at::cuda::philox

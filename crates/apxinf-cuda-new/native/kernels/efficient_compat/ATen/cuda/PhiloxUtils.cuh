#pragma once
// Inference-only carrier needed to parse the pristine upstream forward kernel.
// The provider instantiates kSupportsDropout=false and never accepts RNG input.
#include <cstdint>
#include <tuple>
namespace at {
struct PhiloxCudaState {
  std::uint64_t seed, offset;
  bool captured_ = false;
  __host__ __device__ PhiloxCudaState(std::uint64_t s, std::uint64_t o)
      : seed(s), offset(o) {}
};
namespace cuda::philox {
__device__ inline std::tuple<std::uint64_t, std::uint64_t> unpack(
    const PhiloxCudaState& state) { return {state.seed, state.offset}; }
}
}

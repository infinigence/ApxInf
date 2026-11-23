#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Shared device-side math helpers for custom MUSA operators.
//
// A port of `crates/apxinf-cuda/kernels/custom/math.cuh`, and the definitions
// have to stay bit-compatible with it: these are the functions the eager bridge
// also spelled out in Python (the torch bridge that recorded the reference; it
// is not part of this tree), and the recorded reference tensors were produced
// by that spelling. `gelu_tanh` here is the tanh approximation, not erf --
// `_gelu_tanh` in the bridge says so explicitly, and the two must not drift
// apart.

constexpr int kThreads = 256;

// The highest `gridDim.y` a launch may use. Kernels that put the row on
// `gridDim.y` loop `row += gridDim.y` when there are more rows than that, so
// the limit costs occupancy at worst rather than coverage.
constexpr int kMaxGridY = 65535;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

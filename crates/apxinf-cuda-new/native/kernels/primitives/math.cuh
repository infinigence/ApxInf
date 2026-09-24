#pragma once

// Copyright 2026 apxinf contributors.
// Shared device-side math helpers for custom CUDA operators.

constexpr int kThreads = 256;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "../custom/pillow_bicubic_u8.cuh"

namespace {
__global__ void rgb_u8_to_temporal2_merge2_rect_bf16_kernel(
    const uint8_t* __restrict__ rgb, uint16_t* __restrict__ patches,
    const uint16_t* __restrict__ lut, int grid_h, int grid_w) {
  constexpr int kPatch = 16;
  constexpr int kArea = 256;
  constexpr int kRowWidth = 3 * 2 * kArea;
  const int row = blockIdx.x;
  int rem = row;
  const int merge_x = rem & 1;
  rem >>= 1;
  const int merge_y = rem & 1;
  rem >>= 1;
  const int groups_w = grid_w / 2;
  const int group_x = rem % groups_w;
  const int group_y = rem / groups_w;
  const int patch_y = group_y * 2 + merge_y;
  const int patch_x = group_x * 2 + merge_x;
  const int dy = threadIdx.x / kPatch;
  const int dx = threadIdx.x % kPatch;
  const int64_t pixel = ((static_cast<int64_t>(patch_y) * kPatch + dy) *
                         (static_cast<int64_t>(grid_w) * kPatch) +
                         static_cast<int64_t>(patch_x) * kPatch + dx) * 3;
  const int64_t base = static_cast<int64_t>(row) * kRowWidth + threadIdx.x;
#pragma unroll
  for (int channel = 0; channel < 3; ++channel) {
    const uint16_t normalized = lut[rgb[pixel + channel]];
    patches[base + channel * 2 * kArea] = normalized;
    patches[base + channel * 2 * kArea + kArea] = normalized;
  }
}

}  // namespace

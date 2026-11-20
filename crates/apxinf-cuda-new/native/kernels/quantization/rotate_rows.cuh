// Copyright 2026 ApxInf contributors.
// ConvRot activation rotation fused with row-wise INT8 quantization.
//
// Contract (ROTATE_ROWWISE_I8): for each row and each 256-column group g,
//   y[g*256 + j] = sum_i x[g*256 + i] * H[i][j],   H = (H4 x H4 x H4 x H4) / 16
// with H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] (symmetric, H*H == I).
// The row is then quantized exactly like ROWWISE_I8: scale = max(amax/127, 1e-12),
// q = clamp(roundf(y / scale), -128, 127).
//
// H factors into four radix-4 stages (one per base-4 digit of the column index),
// so one warp rotates one group in shared memory with four butterfly passes.
// Rotation is FP32 end to end and is never rounded to BF16. The row maximum
// needs every rotated value before any can be quantized, so the second pass
// recomputes the rotation from the same input instead of buffering a whole
// FP32 row (up to 56 KiB), which would need a dynamic shared-memory opt-in.

// Included inside the anonymous namespace of adapters/quantization/launch.cu, like the neighbouring
// kernels/quantization/*.cuh headers; it therefore declares no namespace of its own and relies on
// launch.cu for cuda_bf16.h, cuda_runtime.h, <cstdint> and block_max_parallel_unsafe.

constexpr int kRotateGroup = 256;
constexpr int kRotateThreads = 256;  // 8 warps, one group per warp at a time
constexpr int kRotateWarps = kRotateThreads / 32;

__device__ __forceinline__ void hadamard4(float& a, float& b, float& c,
                                          float& d) {
  const float y0 = a + b + c - d;
  const float y1 = a + b - c + d;
  const float y2 = a - b + c + d;
  const float y3 = -a + b + c + d;
  a = y0;
  b = y1;
  c = y2;
  d = y3;
}

// Rotates the 256 floats in `tile` in place; `tile` is private to the calling
// warp. The result is scaled by 16 relative to H (applied by the caller).
__device__ __forceinline__ void rotate_group_unscaled(float* tile) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int stride = 1; stride < kRotateGroup; stride *= 4) {
    __syncwarp();
#pragma unroll
    for (int quad = lane; quad < kRotateGroup / 4; quad += 32) {
      const int base = (quad / stride) * 4 * stride + (quad % stride);
      hadamard4(tile[base], tile[base + stride], tile[base + 2 * stride],
                tile[base + 3 * stride]);
    }
  }
  __syncwarp();
}

// Loads this lane's 8 consecutive BF16 values of group `group` into the tile,
// rotates the group, and leaves the rotated (unscaled) values in the tile.
__device__ __forceinline__ void load_and_rotate(
    const __nv_bfloat16* row, int group, float* tile) {
  const int lane = threadIdx.x & 31;
  const uint4 packed = *reinterpret_cast<const uint4*>(
      row + static_cast<int64_t>(group) * kRotateGroup + lane * 8);
  const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tile[lane * 8 + i] = __bfloat162float(values[i]);
  }
  rotate_group_unscaled(tile);
}

__global__ void __launch_bounds__(kRotateThreads)
    rotate_quantize_rows_bf16_int8_kernel(const __nv_bfloat16* __restrict__ input,
                                          int8_t* __restrict__ output,
                                          float* __restrict__ scales, int rows,
                                          int cols) {
  __shared__ float tiles[kRotateWarps][kRotateGroup];
  __shared__ float scratch[kRotateWarps];
  const int row = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int groups = cols / kRotateGroup;
  float* tile = tiles[warp];
  const __nv_bfloat16* in_row = input + static_cast<int64_t>(row) * cols;
  int8_t* out_row = output + static_cast<int64_t>(row) * cols;
  constexpr float kUnscale = 1.0f / 16.0f;  // exact power of two

  float maximum = 0.0f;
  for (int group = warp; group < groups; group += kRotateWarps) {
    load_and_rotate(in_row, group, tile);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      maximum = fmaxf(maximum, fabsf(tile[lane * 8 + i] * kUnscale));
    }
    __syncwarp();  // the next group overwrites the tile
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;

  for (int group = warp; group < groups; group += kRotateWarps) {
    load_and_rotate(in_row, group, tile);
    uint64_t packed = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float quantized = roundf(tile[lane * 8 + i] * kUnscale / scale);
      const int8_t q =
          static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, quantized)));
      packed |= static_cast<uint64_t>(static_cast<uint8_t>(q)) << (8 * i);
    }
    *reinterpret_cast<uint64_t*>(out_row +
                                 static_cast<int64_t>(group) * kRotateGroup +
                                 lane * 8) = packed;
    __syncwarp();
  }
}

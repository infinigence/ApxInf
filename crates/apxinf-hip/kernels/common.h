// Definitions shared by every apxinf-hip translation unit: status codes, dtype
// codes, BF16 conversion, grid helpers and a wave-size-agnostic block reduction.
//
// BF16 conversion is done by hand rather than through <hip/hip_bf16.h> so the
// rounding is exactly IEEE round-to-nearest-even with quiet NaNs, the same as
// `half::bf16::from_f32` on the host and `__float2bfloat16` in CUDA.
//
// Nothing here uses wave intrinsics or assumes a wave size. Reductions go
// through shared memory with power-of-two block sizes, so the same code is
// correct on wave32 (RDNA, e.g. gfx1151) and wave64 (CDNA) devices.

#pragma once

#include <hip/hip_runtime.h>

#include <math.h>
#include <stdint.h>

namespace apxinf_hip {

constexpr int kDTypeF32 = 0;
constexpr int kDTypeBF16 = 1;

constexpr int kBlasStatusBase = 100000;
constexpr int kInvalidArgument = 200000;

constexpr int kElementwiseBlock = 256;
constexpr int kReduceBlock = 256;
constexpr int kAttentionBlock = 128;
// Grid-stride loops cover any element count; capping the grid keeps launch
// dimensions far from the hardware limit without affecting correctness.
constexpr int64_t kMaxGrid = 1 << 20;

__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((bits >> 16) | 0x40u);  // quiet NaN
  }
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

// Round a float to BF16 and back: the value a BF16 store then load observes.
__device__ __forceinline__ float round_bf16(float value) {
  return bf16_to_f32(f32_to_bf16(value));
}

template <typename T>
struct Elem;

template <>
struct Elem<float> {
  __device__ static float load(const float* p, int64_t i) { return p[i]; }
  __device__ static void store(float* p, int64_t i, float v) { p[i] = v; }
};

template <>
struct Elem<uint16_t> {
  __device__ static float load(const uint16_t* p, int64_t i) { return bf16_to_f32(p[i]); }
  __device__ static void store(uint16_t* p, int64_t i, float v) { p[i] = f32_to_bf16(v); }
};

__device__ __forceinline__ int64_t global_index() {
  return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int64_t grid_stride() {
  return static_cast<int64_t>(gridDim.x) * blockDim.x;
}

// Sum `value` over the block. `scratch` holds blockDim.x floats; blockDim.x is
// a power of two. Every thread gets the total. The trailing barrier makes the
// scratch buffer safe to reuse by the next call.
__device__ __forceinline__ float block_sum(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

inline unsigned grid_for(int64_t n, int block) {
  int64_t blocks = (n + block - 1) / block;
  if (blocks > kMaxGrid) blocks = kMaxGrid;
  return static_cast<unsigned>(blocks < 1 ? 1 : blocks);
}

inline int last_launch_status() { return static_cast<int>(hipGetLastError()); }

inline hipStream_t as_stream(void* stream) { return static_cast<hipStream_t>(stream); }

}  // namespace apxinf_hip

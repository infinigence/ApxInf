// Copyright 2026 ApxInf contributors.
//
// The M=1 projections: FP8 and NVFP4 GEMV.
//
// Moved verbatim out of kernels/custom/mlp_ops.cu, which was named after a call
// site rather than a mathematical family (doc/cuda-new-design.md section 1.3).
// Every kernel and both host launches below are that code unchanged apart from
// the enclosing namespace.
//
// One exception, and it removes code rather than changing any: the unreferenced
// `__constant__ float kE2M1Table[16]` did not come along. It had no reader --
// `nvfp4_gemv_kernel` decodes by bit placement and `nvfp4_dp4a_gemv_kernel`
// through `unpack_e2m1_int8` -- and the comment on `nvfp4_gemv_kernel` records
// why the table approach was rejected in round-01. The comment stays; the dead
// array does not.
//
// These are GEMMs, but not members of the tunable GEMM family: one
// implementation each, no candidates to measure between, so they keep flat
// parameters per doc/cuda-new-design.md section 5 and launch from here rather
// than through apxinf_gemm_prepare.

#include "gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace apxinf::cuda_new::gemv {
namespace {

// One warp per output row, four rows per block, 16-byte loads.
//
// The shape of this kernel is set by what the first attempt got wrong. One
// block per row with scalar loads gave each thread ~20 bytes of work over
// 10240 blocks, and launch plus per-element conversion overhead buried the
// memory traffic it was trying to optimize. Here each thread issues 16-byte
// vector loads and carries ~160 bytes, and the block count drops 4x.
//
// The activation is read from global rather than staged in shared: staging
// makes every block re-read the same K bytes, which at N=10240 is 52 MB of
// redundant traffic -- as much as the weights.
//
// Paired conversion (fp8x2 -> half2): one cvt per two elements. Round-00
// ncu had the scalar-convert version at 58% SM; pairing dropped it to 24.5%
// (141.7 vs 153 us). fp8->half is exact and the f32 accumulation order is
// unchanged, so the sum is bit-identical to the scalar version. A second
// accumulator chain (ILP) was tried on top and rejected: 89.38 vs 89.79
// ms/tok end to end -- the kernel is latency-bound on loads, not FMA -- and
// the reordered sum moved generated tokens, breaking the bit-identity gate
// (round-02).
//
// Software prefetch instead: issue the next iteration's two loads before
// converting the current one, so the load latency overlaps the 32 FMAs.
// The values and their accumulation order are untouched -- only the issue
// distance moves -- so this stays bit-identical.
__global__ void fp8_gemv_kernel(const uint4* __restrict__ weight,
                                const uint4* __restrict__ activation,
                                __nv_bfloat16* __restrict__ output, int n,
                                int k_vectors, float alpha) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n) return;

  const uint4* weight_row = weight + (long long)row * k_vectors;
  float sum = 0.0f;
  int index = lane;
  uint4 w = index < k_vectors ? weight_row[index] : uint4{0, 0, 0, 0};
  uint4 a = index < k_vectors ? activation[index] : uint4{0, 0, 0, 0};
  while (index < k_vectors) {
    const int next = index + 32;
    uint4 w_next{0, 0, 0, 0};
    uint4 a_next{0, 0, 0, 0};
    if (next < k_vectors) {
      w_next = weight_row[next];
      a_next = activation[next];
    }
    const __nv_fp8x2_storage_t* wp =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
    const __nv_fp8x2_storage_t* ap =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(&a);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
      const __half2 wh =
          __half2(__nv_cvt_fp8x2_to_halfraw2(wp[pair], __NV_E4M3));
      const __half2 ah =
          __half2(__nv_cvt_fp8x2_to_halfraw2(ap[pair], __NV_E4M3));
      sum += __low2float(wh) * __low2float(ah);
      sum += __high2float(wh) * __high2float(ah);
    }
    w = w_next;
    a = a_next;
    index = next;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}

__global__ void fp8_gemv_shared_kernel(const uint4* __restrict__ weight,
                                      const uint4* __restrict__ activation,
                                      __nv_bfloat16* __restrict__ output, int rows,
                                      int vectors, float alpha) {
  extern __shared__ float activation_shared[];
  for (int index = threadIdx.x; index < vectors; index += blockDim.x) {
    const uint4 packed = activation[index];
    const auto* codes = reinterpret_cast<const __nv_fp8_storage_t*>(&packed);
#pragma unroll
    for (int element = 0; element < 16; ++element)
      activation_shared[(index / 32) * 512 + element * 32 + index % 32] =
          __half2float(__nv_cvt_fp8_to_halfraw(codes[element], __NV_E4M3));
  }
  __syncthreads();
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int index = lane; index < vectors; index += 32) {
    const uint4 packed = weight[(long long)row * vectors + index];
    const auto* codes = reinterpret_cast<const __nv_fp8_storage_t*>(&packed);
#pragma unroll
    for (int element = 0; element < 16; ++element)
      sum += __half2float(__nv_cvt_fp8_to_halfraw(codes[element], __NV_E4M3)) *
             activation_shared[(index / 32) * 512 + element * 32 + lane];
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}

__device__ __forceinline__ float from_e4m3(uint8_t code) {
  return __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
}

__device__ __forceinline__ int unpack_e2m1_int8(unsigned selector) {
  const unsigned magnitudes = __byte_perm(0x03020100, 0x0c080604, selector & 0x7777);
  const unsigned signs = __byte_perm(0x0000ff00, 0, (selector >> 3) & 0x1111);
  return static_cast<int>(__vsub4(magnitudes ^ signs, signs));
}

// One warp per output row, four rows per block, 16-byte loads.
//
// Each uint4 carries 32 FP4 values, which is exactly two scale blocks, so the
// scale lookup is two loads per vector rather than one per element.
//
// FP4 decodes by bit placement, not by table or cvt. sm_110 has no FP4
// hardware convert: __nv_cvt_fp4x2_to_halfraw2 compiles to a LOP3 sea that
// left this kernel at 96% SM / 6% memory, 994 us for the down projection
// (round-01 ncu, 01-gemv-hwcvt.ncu-rep). A __constant__ table per nibble is
// as bad through a different pipe: 32 divergent constant-cache reads per
// warp (round-01, 01-gemv-slow.ncu-rep, 4.7 ms).
//
// The placement trick: an E2M1 nibble scattered into a binary16 as
// [sign<<15 | exp<<10..9 | man<<9] reads as the FP4 value scaled by 2^-14,
// for normals and subnormals both. One shift-mask pair per element plus one
// exact power-of-two multiply recovers the value; all of it dual-issues as
// plain ALU.
__global__ void nvfp4_gemv_kernel(const uint4* __restrict__ weight,
                                  const uint8_t* __restrict__ weight_scales,
                                  const uint4* __restrict__ activation,
                                  const uint8_t* __restrict__ activation_scales,
                                  __nv_bfloat16* __restrict__ output, int n,
                                  int k_vectors, float alpha) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n) return;

  const uint4* weight_row = weight + (long long)row * k_vectors;
  const uint8_t* weight_scale_row =
      weight_scales + (long long)row * k_vectors * 2;

  // Both nibbles of one packed byte, placed into a half2. Exact.
  const auto pair_to_half2 = [](uint32_t byte) -> __half2 {
    const uint32_t bits = ((byte & 0x07u) << 9) | ((byte & 0x08u) << 12) |
                          ((byte & 0x70u) << 21) | ((byte & 0x80u) << 24);
    return *reinterpret_cast<const __half2*>(&bits);
  };
  constexpr float kRescale = 16384.0f;  // 2^14, exact

  float sum = 0.0f;
  for (int index = lane; index < k_vectors; index += 32) {
    const uint4 w = weight_row[index];
    const uint4 a = activation[index];
    const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w);
    const uint8_t* ab = reinterpret_cast<const uint8_t*>(&a);

#pragma unroll
    for (int block = 0; block < 2; ++block) {
      float accumulator = 0.0f;
#pragma unroll
      for (int byte = 0; byte < 8; ++byte) {
        const int offset = block * 8 + byte;
        const __half2 wh = pair_to_half2(wb[offset]);
        const __half2 ah = pair_to_half2(ab[offset]);
        // One 2^28 rescale per product pair keeps the f32 accumulation
        // order of the reference kernel: (w*2^-14)*(a*2^-14)*2^28 == w*a
        // exactly, because every intermediate is a power-of-two scaling
        // of an exactly-representable value.
        accumulator = fmaf(__low2float(wh) * __low2float(ah),
                           kRescale * kRescale, accumulator);
        accumulator = fmaf(__high2float(wh) * __high2float(ah),
                           kRescale * kRescale, accumulator);
      }
      const int scale_index = index * 2 + block;
      sum += accumulator * from_e4m3(weight_scale_row[scale_index]) *
             from_e4m3(activation_scales[scale_index]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}

__global__ void nvfp4_dp4a_gemv_kernel(
    const uint4* __restrict__ weight, const uint8_t* __restrict__ weight_scales,
    const uint4* __restrict__ activation, const uint8_t* __restrict__ activation_scales,
    __nv_bfloat16* __restrict__ output, int n, int k_vectors, float alpha) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n) return;
  const uint4* weight_row = weight + (long long)row * k_vectors;
  const uint8_t* weight_scale_row = weight_scales + (long long)row * k_vectors * 2;
  float sum = 0.0f;
  for (int index = lane; index < k_vectors; index += 32) {
    const uint4 w = weight_row[index];
    const uint4 a = activation[index];
    const auto* weight_words = reinterpret_cast<const uint16_t*>(&w);
    const auto* activation_words = reinterpret_cast<const uint16_t*>(&a);
#pragma unroll
    for (int block = 0; block < 2; ++block) {
      int accumulator = 0;
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        accumulator = __dp4a(
            unpack_e2m1_int8(weight_words[block * 4 + word]),
            unpack_e2m1_int8(activation_words[block * 4 + word]), accumulator);
      }
      const int scale_index = index * 2 + block;
      sum += static_cast<float>(accumulator) * 0.25f *
             from_e4m3(weight_scale_row[scale_index]) *
             from_e4m3(activation_scales[scale_index]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}
}  // namespace

int nvfp4_gemv(const void* weight, const void* weight_scales,
               const void* activation, const void* activation_scales,
               void* output, int n, int k, float alpha, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return -1;
  // 16-byte loads cover 32 FP4 values, which is two whole scale blocks.
  if (k % 32 != 0) return -2;
  constexpr int kRowsPerBlock = 4;
  const int threads = kRowsPerBlock * 32;
  const int blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
  const char* dp4a_flag = std::getenv("APXINF_QWEN38_NVFP4_DP4A");
  if (dp4a_flag != nullptr && std::strcmp(dp4a_flag, "1") == 0 &&
      reinterpret_cast<uintptr_t>(weight) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(activation) % 16 == 0) {
    nvfp4_dp4a_gemv_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint4*>(weight),
        static_cast<const uint8_t*>(weight_scales),
        static_cast<const uint4*>(activation),
        static_cast<const uint8_t*>(activation_scales),
        static_cast<__nv_bfloat16*>(output), n, k / 32, alpha);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
  }
  nvfp4_gemv_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint4*>(weight),
      static_cast<const uint8_t*>(weight_scales),
      static_cast<const uint4*>(activation),
      static_cast<const uint8_t*>(activation_scales),
      static_cast<__nv_bfloat16*>(output), n, k / 32, alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int fp8_gemv(const void* weight, const void* activation, void* output, int n,
             int k, float alpha, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return -1;
  // 16-byte loads need K to be a multiple of 16. Every projection in this
  // model satisfies that; reject rather than silently scalarize.
  if (k % 16 != 0) return -2;
  // FP8 decode projections are memory-bound and launch thousands of small
  // output-row blocks. Eight independent warps per block cuts the grid in
  // half while preserving the one-warp-per-row reduction.
  constexpr int kRowsPerBlock = 8;
  const int threads = kRowsPerBlock * 32;
  const int blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
  const char* shared = std::getenv("APXINF_FP8_GEMV_SHARED");
  if (shared != nullptr && std::strcmp(shared, "1") == 0 &&
      k % 512 == 0 && k <= 8192 && n <= 6144) {
    fp8_gemv_shared_kernel<<<blocks, threads, k * sizeof(float), stream>>>(
        static_cast<const uint4*>(weight), static_cast<const uint4*>(activation),
        static_cast<__nv_bfloat16*>(output), n, k / 16, alpha);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
  }
  fp8_gemv_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint4*>(weight), static_cast<const uint4*>(activation),
      static_cast<__nv_bfloat16*>(output), n, k / 16, alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}
}  // namespace apxinf::cuda_new::gemv

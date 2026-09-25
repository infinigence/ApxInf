// Copyright 2026 ApxInf contributors.
//
// NVFP4 (packed E2M1 data, unsigned-E4M3 block scales) GEMM for the SM100
// family. Compiled for sm_110a on Jetson Thor, where CUTLASS enables the same
// tcgen05 block-scaled MMA path via CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED.
//
// Tile selection was measured on Thor (20 SMs, 32 MiB L2):
//   * tile N=256 aborts the kernel on this device -- excluded entirely.
//   * cluster 2x1 is worth roughly +30% over 1x1 on large-M shapes.
//   * deep-K shapes lose >2x once the activation operand exceeds L2, which the
//     caller controls by chunking M, not by the tile choice.

#include "gemm_nvfp4_sm100.h"

#include <cuda_bf16.h>

#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace apxinf::cuda::cutlass_ops {
namespace {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;

// Both operands are K-contiguous: A is [M, K] row-major and B is the
// checkpoint's native [N, K], described to CUTLASS as a column-major [K, N].
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kAlignA = 32;
constexpr int kAlignB = 32;
constexpr int kAlignD = 8;

template <class MmaTile, class Cluster>
struct Kernel {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, float, void,
      LayoutD, kAlignD, ElementD, LayoutD, kAlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, ElementA,
      LayoutA, kAlignA, ElementB, LayoutB, kAlignB, ElementAcc, MmaTile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using Universal = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Universal>;
};

// Ordered best-first for large M, which is where the choice matters; the
// autotuner still times all of them.
using Tactic0 = Kernel<Shape<_256, _128, _256>, Shape<_2, _1, _1>>;
using Tactic1 = Kernel<Shape<_128, _128, _256>, Shape<_2, _1, _1>>;
using Tactic2 = Kernel<Shape<_128, _128, _256>, Shape<_1, _1, _1>>;
using Tactic3 = Kernel<Shape<_128, _128, _128>, Shape<_1, _1, _1>>;

constexpr int kTacticCount = 4;

template <class Cfg>
int launch(const void* a, const void* a_sf, const void* b, const void* b_sf,
           void* out, void* workspace, size_t workspace_bytes, int m, int n,
           int k, float alpha, cudaStream_t stream) {
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using SfConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename ElementA::ScaleFactorType;

  auto problem = cute::make_shape(m, n, k, 1);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {reinterpret_cast<typename ElementA::DataType const*>(a), stride_a,
       reinterpret_cast<typename ElementB::DataType const*>(b), stride_b,
       reinterpret_cast<ElementSF const*>(a_sf),
       SfConfig::tile_atom_to_shape_SFA(problem),
       reinterpret_cast<ElementSF const*>(b_sf),
       SfConfig::tile_atom_to_shape_SFB(problem)},
      {{alpha, 0.0f},
       nullptr,
       stride_d,
       reinterpret_cast<ElementD*>(out),
       stride_d}};

  Gemm gemm;
  if (Gemm::get_workspace_size(args) > workspace_bytes) return -3;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
    return -2;
  }
  return gemm.run(stream) == cutlass::Status::kSuccess ? 0 : -4;
}

template <class Cfg>
size_t workspace_for(int m, int n, int k) {
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using SfConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename ElementA::ScaleFactorType;

  auto problem = cute::make_shape(m, n, k, 1);
  ElementSF const* no_scales = nullptr;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {nullptr, cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}), nullptr,
       cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}),
       no_scales, SfConfig::tile_atom_to_shape_SFA(problem),
       no_scales, SfConfig::tile_atom_to_shape_SFB(problem)},
      {{1.0f, 0.0f}, nullptr,
       cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}), nullptr,
       cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1})}};
  return Gemm::get_workspace_size(args);
}

// The atom layout depends only on SFVecSize, so tactic 0's view is
// authoritative for every tactic.
using CanonicalSfConfig =
    typename Tactic0::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Write one block scale in whichever layout the consumer needs. The GEMM
// reads the tcgen05 atom layout; the GEMV indexes a plain row-major grid.
__device__ __forceinline__ void store_block_scale(
    uint8_t* __restrict__ scales, CanonicalSfConfig::LayoutSF layout,
    bool row_major, int row, int block, int sf_vec, int k_blocks,
    uint8_t code) {
  if (row_major) {
    scales[(long long)row * k_blocks + block] = code;
  } else {
    auto tensor = cute::make_tensor(scales, layout);
    tensor(row, block * sf_vec, 0) = code;
  }
}


__global__ void scatter_block_scales_kernel(const uint8_t* __restrict__ src,
                                            uint8_t* __restrict__ dst,
                                            int rows, int k, int sf_vec,
                                            int k_blocks,
                                            CanonicalSfConfig::LayoutSF layout) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)rows * k_blocks;
  if (index >= total) return;
  const int row = static_cast<int>(index / k_blocks);
  const int block = static_cast<int>(index % k_blocks);
  // The atom layout is addressed in element coordinates along K; every k
  // inside one block aliases to the same scale entry.
  auto tensor = cute::make_tensor(dst, layout);
  tensor(row, block * sf_vec, 0) = src[index];
}

// E2M1 magnitudes, indexed by the low three bits of the code.
__device__ __forceinline__ uint8_t quantize_e2m1(float value) {
  const uint8_t sign = value < 0.0f ? 0x8 : 0x0;
  const float magnitude = fabsf(value);
  // Round to nearest representable magnitude. The ladder is short enough that
  // comparing against midpoints beats any arithmetic trick.
  uint8_t code;
  if (magnitude < 0.25f) code = 0;
  else if (magnitude < 0.75f) code = 1;   // 0.5
  else if (magnitude < 1.25f) code = 2;   // 1.0
  else if (magnitude < 1.75f) code = 3;   // 1.5
  else if (magnitude < 2.5f) code = 4;    // 2.0
  else if (magnitude < 3.5f) code = 5;    // 3.0
  else if (magnitude < 5.0f) code = 6;    // 4.0
  else code = 7;                          // 6.0
  return sign | code;
}

// Encode a non-negative float as E4M3 (bias 7). Scale values are never
// negative, so this matches the unsigned encoding the kernel reads.
__device__ __forceinline__ uint8_t quantize_e4m3_nonnegative(float value) {
  if (!(value > 0.0f)) return 0;
  int exponent;
  float mantissa = frexpf(value, &exponent);  // value = mantissa * 2^exponent
  mantissa *= 2.0f;
  exponent -= 1;                              // mantissa now in [1, 2)
  int biased = exponent + 7;
  if (biased <= 0) return 0;                  // underflows to zero
  int fraction = __float2int_rn((mantissa - 1.0f) * 8.0f);
  if (fraction > 7) {
    fraction = 0;
    ++biased;
  }
  if (biased > 15) {                          // saturate at the E4M3 maximum
    biased = 15;
    fraction = 7;
  }
  return static_cast<uint8_t>((biased << 3) | fraction);
}

__device__ __forceinline__ float dequantize_e4m3_nonnegative(uint8_t code) {
  const int exponent = (code >> 3) & 0x0F;
  const float fraction = static_cast<float>(code & 0x07);
  if (exponent == 0) return fraction / 8.0f * exp2f(-6.0f);
  return (1.0f + fraction / 8.0f) * exp2f(static_cast<float>(exponent - 7));
}

// One thread per block of `sf_vec` activation elements.
__global__ void quantize_activation_kernel(
    const __nv_bfloat16* __restrict__ src, uint8_t* __restrict__ packed,
    uint8_t* __restrict__ scales, int rows, int k, int sf_vec, int k_blocks,
    float input_scale, CanonicalSfConfig::LayoutSF layout, bool row_major) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)rows * k_blocks;
  if (index >= total) return;
  const int row = static_cast<int>(index / k_blocks);
  const int block = static_cast<int>(index % k_blocks);
  const long long base = (long long)row * k + (long long)block * sf_vec;

  float amax = 0.0f;
  for (int offset = 0; offset < sf_vec; ++offset) {
    amax = fmaxf(amax, fabsf(__bfloat162float(src[base + offset])));
  }

  // The block scale is stored relative to the per-tensor input_scale, so the
  // GEMM recovers absolute magnitudes by folding input_scale into alpha.
  const float block_scale = amax / 6.0f;
  const uint8_t scale_code = quantize_e4m3_nonnegative(block_scale / input_scale);
  store_block_scale(scales, layout, row_major, row, block, sf_vec, k_blocks,
                    scale_code);

  // Quantize against the scale that was actually stored, not the ideal one:
  // the E4M3 rounding is part of the encoding and must not be double-counted.
  const float effective = dequantize_e4m3_nonnegative(scale_code) * input_scale;
  const float inverse = effective > 0.0f ? 1.0f / effective : 0.0f;

  uint8_t* out = packed + ((long long)row * k + (long long)block * sf_vec) / 2;
  for (int offset = 0; offset < sf_vec; offset += 2) {
    const float low = __bfloat162float(src[base + offset]) * inverse;
    const float high = __bfloat162float(src[base + offset + 1]) * inverse;
    out[offset / 2] =
        static_cast<uint8_t>(quantize_e2m1(low) | (quantize_e2m1(high) << 4));
  }
}


// Quantize one block of `sf_vec` already-computed activation values into the
// packed operand and its block scale. Shared by every producer so the scale
// convention cannot drift between them.
__device__ __forceinline__ void emit_nvfp4_block(
    const float* values, int sf_vec, float input_scale, uint8_t* out,
    uint8_t& scale_code) {
  float amax = 0.0f;
  for (int offset = 0; offset < sf_vec; ++offset) {
    amax = fmaxf(amax, fabsf(values[offset]));
  }
  scale_code = quantize_e4m3_nonnegative(amax / 6.0f / input_scale);
  const float effective = dequantize_e4m3_nonnegative(scale_code) * input_scale;
  const float inverse = effective > 0.0f ? 1.0f / effective : 0.0f;
  for (int offset = 0; offset < sf_vec; offset += 2) {
    out[offset / 2] = static_cast<uint8_t>(
        quantize_e2m1(values[offset] * inverse) |
        (quantize_e2m1(values[offset + 1] * inverse) << 4));
  }
}

// One block per row: reduce for RMSNorm, then quantize that row in place
// without ever materializing the normalized BF16 tensor.
__global__ void quantize_rms_norm_kernel(
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ norm_weight,
    uint8_t* __restrict__ packed, uint8_t* __restrict__ scales, int rows,
    int k, int sf_vec, int k_blocks, float epsilon, float input_scale,
    CanonicalSfConfig::LayoutSF layout, bool row_major) {
  extern __shared__ float shared[];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long long base = (long long)row * k;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < k; index += blockDim.x) {
    const float value = __bfloat162float(src[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    const int warps = (blockDim.x + 31) / 32;
    float total = threadIdx.x < warps ? shared[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xFFFFFFFFu, total, offset);
    }
    if (threadIdx.x == 0) shared[0] = total;
  }
  __syncthreads();
  const float norm = rsqrtf(shared[0] / static_cast<float>(k) + epsilon);

  float values[32];
  for (int block = threadIdx.x; block < k_blocks; block += blockDim.x) {
    const long long offset = base + (long long)block * sf_vec;
    for (int index = 0; index < sf_vec; ++index) {
      values[index] = __bfloat162float(src[offset + index]) * norm *
                      (1.0f + __bfloat162float(norm_weight[block * sf_vec + index]));
    }
    uint8_t code;
    emit_nvfp4_block(values, sf_vec, input_scale, packed + offset / 2, code);
    store_block_scale(scales, layout, row_major, row, block, sf_vec, k_blocks,
                      code);
  }
}

// One thread per block of `sf_vec` outputs: read gate and up, apply SwiGLU,
// quantize. The BF16 intermediate never exists.
__global__ void quantize_swiglu_kernel(const __nv_bfloat16* __restrict__ src,
                                       uint8_t* __restrict__ packed,
                                       uint8_t* __restrict__ scales, int rows,
                                       int k, int sf_vec, int k_blocks,
                                       float input_scale,
                                       CanonicalSfConfig::LayoutSF layout,
                                       bool row_major) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)rows * k_blocks;
  if (index >= total) return;
  const int row = static_cast<int>(index / k_blocks);
  const int block = static_cast<int>(index % k_blocks);
  const long long gate_base = (long long)row * 2 * k + (long long)block * sf_vec;
  const long long up_base = gate_base + k;

  float values[32];
  for (int offset = 0; offset < sf_vec; ++offset) {
    const float gate = __bfloat162float(src[gate_base + offset]);
    const float up = __bfloat162float(src[up_base + offset]);
    values[offset] = gate / (1.0f + __expf(-gate)) * up;
  }
  uint8_t code;
  emit_nvfp4_block(values, sf_vec, input_scale,
                   packed + ((long long)row * k + (long long)block * sf_vec) / 2,
                   code);
  store_block_scale(scales, layout, row_major, row, block, sf_vec, k_blocks,
                    code);
}

}  // namespace

int nvfp4_gemm_tactic_count() { return kTacticCount; }

bool nvfp4_gemm_tactic_supported(int tactic, int m, int n, int k, int sf_vec) {
  if (tactic < 0 || tactic >= kTacticCount) return false;
  if (sf_vec != 16) return false;
  // The block-scaled MMA consumes 256 elements of K per instruction for the
  // _256 tiles; a shorter K would leave the scale atom partially covered.
  if (k % 128 != 0 || n % 64 != 0) return false;
  if (tactic == 3) return true;  // 128x128x128 has the loosest K requirement
  return k % 256 == 0;
}

size_t nvfp4_gemm_workspace_bytes(int m, int n, int k, int sf_vec, int tactic) {
  (void)sf_vec;
  switch (tactic) {
    case 0: return workspace_for<Tactic0>(m, n, k);
    case 1: return workspace_for<Tactic1>(m, n, k);
    case 2: return workspace_for<Tactic2>(m, n, k);
    case 3: return workspace_for<Tactic3>(m, n, k);
    default: return 0;
  }
}

int nvfp4_gemm_bf16(const void* a, const void* a_sf, const void* b,
                    const void* b_sf, void* out, void* workspace,
                    size_t workspace_bytes, int m, int n, int k, int sf_vec,
                    float alpha, int tactic, cudaStream_t stream) {
  if (sf_vec != 16) return -10;
  switch (tactic) {
    case 0:
      return launch<Tactic0>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    case 1:
      return launch<Tactic1>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    case 2:
      return launch<Tactic2>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    case 3:
      return launch<Tactic3>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    default:
      return -11;
  }
}

size_t nvfp4_scale_buffer_bytes(int rows, int k, int sf_vec) {
  if (sf_vec != 16) return 0;
  // SFB is built from (N, K); SFA from (M, K). Both go through the same atom,
  // so either entry point gives the same size for a given row count.
  auto layout = CanonicalSfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, rows, k, 1));
  return cute::cosize(layout);
}

int nvfp4_scatter_block_scales(const void* src_row_major, void* dst_atom,
                               int rows, int k, int sf_vec,
                               cudaStream_t stream) {
  if (sf_vec != 16) return -10;
  const int k_blocks = (k + sf_vec - 1) / sf_vec;
  auto layout = CanonicalSfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, rows, k, 1));
  const size_t bytes = cute::cosize(layout);
  cudaError_t status = cudaMemsetAsync(dst_atom, 0, bytes, stream);
  if (status != cudaSuccess) return -20;
  const long long total = (long long)rows * k_blocks;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  scatter_block_scales_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const uint8_t*>(src_row_major),
      static_cast<uint8_t*>(dst_atom), rows, k, sf_vec, k_blocks, layout);
  return cudaGetLastError() == cudaSuccess ? 0 : -21;
}

int nvfp4_quantize_activation(const void* src_bf16, void* dst_packed,
                              void* dst_scales, int rows, int k, int sf_vec,
                              float input_scale, int row_major_scales,
                              cudaStream_t stream) {
  if (sf_vec != 16) return -10;
  if (k % sf_vec != 0) return -11;
  if (!(input_scale > 0.0f)) return -12;
  const int k_blocks = k / sf_vec;
  auto layout = CanonicalSfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, rows, k, 1));
  const size_t bytes = row_major_scales
                           ? (size_t)rows * k_blocks
                           : cute::cosize(layout);
  // Padding entries must read as zero rather than as stale data.
  if (cudaMemsetAsync(dst_scales, 0, bytes, stream) != cudaSuccess) return -20;
  const long long total = (long long)rows * k_blocks;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  quantize_activation_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(src_bf16),
      static_cast<uint8_t*>(dst_packed), static_cast<uint8_t*>(dst_scales),
      rows, k, sf_vec, k_blocks, input_scale, layout,
      row_major_scales != 0);
  return cudaGetLastError() == cudaSuccess ? 0 : -21;
}

int nvfp4_quantize_rms_norm(const void* src_bf16, const void* norm_weight,
                            void* dst_packed, void* dst_scales, int rows,
                            int k, int sf_vec, float epsilon,
                            float input_scale, int row_major_scales,
                            cudaStream_t stream) {
  if (sf_vec != 16 || k % sf_vec != 0) return -10;
  if (!(input_scale > 0.0f)) return -12;
  const int k_blocks = k / sf_vec;
  auto layout = CanonicalSfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, rows, k, 1));
  const size_t scale_bytes = row_major_scales
                                 ? (size_t)rows * k_blocks
                                 : cute::cosize(layout);
  if (cudaMemsetAsync(dst_scales, 0, scale_bytes, stream) != cudaSuccess) {
    return -20;
  }
  const int threads = k_blocks >= 256 ? 256 : ((k_blocks + 31) / 32) * 32;
  const int warps = (threads + 31) / 32;
  quantize_rms_norm_kernel<<<rows, threads, warps * sizeof(float), stream>>>(
      static_cast<const __nv_bfloat16*>(src_bf16),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<uint8_t*>(dst_packed), static_cast<uint8_t*>(dst_scales),
      rows, k, sf_vec, k_blocks, epsilon, input_scale, layout,
      row_major_scales != 0);
  return cudaGetLastError() == cudaSuccess ? 0 : -21;
}

int nvfp4_quantize_swiglu(const void* src_bf16, void* dst_packed,
                          void* dst_scales, int rows, int k, int sf_vec,
                          float input_scale, int row_major_scales,
                          cudaStream_t stream) {
  if (sf_vec != 16 || k % sf_vec != 0) return -10;
  if (!(input_scale > 0.0f)) return -12;
  const int k_blocks = k / sf_vec;
  auto layout = CanonicalSfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, rows, k, 1));
  const size_t scale_bytes = row_major_scales
                                 ? (size_t)rows * k_blocks
                                 : cute::cosize(layout);
  if (cudaMemsetAsync(dst_scales, 0, scale_bytes, stream) != cudaSuccess) {
    return -20;
  }
  const long long total = (long long)rows * k_blocks;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  quantize_swiglu_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(src_bf16),
      static_cast<uint8_t*>(dst_packed), static_cast<uint8_t*>(dst_scales),
      rows, k, sf_vec, k_blocks, input_scale, layout,
      row_major_scales != 0);
  return cudaGetLastError() == cudaSuccess ? 0 : -21;
}

}  // namespace apxinf::cuda::cutlass_ops

// Copyright 2026 ApxInf contributors.
// SM100 BF16 widened paired gate/up projection with a TMEM-to-register epilogue.
//
// The static weight is tiled as [gate256, up256].  A single explicit-2SM
// 128x512x64 SM100 GEMM therefore consumes each A K-stage once and computes
// adjacent gate/up accumulator regions. Each small epilogue subtile loads
// gate and up from TMEM, rounds both scaled values through BF16 RNE, applies
// the production GeGLU arithmetic, and writes compact BF16 output.  There is
// no full-N BF16 temporary.  The production entry is exact-shape only and
// requires a load-time validated physical interleaved weight.

#if defined(__CUDA_ARCH_FEAT_SM101_ALL)
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "gemm_bf16_sm100.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/callbacks.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

namespace apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail {

constexpr int APXINF_DUAL_GEGLU_WIDE_PAIR_IMPLEMENTED = 1;
constexpr int APXINF_DUAL_GEGLU_PAIRED_EPILOGUE_IMPLEMENTED = 1;

constexpr int kN = 16384;
constexpr int kFullN = 2 * kN;
constexpr int kK = 2048;
constexpr int kPairColumns = 256;

constexpr bool valid_production_m(int m) { return m == 522 || m == 533; }
static_assert(valid_production_m(522));
static_assert(valid_production_m(533));
static_assert(!valid_production_m(521));
static_assert(!valid_production_m(534));

#ifndef APXINF_DUAL_GEGLU_CLUSTER_M
#define APXINF_DUAL_GEGLU_CLUSTER_M 2
#endif
#ifndef APXINF_DUAL_GEGLU_CLUSTER_N
#define APXINF_DUAL_GEGLU_CLUSTER_N 2
#endif
#ifndef APXINF_DUAL_GEGLU_FAST_TANH
#define APXINF_DUAL_GEGLU_FAST_TANH 0
#endif
#ifndef APXINF_DUAL_GEGLU_ROUND_UP_HALF
#define APXINF_DUAL_GEGLU_ROUND_UP_HALF 1
#endif
#ifndef APXINF_DUAL_GEGLU_MONOLITHIC
#define APXINF_DUAL_GEGLU_MONOLITHIC 0
#endif
#ifndef APXINF_DUAL_GEGLU_GEMM_ALPHA
#define APXINF_DUAL_GEGLU_GEMM_ALPHA 1.0f
#endif
#ifndef APXINF_DUAL_GEGLU_OUTPUT_SCALE
#define APXINF_DUAL_GEGLU_OUTPUT_SCALE 1.0f
#endif
#ifndef APXINF_DUAL_GEGLU_SATURATE_TANH
#define APXINF_DUAL_GEGLU_SATURATE_TANH 0
#endif
#ifndef APXINF_DUAL_GEGLU_TILE_M
#define APXINF_DUAL_GEGLU_TILE_M 256
#endif
#ifndef APXINF_DUAL_GEGLU_TILE_N
#define APXINF_DUAL_GEGLU_TILE_N 256
#endif
#ifndef APXINF_DUAL_GEGLU_TILE_K
#define APXINF_DUAL_GEGLU_TILE_K 128
#endif
#ifndef APXINF_DUAL_GEGLU_LEGACY_EPI_M
#define APXINF_DUAL_GEGLU_LEGACY_EPI_M 0
#endif
#ifndef APXINF_DUAL_GEGLU_LEGACY_EPI_N
#define APXINF_DUAL_GEGLU_LEGACY_EPI_N 0
#endif
#ifndef APXINF_DUAL_GEGLU_STAGE_COUNT
#define APXINF_DUAL_GEGLU_STAGE_COUNT 0
#endif
#ifndef APXINF_DUAL_GEGLU_WIDE_STAGE_COUNT
#define APXINF_DUAL_GEGLU_WIDE_STAGE_COUNT 3
#endif
#ifndef APXINF_DUAL_GEGLU_EPI_M
#define APXINF_DUAL_GEGLU_EPI_M 64
#endif
#ifndef APXINF_DUAL_GEGLU_EPI_N
#define APXINF_DUAL_GEGLU_EPI_N 64
#endif
#ifndef APXINF_DUAL_GEGLU_DOUBLE_BUFFER
#define APXINF_DUAL_GEGLU_DOUBLE_BUFFER 0
#endif
#ifndef APXINF_DUAL_GEGLU_TRACE_OWNERS
#define APXINF_DUAL_GEGLU_TRACE_OWNERS 0
#endif
#ifndef APXINF_DUAL_GEGLU_REGISTER_RENDEZVOUS
#define APXINF_DUAL_GEGLU_REGISTER_RENDEZVOUS 1
#endif
#ifndef APXINF_DUAL_GEGLU_MAINLOOP_SCHEDULE
// 0=auto, 1=explicit SM100 1-SM TMA, 2=explicit SM100 2-SM TMA.
#define APXINF_DUAL_GEGLU_MAINLOOP_SCHEDULE 0
#endif
#ifndef APXINF_DUAL_GEGLU_STREAMED_EVT
// Compute each output element as soon as possible instead of keeping a full
// scaled-up fragment live across the GELU loop.
#define APXINF_DUAL_GEGLU_STREAMED_EVT 0
#endif

constexpr float kGemmAlpha = APXINF_DUAL_GEGLU_GEMM_ALPHA;
constexpr float kOutputScale = APXINF_DUAL_GEGLU_OUTPUT_SCALE;
constexpr float kInverseOutputScale = 1.0f / kOutputScale;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    cudaError_t status_ = (expr);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status_));                                 \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(expr)                                                       \
  do {                                                                          \
    cublasStatus_t status_ = (expr);                                              \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
                   static_cast<int>(status_));                                    \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

__global__ void init_fp8(__nv_fp8_e4m3* data, size_t count, int salt) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    int value = static_cast<int>((index * 17 + static_cast<size_t>(salt)) % 23) - 11;
    // Use production-like quantized magnitudes.  With the real GEMM alpha
    // (~2.7e-4 for language layer 0), the former 0.03125 multiplier made the
    // entire scaled GeGLU output collapse to FP8 zero and could not exercise
    // the scale path meaningfully.
    data[index] = __nv_fp8_e4m3(static_cast<float>(value) * 2.0f);
  }
}

__global__ void init_bf16(__nv_bfloat16* data, size_t count, int salt) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    int value = static_cast<int>((index * 17 + static_cast<size_t>(salt)) % 31) - 15;
    data[index] = __float2bfloat16_rn(static_cast<float>(value) * 0.015625f);
  }
}

__global__ void pack_gate_up_tiles(
    const __nv_fp8_e4m3* source, __nv_fp8_e4m3* packed, size_t count) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int row = static_cast<int>(index / kFullN);
  const int dst_col = static_cast<int>(index - static_cast<size_t>(row) * kFullN);
  const int tile = dst_col / (2 * kPairColumns);
  const int within = dst_col - tile * (2 * kPairColumns);
  const int src_col = within < kPairColumns
      ? tile * kPairColumns + within
      : kN + tile * kPairColumns + (within - kPairColumns);
  packed[index] = source[static_cast<size_t>(row) * kFullN + src_col];
}

__global__ void pack_gate_up_tiles_bf16(
    const __nv_bfloat16* source, __nv_bfloat16* packed, size_t count) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int row = static_cast<int>(index / kFullN);
  const int dst_col = static_cast<int>(index - static_cast<size_t>(row) * kFullN);
  const int tile = dst_col / (2 * kPairColumns);
  const int within = dst_col - tile * (2 * kPairColumns);
  const int src_col = within < kPairColumns
      ? tile * kPairColumns + within
      : kN + tile * kPairColumns + (within - kPairColumns);
  packed[index] = source[static_cast<size_t>(row) * kFullN + src_col];
}

__global__ void evict_l2(uint32_t* data, size_t count, uint32_t seed) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) data[index] = data[index] * 1664525u + seed + index;
}

__device__ __forceinline__ float production_gelu(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

__device__ __forceinline__ float candidate_gelu(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  const float argument =
      kAlpha * (value + 0.044715f * value * value * value);
#if APXINF_DUAL_GEGLU_FAST_TANH
  return 0.5f * value * (1.0f + __tanhf(argument));
#elif APXINF_DUAL_GEGLU_SATURATE_TANH
  // CUDA's correctly rounded float tanh is already exactly +/-1 outside
  // this conservative interval.  Bypass the long-latency libdevice path in
  // the common saturated tail while preserving the same float value.
  const float tanh_value = fabsf(argument) >= 10.0f
      ? copysignf(1.0f, argument)
      : tanhf(argument);
  return 0.5f * value * (1.0f + tanh_value);
#else
  return production_gelu(value);
#endif
}

struct alignas(8) Fp8x8 {
  __nv_fp8x2_e4m3 pair0;
  __nv_fp8x2_e4m3 pair1;
  __nv_fp8x2_e4m3 pair2;
  __nv_fp8x2_e4m3 pair3;
};

__global__ void geglu_quant_packed8(
    const half* gate_up, __nv_fp8_e4m3* output, int rows, int inner,
    float inverse_scale) {
  const int groups_per_row = inner / 8;
  const int group_count = rows * groups_per_row;
  int group_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; group_index < group_count; group_index += stride) {
    const int row = group_index / groups_per_row;
    const int group_col = group_index - row * groups_per_row;
    const int pair_col = group_col * 4;
    const half* row_input = gate_up + static_cast<int64_t>(row) * kFullN;
    const half2* gate2 = reinterpret_cast<const half2*>(row_input);
    const half2* up2 = reinterpret_cast<const half2*>(row_input + inner);
    const float2 gate0 = __half22float2(gate2[pair_col]);
    const float2 gate1 = __half22float2(gate2[pair_col + 1]);
    const float2 gate2_value = __half22float2(gate2[pair_col + 2]);
    const float2 gate3 = __half22float2(gate2[pair_col + 3]);
    const float2 up0 = __half22float2(up2[pair_col]);
    const float2 up1 = __half22float2(up2[pair_col + 1]);
    const float2 up2_value = __half22float2(up2[pair_col + 2]);
    const float2 up3 = __half22float2(up2[pair_col + 3]);
    float2 value0, value1, value2, value3;
    value0.x = production_gelu(gate0.x) * up0.x * inverse_scale;
    value0.y = production_gelu(gate0.y) * up0.y * inverse_scale;
    value1.x = production_gelu(gate1.x) * up1.x * inverse_scale;
    value1.y = production_gelu(gate1.y) * up1.y * inverse_scale;
    value2.x = production_gelu(gate2_value.x) * up2_value.x * inverse_scale;
    value2.y = production_gelu(gate2_value.y) * up2_value.y * inverse_scale;
    value3.x = production_gelu(gate3.x) * up3.x * inverse_scale;
    value3.y = production_gelu(gate3.y) * up3.y * inverse_scale;
    value0.x = fminf(448.0f, fmaxf(-448.0f, value0.x));
    value0.y = fminf(448.0f, fmaxf(-448.0f, value0.y));
    value1.x = fminf(448.0f, fmaxf(-448.0f, value1.x));
    value1.y = fminf(448.0f, fmaxf(-448.0f, value1.y));
    value2.x = fminf(448.0f, fmaxf(-448.0f, value2.x));
    value2.y = fminf(448.0f, fmaxf(-448.0f, value2.y));
    value3.x = fminf(448.0f, fmaxf(-448.0f, value3.x));
    value3.y = fminf(448.0f, fmaxf(-448.0f, value3.y));
    reinterpret_cast<Fp8x8*>(output + static_cast<int64_t>(row) * inner)
        [group_col] = {static_cast<__nv_fp8x2_e4m3>(value0),
                       static_cast<__nv_fp8x2_e4m3>(value1),
                       static_cast<__nv_fp8x2_e4m3>(value2),
                       static_cast<__nv_fp8x2_e4m3>(value3)};
  }
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

__global__ void geglu_bf16_packed4(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output,
    int rows, int inner) {
  const int quads_per_row = inner / 4;
  const int quad_count = rows * quads_per_row;
  int quad_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (; quad_index < quad_count; quad_index += stride) {
    const int row = quad_index / quads_per_row;
    const int col_quad = quad_index - row * quads_per_row;
    const __nv_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    const Bf16x4 gate = reinterpret_cast<const Bf16x4*>(row_input)[col_quad];
    const Bf16x4 up =
        reinterpret_cast<const Bf16x4*>(row_input + inner)[col_quad];
    reinterpret_cast<Bf16x4*>(
        output + static_cast<int64_t>(row) * inner)[col_quad] = Bf16x4{
      __floats2bfloat162_rn(
          production_gelu(__bfloat162float(gate.low.x)) * __bfloat162float(up.low.x),
          production_gelu(__bfloat162float(gate.low.y)) * __bfloat162float(up.low.y)),
      __floats2bfloat162_rn(
          production_gelu(__bfloat162float(gate.high.x)) * __bfloat162float(up.high.x),
          production_gelu(__bfloat162float(gate.high.y)) * __bfloat162float(up.high.y))};
  }
}

__global__ void geglu_quant_interleaved256(
    const half* gate_up, __nv_fp8_e4m3* output, int rows, int inner,
    float inverse_scale) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(rows) * inner;
  if (index >= count) return;
  const int row = static_cast<int>(index / inner);
  const int col = static_cast<int>(index - static_cast<size_t>(row) * inner);
  const int tile = col / kPairColumns;
  const int within = col - tile * kPairColumns;
  const int gate_col = tile * (2 * kPairColumns) + within;
  const int up_col = gate_col + kPairColumns;
  const half* input_row = gate_up + static_cast<size_t>(row) * kFullN;
  float value = production_gelu(__half2float(input_row[gate_col])) *
      __half2float(input_row[up_col]) * inverse_scale;
  value = fminf(448.0f, fmaxf(-448.0f, value));
  output[index] = __nv_fp8_e4m3(value);
}

template <class T>
struct Identity {
  CUTLASS_HOST_DEVICE T operator()(T const& value) const { return value; }
};

template <class T>
struct ProductionGelu;

template <>
struct ProductionGelu<float> {
  CUTLASS_DEVICE float operator()(float const& value) const {
    return candidate_gelu(value);
  }
};

template <class T, int N>
struct ProductionGelu<cutlass::Array<T, N>> {
  CUTLASS_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& input) const {
    cutlass::Array<T, N> output;
    ProductionGelu<T> op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) output[i] = op(input[i]);
    return output;
  }
};

template <class T>
struct ProductionGeGlu;

struct GeGluScaleArguments {
  float alpha = 1.0f;
  float inverse_scale = 1.0f;
};

template <>
struct ProductionGeGlu<float> {
  using Arguments = GeGluScaleArguments;

  CUTLASS_DEVICE float operator()(
      float const& gate, float const& up,
      Arguments const& arguments) const {
    const float scaled_up = up * arguments.alpha;
#if APXINF_DUAL_GEGLU_ROUND_UP_HALF
    const float rounded_up = __half2float(__float2half_rn(scaled_up));
#else
    const float rounded_up = scaled_up;
#endif
    return candidate_gelu(gate) * rounded_up * arguments.inverse_scale;
  }
};

template <class T, int N>
struct ProductionGeGlu<cutlass::Array<T, N>> {
  using Arguments = GeGluScaleArguments;

  CUTLASS_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& gate,
      cutlass::Array<T, N> const& up,
      Arguments const& arguments) const {
    cutlass::Array<T, N> output;
#if APXINF_DUAL_GEGLU_STREAMED_EVT
    static_assert(APXINF_DUAL_GEGLU_ROUND_UP_HALF == 0,
                  "streamed EVT currently preserves the exact float path only");
    ProductionGelu<T> gelu;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      // Preserve the production arithmetic order exactly:
      // gelu(gate) * (up * alpha) * inverse_scale.
      output[i] = gelu(gate[i]) * (up[i] * arguments.alpha) *
                  arguments.inverse_scale;
    }
#else
    cutlass::Array<T, N> scaled_up;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) scaled_up[i] = up[i] * arguments.alpha;
#if APXINF_DUAL_GEGLU_ROUND_UP_HALF
    cutlass::NumericArrayConverter<
        cutlass::half_t, T, N,
        cutlass::FloatRoundStyle::round_to_nearest> to_half;
    cutlass::NumericArrayConverter<
        T, cutlass::half_t, N,
        cutlass::FloatRoundStyle::round_to_nearest> to_compute;
    cutlass::Array<T, N> rounded_up = to_compute(to_half(scaled_up));
#else
    cutlass::Array<T, N> rounded_up = scaled_up;
#endif
    ProductionGelu<T> gelu;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      output[i] = gelu(gate[i]) * rounded_up[i] * arguments.inverse_scale;
    }
#endif
    return output;
  }
};

using ElementInput = cutlass::bfloat16_t;
using ElementSource = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;

using GateGeluEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<ProductionGelu, ElementCompute,
                                            ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementSource>>;
using RoundedUpEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<Identity, ElementSource,
                                            ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90AccFetch>;
using RawUpEVT = std::conditional_t<APXINF_DUAL_GEGLU_ROUND_UP_HALF != 0, RoundedUpEVT,
                                    cutlass::epilogue::fusion::Sm90AccFetch>;
using ScaledUpEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute,
                                            ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<
        ElementCompute, Stride<_0, _0, int64_t>>,
    RawUpEVT>;
using GeGluCoreEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute,
                                            ElementCompute, kRound>,
    GateGeluEVT, ScaledUpEVT>;
using GeGluEVTBase = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementOutput,
                                            ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<
        ElementCompute, Stride<_0, _0, int64_t>>,
    GeGluCoreEVT>;
using MonolithicGeGluEVT = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<ProductionGeGlu, ElementOutput,
                                            ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementSource>,
    cutlass::epilogue::fusion::Sm90AccFetch>;
using SelectedGeGluEVT = std::conditional_t<APXINF_DUAL_GEGLU_MONOLITHIC != 0,
                                            MonolithicGeGluEVT, GeGluEVTBase>;

struct GeGluEVT : SelectedGeGluEVT {
  using SelectedGeGluEVT::SelectedGeGluEVT;
};

struct GeGluOperation : cutlass::epilogue::fusion::FusionOperation {
  using ElementOutput = apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::ElementOutput;
  using ElementCompute = apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::ElementCompute;
  using ElementSource = apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::ElementSource;
  static constexpr bool IsSourceSupported = true;
};

}  // namespace apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail

namespace cutlass::epilogue::fusion {
template <>
struct FusionCallbacksTraits<apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::GeGluEVT> {
  using DispatchPolicy = void;
  using Callbacks = apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::GeGluEVT;
  using Operation = apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::GeGluOperation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::ElementCompute;
};
}  // namespace cutlass::epilogue::fusion

namespace apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail {

template <class TileShape, class ClusterShape>
struct FusedUpGeGlu {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentInput = 16;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 16;
  using EpilogueTile = std::conditional_t<
      APXINF_DUAL_GEGLU_LEGACY_EPI_M == 0,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Shape<Int<APXINF_DUAL_GEGLU_LEGACY_EPI_M>, Int<APXINF_DUAL_GEGLU_LEGACY_EPI_N>>>;
  using EpilogueSchedule = std::conditional_t<
      APXINF_DUAL_GEGLU_LEGACY_EPI_M == 0,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::epilogue::TmaWarpSpecialized1Sm>;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, EpilogueTile,
          ElementAccumulator, ElementCompute, ElementSource, LayoutC,
          AlignmentC, ElementOutput, LayoutD, AlignmentD,
          EpilogueSchedule,
          GeGluEVT>::CollectiveOp;
#if APXINF_DUAL_GEGLU_MAINLOOP_SCHEDULE == 0
  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
#elif APXINF_DUAL_GEGLU_MAINLOOP_SCHEDULE == 1
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
#elif APXINF_DUAL_GEGLU_MAINLOOP_SCHEDULE == 2
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
#else
#error "unsupported APXINF_DUAL_GEGLU_MAINLOOP_SCHEDULE"
#endif
#if APXINF_DUAL_GEGLU_STAGE_COUNT == 0
  using MainloopStages = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
#else
  using MainloopStages =
      cutlass::gemm::collective::StageCount<APXINF_DUAL_GEGLU_STAGE_COUNT>;
#endif
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementInput,
          LayoutA, AlignmentInput, ElementInput, LayoutB, AlignmentInput,
          ElementAccumulator, TileShape, ClusterShape,
          MainloopStages, MainloopSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <class TileShape, class ClusterShape>
struct WidePairGemm {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentInput = 16;
  static constexpr int AlignmentOutput = 8;
  // SM100's physical BF16 2SM UMMA atom is limited to N<=256.  The ordinary
  // builder incorrectly tries to create one N512 atom and fails statically.
  // Build a legal N256 atom first, then let CollectiveMma partition the N512
  // tile into two MMA_N regions.  One shared A descriptor/stage feeds both
  // N256 B regions and cute::gemm iterates the resulting MMA_N mode.
  using LegalAtomTile = Shape<Int<128>, Int<256>, Int<64>>;
  // D8 streamed tile: one owner warp holds only eight FP32 accumulator values
  // per lane at a time. Gate and up rendezvous as FP16 in 1 KiB of shared
  // memory instead of co-residing as complete 64x64 register fragments.
  using PairedEpilogueTile =
      Shape<Int<APXINF_DUAL_GEGLU_EPI_M>, Int<APXINF_DUAL_GEGLU_EPI_N>>;
  using EpilogueSeedOperation = cutlass::epilogue::fusion::ScaledAcc<
      ElementOutput, float, float>;
  // Ask the supported physical-N256 builder only for its proven TMEM load op,
  // epilogue tile and row-major stride types.  The generated operation is not
  // invoked: PairedEpilogue below owns the full store semantics.
  using PhysicalNoSmemEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
          LegalAtomTile, ClusterShape, PairedEpilogueTile,
          float, float, void, LayoutD, AlignmentOutput, ElementOutput,
          LayoutD, AlignmentOutput,
          cutlass::epilogue::NoSmemWarpSpecialized2Sm,
          EpilogueSeedOperation>::CollectiveOp;

  class PairedEpilogue {
   public:
    using DispatchPolicy = cutlass::epilogue::Sm100NoSmem;
    using EpilogueTile = typename PhysicalNoSmemEpilogue::EpilogueTile;
    using ElementC = void;
    using StrideC = typename PhysicalNoSmemEpilogue::StrideC;
    using ElementD = ElementOutput;
    using StrideD = typename PhysicalNoSmemEpilogue::StrideD;
    using CopyOpT2R = typename PhysicalNoSmemEpilogue::CopyOpT2R;
    using ThreadEpilogueOp =
        typename PhysicalNoSmemEpilogue::ThreadEpilogueOp;
    using GmemTiledCopyC = void;
    using GmemTiledCopyD = void;

    static constexpr int ThreadCount = 128;
    static constexpr uint32_t TmaTransactionBytes = 0;
    // The ownership trace for the legal 64x64 epilogue proves that Gate and
    // Up always have the same owner warp for each logical tile.  The register
    // path therefore needs no inter-warp shared rendezvous at all.  Retain a
    // one-element placeholder because an empty Array is not portable C++.
    static constexpr int RendezvousElements =
        APXINF_DUAL_GEGLU_REGISTER_RENDEZVOUS ? 1 : size(EpilogueTile{});
    static constexpr int RendezvousStages = APXINF_DUAL_GEGLU_DOUBLE_BUFFER ? 2 : 1;
    struct SharedStorage {
      alignas(128) cutlass::Array<
          cutlass::bfloat16_t, RendezvousElements * RendezvousStages> gate;
      alignas(128) cutlass::Array<
          cutlass::bfloat16_t, RendezvousElements * RendezvousStages> up;
    };

    struct Arguments {
      float alpha = 1.0f;
      float inverse_scale = 1.0f;
      ElementD* ptr_D = nullptr;
      StrideD dD{};
    };
    using Params = Arguments;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const&, Arguments const& args, void*) {
      return args;
    }
    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
      return 0;
    }
    template <class ProblemShape>
    static cutlass::Status initialize_workspace(
        ProblemShape const&, Arguments const&, void*, cudaStream_t,
        cutlass::CudaHostAdapter* = nullptr) {
      return cutlass::Status::kSuccess;
    }
    template <class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape,
                              Arguments const& args) {
      auto shape = append<4>(problem_shape, 1);
      auto [m, n, k, l] = shape;
      bool exact_shape = valid_production_m(m) && n == kFullN && k == kK && l == 1;
      bool valid_args = args.ptr_D != nullptr && std::isfinite(args.alpha) &&
                        std::isfinite(args.inverse_scale);
      return exact_shape && valid_args;
    }

    CUTLASS_DEVICE PairedEpilogue(Params const& params, SharedStorage& shared)
        : params_(params), shared_(&shared) {}

    template <bool ReuseTmem = false, class AccumulatorPipeline,
              class AccumulatorPipelineState, class ProblemShapeMNKL,
              class CtaTileMNK, class CtaCoordMNKL, class AccEngine,
              class AccLayout>
    CUTLASS_DEVICE auto operator()(
        AccumulatorPipeline acc_pipeline,
        AccumulatorPipelineState acc_pipe_consumer_state,
        ProblemShapeMNKL problem_shape_mnkl, CtaTileMNK cta_tile_mnk,
        CtaCoordMNKL cta_coord_mnkl,
        cute::Tensor<AccEngine, AccLayout> accumulators, SharedStorage&) {
      using namespace cute;
      using Accumulator = typename AccEngine::value_type;
      static_assert(is_tmem<AccEngine>::value,
                    "paired epilogue requires TMEM accumulators");
      static_assert(rank(AccLayout{}) == 3,
                    "expected [MMA,MMA_M,MMA_N] accumulator layout");
      static_assert(size<1>(AccLayout{}) == 1,
                    "paired epilogue requires MMA_M extent one");
      static_assert(size<2>(AccLayout{}) == 2,
                    "paired epilogue requires gate/up MMA_N extent two");
      static_assert(rank(ProblemShapeMNKL{}) == 4);
      static_assert(rank(CtaCoordMNKL{}) == 4);
      static_assert(!ReuseTmem,
                    "D0 deliberately gates out overlapping-TMEM scheduling");

      auto [M, N_full, K, L] = problem_shape_mnkl;
      auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
      auto compact_problem = make_shape(M, N_full / 2, L);
      auto compact_cta_tile = make_shape(get<0>(cta_tile_mnk),
                                         shape_div(get<1>(cta_tile_mnk), _2{}));
      auto compact_coord = make_coord(m_coord, n_coord, l_coord);

      Tensor mD = make_tensor(make_gmem_ptr(params_.ptr_D), compact_problem,
                              append<3>(params_.dD, _0{}));
      Tensor gD = local_tile(mD, compact_cta_tile, compact_coord);
      Tensor gD_epi = flat_divide(gD, EpilogueTile{});

      // The logical N512 mainloop is physically N256 x MMA_N=2.  Both slices
      // have identical M/N layouts and therefore share one T2R partition.
      Tensor tGate = accumulators(make_coord(_, _), _0{}, _0{});
      Tensor tUp = accumulators(make_coord(_, _), _0{}, _1{});
      Tensor tGate_epi = flat_divide(tGate, EpilogueTile{});
      Tensor tUp_epi = flat_divide(tUp, EpilogueTile{});
      // Build both ownership maps explicitly.  MMA_N0 and MMA_N1 may reside in
      // different TMEM subpartitions, so a warp must never blindly load both.
      auto gate_warp_partitioner =
          make_tmem_warp_partitioner(tGate_epi(_, _, _0{}, _0{}));
      auto up_warp_partitioner =
          make_tmem_warp_partitioner(tUp_epi(_, _, _0{}, _0{}));
      static_assert(size(typename decltype(gate_warp_partitioner)::TiledLayout_TV{}) <= 4);
      static_assert(size(typename decltype(up_warp_partitioner)::TiledLayout_TV{}) <= 4);
      auto tiled_t2r = make_tmem_copy(CopyOpT2R{},
                                      tGate_epi(_, _, _0{}, _0{}));
      int thread_idx = threadIdx.x % ThreadCount;
      int warp_idx = thread_idx / 32;
      auto thread_t2r = tiled_t2r.get_slice(thread_idx);
      Tensor tTR_tGate = thread_t2r.partition_S(tGate_epi);
      Tensor tTR_tUp = thread_t2r.partition_S(tUp_epi);
      Tensor tTR_gD = thread_t2r.partition_D(gD_epi);
      Tensor rAcc = make_tensor<Accumulator>(
          shape(tTR_gD(_, _, _, _0{}, _0{})));
      Tensor rGate = make_tensor<cutlass::bfloat16_t>(shape(rAcc));
      Tensor rUp = make_tensor<cutlass::bfloat16_t>(shape(rAcc));
      Tensor rD = make_tensor<ElementD>(shape(rAcc));

      Tensor coordD = make_identity_tensor(compact_problem);
      Tensor cD = local_tile(coordD, compact_cta_tile, compact_coord);
      Tensor tTR_cD = thread_t2r.partition_D(flat_divide(cD, EpilogueTile{}));
      constexpr auto common_layout =
          decltype(max_common_layout(tTR_gD(_, _, _, _0{}, _0{}), rD)){};
      constexpr int Vector = cute::min(Int<AlignmentOutput>{},
                                       size(common_layout));
      using VectorType = uint_bit_t<Vector * sizeof_bits_v<ElementD>>;

      constexpr int NumEpiM = CUTE_STATIC_V(size<3>(tTR_tGate));
      constexpr int NumEpiN = CUTE_STATIC_V(size<4>(tTR_tGate));
      auto synchronize = []() CUTLASS_LAMBDA_FUNC_INLINE {
        cutlass::arch::NamedBarrier::sync(
            ThreadCount,
            cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      };
      for (int epi_n = 0; epi_n < NumEpiN; ++epi_n) {
        for (int epi_m = 0; epi_m < NumEpiM; ++epi_m) {
          int linear_epi = epi_n * NumEpiM + epi_m;
          int rendezvous_stage =
              APXINF_DUAL_GEGLU_DOUBLE_BUFFER ? (linear_epi & 1) : 0;
          Tensor sGate = make_tensor(
              make_smem_ptr(shared_->gate.data() +
                            rendezvous_stage * RendezvousElements),
              make_layout(EpilogueTile{}));
          Tensor sUp = make_tensor(
              make_smem_ptr(shared_->up.data() +
                            rendezvous_stage * RendezvousElements),
              make_layout(EpilogueTile{}));
          Tensor tTR_sGate = thread_t2r.partition_D(sGate);
          Tensor tTR_sUp = thread_t2r.partition_D(sUp);
          Tensor tGateTile = tTR_tGate(_, _, _, epi_m, epi_n);
          Tensor tUpTile = tTR_tUp(_, _, _, epi_m, epi_n);
          int gate_owner = (tGateTile.data().dp_ / 32) % 4;
          int up_owner = (tUpTile.data().dp_ / 32) % 4;
          assert(gate_owner >= 0 && gate_owner < 4);
          assert(up_owner >= 0 && up_owner < 4);
#if APXINF_DUAL_GEGLU_TRACE_OWNERS
          if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
              thread_idx == 0) {
            printf("dual_geglu_owner linear=%d epi_m=%d epi_n=%d gate=%d up=%d\\n",
                   linear_epi, epi_m, epi_n, gate_owner, up_owner);
          }
#endif

#if APXINF_DUAL_GEGLU_REGISTER_RENDEZVOUS
          // Runtime ownership tracing established the fixed sequence
          // (gate,up)=(0,0),(0,0),(2,2),(2,2).  Keep the equality as a device
          // fail-close: if a future CUTLASS layout changes ownership we must
          // not silently issue an illegal TMEM load.
          assert(gate_owner == up_owner);
          if (warp_idx == gate_owner) {
            copy(tiled_t2r, tGateTile, rAcc);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rAcc); ++i) {
              rGate(i) = cutlass::bfloat16_t(rAcc(i) * params_.alpha);
            }
            copy(tiled_t2r, tUpTile, rAcc);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rAcc); ++i) {
              rUp(i) = cutlass::bfloat16_t(rAcc(i) * params_.alpha);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rD); ++i) {
              float gate = static_cast<float>(rGate(i));
              float up = static_cast<float>(rUp(i));
              float value = production_gelu(gate) * up;
              rD(i) = ElementD(value);
            }

            Tensor coord = tTR_cD(_, _, _, epi_m, epi_n);
            Tensor pred = lazy::transform(
                coord, [&](auto const& c) CUTLASS_LAMBDA_FUNC_INLINE {
                  return elem_less(c, compact_problem);
                });
            Tensor dst = recast<VectorType>(
                coalesce(tTR_gD(_, _, _, epi_m, epi_n)));
            Tensor src = recast<VectorType>(coalesce(rD));
            Tensor vec_pred = tensor<1>(zipped_divide(
                coalesce(pred), common_layout.compose(Int<Vector>{})));
            copy_if(vec_pred, src, dst);
          }
#else
          if (warp_idx == gate_owner) {
            copy(tiled_t2r, tGateTile, rAcc);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rAcc); ++i) {
              rGate(i) = cutlass::bfloat16_t(rAcc(i) * params_.alpha);
            }
            copy(rGate, tTR_sGate);
          }
          if (warp_idx == up_owner) {
            copy(tiled_t2r, tUpTile, rAcc);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rAcc); ++i) {
              rUp(i) = cutlass::bfloat16_t(rAcc(i) * params_.alpha);
            }
            copy(rUp, tTR_sUp);
          }
          cutlass::arch::fence_view_async_shared();
          synchronize();

          bool last = epi_m == NumEpiM - 1 && epi_n == NumEpiN - 1;
          if (last) {
            cutlass::arch::fence_view_async_tmem_load();
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }

          // Gate owner is the output consumer. Its register view addresses the
          // same D8 coordinates populated by both owner warps in shared.
          if (warp_idx == gate_owner) {
            copy(tTR_sGate, rGate);
            copy(tTR_sUp, rUp);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rD); ++i) {
              float gate = static_cast<float>(rGate(i));
              float up = static_cast<float>(rUp(i));
              float value = production_gelu(gate) * up;
              rD(i) = ElementD(value);
            }

            Tensor coord = tTR_cD(_, _, _, epi_m, epi_n);
            Tensor pred = lazy::transform(
                coord, [&](auto const& c) CUTLASS_LAMBDA_FUNC_INLINE {
                  return elem_less(c, compact_problem);
                });
            Tensor dst = recast<VectorType>(
                coalesce(tTR_gD(_, _, _, epi_m, epi_n)));
            Tensor src = recast<VectorType>(coalesce(rD));
            Tensor vec_pred = tensor<1>(zipped_divide(
                coalesce(pred), common_layout.compose(Int<Vector>{})));
            copy_if(vec_pred, src, dst);
          }
          if constexpr (!APXINF_DUAL_GEGLU_DOUBLE_BUFFER) {
            synchronize();
          }
#endif
        }
      }
#if APXINF_DUAL_GEGLU_REGISTER_RENDEZVOUS
      // Non-owner warps skip most of the loop and could otherwise release the
      // accumulator pipeline while owner warp 2 is still reading TMEM.  One
      // final full-CTA rendezvous is sufficient: it covers every TMEM read,
      // every output store, and the persistent CTA's next work tile.
      synchronize();
      cutlass::arch::fence_view_async_tmem_load();
      acc_pipeline.consumer_release(acc_pipe_consumer_state);
      ++acc_pipe_consumer_state;
#else
      // With ping-pong shared buffers, barrier i+1 proves consumer i finished
      // before producers can reuse buffer i at i+2. One final rendezvous is
      // still required before a persistent CTA advances to its next work tile.
      if constexpr (APXINF_DUAL_GEGLU_DOUBLE_BUFFER) {
        synchronize();
      }
#endif
      return make_tuple(acc_pipe_consumer_state);
    }

   private:
    Params const& params_;
    SharedStorage* shared_;
  };

  using CollectiveEpilogue =
      cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          PairedEpilogue>;
  using AtomCollective =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementInput,
          LayoutA, AlignmentInput, ElementInput, LayoutB, AlignmentInput,
          float, LegalAtomTile, ClusterShape,
          cutlass::gemm::collective::StageCount<APXINF_DUAL_GEGLU_WIDE_STAGE_COUNT>,
          cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>::CollectiveOp;
  using DispatchPolicy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecialized<
      APXINF_DUAL_GEGLU_WIDE_STAGE_COUNT, 2, 2, ClusterShape>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy, TileShape,
      ElementInput, typename AtomCollective::StrideA,
      ElementInput, typename AtomCollective::StrideB,
      typename AtomCollective::TiledMma,
      typename AtomCollective::GmemTiledCopyA,
      typename AtomCollective::SmemLayoutAtomA,
      typename AtomCollective::SmemCopyAtomA,
      typename AtomCollective::TransformA,
      typename AtomCollective::GmemTiledCopyB,
      typename AtomCollective::SmemLayoutAtomB,
      typename AtomCollective::SmemCopyAtomB,
      typename AtomCollective::TransformB>;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// Production entry: exact BF16 M522/M533 language Gate/Up only. The input weight
// must already use the physical [gate256,up256] column order validated by the
// load-time raw-bit oracle. This path allocates no workspace and launches one
// graph-capturable node.
int production_dual_geglu_bf16(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, cudaStream_t stream) {
  if (activation == nullptr || interleaved_weight == nullptr || output == nullptr ||
      stream == nullptr || !valid_production_m(m) || n != kN || k != kK ||
      full_n != kFullN || full_n != 2 * n ||
      (reinterpret_cast<uintptr_t>(activation) & 0xf) != 0 ||
      (reinterpret_cast<uintptr_t>(interleaved_weight) & 0xf) != 0 ||
      (reinterpret_cast<uintptr_t>(output) & 0xf) != 0) {
    return -6;
  }

  using Gemm = WidePairGemm<Shape<_128, _512, _64>, Shape<_2, _2, _1>>;
  using Device = typename Gemm::Device;
  using Kernel = typename Gemm::Kernel;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideD = typename Kernel::StrideD;
  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(full_n, k, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, 1));
  typename Kernel::MainloopArguments mainloop{
      static_cast<ElementInput const*>(activation), stride_a,
      static_cast<ElementInput const*>(interleaved_weight), stride_b};
  typename Kernel::EpilogueArguments epilogue{
      1.0f, 1.0f, static_cast<ElementOutput*>(output), stride_d};
  cutlass::KernelHardwareInfo hardware;
  cudaError_t cuda_status = cudaDeviceGetAttribute(
      &hardware.sm_count, cudaDevAttrMultiProcessorCount, 0);
  if (cuda_status != cudaSuccess) return -8;
  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, full_n, k, 1}, mainloop, epilogue, hardware, {}};
  Device operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  if (operation.get_workspace_size(arguments) != 0) return -4;
  if (operation.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess)
    return -2;
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}

}  // namespace apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail

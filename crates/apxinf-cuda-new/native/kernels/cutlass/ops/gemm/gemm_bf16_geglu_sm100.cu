// Copyright 2026 apxinf contributors.
// Native SM100/SM110 BF16 up GEMM + exact BF16 GeGLU EVT.

#if defined(__CUDA_ARCH_FEAT_SM101_ALL)
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif

#include <cuda_bf16.h>
#include "gemm_bf16_sm100.h"

#include <cuda_runtime.h>
#include <math.h>

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

namespace apxinf_cuda_cutlass_bf16_detail {

struct GeGluArguments {};

CUTLASS_DEVICE float production_gelu(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
      (1.0f + tanhf(kAlpha *
          (value + 0.044715f * value * value * value)));
}

template <class T>
struct ProductionGeGlu;

template <>
struct ProductionGeGlu<float> {
  using Arguments = GeGluArguments;

  CUTLASS_DEVICE float operator()(
      float const& gate, float const& up, Arguments const&) const {
    const float rounded_up = __bfloat162float(__float2bfloat16_rn(up));
    return production_gelu(gate) * rounded_up;
  }
};

template <class T, int N>
struct ProductionGeGlu<cutlass::Array<T, N>> {
  using Arguments = GeGluArguments;

  CUTLASS_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& gate,
      cutlass::Array<T, N> const& up,
      Arguments const&) const {
    cutlass::NumericArrayConverter<
        cutlass::bfloat16_t, T, N,
        cutlass::FloatRoundStyle::round_to_nearest> to_bf16;
    cutlass::NumericArrayConverter<
        T, cutlass::bfloat16_t, N,
        cutlass::FloatRoundStyle::round_to_nearest> to_compute;
    cutlass::Array<T, N> rounded_up = to_compute(to_bf16(up));
    cutlass::Array<T, N> output;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < N; ++index) {
      output[index] = production_gelu(gate[index]) * rounded_up[index];
    }
    return output;
  }
};

using ElementSource = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementCompute = float;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;
using GeGluEVTBase = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        ProductionGeGlu, ElementOutput, ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementSource>,
    cutlass::epilogue::fusion::Sm90AccFetch>;

struct GeGluEVT : GeGluEVTBase {
  using GeGluEVTBase::GeGluEVTBase;
};

struct GeGluOperation : cutlass::epilogue::fusion::FusionOperation {
  using ElementOutput = apxinf_cuda_cutlass_bf16_detail::ElementOutput;
  using ElementCompute = apxinf_cuda_cutlass_bf16_detail::ElementCompute;
  using ElementSource = apxinf_cuda_cutlass_bf16_detail::ElementSource;
  static constexpr bool IsSourceSupported = true;
};

}  // namespace apxinf_cuda_cutlass_bf16_detail

namespace cutlass::epilogue::fusion {
template <>
struct FusionCallbacksTraits<apxinf_cuda_cutlass_bf16_detail::GeGluEVT> {
  using DispatchPolicy = void;
  using Callbacks = apxinf_cuda_cutlass_bf16_detail::GeGluEVT;
  using Operation = apxinf_cuda_cutlass_bf16_detail::GeGluOperation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = apxinf_cuda_cutlass_bf16_detail::ElementCompute;
};
}  // namespace cutlass::epilogue::fusion

namespace apxinf::cuda::cutlass_ops {

template <typename TileShape, typename ClusterShape,
          typename MainloopSchedule>
struct Bf16GemmGeGlu {
  using ElementInput = cutlass::bfloat16_t;
  using ElementSource = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentInput = 16;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementSource, LayoutC,
          AlignmentC, ElementOutput, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto,
          apxinf_cuda_cutlass_bf16_detail::GeGluEVT>::CollectiveOp;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
          ElementInput, LayoutA, AlignmentInput,
          ElementInput, LayoutB, AlignmentInput,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(sizeof(
                  typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename Gemm>
int launch_bf16_geglu(
    const void* activation, const void* up_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, cudaStream_t stream) {
  using Device = typename Gemm::Device;
  using Kernel = typename Gemm::Kernel;
  using ElementInput = typename Gemm::ElementInput;
  using ElementSource = typename Gemm::ElementSource;
  using ElementOutput = typename Gemm::ElementOutput;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(full_n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(m, full_n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, 1));
  typename Kernel::MainloopArguments mainloop{
      static_cast<ElementInput const*>(activation), stride_a,
      static_cast<ElementInput const*>(up_weight), stride_b};
  typename Kernel::EpilogueArguments epilogue{
      {{}, {}, {}}, static_cast<ElementSource const*>(gate), stride_c,
      static_cast<ElementOutput*>(output), stride_d};
  cutlass::KernelHardwareInfo hardware;
  cudaDeviceGetAttribute(&hardware.sm_count, cudaDevAttrMultiProcessorCount, 0);
  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1}, mainloop, epilogue, hardware, {}};

  Device operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  if (operation.get_workspace_size(arguments) != 0) return -4;
  if (operation.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess)
    return -2;
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}

int bf16_gemm_geglu(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, int tactic,
    cudaStream_t stream) {
  if (activation == nullptr || packed_weight == nullptr || gate == nullptr ||
      output == nullptr || m != 789 || n != 16384 || k != 2048 ||
      full_n != 32768) {
    return -6;
  }
  const auto* up_weight = static_cast<const uint8_t*>(packed_weight) +
      static_cast<size_t>(n) * sizeof(cutlass::bfloat16_t);
  switch (tactic) {
    case 0:
      return launch_bf16_geglu<Bf16GemmGeGlu<
          Shape<_128, _256, _64>, Shape<_1, _2, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>>(
          activation, up_weight, gate, output, m, n, k, full_n, stream);
    case 1:
      return launch_bf16_geglu<Bf16GemmGeGlu<
          Shape<_256, _256, _64>, Shape<_2, _2, _1>,
          cutlass::gemm::collective::KernelScheduleAuto>>(
          activation, up_weight, gate, output, m, n, k, full_n, stream);
    case 2:
      return launch_bf16_geglu<Bf16GemmGeGlu<
          Shape<_256, _256, _64>, Shape<_2, _2, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>(
          activation, up_weight, gate, output, m, n, k, full_n, stream);
    default:
      return -5;
  }
}

}  // namespace apxinf::cuda::cutlass_ops

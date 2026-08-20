// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
// Raw-pointer adaptation of an upstream CUTLASS SM100 FP8 rowwise GEMM.

#if defined(__CUDA_ARCH_FEAT_SM101_ALL)
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif

#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

namespace apxinf::cuda::cutlass_ops {
template <typename TileShape, typename ClusterShape>
struct Fp8Gemm {
  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  // ApxInf stores linear weights physically as contiguous [K, N]. The original
  // wrapper passed a non-contiguous PyTorch transpose whose same
  // logical [K, N] tensor was physically [N, K], so its ColumnMajor tag was
  // correct there but transposed ApxInf weights a second time.  Keep the
  // shared ApxInf/cuBLASLt [K, N] allocation and describe it as RowMajor.
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentInput = 16;
  static constexpr int AlignmentOutput = 8;

  using FusionOperation = cutlass::epilogue::fusion::ScaledAcc<
      ElementOutput, float, float>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      float,
      void,
      LayoutD,
      AlignmentOutput,
      ElementOutput,
      LayoutD,
      AlignmentOutput,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOperation>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      ElementInput,
      LayoutA,
      AlignmentInput,
      ElementInput,
      LayoutB,
      AlignmentInput,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename Gemm>
int launch(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  using Device = typename Gemm::Device;
  using Kernel = typename Gemm::Kernel;
  using ElementInput = typename Gemm::ElementInput;
  using ElementOutput = typename Gemm::ElementOutput;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideD = typename Kernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n, k, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, 1));
  typename Kernel::MainloopArguments mainloop{
      static_cast<ElementInput const*>(activation), stride_a,
      static_cast<ElementInput const*>(weight), stride_b};
  typename Kernel::EpilogueArguments epilogue{
      {alpha, 0.0f, nullptr, nullptr}, nullptr, stride_d,
      static_cast<ElementOutput*>(output), stride_d};
  cutlass::KernelHardwareInfo hardware;
  cudaDeviceGetAttribute(&hardware.sm_count, cudaDevAttrMultiProcessorCount, 0);
  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1}, mainloop, epilogue, hardware, {}};

  Device operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  size_t required = operation.get_workspace_size(arguments);
  // These serial-K static-inference tactics must remain workspace-free so
  // execution is safe inside CUDA Graph capture. Reject any future CUTLASS
  // schedule that silently changes that contract.
  if (required != 0) return -4;
  if (operation.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess)
    return -2;
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}
int fp8_gemm_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, int tactic, cudaStream_t stream) {
  switch (tactic) {
    case 0:
      return launch<Fp8Gemm<Shape<_64, _64, _128>, Shape<_1, _4, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 1:
      return launch<Fp8Gemm<Shape<_64, _64, _128>, Shape<_1, _1, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 2:
      return launch<Fp8Gemm<Shape<_128, _128, _128>, Shape<_2, _1, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 3:
      return launch<Fp8Gemm<Shape<_256, _128, _64>, Shape<_2, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 4:
      return launch<Fp8Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 5:
      return launch<Fp8Gemm<Shape<_256, _128, _128>, Shape<_2, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 6:
      // Safe tall-N auto-scheduled candidate. Unlike the former
      // tactic 6, this is a regular one-SM schedule and is graph-replay safe.
      return launch<Fp8Gemm<Shape<_128, _256, _128>, Shape<_1, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 7:
      // Alternate wide auto-scheduled candidate. Keep the 2x1
      // cluster while avoiding the explicit two-SM epilogue that wedges the
      // current Thor-U driver during graph replay.
      return launch<Fp8Gemm<Shape<_256, _128, _128>, Shape<_2, _1, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    default:
      return -5;
  }
}

}  // namespace apxinf::cuda::cutlass_ops

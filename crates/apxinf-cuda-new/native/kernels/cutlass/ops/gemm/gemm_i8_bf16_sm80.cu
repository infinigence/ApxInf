/*
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * Copyright 2026 apxinf contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Raw-pointer adaptation of an upstream SM87 W8A8 GEMM. The custom epilogue
// applies one FP32 activation scale per row and one FP32 weight scale per
// output column, then writes BF16 directly. This removes the materialized
// INT32 matrix and the standalone dequantization kernel from every aligned
// Static-shape W8A8 linear layer.

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/numeric_types.h>

#include "extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "extensions/gemm/gemm_universal_base_compat.h"
#include "extensions/gemm/gemm_with_epilogue_visitor.h"

namespace apxinf::cuda::cutlass_ops {

template <typename ThreadblockShape, typename WarpShape, int NumStages>
cudaError_t run_w8a8_bf16(
    int8_t* activation,
    int8_t* weight_output_major,
    float* row_scales,
    float* column_scales,
    cutlass::bfloat16_t* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;
  using ElementOutput = cutlass::bfloat16_t;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  using DefaultGemmConf =
      cutlass::gemm::device::DefaultGemmConfiguration<
          OperatorClass,
          ArchTag,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementCompute>;
  using EpilogueOutputOp = typename DefaultGemmConf::EpilogueOutputOp;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInputA,
      cutlass::layout::RowMajor,
      DefaultGemmConf::kAlignmentA,
      ElementInputB,
      cutlass::layout::ColumnMajor,
      DefaultGemmConf::kAlignmentB,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      ThreadblockSwizzle,
      NumStages,
      true,
      typename DefaultGemmConf::Operator>::GemmKernel;

  using AlphaColTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIterator<
          cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
              typename DefaultGemmKernel::Epilogue::OutputTileIterator::
                  ThreadMap::Shape,
              typename DefaultGemmKernel::Epilogue::OutputTileIterator::
                  ThreadMap::Count,
              DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::
                  kThreads,
              DefaultGemmKernel::Epilogue::OutputTileIterator::
                  kElementsPerAccess,
              cutlass::sizeof_bits<ElementOutput>::value>,
          ElementCompute>;

  using EpilogueVisitor =
      cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
          ThreadblockShape,
          DefaultGemmKernel::kThreadCount,
          AlphaColTileIterator,
          typename DefaultGemmKernel::Epilogue::OutputTileIterator,
          ElementAccumulator,
          ElementCompute,
          EpilogueOutputOp>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<
          EpilogueVisitor,
          typename DefaultGemmKernel::Epilogue>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
      typename DefaultGemmKernel::Mma,
      Epilogue,
      ThreadblockSwizzle>;
  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  typename EpilogueOutputOp::Params linear_scaling;
  typename EpilogueVisitor::Arguments visitor_args{linear_scaling};
  typename Gemm::Arguments args{
      {m, n, k},
      {activation, k},
      {weight_output_major, k},
      {column_scales, 0},
      {row_scales, 0},
      {static_cast<ElementOutput*>(nullptr), 0},
      {output, n},
      visitor_args};

  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  // All dispatched calls use a single serial-K partition, so this kernel has
  // no auxiliary workspace. Refuse silently changing that contract.
  if (gemm.get_workspace_size(args) != 0) {
    return cudaErrorInvalidValue;
  }
  if (gemm(args, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}

cudaError_t w8a8_gemm_bf16(
    const void* activation,
    const void* weight_output_major,
    const void* row_scales,
    const void* column_scales,
    void* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  if (activation == nullptr || weight_output_major == nullptr ||
      row_scales == nullptr || column_scales == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (k % 16) != 0) {
    return cudaErrorInvalidValue;
  }

  // The inherited CUTLASS 2.x compatibility layer models all TensorRefs as
  // mutable even though A, B, and scales are read-only in the kernel.
  auto* a = const_cast<int8_t*>(static_cast<const int8_t*>(activation));
  auto* b =
      const_cast<int8_t*>(static_cast<const int8_t*>(weight_output_major));
  auto* a_scales =
      const_cast<float*>(static_cast<const float*>(row_scales));
  auto* b_scales =
      const_cast<float*>(static_cast<const float*>(column_scales));
  auto* d = static_cast<cutlass::bfloat16_t*>(output);

  // Orin-specific dispatch. Static inference uses the small-M branches for
  // the denoiser and the medium-M branch for the vision/language prefix.
  if (m <= 64 && n <= 4096) {
    return run_w8a8_bf16<
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        5>(a, b, a_scales, b_scales, d, m, n, k, stream);
  }
  if (m <= 64) {
    return run_w8a8_bf16<
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        3>(a, b, a_scales, b_scales, d, m, n, k, stream);
  }
  return run_w8a8_bf16<
      cutlass::gemm::GemmShape<128, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>,
      5>(a, b, a_scales, b_scales, d, m, n, k, stream);
}

}  // namespace apxinf::cuda::cutlass_ops

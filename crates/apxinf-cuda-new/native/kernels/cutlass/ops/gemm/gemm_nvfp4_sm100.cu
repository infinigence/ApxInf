// SPDX-License-Identifier: Apache-2.0
//
// Adapted from FlashRT's cutlass_fp4_gemm_variants.cu (Apache-2.0), which is
// based on CUTLASS example 72a. Allocation and initialization are deliberately
// split from run so the hot path and CUDA Graph capture perform no allocation.

#include "gemm_nvfp4_sm100.h"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

#include <memory>

namespace apxinf::cuda::cutlass_ops {
namespace {

using namespace cute;

__device__ unsigned char packed_nibble(const unsigned char* values,
                                       size_t logical) {
  const unsigned char byte = values[logical / 2];
  return (logical & 1U) == 0 ? (byte & 0xfU) : (byte >> 4);
}

__global__ void pack_b_kernel(const unsigned char* source,
                              unsigned char* destination, int n, int k) {
  const size_t byte_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t packed_bytes = static_cast<size_t>(n) * k / 2;
  if (byte_index >= packed_bytes) return;
  const size_t physical0 = byte_index * 2;
  const size_t column = physical0 / static_cast<size_t>(k);
  const size_t inner = physical0 % static_cast<size_t>(k);
  const size_t source0 = inner * static_cast<size_t>(n) + column;
  const size_t source1 =
      (inner + 1) * static_cast<size_t>(n) + column;
  destination[byte_index] =
      packed_nibble(source, source0) | (packed_nibble(source, source1) << 4);
}

__global__ void pack_scales_kernel(const unsigned char* source,
                                   unsigned char* destination, int rows,
                                   int k_blocks) {
  const size_t logical =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(rows) * k_blocks;
  if (logical >= count) return;
  const size_t row = logical / static_cast<size_t>(k_blocks);
  const size_t block = logical % static_cast<size_t>(k_blocks);
  const size_t column_supertiles = (static_cast<size_t>(k_blocks) + 3) / 4;
  const size_t offset =
      (row / 128 * column_supertiles + block / 4) * 512 +
      (row % 32) * 16 + (row / 32 % 4) * 4 + block % 4;
  destination[offset] = source[logical];
}

struct Runner {
  virtual ~Runner() = default;
  virtual int run(cudaStream_t stream) = 0;
};

template <class MmaTile, class Cluster>
struct VariantRunner final : Runner {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using Accumulator = float;
  using Arch = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OperatorClass, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, Accumulator,
      Accumulator, ElementC, LayoutC, AlignmentC, ElementD, LayoutD,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB,
      AlignmentB, Accumulator, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
  using Arguments = typename Gemm::Arguments;

  Gemm gemm;

  static Arguments arguments(const void* a, const void* a_scales,
                             const void* b, const void* b_scales,
                             void* output, int m, int n, int k, float alpha) {
    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    auto layout_sfa =
        ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto layout_sfb =
        ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    using AData = typename ElementA::DataType;
    using AScale = typename ElementA::ScaleFactorType;
    using BData = typename ElementB::DataType;
    using BScale = typename ElementB::ScaleFactorType;
    return Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {reinterpret_cast<const AData*>(a), stride_a,
         reinterpret_cast<const BData*>(b), stride_b,
         reinterpret_cast<const AScale*>(a_scales), layout_sfa,
         reinterpret_cast<const BScale*>(b_scales), layout_sfb},
        {{alpha, 0.0F}, reinterpret_cast<ElementC*>(output), stride_c,
         reinterpret_cast<ElementD*>(output), stride_d}};
  }

  static size_t workspace_size(int m, int n, int k) {
    auto args = arguments(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k,
                          1.0F);
    return Gemm::get_workspace_size(args);
  }

  static int create(const void* a, const void* a_scales, const void* b,
                    const void* b_scales, void* output, int m, int n, int k,
                    float alpha, void* workspace, cudaStream_t stream,
                    void** result) {
    auto runner = std::make_unique<VariantRunner>();
    auto args = arguments(a, a_scales, b, b_scales, output, m, n, k, alpha);
    auto status = runner->gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(status) | 0x10000;
    }
    status = runner->gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(status) | 0x20000;
    }
    *result = runner.release();
    return 0;
  }

  int run(cudaStream_t stream) override {
    const auto status = gemm.run(stream);
    return status == cutlass::Status::kSuccess
               ? 0
               : (static_cast<int>(status) | 0x30000);
  }
};

// FlashRT V1: strong Pi0.5 down-projection configuration.
using V1 = VariantRunner<Shape<_128, _256, _128>, Shape<_2, _1, _1>>;
// FlashRT V6: strong QKV/O and general wide-N configuration.
using V6 = VariantRunner<Shape<_128, _256, _128>, Shape<_1, _1, _1>>;
// FlashRT V8: strong Pi0.5 gate/up configuration.
using V8 = VariantRunner<Shape<_128, _256, _256>, Shape<_1, _1, _1>>;

}  // namespace

int nvfp4_num_configurations() { return 3; }

size_t nvfp4_scale_workspace_size(int rows, int k) {
  const size_t row_supertiles = (static_cast<size_t>(rows) + 127) / 128;
  const size_t k_blocks = static_cast<size_t>(k) / 16;
  const size_t column_supertiles = (k_blocks + 3) / 4;
  return row_supertiles * column_supertiles * 512;
}

size_t nvfp4_packed_b_size(int n, int k) {
  return static_cast<size_t>(n) * k / 2;
}

cudaError_t nvfp4_pack_scales(const void* source, void* destination, int rows,
                              int k, cudaStream_t stream) {
  const size_t count = static_cast<size_t>(rows) * (k / 16);
  const int blocks = static_cast<int>((count + 255) / 256);
  pack_scales_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const unsigned char*>(source),
      static_cast<unsigned char*>(destination), rows, k / 16);
  return cudaGetLastError();
}

cudaError_t nvfp4_pack_b(const void* source, void* destination, int n, int k,
                         cudaStream_t stream) {
  const size_t count = nvfp4_packed_b_size(n, k);
  const int blocks = static_cast<int>((count + 255) / 256);
  pack_b_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const unsigned char*>(source),
      static_cast<unsigned char*>(destination), n, k);
  return cudaGetLastError();
}

size_t nvfp4_workspace_size(int configuration, int m, int n, int k) {
  switch (configuration) {
    case 0:
      return V1::workspace_size(m, n, k);
    case 1:
      return V6::workspace_size(m, n, k);
    case 2:
      return V8::workspace_size(m, n, k);
    default:
      return static_cast<size_t>(-1);
  }
}

int nvfp4_create(int configuration, const void* a, const void* a_scales,
                 const void* b, const void* b_scales, void* output, int m,
                 int n, int k, float alpha, void* workspace,
                 cudaStream_t stream, void** runner) {
  if (runner == nullptr) return -1;
  *runner = nullptr;
  switch (configuration) {
    case 0:
      return V1::create(a, a_scales, b, b_scales, output, m, n, k, alpha,
                        workspace, stream, runner);
    case 1:
      return V6::create(a, a_scales, b, b_scales, output, m, n, k, alpha,
                        workspace, stream, runner);
    case 2:
      return V8::create(a, a_scales, b, b_scales, output, m, n, k, alpha,
                        workspace, stream, runner);
    default:
      return -1;
  }
}

int nvfp4_run(void* runner, cudaStream_t stream) {
  if (runner == nullptr) return -1;
  return static_cast<Runner*>(runner)->run(stream);
}

void nvfp4_destroy(void* runner) noexcept {
  delete static_cast<Runner*>(runner);
}

}  // namespace apxinf::cuda::cutlass_ops

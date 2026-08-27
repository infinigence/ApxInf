// Copyright 2026 apxinf contributors.
// SM89 BF16 gate/up GEMM with a GeGLU visitor epilogue.
//
// The weight must be row-major [K, 2*N] with column-pair layout
// [g0,u0,g1,u1,...]. The epilogue consumes adjacent gate/up accumulator
// values and writes the compressed [M, N] GeGLU output directly.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/numeric_types.h>

#include "extensions/gemm/gemm_universal_base_compat.h"
#include "extensions/gemm/gemm_with_epilogue_visitor.h"

namespace apxinf::cuda::cutlass_ops {

template <
    typename ThreadblockShape_,
    int ThreadCount,
    typename ScaleTileIterator_,
    typename OutputTileIterator_>
class Bf16InterleavedGeGluVisitor {
 public:
  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;
  using ScaleTileIterator = ScaleTileIterator_;
  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = cutlass::epilogue::thread::LinearCombination<
      cutlass::bfloat16_t,
      OutputTileIterator::kElementsPerAccess,
      float,
      float>;
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using AccumulatorFragment = cutlass::Array<ElementAccumulator, kElementsPerAccess>;

  struct Arguments {
    int output_cols;
    CUTLASS_HOST_DEVICE Arguments(int output_cols_ = 0) : output_cols(output_cols_) {}
  };

  struct Params {
    int output_cols;
    CUTLASS_HOST_DEVICE Params() : output_cols(0) {}
    CUTLASS_HOST_DEVICE Params(Arguments const& args) : output_cols(args.output_cols) {}
  };

  struct SharedStorage {};

 private:
  Params const& params_;
  cutlass::MatrixCoord extent_;
  ElementOutput* ptr_D_;
  OutputTileIterator iterator_D_;

 public:
  CUTLASS_DEVICE Bf16InterleavedGeGluVisitor(
      Params const& params,
      SharedStorage&,
      cutlass::MatrixCoord const& problem_size,
      int thread_idx,
      int,
      int,
      typename ScaleTileIterator::Params,
      typename OutputTileIterator::Params,
      typename OutputTileIterator::Params params_D,
      bool,
      bool,
      bool,
      typename ScaleTileIterator::Element*,
      typename ScaleTileIterator::Element*,
      ElementOutput*,
      ElementOutput* ptr_D,
      cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0, 0),
      int = 0,
      cutlass::MatrixCoord const& = cutlass::MatrixCoord(0, 0))
      : params_(params),
        extent_(problem_size),
        ptr_D_(ptr_D),
        iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset) {}

  CUTLASS_DEVICE void set_k_partition(int, int) {}
  CUTLASS_DEVICE void set_batch_index(int) {}
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) {}
  CUTLASS_DEVICE void begin_row(int) {}

  CUTLASS_DEVICE static float gelu(float x) {
    constexpr float kBeta = 0.7978845608028654f;
    constexpr float kAlpha = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(kBeta * (x + kAlpha * x * x * x)));
  }

  CUTLASS_DEVICE void visit(
      int,
      int row_idx,
      int column_idx,
      int,
      AccumulatorFragment const& accum) {
    using ThreadMap = typename OutputTileIterator::ThreadMap;
    int row = iterator_D_.thread_start_row() + ThreadMap::iteration_offset(row_idx).row();
    int col = iterator_D_.thread_start_column() + ThreadMap::iteration_offset(column_idx).column();
    if (row >= extent_.row()) {
      return;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e + 1 < kElementsPerAccess; e += 2) {
      int full_col = col + e;
      if (full_col + 1 < extent_.column()) {
        float gate = accum[e];
        float up = __bfloat162float(__float2bfloat16_rn(accum[e + 1]));
        ptr_D_[row * params_.output_cols + full_col / 2] =
            cutlass::bfloat16_t(gelu(gate) * up);
      }
    }
  }

  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void end_step(int) { ++iterator_D_; }
  CUTLASS_DEVICE void end_epilogue() {}
};

template <typename ThreadblockShape, typename WarpShape, int NumStages>
cudaError_t run_interleaved_geglu(
    const void* activation,
    const void* interleaved_weight,
    void* output,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  using ElementInput = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ArchTag = cutlass::arch::Sm80;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
  using DefaultGemmConf = cutlass::gemm::device::DefaultGemmConfiguration<
      OperatorClass,
      ArchTag,
      ElementInput,
      ElementInput,
      ElementOutput,
      ElementCompute>;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      8,
      ElementAccumulator,
      ElementCompute>;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInput,
      cutlass::layout::RowMajor,
      8,
      ElementInput,
      cutlass::layout::RowMajor,
      8,
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

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::Count,
          DefaultGemmKernel::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          DefaultGemmKernel::Epilogue::OutputTileIterator::kElementsPerAccess,
          cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  using EpilogueVisitor = Bf16InterleavedGeGluVisitor<
      ThreadblockShape,
      DefaultGemmKernel::kThreadCount,
      AlphaColTileIterator,
      typename DefaultGemmKernel::Epilogue::OutputTileIterator>;
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
      EpilogueVisitor,
      typename DefaultGemmKernel::Epilogue>::Epilogue;
  using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
      typename DefaultGemmKernel::Mma,
      Epilogue,
      ThreadblockSwizzle>;
  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  auto* a = const_cast<ElementInput*>(static_cast<ElementInput const*>(activation));
  auto* b = const_cast<ElementInput*>(static_cast<ElementInput const*>(interleaved_weight));
  auto* d = static_cast<ElementOutput*>(output);
  typename EpilogueVisitor::Arguments visitor_args{n};
  typename Gemm::Arguments args{
      {m, 2 * n, k},
      {a, k},
      {b, 2 * n},
      {nullptr, 0},
      {nullptr, 0},
      {nullptr, 0},
      {d, n},
      visitor_args};

  Gemm gemm;
  auto can = gemm.can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "sm89 geglu can_implement failed m=%d n=%d k=%d status=%d\n", m, n, k, int(can));
    return cudaErrorInvalidValue;
  }
  auto workspace = gemm.get_workspace_size(args);
  if (workspace != 0) {
    std::fprintf(stderr, "sm89 geglu workspace nonzero m=%d n=%d k=%d workspace=%zu\n", m, n, k, workspace);
    return cudaErrorInvalidValue;
  }
  auto launched = gemm(args, nullptr, stream);
  if (launched != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "sm89 geglu launch failed m=%d n=%d k=%d status=%d\n", m, n, k, int(launched));
    return cudaErrorUnknown;
  }
  return cudaGetLastError();
}

namespace bf16_sm89_detail {

int interleaved_geglu(
    const void* activation,
    const void* interleaved_weight,
    void* output,
    int m,
    int n,
    int k,
    int full_n,
    int tactic,
    cudaStream_t stream) {
  (void)tactic;
  if (activation == nullptr || interleaved_weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || full_n != 2 * n) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (n == 4096 && k == 1024) {
    return static_cast<int>(::apxinf::cuda::cutlass_ops::run_interleaved_geglu<
        cutlass::gemm::GemmShape<32, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        3>(activation, interleaved_weight, output, m, n, k, stream));
  }
  if (n == 16384 && k == 2048) {
    return static_cast<int>(::apxinf::cuda::cutlass_ops::run_interleaved_geglu<
        cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>,
        3>(activation, interleaved_weight, output, m, n, k, stream));
  }
  return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace bf16_sm89_detail
}  // namespace apxinf::cuda::cutlass_ops

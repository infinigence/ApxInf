// Copyright 2026 apxinf contributors.
// BF16 GEMM with BF16 accumulator rounding followed by squared ReLU
// (relu(x)^2 = fmaxf(x,0)^2), the Cosmos3-Edge MoT MLP activation.

#if defined(__CUDA_ARCH_FEAT_SM101_ALL)
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif

#include <cuda_bf16.h>


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

namespace apxinf_cuda_new_relu2_detail {

struct Relu2Arguments {};

CUTLASS_DEVICE float production_relu2(float acc) {
  const float value = __bfloat162float(__float2bfloat16_rn(acc));
  return fmaxf(value, 0.0f) * fmaxf(value, 0.0f);
}

template <class T>
struct StagedRelu2;
template <>
struct StagedRelu2<float> {
  using Arguments = Relu2Arguments;
  CUTLASS_DEVICE float operator()(float const& acc, Arguments const&) const {
    return production_relu2(acc);
  }
};
template <class T, int N>
struct StagedRelu2<cutlass::Array<T, N>> {
  using Arguments = Relu2Arguments;
  CUTLASS_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& acc, Arguments const&) const {
    cutlass::Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) result[i] = production_relu2(acc[i]);
    return result;
  }
};

using ElementSource = void;
using ElementOutput = cutlass::bfloat16_t;
using ElementCompute = float;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;
using Relu2EVTBase = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        StagedRelu2, ElementOutput, ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90AccFetch>;

struct Relu2EVT : Relu2EVTBase {
  using Relu2EVTBase::Relu2EVTBase;
};

struct Relu2Operation : cutlass::epilogue::fusion::FusionOperation {
  using ElementOutput = apxinf_cuda_new_relu2_detail::ElementOutput;
  using ElementCompute = apxinf_cuda_new_relu2_detail::ElementCompute;
  using ElementSource = apxinf_cuda_new_relu2_detail::ElementSource;
  static constexpr bool IsSourceSupported = false;
};

}  // namespace apxinf_cuda_new_relu2_detail

namespace cutlass::epilogue::fusion {
template <>
struct FusionCallbacksTraits<apxinf_cuda_new_relu2_detail::Relu2EVT> {
  using DispatchPolicy = void;
  using Callbacks = apxinf_cuda_new_relu2_detail::Relu2EVT;
  using Operation = apxinf_cuda_new_relu2_detail::Relu2Operation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = apxinf_cuda_new_relu2_detail::ElementCompute;
};
}  // namespace cutlass::epilogue::fusion

namespace apxinf::cuda_new::cutlass_ops {

template <typename TileShape, typename ClusterShape,
          typename MainloopSchedule,
          int Swizzle_ = 0,
          cutlass::gemm::kernel::detail::RasterOrderOptions Raster_ =
              cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic>
struct StagedRelu2Gemm {
  static constexpr int Swizzle = Swizzle_;
  static constexpr auto Raster = Raster_;
  using ElementInput = cutlass::bfloat16_t;
  using ElementSource = void;
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
          apxinf_cuda_new_relu2_detail::Relu2EVT>::CollectiveOp;
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

}  // namespace apxinf::cuda_new::cutlass_ops

#include <algorithm>
#include <memory>
#include <stdexcept>
namespace apxinf::cuda_new::cutlass_ops {
struct Relu2Execution {
  virtual ~Relu2Execution() = default;
  virtual int run(cudaStream_t) = 0;
};
template <class Gemm>
struct Relu2Plan : Relu2Execution {
  using Kernel = typename Gemm::Kernel;
  typename Gemm::Device device;
  void* workspace = nullptr;
  ~Relu2Plan() override {
    if (workspace) cudaFree(workspace);
  }
  static typename Kernel::Arguments arguments(
      const void* a, const void* b, void* d) {
    constexpr int m = 1971, n = 9216, k = 2048;
    auto sa = cutlass::make_cute_packed_stride(
        typename Kernel::StrideA{}, cute::make_shape(m, k, 1));
    auto sb = cutlass::make_cute_packed_stride(
        typename Kernel::StrideB{}, cute::make_shape(n, k, 1));
    auto sc = cutlass::make_cute_packed_stride(
        typename Kernel::StrideC{}, cute::make_shape(m, n, 1));
    auto sd = cutlass::make_cute_packed_stride(
        typename Kernel::StrideD{}, cute::make_shape(m, n, 1));
    typename Kernel::MainloopArguments mainloop{
        static_cast<const typename Gemm::ElementInput*>(a), sa,
        static_cast<const typename Gemm::ElementInput*>(b), sb};
    auto epilogue = [&]() {
      using Args = typename Kernel::EpilogueArguments;
      // Tree visitor arguments are ordered child (accumulator), then node.
      return Args{{}, nullptr, sc,
                  static_cast<typename Gemm::ElementOutput*>(d), sd};
    }();
    cutlass::KernelHardwareInfo hw;
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaDeviceGetAttribute(&hw.sm_count, cudaDevAttrMultiProcessorCount,
                               device_id) != cudaSuccess)
      throw std::runtime_error("relu2 device query failed");
    // The SM100 default scheduler is the CLC persistent tile scheduler; the
    // swizzle/raster pair decides which of A (N-major) or B (M-major) panels
    // stay resident in L2 across consecutive tiles.
    return {cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
            mainloop, epilogue, hw, {Gemm::Swizzle, Gemm::Raster}};
  }
  static size_t bytes() {
    return Gemm::Device::get_workspace_size(arguments(nullptr, nullptr, nullptr));
  }
  void initialize(const void* a, const void* b, void* d, size_t limit,
                  cudaStream_t stream) {
    auto args = arguments(a, b, d);
    if (device.can_implement(args) != cutlass::Status::kSuccess)
      throw std::runtime_error("relu2 unsupported arguments");
    const auto size = device.get_workspace_size(args);
    if (size > limit) throw std::runtime_error("relu2 workspace exceeds limit");
    if (size && cudaMalloc(&workspace, size) != cudaSuccess)
      throw std::runtime_error("relu2 workspace allocation failed");
    if (device.initialize(args, workspace, stream) != cutlass::Status::kSuccess)
      throw std::runtime_error("relu2 preparation failed");
  }
  int run(cudaStream_t stream) override {
    return device.run(stream) == cutlass::Status::kSuccess ? 0 : -1;
  }
};
using P0 = Relu2Plan<
    StagedRelu2Gemm<Shape<_128, _256, _64>, Shape<_1, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>>;
using P1 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _256, _64>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P2 = Relu2Plan<
    StagedRelu2Gemm<Shape<_128, _128, _64>, Shape<_1, _1, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>>;
using P3 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _128, _64>, Shape<_2, _1, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P4 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _192, _64>, Shape<_2, _1, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P5 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
// Config 5 plus explicit tile-scheduler policies. The N-major raster keeps the
// small A (activation) panel resident; the M-major raster keeps the 36 MB
// weight panel resident across the four M-cluster rows.
using P6 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, 8,
                    cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM>>;
using P7 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, 8,
                    cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN>>;
using P8 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, 4,
                    cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM>>;
using P9 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, 0,
                    cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM>>;
using P10 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, 16,
                    cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM>>;
// N=192 divides 9216 exactly and won the eager/session sweep over 256: the smaller
// N tile (closer to nvjet's 200) leaves more TMEM/epilogue slack. Variants
// probe 208 (closer to nvjet but with a partial N tile), K=128, and the
// 2x2 cluster for A multicast.
using P11 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _208, _64>, Shape<_2, _1, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P12 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _192, _128>, Shape<_2, _1, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P13 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _208, _128>, Shape<_2, _1, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P14 = Relu2Plan<
    StagedRelu2Gemm<Shape<_256, _192, _64>, Shape<_2, _2, _1>,
                    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
size_t relu2_workspace() {
  return std::max({P0::bytes(), P1::bytes(), P2::bytes(), P3::bytes(), P4::bytes(),
                   P5::bytes(), P6::bytes(), P7::bytes(), P8::bytes(), P9::bytes(),
                   P10::bytes(), P11::bytes(), P12::bytes(), P13::bytes(),
                   P14::bytes()});
}
template <class Plan>
Relu2Execution* make_relu2(const void* a, const void* b, void* d, size_t limit,
                           cudaStream_t stream) {
  auto owned = std::make_unique<Plan>();
  owned->initialize(a, b, d, limit, stream);
  return owned.release();
}
Relu2Execution* prepare_relu2(const void* a, const void* b, void* d, int config,
                              size_t limit, cudaStream_t stream) {
  switch (config) {
    case 0: return make_relu2<P0>(a, b, d, limit, stream);
    case 1: return make_relu2<P1>(a, b, d, limit, stream);
    case 2: return make_relu2<P2>(a, b, d, limit, stream);
    case 3: return make_relu2<P3>(a, b, d, limit, stream);
    case 4: return make_relu2<P4>(a, b, d, limit, stream);
    case 5: return make_relu2<P5>(a, b, d, limit, stream);
    case 6: return make_relu2<P6>(a, b, d, limit, stream);
    case 7: return make_relu2<P7>(a, b, d, limit, stream);
    case 8: return make_relu2<P8>(a, b, d, limit, stream);
    case 9: return make_relu2<P9>(a, b, d, limit, stream);
    case 10: return make_relu2<P10>(a, b, d, limit, stream);
    case 11: return make_relu2<P11>(a, b, d, limit, stream);
    case 12: return make_relu2<P12>(a, b, d, limit, stream);
    case 13: return make_relu2<P13>(a, b, d, limit, stream);
    case 14: return make_relu2<P14>(a, b, d, limit, stream);
    default: throw std::runtime_error("invalid relu2 configuration");
  }
}
int enqueue_relu2(Relu2Execution* plan, cudaStream_t stream) {
  return plan->run(stream);
}
void destroy_relu2(Relu2Execution* plan) { delete plan; }
}  // namespace apxinf::cuda_new::cutlass_ops

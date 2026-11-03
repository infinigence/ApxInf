// Copyright 2026 apxinf contributors.
// BF16 GEMM with BF16 accumulator rounding followed by exact erf GELU.

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

namespace apxinf_cuda_new_exact_gelu_detail {

struct ExactGeluArguments {};

CUTLASS_DEVICE float production_gelu(float acc) {
 const float value=__bfloat162float(__float2bfloat16_rn(acc));
 return 0.5f*value*(1.0f+erff(value*0.7071067811865475f));
}
template<class T> struct StagedGelu;
template<> struct StagedGelu<float> {
 using Arguments=ExactGeluArguments;
 CUTLASS_DEVICE float operator()(float const& acc,Arguments const&)const{return production_gelu(acc);}
};
template<class T,int N>struct StagedGelu<cutlass::Array<T,N>> {
 using Arguments=ExactGeluArguments;
 CUTLASS_DEVICE cutlass::Array<T,N> operator()(cutlass::Array<T,N>const& acc,Arguments const&)const{
 cutlass::Array<T,N> result;
 CUTLASS_PRAGMA_UNROLL
 for(int i=0;i<N;++i)result[i]=production_gelu(acc[i]);return result;}
};

constexpr size_t kLookupBytes = 65536 * sizeof(uint16_t);
struct LookupGeluArguments { const uint16_t* table = nullptr; };
__global__ void initialize_exact_gelu_table(uint16_t* table) {
  const unsigned bits = blockIdx.x * blockDim.x + threadIdx.x;
  if (bits < 65536) {
    const float value = __bfloat162float(__ushort_as_bfloat16(uint16_t(bits)));
    table[bits] = __bfloat16_as_ushort(__float2bfloat16_rn(production_gelu(value)));
  }
}
CUTLASS_DEVICE float lookup_gelu(float acc, const uint16_t* table) {
  const unsigned bits = __bfloat16_as_ushort(__float2bfloat16_rn(acc));
  return __bfloat162float(__ushort_as_bfloat16(__ldg(table + bits)));
}
template<class T> struct LookupGelu;
template<> struct LookupGelu<float> {
  using Arguments = LookupGeluArguments;
  CUTLASS_DEVICE float operator()(float const& acc, Arguments const& args) const {
    return lookup_gelu(acc, args.table);
  }
};
template<class T, int N> struct LookupGelu<cutlass::Array<T, N>> {
  using Arguments = LookupGeluArguments;
  CUTLASS_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& acc, Arguments const& args) const {
    cutlass::Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) result[i] = lookup_gelu(acc[i], args.table);
    return result;
  }
};

using ElementSource = void;
using ElementOutput = cutlass::bfloat16_t;
using ElementCompute = float;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;
using ExactGeluEVTBase = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        StagedGelu, ElementOutput, ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90AccFetch>;

struct ExactGeluEVT : ExactGeluEVTBase {
  using ExactGeluEVTBase::ExactGeluEVTBase;
};

using LookupGeluEVTBase = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        LookupGelu, ElementOutput, ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90AccFetch>;
struct LookupGeluEVT : LookupGeluEVTBase {
  using LookupGeluEVTBase::LookupGeluEVTBase;
};

struct ExactGeluOperation : cutlass::epilogue::fusion::FusionOperation {
  using ElementOutput = apxinf_cuda_new_exact_gelu_detail::ElementOutput;
  using ElementCompute = apxinf_cuda_new_exact_gelu_detail::ElementCompute;
  using ElementSource = apxinf_cuda_new_exact_gelu_detail::ElementSource;
  static constexpr bool IsSourceSupported = false;
};

}  // namespace apxinf_cuda_new_exact_gelu_detail

namespace cutlass::epilogue::fusion {
template <>
struct FusionCallbacksTraits<apxinf_cuda_new_exact_gelu_detail::ExactGeluEVT> {
  using DispatchPolicy = void;
  using Callbacks = apxinf_cuda_new_exact_gelu_detail::ExactGeluEVT;
  using Operation = apxinf_cuda_new_exact_gelu_detail::ExactGeluOperation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = apxinf_cuda_new_exact_gelu_detail::ElementCompute;
};
template <>
struct FusionCallbacksTraits<apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT> {
  using DispatchPolicy = void;
  using Callbacks = apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT;
  using Operation = apxinf_cuda_new_exact_gelu_detail::ExactGeluOperation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = float;
};
}  // namespace cutlass::epilogue::fusion

namespace apxinf::cuda_new::cutlass_ops {

template <typename TileShape, typename ClusterShape,
          typename MainloopSchedule,
          typename Fusion = apxinf_cuda_new_exact_gelu_detail::ExactGeluEVT>
struct StagedGeluGemm {
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
          Fusion>::CollectiveOp;
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


} // namespace apxinf::cuda_new::cutlass_ops

#include <algorithm>
#include <memory>
#include <stdexcept>
namespace apxinf::cuda_new::cutlass_ops {
struct ExactGeluExecution {
  virtual ~ExactGeluExecution() = default;
  virtual int run(cudaStream_t) = 0;
};
template<class Gemm, bool UseLookup = false> struct ExactGeluPlan : ExactGeluExecution {
  using Kernel = typename Gemm::Kernel;
  typename Gemm::Device device;
  void* workspace = nullptr;
  uint16_t* table = nullptr;
  ~ExactGeluPlan() override {
    if (workspace) cudaFree(workspace);
    if (table) cudaFree(table);
  }
  static typename Kernel::Arguments arguments(const void* a, const void* b, void* d, const uint16_t* table = nullptr) {
    constexpr int m=1764,n=8192,k=2048;
    auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},cute::make_shape(m,k,1));
    auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},cute::make_shape(n,k,1));
    auto sc=cutlass::make_cute_packed_stride(typename Kernel::StrideC{},cute::make_shape(m,n,1));
    auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},cute::make_shape(m,n,1));
    typename Kernel::MainloopArguments mainloop{static_cast<const typename Gemm::ElementInput*>(a),sa,
        static_cast<const typename Gemm::ElementInput*>(b),sb};
    auto epilogue = [&]() {
      using Args = typename Kernel::EpilogueArguments;
      // Tree visitor arguments are ordered child (accumulator), then node.
      if constexpr (UseLookup) {
        return Args{{{}, {table}}, nullptr, sc,
                    static_cast<typename Gemm::ElementOutput*>(d), sd};
      } else {
        return Args{{{}, {}}, nullptr, sc,
                    static_cast<typename Gemm::ElementOutput*>(d), sd};
      }
    }();
    cutlass::KernelHardwareInfo hw;
    int device_id=0;
    if(cudaGetDevice(&device_id)!=cudaSuccess || cudaDeviceGetAttribute(&hw.sm_count,cudaDevAttrMultiProcessorCount,device_id)!=cudaSuccess)
      throw std::runtime_error("GELU device query failed");
    return {cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},mainloop,epilogue,hw,{}};
  }
  static size_t bytes() {
    return Gemm::Device::get_workspace_size(arguments(nullptr,nullptr,nullptr)) +
        (UseLookup ? apxinf_cuda_new_exact_gelu_detail::kLookupBytes : 0);
  }
  void initialize(const void* a,const void* b,void* d,size_t limit,cudaStream_t stream) {
    auto args=arguments(a,b,d);
    if(device.can_implement(args)!=cutlass::Status::kSuccess)throw std::runtime_error("GELU unsupported arguments");
    const auto size=device.get_workspace_size(args);
    const size_t table_bytes = UseLookup ? apxinf_cuda_new_exact_gelu_detail::kLookupBytes : 0;
    if (table_bytes > limit || size > limit - table_bytes)
      throw std::runtime_error("GELU workspace exceeds limit");
    if constexpr (UseLookup) {
      if (cudaMalloc(&table, table_bytes) != cudaSuccess)
        throw std::runtime_error("GELU table allocation failed");
      apxinf_cuda_new_exact_gelu_detail::initialize_exact_gelu_table<<<256,256,0,stream>>>(table);
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("GELU table initialization failed");
    }
    if(size && cudaMalloc(&workspace,size)!=cudaSuccess)throw std::runtime_error("GELU workspace allocation failed");
    if(device.initialize(arguments(a,b,d,table),workspace,stream)!=cutlass::Status::kSuccess)throw std::runtime_error("GELU preparation failed");
  }
  int run(cudaStream_t stream) override { return device.run(stream)==cutlass::Status::kSuccess ? 0 : -1; }
};
using P0=ExactGeluPlan<StagedGeluGemm<Shape<_128,_256,_64>,Shape<_1,_2,_1>,cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>>;
using P1=ExactGeluPlan<StagedGeluGemm<Shape<_256,_256,_64>,Shape<_2,_2,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P2=ExactGeluPlan<StagedGeluGemm<Shape<_128,_128,_64>,Shape<_1,_1,_1>,cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>>;
using P3=ExactGeluPlan<StagedGeluGemm<Shape<_256,_128,_64>,Shape<_2,_1,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P4=ExactGeluPlan<StagedGeluGemm<Shape<_256,_192,_64>,Shape<_2,_1,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using P5=ExactGeluPlan<StagedGeluGemm<Shape<_256,_256,_128>,Shape<_2,_2,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>;
using L0=ExactGeluPlan<StagedGeluGemm<Shape<_128,_256,_64>,Shape<_1,_2,_1>,cutlass::gemm::KernelTmaWarpSpecialized1SmSm100, apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT>, true>;
using L1=ExactGeluPlan<StagedGeluGemm<Shape<_256,_256,_64>,Shape<_2,_2,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT>, true>;
using L2=ExactGeluPlan<StagedGeluGemm<Shape<_128,_128,_64>,Shape<_1,_1,_1>,cutlass::gemm::KernelTmaWarpSpecialized1SmSm100, apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT>, true>;
using L3=ExactGeluPlan<StagedGeluGemm<Shape<_256,_128,_64>,Shape<_2,_1,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT>, true>;
using L4=ExactGeluPlan<StagedGeluGemm<Shape<_256,_192,_64>,Shape<_2,_1,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT>, true>;
using L5=ExactGeluPlan<StagedGeluGemm<Shape<_256,_256,_128>,Shape<_2,_2,_1>,cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, apxinf_cuda_new_exact_gelu_detail::LookupGeluEVT>, true>;
size_t exact_gelu_lut_workspace() { return std::max({L0::bytes(),L1::bytes(),L2::bytes(),L3::bytes(),L4::bytes(),L5::bytes()}); }
size_t exact_gelu_workspace() { return std::max({P0::bytes(),P1::bytes(),P2::bytes(),P3::bytes(),P4::bytes(),P5::bytes()}); }
template<class Plan> ExactGeluExecution* make_exact_gelu(const void*a,const void*b,void*d,size_t limit,cudaStream_t stream) {
  auto owned=std::make_unique<Plan>();owned->initialize(a,b,d,limit,stream);return owned.release();
}
ExactGeluExecution* prepare_exact_gelu(const void*a,const void*b,void*d,int config,size_t limit,cudaStream_t stream) {
  switch(config) {
    case 0:return make_exact_gelu<P0>(a,b,d,limit,stream);
    case 1:return make_exact_gelu<P1>(a,b,d,limit,stream);
    case 2:return make_exact_gelu<P2>(a,b,d,limit,stream);
    case 3:return make_exact_gelu<P3>(a,b,d,limit,stream);
    case 4:return make_exact_gelu<P4>(a,b,d,limit,stream);
    case 5:return make_exact_gelu<P5>(a,b,d,limit,stream);
    default:throw std::runtime_error("invalid exact GELU configuration");
  }
}
ExactGeluExecution* prepare_exact_gelu_lut(const void*a,const void*b,void*d,int config,size_t limit,cudaStream_t stream) {
  switch(config) {
    case 0:return make_exact_gelu<L0>(a,b,d,limit,stream);
    case 1:return make_exact_gelu<L1>(a,b,d,limit,stream);
    case 2:return make_exact_gelu<L2>(a,b,d,limit,stream);
    case 3:return make_exact_gelu<L3>(a,b,d,limit,stream);
    case 4:return make_exact_gelu<L4>(a,b,d,limit,stream);
    case 5:return make_exact_gelu<L5>(a,b,d,limit,stream);
    default:throw std::runtime_error("invalid exact GELU configuration");
  }
}
int enqueue_exact_gelu(ExactGeluExecution* plan,cudaStream_t stream) { return plan->run(stream); }
void destroy_exact_gelu(ExactGeluExecution* plan) { delete plan; }
} // namespace apxinf::cuda_new::cutlass_ops

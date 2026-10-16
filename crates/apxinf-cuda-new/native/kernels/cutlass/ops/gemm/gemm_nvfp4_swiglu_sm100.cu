// Copyright 2026 ApxInf contributors.
//
// CUDA/CUTLASS translation of the CuTe-DSL AOT operator
// the NVFP4 GEMM + SwiGLU + requantization operator.
//
//   c[M, I] (NVFP4) + c_sf = requant( SwiGLU( a[M,K] @ b[2I,K]^T ) )
//
// Accepted at 100.000000% bit-equality against both AOT oracles
// (`down_input_scale` 0.02 and 0.004 -- 35,651,584/35,651,584 e2m1 nibbles and
// 2,228,224/2,228,224 e4m3 scales on each). The configuration below is the one
// that was accepted ("tactic 19"): MMA tile 256x256x256, cluster 2x1, 4
// mainloop stages, `max_swizzle_size = 4`, heuristic raster order, reversed
// TMEM reuse and the `uint4`-vectorised sub-byte store. It is pinned here
// rather than left to a tuner on purpose -- the tuner would have to re-derive
// a bit-exactness property, not just a latency.
//
#include <cuda_runtime.h>

#include <mutex>

#include <cstdint>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

#include "gemm_nvfp4_swiglu_epilogue.cuh"

namespace apxinf::cuda_new::cutlass_ops {
namespace nvfp4_swiglu {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;     // A is [M, K], K-contiguous
using LayoutB = cutlass::layout::ColumnMajor;  // B is [N, K], K-contiguous
using LayoutD = cutlass::layout::RowMajor;
constexpr int kAlignAB = 32;

// Runtime scalars + output pointers the epilogue needs.
struct EpilogueArgs {
  uint8_t* ptr_c = nullptr;      // [M, I/2] packed e2m1 pairs
  uint8_t* ptr_c_sf = nullptr;   // tcgen05 atom-layout e4m3 scales
  const float* ptr_alpha = nullptr;
  const float* ptr_input_global_scale = nullptr;
  const float* ptr_down_scale = nullptr;
  int interm = 0;                // I, output width
};

// c_sf atom layout (32,4,rest_m,4,rest_k,l):(16,4,512*rest_k,1,512,...).
// Printed from CUTLASS's Sm1xxBlkScaledConfig and cross-checked against the
// exporter's make_ordered_layout((32,4,m//128,4,N//64,l),(2,1,4,0,3,5)).
CUTLASS_DEVICE size_t sfc_offset(int row, int scale_col, int rest_k) {
  return size_t(row & 31) * 16 + size_t((row >> 5) & 3) * 4 +
         size_t(row >> 7) * (512u * unsigned(rest_k)) +
         size_t(scale_col & 3) + size_t(scale_col >> 2) * 512;
}

template <class MmaTile, class Cluster, int Stages = 0, int Mode = 0>
struct Config {
  using EpilogueTile = Shape<_128, _64>;

  // Seed builder: used only for its proven CopyOpT2R / stride types. Its
  // generated epilogue op is never invoked.
  using EpiSchedule = std::conditional_t<
      cute::size<0>(Cluster{}) == 2,
      cutlass::epilogue::NoSmemWarpSpecialized2Sm,
      cutlass::epilogue::NoSmemWarpSpecialized1Sm>;
  using SeedEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile,
          Cluster, EpilogueTile, float, float, void, LayoutD, 8,
          cutlass::bfloat16_t, LayoutD, 8, EpiSchedule,
          cutlass::epilogue::fusion::ScaledAcc<cutlass::bfloat16_t, float,
                                               float>>::CollectiveOp;

  class SwigluQuantEpilogue {
   public:
    using DispatchPolicy = cutlass::epilogue::Sm100NoSmem;
    using EpilogueTile = typename SeedEpilogue::EpilogueTile;
    using ElementC = void;
    using StrideC = typename SeedEpilogue::StrideC;
    using ElementD = cutlass::float_e2m1_t;
    using StrideD = typename SeedEpilogue::StrideD;
    using CopyOpT2R = typename SeedEpilogue::CopyOpT2R;
    using ThreadEpilogueOp = typename SeedEpilogue::ThreadEpilogueOp;
    using GmemTiledCopyC = void;
    using GmemTiledCopyD = void;

    static constexpr int ThreadCount = 128;
    static constexpr uint32_t TmaTransactionBytes = 0;

    struct SharedStorage {
      // No rendezvous needed: one thread owns one row and both the up and the
      // gate epilogue tile of that row. An empty struct is avoided because the
      // kernel takes its address.
      int unused;
    };

    using Arguments = EpilogueArgs;
    using Params = EpilogueArgs;

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const&,
                                                    Arguments const& args,
                                                    void*) {
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
      return l == 1 && n == 2 * args.interm && (m % 128) == 0 &&
             (args.interm % 128) == 0 && args.ptr_c != nullptr &&
             args.ptr_c_sf != nullptr && args.ptr_alpha != nullptr &&
             args.ptr_input_global_scale != nullptr &&
             args.ptr_down_scale != nullptr;
    }

    CUTLASS_DEVICE SwigluQuantEpilogue(Params const& params, SharedStorage&)
        : params_(params) {}

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
      static_assert(is_tmem<AccEngine>::value, "need TMEM accumulators");
      // ReuseTmem (= the mainloop's IsOverlappingAccum) is tolerated rather
      // than exploited: the two accumulator phases overlap in TMEM columns, and
      // releasing only after every epilogue tile has been read is always safe,
      // it just forfeits the AOT's reverse-subtile early release. Measured
      // below; see the report for the cost.

      auto [M, N_full, K, L] = problem_shape_mnkl;
      auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;

      Tensor tAcc = accumulators(make_coord(_, _), _0{}, _0{});
      Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});
      auto tiled_t2r = make_tmem_copy(CopyOpT2R{}, tAcc_epi(_, _, _0{}, _0{}));
      const int thread_idx = threadIdx.x % ThreadCount;
      auto thread_t2r = tiled_t2r.get_slice(thread_idx);
      Tensor tTR_tAcc = thread_t2r.partition_S(tAcc_epi);

      constexpr int kNumEpiN = CUTE_STATIC_V(size<4>(tTR_tAcc));
      static_assert(kNumEpiN % 2 == 0, "gated activation needs even epi_n");
      static_assert(CUTE_STATIC_V(size<3>(tTR_tAcc)) == 1,
                    "epilogue tile must span the full CTA M extent");

      // The T2R destination shape must come from partition_D, not from the
      // source partition, or the CopyAtom cannot vectorize into registers.
      Tensor cTile = make_identity_tensor(
          make_shape(get<0>(cta_tile_mnk), get<1>(cta_tile_mnk)));
      Tensor tTR_c = thread_t2r.partition_D(flat_divide(cTile, EpilogueTile{}));
      Tensor rUp = make_tensor<float>(shape(tTR_c(_, _, _, _0{}, _0{})));
      Tensor rGate = make_tensor<float>(shape(rUp));
      constexpr int kVals = CUTE_STATIC_V(size(rUp));
      static_assert(kVals % 16 == 0, "fragment must be whole scale blocks");

      // Measured with nvfp4_probe2: thread t owns row t of the epilogue tile
      // and register i maps to N offset i. Nothing here is inferred.
      const int row = int(get<0>(cta_tile_mnk)) * m_coord + thread_idx;
      const int interm = params_.interm;
      const int out_n_tile = int(shape_div(get<1>(cta_tile_mnk), _2{})) * n_coord;
      const int sf_rest_k = interm / (16 * 4);

      const float alpha =
          params_.ptr_alpha[0] * params_.ptr_input_global_scale[0];
      const float down_scale = params_.ptr_down_scale[0];
      const float norm_const = apx_rcp_approx(down_scale);

      // TMEM reuse (ReuseTmem == mainloop IsOverlappingAccum, 1 accumulator
      // stage): phase 0 lives in columns [0,256) and phase 1 in [192,448), so
      // they share exactly the last epilogue-tile column of phase 0. While
      // draining phase 0 the epilogue therefore walks the gate/up pairs in
      // REVERSE so the shared column is read first, then releases the
      // accumulator immediately -- the MMA for the next tile can start while
      // this epilogue still has a pair to quantize. Without this the MMA and
      // the epilogue serialize completely; measured cost was 1.25x -> 1.00x.
      constexpr int kPairs = kNumEpiN / 2;
      bool reverse_pairs = false;
      if constexpr (ReuseTmem) {
        static_assert(AccumulatorPipeline::Stages == 1,
                      "TMEM reuse requires a single accumulator stage");
        reverse_pairs = acc_pipe_consumer_state.phase() == 0;
      }
      CUTLASS_PRAGMA_NO_UNROLL
      for (int iter = 0; iter < kPairs; ++iter) {
        const int pair = reverse_pairs ? kPairs - 1 - iter : iter;
        copy(tiled_t2r, tTR_tAcc(_, _, _, _0{}, 2 * pair), rUp);
        copy(tiled_t2r, tTR_tAcc(_, _, _, _0{}, 2 * pair + 1), rGate);
        const bool release = ReuseTmem ? iter == 0 : iter == kPairs - 1;
        if (release) {
          cutlass::arch::fence_view_async_tmem_load();
          acc_pipeline.consumer_release(acc_pipe_consumer_state);
          ++acc_pipe_consumer_state;
        }
        if (row >= M) continue;

        float out[kVals];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kVals; i += 2) {
          // Mode 1 is a measurement-only variant: same TMEM traffic and same
          // stores, no transcendentals / divisions / amax. Used to separate
          // "epilogue ALU bound" from "mainloop pipelining bound".
          if constexpr (Mode == 1) {
            out[i] = rUp(i) + rGate(i);
            out[i + 1] = rUp(i + 1) + rGate(i + 1);
          } else {
            apx_swiglu_pair(rUp(i), rUp(i + 1), rGate(i), rGate(i + 1), alpha,
                            out[i], out[i + 1]);
          }
        }

        const int out_n = out_n_tile + pair * kVals;
        uint8_t* dst = params_.ptr_c + size_t(row) * (interm / 2) + out_n / 2;
        uint32_t sf_word = 0;
        // The 64 outputs are kVals/2 contiguous bytes of one output row, so
        // they are assembled in registers and written with 16-byte stores.
        // Writing them a nibble-pair at a time cost 32 single-byte STGs per
        // thread per epilogue tile and showed up as a 6% whole-kernel
        // regression (measured; see the report).
        static_assert(kVals % 32 == 0, "need whole 16-byte output vectors");
        uint32_t words[kVals / 8];
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kVals / 16; ++v) {
          float s = 1.0f;
          if constexpr (Mode != 1) {
            float amax = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < 16; ++e) {
              amax = apx_fmax_nan(amax, fabsf(out[v * 16 + e]));
            }
            const uint8_t sfc =
                apx_f32_to_e4m3(norm_const * (amax * 0.16666667163372039795f));
            sf_word |= uint32_t(sfc) << (8 * v);
            const float eff = apx_e4m3_to_f32(sfc) * down_scale;
            s = 0.0f;
            if (eff > 0.0f) s = apx_div_rn(1.0f, eff);
            s = apx_fmin_nan(s, 3.40282346638528859812e38f);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int w = 0; w < 2; ++w) {
            uint32_t acc = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) {
              const int e = w * 8 + j * 2;
              acc |= uint32_t(apx_f32x2_to_e2m1x2(out[v * 16 + e] * s,
                                                  out[v * 16 + e + 1] * s))
                     << (8 * j);
            }
            words[v * 2 + w] = acc;
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int q = 0; q < kVals / 32; ++q) {
          reinterpret_cast<uint4*>(dst)[q] =
              make_uint4(words[q * 4], words[q * 4 + 1], words[q * 4 + 2],
                         words[q * 4 + 3]);
        }
        *reinterpret_cast<uint32_t*>(
            params_.ptr_c_sf + sfc_offset(row, out_n / 16, sf_rest_k)) = sf_word;
      }
      return make_tuple(acc_pipe_consumer_state);
    }

   private:
    Params params_;
  };

  using CollectiveEpilogue =
      cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          SwigluQuantEpilogue>;
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
          ElementA, LayoutA, kAlignAB, ElementB, LayoutB, kAlignAB, ElementAcc,
          MmaTile, Cluster,
          std::conditional_t<
              Stages == 0,
              cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                  sizeof(typename CollectiveEpilogue::TensorStorage))>,
              cutlass::gemm::collective::StageCount<Stages == 0 ? 1 : Stages>>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// The accepted configuration, and only it. The tuning sweep that produced it
// lived in the scratch harness (33 tactics); keeping the losers here would
// invite a tuner to pick one, and the property being protected is
// bit-exactness against the AOT oracle, not latency.
using Mma256 = Shape<_256, _256, _256>;
using C21 = Shape<_2, _1, _1>;
using T256s4 = Config<Mma256, C21, 4>;

using Accepted = T256s4;
constexpr int kRasterHeuristic = 0;
constexpr int kMaxSwizzleSize = 4;

// `initialize()` does host-side parameter setup and clears the tile-scheduler
// workspace on the stream. The pointers are captured by `initialize`, so it
// has to run per call; the workspace clear is the price. This matches
// `gemm_nvfp4_sm100.cu`'s non-prepared launch path.
int launch(const void* a, const void* a_sf, const void* b, const void* b_sf,
           void* c, void* c_sf, const void* alpha,
           const void* input_global_scale, const void* down_scale, int m,
           int interm, int k, void* workspace, size_t workspace_bytes,
           cudaStream_t stream) {
  using Kernel = typename Accepted::Kernel;
  using Device = typename Accepted::Device;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using SfCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename ElementA::ScaleFactorType;

  const int n = 2 * interm;
  auto problem = cute::make_shape(m, n, k, 1);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});

  typename Kernel::MainloopArguments mainloop{
      reinterpret_cast<typename ElementA::DataType const*>(a), stride_a,
      reinterpret_cast<typename ElementB::DataType const*>(b), stride_b,
      reinterpret_cast<ElementSF const*>(a_sf),
      SfCfg::tile_atom_to_shape_SFA(problem),
      reinterpret_cast<ElementSF const*>(b_sf),
      SfCfg::tile_atom_to_shape_SFB(problem)};
  EpilogueArgs epilogue{static_cast<uint8_t*>(c),
                        static_cast<uint8_t*>(c_sf),
                        static_cast<const float*>(alpha),
                        static_cast<const float*>(input_global_scale),
                        static_cast<const float*>(down_scale),
                        interm};
  cutlass::KernelHardwareInfo hw;
  if (cudaDeviceGetAttribute(&hw.sm_count, cudaDevAttrMultiProcessorCount, 0) !=
      cudaSuccess) {
    return -8;
  }
  using RasterOrder = cutlass::gemm::kernel::detail::RasterOrderOptions;
  typename Kernel::TileSchedulerArguments sched{};
  sched.max_swizzle_size = kMaxSwizzleSize;
  sched.raster_order = RasterOrder::Heuristic;
  static_assert(kRasterHeuristic == 0, "accepted raster order is Heuristic");
  typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {m, n, k, 1},
                                  mainloop,
                                  epilogue,
                                  hw,
                                  sched};
  Device op;
  if (op.can_implement(args) != cutlass::Status::kSuccess) return -1;
  if (Device::get_workspace_size(args) > workspace_bytes) return -4;
  if (op.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
    return -2;
  }
  return op.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}

size_t workspace_bytes_for(int m, int interm, int k) {
  using Kernel = typename Accepted::Kernel;
  using Device = typename Accepted::Device;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using SfCfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename ElementA::ScaleFactorType;
  const int n = 2 * interm;
  auto problem = cute::make_shape(m, n, k, 1);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  typename Kernel::MainloopArguments mainloop{
      nullptr, stride_a, nullptr, stride_b, nullptr,
      SfCfg::tile_atom_to_shape_SFA(problem), nullptr,
      SfCfg::tile_atom_to_shape_SFB(problem)};
  EpilogueArgs epilogue{};
  epilogue.interm = interm;
  cutlass::KernelHardwareInfo hw;
  typename Kernel::TileSchedulerArguments sched{};
  sched.max_swizzle_size = kMaxSwizzleSize;
  typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                  {m, n, k, 1},
                                  mainloop,
                                  epilogue,
                                  hw,
                                  sched};
  return Device::get_workspace_size(args);
}

}  // namespace nvfp4_swiglu

size_t nvfp4_gemm_swiglu_workspace_bytes(int rows, int n, int k) {
  return nvfp4_swiglu::workspace_bytes_for(rows, n, k);
}

namespace nvfp4_swiglu {
namespace {

// The tile-scheduler scratch depends only on the problem shape, which is fixed
// for the production FC1, so it is allocated once on the first call and reused.
// That first call must happen outside a CUDA graph capture -- which it does,
// because the model warms every operator up before it captures. Keyed on the
// shape so a second shape cannot silently reuse a too-small buffer.
struct Workspace {
  void* ptr = nullptr;
  size_t bytes = 0;
  int rows = 0, n = 0, k = 0;
};
Workspace g_workspace;
std::mutex g_workspace_mutex;

// Returns false only on an allocation failure.
bool reserve(int rows, int n, int k, void** ptr, size_t* bytes) {
  std::lock_guard<std::mutex> guard(g_workspace_mutex);
  if (g_workspace.rows != rows || g_workspace.n != n || g_workspace.k != k) {
    const size_t needed = workspace_bytes_for(rows, n, k);
    if (needed > g_workspace.bytes) {
      void* fresh = nullptr;
      if (needed != 0 && cudaMalloc(&fresh, needed) != cudaSuccess) return false;
      if (g_workspace.ptr != nullptr) cudaFree(g_workspace.ptr);
      g_workspace.ptr = fresh;
      g_workspace.bytes = needed;
    }
    g_workspace.rows = rows;
    g_workspace.n = n;
    g_workspace.k = k;
  }
  *ptr = g_workspace.ptr;
  *bytes = g_workspace.bytes;
  return true;
}

}  // namespace
}  // namespace nvfp4_swiglu

int nvfp4_gemm_swiglu(
    const void* a, const void* b, const void* a_sf, const void* b_sf,
    void* c, void* c_sf, const void* alpha, const void* input_global_scale,
    const void* down_inverse_global_scale, void* workspace,
    size_t workspace_bytes, int rows, int n, int k, cudaStream_t stream) {
  if (rows <= 0 || n <= 0 || k <= 0) return -10;
  if (n % 128 != 0 || k % 128 != 0) return -10;
  if (workspace == nullptr && workspace_bytes == 0) {
    if (!nvfp4_swiglu::reserve(rows, n, k, &workspace, &workspace_bytes)) {
      return -5;
    }
  }
  return nvfp4_swiglu::launch(a, a_sf, b, b_sf, c, c_sf, alpha,
                              input_global_scale, down_inverse_global_scale,
                              rows, n, k, workspace, workspace_bytes, stream);
}

}  // namespace apxinf::cuda_new::cutlass_ops

// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Shared body of the two head_dim-256 single-Q-stage FMHA translation units
// (`fmha_d256_language_sm100.cu` and `fmha_d256_action_sm100.cu`).
//
// It is a HEADER rather than a translation unit on purpose. The two operators
// share every line of code and differ only in the `-DAPXINF_*` flag line they
// are compiled with -- language follows
// `sm100_hd256_2cta_fmha_forward.py` (ASCENDING KV, mask last, no rescale
// threshold, ex2 emulation 14/6/0, `div.rn` normalisation with 0/NaN -> 0.0,
// row_sum over contiguous quarters seeded (s,s) with `acc_scale *= 0.5`) while
// action follows `flash_fwd_sm100.py` (DESCENDING KV, mask first,
// `rescale_threshold = 8.0`, no ex2 emulation, `rcp.approx` normalisation,
// row_sum stride-8 seeded into lane 0 only). Applying action's contract to
// language pins it at 52.87% bit-equal, so the two flag lines are the
// operators' numeric contracts and `devlocal/aot2cuda/harness/MERGED_FLAGS.sh`
// is authoritative for them.
//
// Because those flags configure the vendored FMHA headers WITHOUT taking part
// in name mangling, two identically parameterised instantiations compiled from
// two differently flagged translation units would be a silent ODR violation:
// the linker would keep one body and both operators would run it. Every FMHA
// template reachable from here is therefore parameterised by a type from the
// per-operator namespace `APXINF_D256_NS`, which makes the mangled names
// disjoint. `tests/cuda_arch.rs` asserts the object symbol sets do not
// intersect.
//
// Why a single Q stage at all: the stock collective hardwires
// `ThreadShape<_2,_1,_1>` (two Q stages, two softmax warpgroups, two O
// accumulators). At D = 256 that needs
//   tmem = 2*N + 2*D = 768 columns  (budget 512)
//   smem = (2*M + 3*N) * D * 2 B    = 320.2 KiB (budget 227.0)
// FA4's hd256 kernel uses one Q stage instead:
//   tmem = N + D = 384 columns
//   smem = (M + 2*N) * D * 2 B + barriers = 194.0 KiB
// which is what this instantiates (`ThreadShape<_1,_1,_1>`, 2 KV stages).
//
// FlashAttention / CUTLASS provenance and licences: kernels/cutlass/licenses.

#ifndef APXINF_D256_NS
#error "include fmha_d256_language_sm100.cu / fmha_d256_action_sm100.cu, not this header"
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Must be defined before the mainloop header: K and V alternate in one
// circular buffer, so 2 stages == one K slot + one V slot == 128 KiB at D=256.
#ifndef APXINF_FMHA_KV_STAGES
#define APXINF_FMHA_KV_STAGES 2
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"

#if APXINF_D256_BUILD_ACTION
// apxinf pack-GQA (action geometry): the stride type and the TMA dummy-shape
// specialisation have to be visible BEFORE the SM100 GEMM collective is
// instantiated, hence the early placement.
#ifndef APXINF_PACK_GQA_QPK
#define APXINF_PACK_GQA_QPK 4
#endif
#include "cutlass/gemm/collective/apxinf_tma_dummy_shape.hpp"

namespace APXINF_D256_NS {
using QPerKv = cute::Int<APXINF_PACK_GQA_QPK>;
// M mode is the HIERARCHICAL coordinate (q_per_kv, seq_q) with strides
// (head_dim, q_token_stride); the head mode keeps the nesting with q_per_kv
// collapsed to a static _1 so TMA still sees 5 dims, not 6.
using StrideQPack = cute::tuple<cute::tuple<int, int>, cute::_1,
                                cute::tuple<cute::tuple<cute::_0, int>, int>>;
using StrideOPack = StrideQPack;
}  // namespace APXINF_D256_NS

namespace cutlass::gemm::collective::detail {
template <>
struct ApxinfTmaDummyShape<APXINF_D256_NS::StrideQPack> {
  CUTE_HOST_DEVICE static constexpr auto get() {
    return cute::make_shape(
        cute::make_shape(APXINF_D256_NS::QPerKv{}, int32_t(0)), int32_t(0),
        cute::make_shape(cute::make_shape(cute::_1{}, int32_t(0)), int32_t(0)));
  }
};
}  // namespace cutlass::gemm::collective::detail
#endif  // APXINF_D256_BUILD_ACTION

// The upstream example guard recognizes SM100 and SM103. CUTLASS enables the
// same kernel path for SM101 and SM110, so alias those architecture markers
// only while parsing the kernel header.
#if (defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) ||  \
     defined(CUTLASS_ARCH_MMA_SM110A_ENABLED)) && \
    !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
#define APXINF_FMHA_UNDEF_SM100A 1
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif
#if (defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) ||  \
     defined(CUTLASS_ARCH_MMA_SM110F_ENABLED)) && \
    !defined(CUTLASS_ARCH_MMA_SM100F_ENABLED)
#define APXINF_FMHA_UNDEF_SM100F 1
#define CUTLASS_ARCH_MMA_SM100F_ENABLED 1
#endif
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#if defined(APXINF_FMHA_UNDEF_SM100A)
#undef CUTLASS_ARCH_MMA_SM100A_ENABLED
#undef APXINF_FMHA_UNDEF_SM100A
#endif
#if defined(APXINF_FMHA_UNDEF_SM100F)
#undef CUTLASS_ARCH_MMA_SM100F_ENABLED
#undef APXINF_FMHA_UNDEF_SM100F
#endif
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "device/fmha.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_load_tma_warpspecialized.hpp"
#include "collective/fmha_fusion.hpp"

namespace APXINF_D256_NS {

using namespace cute;

#if APXINF_D256_BUILD_LANGUAGE
using TileShape = Shape<_256, _128, _256>;
using ClusterShape = Shape<_2, _1, _1>;
#else
using TileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;
#endif
using ThreadShape = Shape<_1, _1, _1>;  // single Q stage

using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using ProblemShape =
    cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

using Element = cutlass::bfloat16_t;

#if APXINF_D256_BUILD_LANGUAGE
// FA4's hd256 kernel is a 2-CTA kernel: `qk_mma_tiler[0] = 2 * 128`, so the
// causal trip count is computed on the 256-row MMA tile shared by the CTA pair,
// not on each CTA's own 128 rows. The lower CTA of every pair therefore runs
// one extra, entirely-masked KV tile, and `use_semantic_trip_range` masks both
// trailing tiles. That changes nothing mathematically but it does change the
// online-softmax trajectory's rounding, so reproduce it exactly.
struct CausalMask2Cta : cutlass::fmha::collective::CausalMask<true> {
  using Base = cutlass::fmha::collective::CausalMask<true>;

  template <class BlkCoord, class TileShape_, class ProblemSize>
  CUTLASS_DEVICE int get_trip_count(BlkCoord const& blk_coord,
                                    TileShape_ const& tile_shape,
                                    ProblemSize const& problem_size) {
    int max_blocks_k =
        cute::ceil_div(cute::get<1>(problem_size), cute::size<1>(tile_shape));
    int max_blocks_q = cute::ceil_div((cute::get<0>(blk_coord) + 1) * cute::size<0>(tile_shape),
                                      cute::size<1>(tile_shape));
    return max_blocks_k < max_blocks_q ? max_blocks_k : max_blocks_q;
  }

  template <class BlkCoord, class TileShape_, class ProblemSize>
  CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const& blk_coord,
                                          TileShape_ const& tile_shape,
                                          ProblemSize const& problem_size) {
    int trip = get_trip_count(blk_coord, tile_shape, problem_size);
    // Both trailing 128-column tiles of the 256-row cluster tile carry the
    // causal mask.
    int masked = cute::ceil_div(cute::size<0>(tile_shape),
                                cute::size<1>(tile_shape));
    return trip < masked ? trip : masked;
  }

  template <class BlkCoord, class TileShape_, class ProblemSize>
  CUTLASS_DEVICE int get_unmasked_trip_count(BlkCoord const& blk_coord,
                                            TileShape_ const& tile_shape,
                                            ProblemSize const& problem_size) {
    return get_trip_count(blk_coord, tile_shape, problem_size) -
           get_masked_trip_count(blk_coord, tile_shape, problem_size);
  }
};

// One logical 256-row tile is executed by a two-CTA cluster. Both CTAs must
// receive the same logical coordinate, while the physical grid still has two
// blocks per tile so CUDA can form clusterDim=(2,1,1).
struct ReverseCluster2TileScheduler {
  struct Params {
    dim3 grid;
    int num_m_tiles;
  };

  bool valid_ = true;
  int num_m_tiles_ = 1;

  CUTLASS_DEVICE explicit ReverseCluster2TileScheduler(Params const& params)
      : num_m_tiles_(params.num_m_tiles) {}

  template <class ProblemSize, class ClusterShape_, class TileShape_>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size, cutlass::KernelHardwareInfo,
      ClusterShape_ const& cluster_shape, TileShape_ const& tile_shape) {
    int tiles = int(cute::ceil_div(cute::size<0>(problem_size),
                                   cute::size<0>(tile_shape)));
    int cluster_m = int(cute::size<0>(cluster_shape));
    return Params{dim3(unsigned(tiles * cluster_m),
                       unsigned(cute::size<3,0>(problem_size)),
                       unsigned(cute::size<3,1>(problem_size))), tiles};
  }

  static dim3 get_grid_shape(Params const& params) { return params.grid; }
  CUTLASS_DEVICE bool is_valid() { return valid_; }
  CUTLASS_DEVICE auto get_block_coord() {
    int cluster_m = int(cute::size<0>(ClusterShape{}));
    int tile = int(blockIdx.x) / cluster_m;
    return cute::make_coord(num_m_tiles_ - 1 - tile, cute::_0{},
                            cute::make_coord(blockIdx.y, blockIdx.z));
  }
  CUTLASS_DEVICE ReverseCluster2TileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};
#else
// apxinf pack-GQA: seq_k is `VariableLength` because the action AOT is ONE
// launch over a split-batch view with `seqused_k = {1718, key_tokens - 1718}`.
// Two launches of 8 CTAs cost 2 x 50 us (measured, ncu) because 8 CTAs on 20
// SMs is 0.4 waves -- the critical path is the 14-tile KV loop, not the CTA
// count. One launch of 16 CTAs (0.8 waves) is exactly the AOT grid (2,4,2).
using ProblemShapePack =
    cute::tuple<cute::tuple<QPerKv, int>,
                cutlass::fmha::collective::VariableLength, int,
                cute::tuple<cute::tuple<_1, int>, int>>;

// See the ODR note at the top of this header: `ResidualMask` itself carries no
// per-operator type, and the action instantiation must not share a mangled
// name with any other translation unit's `ResidualMask` mainloop.
struct ActionMask : cutlass::fmha::collective::ResidualMask {};
#endif

template <class Mask, class PSh, class SQ, class SO, class Scheduler>
struct ConfigT {
  using Mainloop =
      cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
          Element, float, float, TileShape, SQ, StrideK, StrideV, Mask,
          ThreadShape, ClusterShape>;
  using Epilogue =
      cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
          Element, float, typename Mainloop::CtaShapePV, SO, StrideLSE,
          cute::false_type, /*NumAccStages=*/1>;
  using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      PSh, Mainloop, Epilogue,
      Scheduler,
      cutlass::fmha::kernel::Sm100FmhaCtxKernelWarpspecialized1QSchedule>;
  using Op = cutlass::fmha::device::FMHA<Kernel>;
};

#if APXINF_D256_BUILD_LANGUAGE
using CausalCfg = ConfigT<CausalMask2Cta, ProblemShape, StrideQ, StrideO,
                          ReverseCluster2TileScheduler>;
#else
using ResidualPackCfg =
    ConfigT<ActionMask, ProblemShapePack, StrideQPack, StrideOPack,
            cutlass::fmha::kernel::ReverseIndividualTileScheduler>;
#endif

struct State {
  void* workspace = nullptr;
  size_t workspace_size = 0;
  float* lse = nullptr;
  size_t lse_size = 0;
  int device = -1;
};

// `cudaMalloc` on demand, exactly once per process+device. Both arms do this
// behind `workspace::may_prepare_native_resources()`, so no allocation can
// happen inside a graph capture.
template <class Cfg, class Arguments>
int finish(Arguments& arguments, State& state, cudaStream_t stream,
           int prepare_only) {
  typename Cfg::Op operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  const size_t required_workspace = Cfg::Op::get_workspace_size(arguments);
  if (required_workspace > state.workspace_size) {
    if (!prepare_only) return -4;
    if (state.workspace) cudaFree(state.workspace);
    state.workspace = nullptr;
    state.workspace_size = 0;
    if (required_workspace != 0 &&
        cudaMalloc(&state.workspace, required_workspace) != cudaSuccess) {
      return -4;
    }
    state.workspace_size = required_workspace;
  }
  if (prepare_only) return 0;
  if (operation.initialize(arguments, state.workspace, stream) !=
      cutlass::Status::kSuccess) {
    return -2;
  }
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}

int device_properties(int& device, int& multiprocessors) {
  device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return -4;
  if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess) {
    return -4;
  }
  return 0;
}

}  // namespace APXINF_D256_NS

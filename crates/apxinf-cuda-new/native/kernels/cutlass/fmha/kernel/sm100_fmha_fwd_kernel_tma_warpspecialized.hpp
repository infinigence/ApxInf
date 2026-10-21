/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cute/layout.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/arch/tmem_allocator_sm100.hpp"

#include "kernel/fmha_options.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/fmha_causal_tile_scheduler.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/fmha_common.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;
using namespace cutlass::fmha::collective;

// ---------------------------------------------------------------------------
// apxinf: macro vocabulary for this header.  EVERY DEFAULT IS UPSTREAM; with no
// -DAPXINF_* on the command line the generated code is numerically identical to
// pristine CUTLASS (proven by run, see
// devlocal/aot2cuda/reports/22-fmha-kernel-epilogue-reconciliation.md).
//
//   APXINF_NUM_REGS_SOFTMAX     192  per-warpgroup register budget of the
//   APXINF_NUM_REGS_CORRECTION   96  stock 16-warp (two-Q-stage) schedule
//   APXINF_NUM_REGS_OTHER        32  `Sm100FmhaCtxKernelWarpspecializedSchedule`
//
//   APXINF_1Q_NUM_REGS_SOFTMAX     256  budgets of the 12-warp single-Q-stage
//   APXINF_1Q_NUM_REGS_CORRECTION  160  schedule
//   APXINF_1Q_NUM_REGS_OTHER        32  `Sm100FmhaCtxKernelWarpspecialized1QSchedule`
//                                       (opt-in: a consumer must name the type)
//
// NOTE the two groups are SEPARATE knobs.  The 1-Q-stage fork spelled its
// budgets `APXINF_REGS_*` while the vision fork spelled the 16-warp ones
// `APXINF_NUM_REGS_*`; those names differ by four characters and mean different
// warp counts, so the old short spelling is a hard error below.
// `APXINF_NO_LSE` and `APXINF_PERSISTENT` are NOT read here -- they are
// consumer-side knobs (ptr_LSE = nullptr, and which TileScheduler to name);
// report 20 §7.1 listed them as kernel-header knobs, which was wrong.
//
// The 16-warp budget is NON-MONOTONE and its optimum has now flipped twice, so
// it must be re-swept whenever the compile regime changes.  Report 30, vision
// hd64 through the repo path (16 rounds, ABBA, control band [1.0164, 1.0190]
// over six replicates of the shipped 200/64/48):
//     200/64/48  controls 1.0183 / 1.0187 / 1.0188 / 1.0190   <- shipped
//     192/96/32  1.0212   (CUTLASS default; the vision fork's post-NDEBUG
//                          optimum on its standalone L=624 driver)
//     184/112/32 1.0199
//     216/56/24  1.0248   (the vision fork's pre-NDEBUG optimum)
// ptxas STACK does not predict any of it: 200/64/48 spills 40 B, 192/96/32
// spills 8 B and is slower, 216/56/24 spills 88 B and is slowest.
// Constraint: 2*SOFTMAX + CORRECTION + OTHER <= 512.
// ---------------------------------------------------------------------------
#if defined(APXINF_REGS_SOFTMAX) || defined(APXINF_REGS_CORRECTION) || defined(APXINF_REGS_OTHER)
#  error "APXINF_REGS_* is gone; use APXINF_1Q_NUM_REGS_* for the 12-warp single-Q-stage schedule, or APXINF_NUM_REGS_* for the stock 16-warp one"
#endif

#ifndef APXINF_NUM_REGS_SOFTMAX
#define APXINF_NUM_REGS_SOFTMAX 192
#endif
#ifndef APXINF_NUM_REGS_CORRECTION
#define APXINF_NUM_REGS_CORRECTION 96
#endif
#ifndef APXINF_NUM_REGS_OTHER
#define APXINF_NUM_REGS_OTHER 32
#endif

#ifndef APXINF_1Q_NUM_REGS_SOFTMAX
#define APXINF_1Q_NUM_REGS_SOFTMAX 256
#endif
#ifndef APXINF_1Q_NUM_REGS_CORRECTION
#define APXINF_1Q_NUM_REGS_CORRECTION 160
#endif
#ifndef APXINF_1Q_NUM_REGS_OTHER
#define APXINF_1Q_NUM_REGS_OTHER 32
#endif

namespace apxinf_detail {

// `CollectiveMainloop::kSingleQStage` only exists on the reconciled forward
// mainloop; the MLA mainloop does not define it.  Detect instead of requiring.
template<class T, class = void>
struct SingleQStage { static constexpr bool value = false; };
template<class T>
struct SingleQStage<T, cute::void_t<decltype(T::kSingleQStage)>> {
  static constexpr bool value = T::kSingleQStage;
};

// Number of rows in mode 0 of the logical problem shape.  Upstream writes
// `get<0>(logical_problem_shape)`, which is an `int` for every flat problem
// shape; pack-GQA makes mode 0 the HIERARCHICAL coordinate (q_per_kv, seq_q),
// for which the row count is `size<0>`, not `get<0>`.  The integral branch is
// textually upstream so the stock path cannot drift.
template<class Shape>
CUTLASS_HOST_DEVICE constexpr int logical_seqlen_q(Shape const& s) {
  if constexpr (cute::is_integral<cute::remove_cvref_t<decltype(cute::get<0>(s))>>::value) {
    return cute::get<0>(s);
  } else {
    return int(cute::size<0>(s));
  }
}

// Guard the mainloop/epilogue pairing only when the epilogue actually declares
// `NumAccStages` (the gen/MLA epilogues do not); otherwise stay out of the way.
template<class T, int Want, class = void>
struct AccStagesMatch { static constexpr bool value = true; };
template<class T, int Want>
struct AccStagesMatch<T, Want, cute::void_t<decltype(T::NumAccStages)>> {
  static constexpr bool value = (T::NumAccStages == Want);
};

}  // namespace apxinf_detail

struct Sm100FmhaCtxKernelWarpspecializedSchedule {

  enum class WarpRole {
    Softmax0,
    Softmax1,
    Correction,
    MMA,
    Load,
    Epilogue,
    Empty
  };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    int wg_idx = warp_idx / 4;                        // warp_idx
    if (wg_idx == 0) return WarpRole::Softmax0;       //   0 -  3
    if (wg_idx == 1) return WarpRole::Softmax1;       //   4 -  7
    if (wg_idx == 2) return WarpRole::Correction;     //   8 - 11
    if (warp_idx == 12) return WarpRole::MMA;         //       12
    if (warp_idx == 13) return WarpRole::Load;        //       13
    if (warp_idx == 14) return WarpRole::Epilogue;    //       14
    return WarpRole::Empty;                           //       15
  }

  static const int NumWarpsSoftmax = 4;
  static const int NumWarpsCorrection = 4;
  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  static const bool kDebugUsingPrintf = false;
  static const int NumRegsSoftmax = APXINF_NUM_REGS_SOFTMAX;
  static const int NumRegsCorrection = APXINF_NUM_REGS_CORRECTION - (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsOther = APXINF_NUM_REGS_OTHER + (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsEmpty = 24;

  static const int NumWarps = 16;

};


struct Sm100MlaFwdCtxKernelWarpspecializedSchedule {

  enum class WarpRole {
    Softmax0,
    Softmax1,
    Correction,
    MMA,
    Load,
    Epilogue,
    Empty
  };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    int wg_idx = warp_idx / 4;                        // warp_idx
    if (wg_idx == 0) return WarpRole::Softmax0;       //   0 -  3
    if (wg_idx == 1) return WarpRole::Softmax1;       //   4 -  7
    if (wg_idx == 2) return WarpRole::Correction;     //   8 - 11
    if (warp_idx == 12) return WarpRole::MMA;         //       12
    if (warp_idx == 13) return WarpRole::Load;        //       13
    if (warp_idx == 14) return WarpRole::Epilogue;    //       14
    return WarpRole::Empty;                           //       15
  }

  static const int NumWarpsSoftmax = 4;
  static const int NumWarpsCorrection = 4;
  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  static const bool kDebugUsingPrintf = false;
  static const int NumRegsSoftmax = 184;
  static const int NumRegsCorrection = 96 - (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsOther = 48 + (kDebugUsingPrintf ? 16 : 0);
  static const int NumRegsEmpty = 24;

  static const int NumWarps = 16;

};

// apxinf: schedule for the single-Q-stage (head_dim 256) mainloop, selected by
// naming this type as `KernelSchedule`.  Purely additive -- nothing below reads
// it unless a consumer asks for it.
//
// Only one softmax warpgroup exists on that path, so the 4 Softmax1 warps of
// the stock 16-warp schedule are pure overhead: they still consume the CTA's
// register pool.  12 warps leaves 65536/384 = 170 regs/thread for ptxas to hand
// out, which is what lets the softmax warpgroup take FA4's 256
// (`num_regs_softmax`) without spilling -- at 16 warps ptxas caps at 128 and
// the softmax warp spilled 67 STL / 63 LDL.
struct Sm100FmhaCtxKernelWarpspecialized1QSchedule {

  enum class WarpRole {
    Softmax0,
    Softmax1,
    Correction,
    MMA,
    Load,
    Epilogue,
    Empty
  };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    int wg_idx = warp_idx / 4;                        // warp_idx
    if (wg_idx == 0) return WarpRole::Softmax0;       //   0 -  3
    if (wg_idx == 1) return WarpRole::Correction;     //   4 -  7
    if (warp_idx == 8) return WarpRole::MMA;          //       8
    if (warp_idx == 9) return WarpRole::Load;         //       9
    if (warp_idx == 10) return WarpRole::Epilogue;    //      10
    return WarpRole::Empty;                           //      11
  }

  static const int NumWarpsSoftmax = 4;
  static const int NumWarpsCorrection = 4;
  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  static const bool kDebugUsingPrintf = false;
  // FA4 hd256 defaults: num_regs_softmax = 256, num_regs_correction = 160,
  // num_regs_other = 32.
  static const int NumRegsSoftmax = APXINF_1Q_NUM_REGS_SOFTMAX;
  static const int NumRegsCorrection = APXINF_1Q_NUM_REGS_CORRECTION;
  static const int NumRegsOther = APXINF_1Q_NUM_REGS_OTHER;
  static const int NumRegsEmpty = 24;

  static const int NumWarps = 12;
};

template<
  class ProblemShapeIn,
  class CollectiveMainloop,
  class CollectiveEpilogue,
  class TileScheduler,
  class KernelSchedule = Sm100FmhaCtxKernelWarpspecializedSchedule
>
struct Sm100FmhaFwdKernelTmaWarpspecialized {

  using TileShape = typename CollectiveMainloop::TileShape;
  using ProblemShape = ProblemShapeIn;

  using WarpRole = typename KernelSchedule::WarpRole;

  constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    return KernelSchedule::warp_idx_to_WarpRole(warp_idx);
  }

  static const int NumWarpsSoftmax = KernelSchedule::NumWarpsSoftmax;
  static const int NumWarpsCorrection = KernelSchedule::NumWarpsCorrection;
  static const int NumWarpsEpilogue = KernelSchedule::NumWarpsEpilogue;
  static const int NumWarpsLoad = KernelSchedule::NumWarpsLoad;
  
  static_assert(NumWarpsEpilogue == CollectiveEpilogue::NumWarpsEpilogue);
  static_assert(NumWarpsLoad == CollectiveEpilogue::NumWarpsLoad);

  static const int NumRegsSoftmax = KernelSchedule::NumRegsSoftmax;
  static const int NumRegsCorrection = KernelSchedule::NumRegsCorrection;
  static const int NumRegsOther = KernelSchedule::NumRegsOther;
  static const int NumRegsEmpty = 24;

  static const int NumWarps = KernelSchedule::NumWarps;

  static constexpr bool IsMla = std::is_same_v<KernelSchedule, Sm100MlaFwdCtxKernelWarpspecializedSchedule>;

  // apxinf: the single-Q-stage mainloop (head_dim 256) has only one softmax
  // warpgroup; the Softmax1 warps idle and donate their registers.  `false`
  // for every two-Q-stage mainloop and for mainloops that never heard of the
  // flag (MLA), so this is inert on the stock path.
  static constexpr bool kSingleQStage = apxinf_detail::SingleQStage<CollectiveMainloop>::value;

  // apxinf: a single-Q-stage mainloop produces ONE O sub-tile per CTA, so the
  // epilogue must be instantiated with NumAccStages = 1.  Pairing them the
  // other way round silently stores the wrong half of O (and over-runs the
  // epilogue's smem), so assert instead of trusting the consumer.
  static_assert(apxinf_detail::AccStagesMatch<CollectiveEpilogue, kSingleQStage ? 1 : 2>::value,
                "CollectiveEpilogue's NumAccStages must match the mainloop's Q stage count");

  // apxinf: a single-Q-stage mainloop has no Softmax1 work, so it may only be
  // driven by a KernelSchedule whose warp_idx_to_WarpRole() never returns
  // Softmax1.  `Sm100FmhaCtxKernelWarpspecialized1QSchedule` is the only such
  // schedule in this header.
  static_assert(! kSingleQStage ||
                    std::is_same_v<KernelSchedule, Sm100FmhaCtxKernelWarpspecialized1QSchedule>,
                "a single-Q-stage mainloop needs Sm100FmhaCtxKernelWarpspecialized1QSchedule");

  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  static constexpr bool kIs2Sm = cute::size(ClusterShape{}) == 2;

  using TmemAllocator = cute::conditional_t<
      kIs2Sm, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;

  struct SharedStorage {
    using UnionType = union {
      typename CollectiveMainloop::TensorStorage mainloop;
      typename CollectiveEpilogue::TensorStorage epilogue;
    };

    using  StructType = struct {
      typename CollectiveMainloop::TensorStorage mainloop;
      typename CollectiveEpilogue::TensorStorage epilogue;
    };

    static constexpr bool IsPersistent = std::is_same_v<TileScheduler, PersistentTileScheduler> || std::is_same_v<TileScheduler, CausalPersistentTileScheduler>;
    using MainloopEpilogueStorage = std::conditional_t<IsPersistent, 
                                                       std::conditional_t<IsMla, 
                                                                          std::conditional_t<CollectiveMainloop::IsOrderLoadEpilogue, UnionType, StructType>,
                                                                          StructType>,
                                                       UnionType>;

    MainloopEpilogueStorage mainloop_epilogue; 

    struct PipelineStorage {
      alignas(16) typename CollectiveMainloop::PipelineQ::SharedStorage load_q;
      alignas(16) typename CollectiveMainloop::PipelineKV::SharedStorage load_kv;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s0;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s1;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s0_corr;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s1_corr;
      alignas(16) typename CollectiveMainloop::PipelineO::SharedStorage mma_corr;
      alignas(16) typename CollectiveMainloop::PipelineE::SharedStorage corr_epi;
      alignas(16) typename CollectiveMainloop::OrderBarrierSoftmax::SharedStorage order_s01;
    } pipelines;

    uint32_t tmem_base_ptr;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    cutlass::KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params tile_scheduler;
  };

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  using ArchTag = cutlass::arch::Sm100;

  static size_t get_workspace_size(Arguments const& args) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        args.problem_shape,
        CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info, ClusterShape{}, TileShape{})
    };
  }

  CUTLASS_DEVICE auto apply_batch(const Params &params, ProblemShape const& problem_shape, int batch_idx) {
    return apply_variable_length(params.problem_shape, batch_idx);
  }

  CUTLASS_DEVICE void operator()(const Params &params, char* smem) {
#if (! defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) && ! defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) && \
    ! defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) && ! defined(CUTLASS_ARCH_MMA_SM103F_ENABLED))
    CUTE_INVALID_CONTROL_PATH("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    TileScheduler tile_scheduler{params.tile_scheduler};

    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_WarpRole(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    bool is_mma_leader_cta =
        (cta_rank_in_cluster % cute::size(typename CollectiveMainloop::CollectiveMmaQK::TiledMma::AtomThrID{})) == 0;

    if (role == WarpRole::Load && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    if (role == WarpRole::Epilogue && lane_predicate) {
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    auto get_epilogue_storage = [&]() {
      if constexpr (IsMla && CollectiveMainloop::IsOrderLoadEpilogue) {
        return reinterpret_cast<typename CollectiveEpilogue::TensorStorage *>(shared_storage.mainloop_epilogue.mainloop.smem_o.data());
      } else {
        return &shared_storage.mainloop_epilogue.epilogue;
      }
    };
    typename CollectiveEpilogue::TensorStorage & epilogue_storage = *get_epilogue_storage();


    typename CollectiveMainloop::PipelineQ::Params pipeline_load_q_params;
    if (role == WarpRole::Load) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Consumer;
    }
    pipeline_load_q_params.is_leader =
        lane_predicate && (role == WarpRole::Load) && is_mma_leader_cta;
    pipeline_load_q_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadQ;
    typename CollectiveMainloop::PipelineQ pipeline_load_q(
      shared_storage.pipelines.load_q,
      pipeline_load_q_params,
      ClusterShape{},  cute::true_type{}, /*mask calc*/cute::false_type{});
    
    typename CollectiveMainloop::PipelineKV::Params pipeline_load_kv_params;
    if (role == WarpRole::Load) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Consumer;
    }
    pipeline_load_kv_params.is_leader =
        lane_predicate && (role == WarpRole::Load) && is_mma_leader_cta;
    pipeline_load_kv_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadK;
    typename CollectiveMainloop::PipelineKV pipeline_load_kv(
      shared_storage.pipelines.load_kv,
      pipeline_load_kv_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s0_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax0) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s0_params.consumer_arv_count =
        NumWarpsSoftmax * cutlass::NumThreadsPerWarp * cute::size(ClusterShape{});
    typename CollectiveMainloop::PipelineS pipeline_mma_s0(
      shared_storage.pipelines.mma_s0,
      pipeline_mma_s0_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s1_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax1) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s1_params.consumer_arv_count =
        NumWarpsSoftmax * cutlass::NumThreadsPerWarp * cute::size(ClusterShape{});
    typename CollectiveMainloop::PipelineS pipeline_mma_s1(
      shared_storage.pipelines.mma_s1,
      pipeline_mma_s1_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s0_corr_params;
    if (role == WarpRole::Softmax0) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s0_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s0_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s0_corr(
      shared_storage.pipelines.s0_corr,
      pipeline_s0_corr_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s1_corr_params;
    if (role == WarpRole::Softmax1) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s1_corr_params.producer_arv_count = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    pipeline_s1_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineC pipeline_s1_corr(
      shared_storage.pipelines.s1_corr,
      pipeline_s1_corr_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineO::Params pipeline_mma_corr_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_corr_params.consumer_arv_count =
        NumWarpsCorrection * cutlass::NumThreadsPerWarp * cute::size(ClusterShape{});
    typename CollectiveMainloop::PipelineO pipeline_mma_corr(
      shared_storage.pipelines.mma_corr,
      pipeline_mma_corr_params,
      ClusterShape{}, /*barrier init*/ cute::true_type{}, /*mask calc*/cute::false_type{});

    typename CollectiveMainloop::PipelineE::Params pipeline_corr_epi_params;
    if (role == WarpRole::Correction) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Producer;
    }
    if (role == WarpRole::Epilogue) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Consumer;
    }
    pipeline_corr_epi_params.producer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_corr_epi_params.consumer_arv_count = NumWarpsEpilogue * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::PipelineE pipeline_corr_epi(
      shared_storage.pipelines.corr_epi,
      pipeline_corr_epi_params,
      /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::OrderBarrierSoftmax::Params params_order_s01;
    params_order_s01.group_id = role == WarpRole::Softmax1 ? 1 : 0;
    params_order_s01.group_size = NumWarpsSoftmax * cutlass::NumThreadsPerWarp;
    typename CollectiveMainloop::OrderBarrierSoftmax order_s01(
      shared_storage.pipelines.order_s01, params_order_s01);

    TmemAllocator tmem_allocator;

    __syncthreads();

    pipeline_load_q.init_masks(ClusterShape{});
    pipeline_load_kv.init_masks(ClusterShape{});
    pipeline_mma_s0.init_masks(ClusterShape{});
    pipeline_mma_s1.init_masks(ClusterShape{});
    pipeline_mma_corr.init_masks(ClusterShape{});

    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_consumer_state;
    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineQ>();

    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_consumer_state;
    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineKV>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_consumer_state;
    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineE>();

    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_consumer_state;
    typename CollectiveMainloop::PipelineO::PipelineState pipeline_mma_corr_producer_state = cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineO>();

    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue{params.epilogue};

    // apxinf: this is upstream's text with one compile-time guard added, and
    // on the stock two-Q-stage path `kSingleQStage` is a compile-time `false`
    // so `! kSingleQStage &&` folds away and it IS upstream verbatim (proven:
    // the whole stock consumer is byte-identical to pristine over 25 arms).
    //
    // On the single-Q-stage path the guard is logically dead --
    // `Sm100FmhaCtxKernelWarpspecialized1QSchedule::warp_idx_to_WarpRole()`
    // never returns Softmax1 -- and it is kept only because it documents that.
    //
    // HISTORY, so nobody re-learns it: in round 22 dropping this dead guard
    // measurably broke the language (hd256, 1-Q-stage, S_PIPE_STAGES=2)
    // operator -- 98.8188% bit-equal with outputs ~1e36 -- with an IDENTICAL
    // register report and an identical arithmetic histogram; only the
    // role-dispatch control flow moved.  That was not a property of this line.
    // It was an unguarded tmem RAW hazard in the multi-S-slot mainloop
    // (PV(i) read P(i) with no barrier against softmax(i)'s store), which
    // ptxas' branch layout happened to tip one way or the other.  Report 24
    // put a real barrier on it, and then re-ran the round-22 bisection: all
    // FIVE dispatch spellings (bare `else`; `else if (Empty)`; `else if (Empty
    // || (1Q && Softmax1))`; that plus `logical_seqlen_q` reverted; and
    // `Softmax0 || Softmax1` here) now give 13873119/13873152 with a
    // byte-identical 33-element MISS list, 3 runs each.  The same bisection on
    // the pre-fix mainloop, in the same session, still reproduced 99.5274% /
    // 99.7636% / 99.0550%.  Report 24 §3.
    if (role == WarpRole::Softmax0 || (! kSingleQStage && role == WarpRole::Softmax1)) {
      warpgroup_reg_set<NumRegsSoftmax>();

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= apxinf_detail::logical_seqlen_q(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        bool is_softmax_0 = role == WarpRole::Softmax0;

        mainloop.softmax(
           is_softmax_0 ? 0 : 1, blk_coord,
           params.mainloop, logical_problem_shape,
           is_softmax_0 ? pipeline_mma_s0 : pipeline_mma_s1,
           is_softmax_0 ? pipeline_mma_s0_consumer_state : pipeline_mma_s1_consumer_state,
           is_softmax_0 ? pipeline_s0_corr : pipeline_s1_corr,
           is_softmax_0 ? pipeline_s0_corr_producer_state : pipeline_s1_corr_producer_state,
           order_s01
         );

       }
    }
    else if (role == WarpRole::Correction) {
      cutlass::arch::warpgroup_reg_dealloc<NumRegsCorrection>();

      bool has_valid = false;

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= apxinf_detail::logical_seqlen_q(logical_problem_shape)) {
          continue;
        }

        has_valid = true;

        if (get<1>(logical_problem_shape) == 0) {
          mainloop.correction_empty(
            blk_coord,
            params.mainloop, logical_problem_shape,
            params.problem_shape,
            epilogue_storage,
            pipeline_corr_epi, pipeline_corr_epi_producer_state,
            epilogue
          );
          continue;
        }

        mainloop.correction(
          blk_coord,
          params.mainloop, logical_problem_shape,
          params.problem_shape,
          epilogue_storage,
          pipeline_s0_corr, pipeline_s0_corr_consumer_state,
          pipeline_s1_corr, pipeline_s1_corr_consumer_state,
          pipeline_mma_corr, pipeline_mma_corr_consumer_state,
          pipeline_corr_epi, pipeline_corr_epi_producer_state,
          epilogue
        );

      }

      if constexpr (NumWarpsEpilogue == 0) {
        static_assert(NumWarpsCorrection == 1);

        if (has_valid) {
          uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
          tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }
      }

    }
    else if (role == WarpRole::MMA) {
      warpgroup_reg_set<NumRegsOther>();

      bool allocated = false;

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= apxinf_detail::logical_seqlen_q(logical_problem_shape)) {
          continue;
        }

        if (!allocated) {
          tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
          __syncwarp();
          allocated = true;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        // A 2-SM TMEM allocation is a cluster-wide operation: the MMA warp in
        // both CTAs must execute allocate/free even though only CTA 0 issues
        // the paired UMMA instructions.  Skipping the peer CTA here leaves the
        // Allocator2Sm handshake incomplete and deadlocks before the first QK.
        if (is_mma_leader_cta) {
          mainloop.mma(
            blk_coord,
            params.mainloop, logical_problem_shape,
            shared_storage.mainloop_epilogue.mainloop,
            pipeline_load_q, pipeline_load_q_consumer_state,
            pipeline_load_kv, pipeline_load_kv_consumer_state,
            pipeline_mma_s0, pipeline_mma_s0_producer_state,
            pipeline_mma_s1, pipeline_mma_s1_producer_state,
            pipeline_mma_corr, pipeline_mma_corr_producer_state
          );
        }

      }
    }
    else if (role == WarpRole::Load) {
      warpgroup_reg_set<NumRegsOther>();

      if constexpr (IsMla && CollectiveMainloop::IsOrderLoadEpilogue) {
        cutlass::arch::NamedBarrier::arrive((NumWarpsLoad + NumWarpsEpilogue) * NumThreadsPerWarp, 
                                      cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= apxinf_detail::logical_seqlen_q(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        mainloop.load(
          blk_coord, logical_problem_shape,
          params.mainloop, params.problem_shape,
          shared_storage.mainloop_epilogue.mainloop,
          pipeline_load_q, pipeline_load_q_producer_state,
          pipeline_load_kv, pipeline_load_kv_producer_state
        );

      }
    }
    else if (role == WarpRole::Epilogue) {
      warpgroup_reg_set<NumRegsOther>();

      bool has_valid = false;

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape = apply_batch(params,
            params.problem_shape, get<2,1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= apxinf_detail::logical_seqlen_q(logical_problem_shape)) {
          continue;
        }

        has_valid = true;

        if constexpr (kIs2Sm) {
          // The scheduler coordinate names the logical 256-row cluster tile,
          // while each CTA stores one 128-row local epilogue tile.
          get<0>(blk_coord) = get<0>(blk_coord) * cute::size<0>(ClusterShape{})
                            + int(cta_rank_in_cluster % cute::size<0>(ClusterShape{}));
          CollectiveEpilogue::consume_only(
              pipeline_corr_epi, pipeline_corr_epi_consumer_state);
        } else {
          epilogue.store(
            blk_coord, logical_problem_shape,
            params.epilogue, params.problem_shape,
            epilogue_storage,
            pipeline_corr_epi, pipeline_corr_epi_consumer_state
          );
        }

      }

      static_assert(NumWarpsEpilogue <= 1);
      if constexpr (NumWarpsEpilogue == 1 && !kIs2Sm) {
        if(has_valid) {
          uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
          tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }
      }

    }
    // apxinf: UPSTREAM TEXT.  The single-Q-stage schedule's
    // warp_idx_to_WarpRole() never returns Softmax1, so no extra disjunct is
    // needed here.  In round 22, adding one (a logically DEAD term) collapsed
    // the role dispatch from a BRX jump table to straight-line branches (BRA
    // 189->186, BRX 2->1, EXIT 18->16) with a bit-identical arithmetic
    // histogram, and that flipped the language operator to 99.5274% /
    // 99.7636% / 99.2912% with outputs ~1e36.  That was a symptom of the
    // unguarded P RAW hazard in the multi-S-slot mainloop, not of this line;
    // report 24 barriered the hazard and re-ran the bisection, and the
    // operator is now invariant to all five spellings.  Keep upstream's text
    // anyway -- it is the smallest diff -- but it is no longer load-bearing.
    else if (role == WarpRole::Empty) {
      warpgroup_reg_set<NumRegsEmpty>();

      /* no-op, donate regs and exit */
    }

    if constexpr (kIs2Sm) {
      cute::cluster_sync();
      if (role == WarpRole::MMA) {
        uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
        tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
      }
    }
#endif
  }

};

}  // namespace cutlass::fmha::kernel

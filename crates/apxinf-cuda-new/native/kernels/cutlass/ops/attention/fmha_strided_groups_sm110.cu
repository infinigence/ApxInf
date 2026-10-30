// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Strided-group attention for the validated 624- and 2200-token contracts:
// BF16 head_dim 64, 16 heads,
// non-causal, three batches over a padded 4072-token view, V carrying a
// different per-token stride than Q/K (the retained interleaved QKV buffer).
//
// It is the stock two-Q-stage collective (`ThreadShape<_2,_1,_1>`) with a
// `ResidualMask` and `PersistentTileScheduler`. The numeric contract follows
// FlashAttention-4's `flash_fwd_sm100.py`:
//   * fully DESCENDING KV, masked (ragged) tile first
//   * `rescale_threshold = 8.0`
//   * ex2 polynomial emulation ON (freq 16 / res 4 / start_frg 1)
//   * `rcp.approx.f32` final normalisation
//   * row_sum stride-8, seeded into LANE 0 only
//   * ResidualMask through FA4's R2P bitmask
// selected entirely by the `-DAPXINF_*` flag line this translation unit is
// compiled with (see `crates/apxinf-cuda-new/build.rs` and
// `devlocal/aot2cuda/harness/MERGED_FLAGS.sh`, which is authoritative).
// Two of those flags (`APXINF_SKIP_PAD_QSTAGE`, `APXINF_P_CHUNKS16`) are
// bit-neutral scheduling knobs. A third, `APXINF_NO_ORDER_S`, used to be on
// this list and is NOT bit-neutral -- see the 🔴 block beside the flag line in
// `build.rs` and report 32 §1.1.
//
// FlashAttention / CUTLASS provenance and licences: kernels/cutlass/licenses.

#include <cuda_runtime.h>

#include <cstddef>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"

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

namespace apxinf::cuda_new::cutlass_ops::fmha_vision {

using namespace cute;

using TileShape = Shape<_256, _128, _64>;

// Every FMHA template this translation unit instantiates is reachable from
// `VisionMask`, so nothing it defines can collide with another differently
// flagged FMHA translation unit's symbols (the FMHA headers are configured by
// `-DAPXINF_*` macros, which do not take part in name mangling -- two
// identically parameterised instantiations compiled with different flag lines
// would be a silent ODR violation). `scripts/check_kernel_architecture.sh`
// does not look for that, so the guard is structural.
struct VisionMask : cutlass::fmha::collective::ResidualMask {};

using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using Element = cutlass::bfloat16_t;

using Mainloop =
    cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
        Element, float, float, TileShape, StrideQ, StrideK, StrideV,
        VisionMask>;
using Epilogue =
    cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
        Element, float, typename Mainloop::TileShapePV, StrideO, StrideLSE>;
template <int Tokens>
using ProblemShape = cute::tuple<
    Int<Tokens>, Int<Tokens>, _64,
    cute::tuple<cute::tuple<_1, _16>, _3>>;

template <int Tokens>
using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape<Tokens>, Mainloop, Epilogue,
#if defined(APXINF_PERSISTENT) && APXINF_PERSISTENT
    cutlass::fmha::kernel::PersistentTileScheduler>;
#else
    cutlass::fmha::kernel::IndividualTileScheduler>;
#endif

template <int Tokens>
using FmhaOp = cutlass::fmha::device::FMHA<Kernel<Tokens>>;

// Device allocations that outlive a launch. Owned by the caller (the adapter)
// so that nothing here is `static`: a captured CUDA graph replays against
// these addresses, and a thread-local would make the lifetime implicit.
struct State {
  void* workspace = nullptr;
  size_t workspace_size = 0;
  float* lse = nullptr;
  size_t lse_size = 0;
  int device = -1;
};

template <int Tokens>
int run_fixed(const void* q, const void* k, const void* v, void* output,
              int qko_token_stride, int v_token_stride,
              int batch_token_count, float scale, cudaStream_t stream,
              State& state, bool prepare_only) {
  using Operation = FmhaOp<Tokens>;
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      qko_token_stride < 16 * 64 || v_token_stride < 16 * 64) {
    return -1;
  }
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return -4;
  if (state.device != -1 && state.device != device) return -4;
  state.device = device;

  // q_per_kv == 1: the vision projection is full MHA, not GQA.
  auto problem = cute::make_tuple(
      Int<Tokens>{}, Int<Tokens>{}, _64{},
      cute::make_tuple(cute::make_tuple(_1{}, _16{}), _3{}));

  // The AOT view is a PADDED buffer: the batch stride is a fixed token count
  // times the per-token stride, independent of the live group length.
  const int batch_tokens = batch_token_count ? batch_token_count : Tokens;

  StrideQ q_stride = make_stride(
      qko_token_stride, _1{},
      make_stride(make_stride(64, 64),
                  qko_token_stride * batch_tokens));
  // O is written densely at `heads * rounded_dim` per token, which for the
  // vision geometry equals `qko_token_stride`; spelled out rather than aliased
  // to `q_stride` so a padded Q view cannot silently pad O too.
  StrideO output_stride = make_stride(
      16 * 64, _1{},
      make_stride(make_stride(64, 64), 16 * 64 * batch_tokens));
  StrideK kv_stride = make_stride(
      qko_token_stride, _1{},
      make_stride(make_stride(_0{}, 64),
                  qko_token_stride * batch_tokens));
  // V may carry a different per-token stride than K (AOT: k 1024, v 3072).
  StrideV v_stride =
      make_stride(v_token_stride, _1{},
                  make_stride(make_stride(_0{}, 64),
                              v_token_stride * batch_tokens));

  constexpr int rounded_query = ((Tokens + 127) / 128) * 128;
  StrideLSE lse_stride = make_stride(
      _1{}, make_stride(make_stride(rounded_query, rounded_query),
                        rounded_query * 16));
#if !(defined(APXINF_NO_LSE) && APXINF_NO_LSE)
  // Only reachable if somebody turns the LSE back on; the vision AOT emits
  // none, so the default build never allocates this and therefore never calls
  // `cudaMalloc` from a path a CUDA graph capture could reach.
  const size_t required_lse =
      size_t(3) * 16 * rounded_query * sizeof(float);
  if (required_lse > state.lse_size) {
    if (!prepare_only) return -4;
    if (state.lse != nullptr) cudaFree(state.lse);
    state.lse = nullptr;
    state.lse_size = 0;
    if (cudaMalloc(&state.lse, required_lse) != cudaSuccess) return -4;
    state.lse_size = required_lse;
  }
#endif

  int multiprocessors = 0;
  if (cudaDeviceGetAttribute(&multiprocessors,
                             cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess) {
    return -4;
  }
  typename Operation::Arguments arguments{
      problem,
      {{static_cast<Element const*>(q), q_stride,
        static_cast<Element const*>(k), kv_stride,
        static_cast<Element const*>(v), v_stride},
       scale,
       1.0f,
       1.0f,
       1.0f,
       1.0f},
      {static_cast<Element*>(output), output_stride,
// The vision AOT emits no LSE, so neither do we: a non-null `ptr_LSE` costs a
// `cutlass::fast_log` and a global store per row.
#if defined(APXINF_NO_LSE) && APXINF_NO_LSE
       nullptr,
#else
       state.lse,
#endif
       lse_stride},
      {0, multiprocessors}};

  Operation operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  const size_t required_workspace = Operation::get_workspace_size(arguments);
  if (required_workspace > state.workspace_size) {
    if (!prepare_only) return -4;
    if (state.workspace != nullptr) cudaFree(state.workspace);
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

int run(const void* q, const void* k, const void* v, void* output, int batches,
        int tokens, int heads, int head_dim, int qko_token_stride,
        int v_token_stride, int batch_token_count, float scale,
        cudaStream_t stream, State& state, bool prepare_only) {
  if (batches != 3 || heads != 16 || head_dim != 64) return -1;
  if (tokens == 624) {
    return run_fixed<624>(q, k, v, output, qko_token_stride, v_token_stride,
                          batch_token_count, scale, stream, state,
                          prepare_only);
  }
  if (tokens == 2200) {
    return run_fixed<2200>(q, k, v, output, qko_token_stride, v_token_stride,
                           batch_token_count, scale, stream, state,
                           prepare_only);
  }
  return -1;
}

}  // namespace apxinf::cuda_new::cutlass_ops::fmha_vision

extern "C" size_t apxinf_attention_strided_groups_state_size() {
  return sizeof(apxinf::cuda_new::cutlass_ops::fmha_vision::State);
}

extern "C" void apxinf_attention_strided_groups_state_destroy(void* state_ptr) {
  using apxinf::cuda_new::cutlass_ops::fmha_vision::State;
  if (state_ptr == nullptr) return;
  auto& state = *static_cast<State*>(state_ptr);
  if (state.device >= 0) cudaSetDevice(state.device);
  if (state.lse != nullptr) cudaFree(state.lse);
  if (state.workspace != nullptr) cudaFree(state.workspace);
  state = State{};
}

// Vision-attention parity entry: a strided padded view with a distinct V
// stride. q/k/out token_stride = 1024, v token_stride = 3072, batch stride =
// batch_token_count * token_stride.
extern "C" int apxinf_attention_strided_groups_forward(
    const void* q, const void* k, const void* v, void* output, int batches,
    int tokens, int heads, int head_dim, int qko_token_stride,
    int v_token_stride, int batch_token_count, float scale,
    cudaStream_t stream, void* state_ptr, int prepare_only) {
  using apxinf::cuda_new::cutlass_ops::fmha_vision::State;
  if (state_ptr == nullptr) return -1;
  return apxinf::cuda_new::cutlass_ops::fmha_vision::run(
      q, k, v, output, batches, tokens, heads, head_dim, qko_token_stride,
      v_token_stride, batch_token_count, scale, stream,
      *static_cast<State*>(state_ptr), prepare_only != 0);
}

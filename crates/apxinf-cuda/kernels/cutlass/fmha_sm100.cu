// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Rust/FFI adaptation of an upstream SM100/SM110 CUTLASS FMHA kernel. Its pinned
// FMHA and CUTLASS headers live alongside this wrapper under kernels/cutlass.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"

// The upstream example guard recognizes SM100 and SM103. CUTLASS enables the
// same kernel path for SM101 and SM110, so alias those architecture markers
// only while parsing the kernel header.
#if (defined(CUTLASS_ARCH_MMA_SM101A_ENABLED) || \
     defined(CUTLASS_ARCH_MMA_SM110A_ENABLED)) && \
    !defined(CUTLASS_ARCH_MMA_SM100A_ENABLED)
#define APXINF_FMHA_UNDEF_SM100A 1
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif
#if (defined(CUTLASS_ARCH_MMA_SM101F_ENABLED) || \
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

using namespace cute;
using TileShape = Shape<_256, _128, _128>;

using StrideQ = cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;
using StrideK = cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>;
using StrideV = StrideK;
using StrideO = StrideQ;
using StrideLSE = cute::tuple<_1, cute::tuple<cute::tuple<int, int>, int>>;
using ProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

namespace apxinf::cuda::cutlass_ops {

template <typename Element, typename ElementOut>
struct FmhaTypes {
  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, float, float, TileShape, StrideQ, StrideK, StrideV,
      cutlass::fmha::collective::NoMask>;
  using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      ElementOut, float, typename Mainloop::TileShapePV, StrideO, StrideLSE>;
  using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
      ProblemShape, Mainloop, Epilogue,
      cutlass::fmha::kernel::IndividualTileScheduler>;
  using FmhaOp = cutlass::fmha::device::FMHA<Kernel>;
};

template <typename Element, typename ElementOut>
static int cutlass_mha(
    const void* q, const void* k, const void* v, void* output,
    int batches, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, int q_token_stride, int kv_token_stride,
    cudaStream_t stream, bool prepare_only) {
  using FmhaOp = typename FmhaTypes<Element, ElementOut>::FmhaOp;
  static thread_local void* workspace = nullptr;
  static thread_local size_t workspace_size = 0;
  static thread_local float* lse = nullptr;
  static thread_local size_t lse_size = 0;
  static thread_local int prepared_device = -1;
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      batches <= 0 || query_tokens <= 0 || key_tokens <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      query_heads % kv_heads != 0) {
    return -1;
  }
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return -4;
  if (prepared_device != -1 && prepared_device != device) {
    if (!prepare_only) return -4;
    if (lse != nullptr) cudaFree(lse);
    if (workspace != nullptr) cudaFree(workspace);
    lse = nullptr;
    lse_size = 0;
    workspace = nullptr;
    workspace_size = 0;
  }
  prepared_device = device;

  int q_per_kv = query_heads / kv_heads;
  int rounded_dim = cutlass::round_up(head_dim, 8);
  if (q_token_stride == 0) q_token_stride = query_heads * rounded_dim;
  if (kv_token_stride == 0) kv_token_stride = kv_heads * rounded_dim;
  if (q_token_stride < query_heads * rounded_dim ||
      kv_token_stride < kv_heads * rounded_dim) {
    return -1;
  }
  auto problem = cute::make_tuple(
      query_tokens, key_tokens, rounded_dim,
      cute::make_tuple(cute::make_tuple(q_per_kv, kv_heads), batches));

  StrideQ q_stride = make_stride(
      q_token_stride, _1{},
      make_stride(make_stride(rounded_dim, q_per_kv * rounded_dim),
                  q_token_stride * query_tokens));
  StrideO output_stride = make_stride(
      query_heads * rounded_dim, _1{},
      make_stride(make_stride(rounded_dim, q_per_kv * rounded_dim),
                  query_heads * rounded_dim * query_tokens));
  StrideK kv_stride = make_stride(
      kv_token_stride, _1{},
      make_stride(make_stride(_0{}, rounded_dim),
                  kv_token_stride * key_tokens));

  int rounded_query = ((query_tokens + 127) / 128) * 128;
  StrideLSE lse_stride = make_stride(
      _1{}, make_stride(make_stride(rounded_query, rounded_query * q_per_kv),
                        rounded_query * query_heads));
  size_t required_lse = static_cast<size_t>(batches) * query_heads *
                        rounded_query * sizeof(float);
  if (required_lse > lse_size) {
    if (!prepare_only) return -4;
    if (lse != nullptr) cudaFree(lse);
    if (cudaMalloc(&lse, required_lse) != cudaSuccess) return -4;
    lse_size = required_lse;
  }

  int multiprocessors = 0;
  if (cudaDeviceGetAttribute(
          &multiprocessors, cudaDevAttrMultiProcessorCount, device) !=
      cudaSuccess) {
    return -4;
  }
  typename FmhaOp::Arguments arguments{
      problem,
      {{static_cast<Element const*>(q), q_stride,
        static_cast<Element const*>(k), kv_stride,
        static_cast<Element const*>(v), kv_stride},
       0.0f, 1.0f, 1.0f, 1.0f, 1.0f},
      {static_cast<ElementOut*>(output), output_stride, lse, lse_stride},
      {0, multiprocessors}};

  FmhaOp operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  size_t required_workspace = FmhaOp::get_workspace_size(arguments);
  if (required_workspace > workspace_size) {
    if (!prepare_only) return -4;
    if (workspace != nullptr) cudaFree(workspace);
    if (cudaMalloc(&workspace, required_workspace) != cudaSuccess) return -4;
    workspace_size = required_workspace;
  }
  if (prepare_only) return 0;
  if (operation.initialize(arguments, workspace, stream) != cutlass::Status::kSuccess)
    return -2;
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}

int prepare_mha_bf16(
    const void* q, const void* k, const void* v, void* output,
    int batches, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, cudaStream_t stream) {
  return cutlass_mha<cutlass::bfloat16_t, cutlass::bfloat16_t>(
      q, k, v, output, batches, query_tokens, key_tokens, query_heads,
      kv_heads, head_dim, 0, 0, stream, true);
}

int mha_bf16(
    const void* q, const void* k, const void* v, void* output,
    int batches, int query_tokens, int key_tokens, int query_heads,
    int kv_heads, int head_dim, cudaStream_t stream) {
  return cutlass_mha<cutlass::bfloat16_t, cutlass::bfloat16_t>(
      q, k, v, output, batches, query_tokens, key_tokens, query_heads,
      kv_heads, head_dim, 0, 0, stream, false);
}

}  // namespace apxinf::cuda::cutlass_ops

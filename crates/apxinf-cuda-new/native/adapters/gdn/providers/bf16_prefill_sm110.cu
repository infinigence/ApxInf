// SPDX-License-Identifier: Apache-2.0
// BF16 first-prefill recurrent attention on the vendored in-tree FlashInfer
// Cake GDN prefill kernel.
//
// The conv/pack/gate prologue and the state transpose come from
// `gdn_bf16_prefill_common.cuh`, identical to the AOT arm, so the two arms
// differ in exactly one kernel launch. This file owns the TMA descriptors,
// which need <cuda.h> and therefore cannot live in the kernel translation
// unit; see gdn_bf16_cuda_launch.cu.

#include <cuda.h>

#include <atomic>
#include <mutex>

#include "bf16_prefill_common.cuh"

extern "C" int32_t apxinf_gdn_bf16_fulldv_set_attr();
extern "C" int32_t apxinf_gdn_bf16_fulldv_launch(
    const void* map_q, const void* map_k, const void* map_v,
    const void* map_out, float* gate_linear, float* beta, int* cu_seqlens,
    float* state, int* state_indices, float* checkpoint_state,
    int* cu_checkpoints, unsigned char* tensormap_workspace,
    long long state_stride, float scale, int num_seqs, int q_heads,
    int v_heads, int total_tiles, int grid_x, cudaStream_t stream);

namespace {
using namespace apxinf_gdn_bf16;

constexpr int kTensorMapWorkspacePerCta = 512;
// Tail of the scratch, for the three optional pointers the kernel still
// dereferences even at USE_STATE_INDICES=0 / ENABLE_CHECKPOINTS=0:
// state_indices (i32), cu_checkpoints (i32), checkpoint_state (f32). Passing
// nullptr for these produced a state 1e27x too large at a single-chunk
// sequence while longer ones were correct.
constexpr size_t kOptionalTailBytes = 256;

// A 3-D tiled descriptor over a [tokens, heads, 128] BF16 tensor.
//
// The kernel indexes (dim, token, head), so the global dimensions are reversed
// against the row-major shape, and cuTensorMapEncodeTiled takes strides for
// axes 1..rank-1 only -- the innermost is implicit. Every field below was
// checked character-for-character against upstream's own host binding.
CUresult encode(CUtensorMap* map, void* base, long long tokens,
                long long heads) {
  cuuint64_t global_dim[3] = {static_cast<cuuint64_t>(D),
                              static_cast<cuuint64_t>(tokens),
                              static_cast<cuuint64_t>(heads)};
  cuuint64_t global_stride[2] = {
      static_cast<cuuint64_t>(heads * D * sizeof(__nv_bfloat16)),
      static_cast<cuuint64_t>(D * sizeof(__nv_bfloat16))};
  unsigned int box_dim[3] = {64u, 64u, 1u};
  unsigned int element_stride[3] = {1u, 1u, 1u};
  return cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, base, global_dim,
      global_stride, box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

std::atomic<bool> attribute_ready{false};
std::mutex attribute_mutex;
}  // namespace

extern "C" int32_t apxinf_gdn_bf16_prefill_sm110_init() {
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return status(error);
  int major = 0, minor = 0, sms = 0;
  if ((error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)) != cudaSuccess)
    return status(error);
  if ((error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device)) != cudaSuccess)
    return status(error);
  if ((error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess)
    return status(error);
  if (major != 11 || minor != 0 || sms != 20)
    return status(cudaErrorInvalidDeviceFunction);
  if (attribute_ready.load(std::memory_order_acquire)) return 0;
  std::lock_guard<std::mutex> lock(attribute_mutex);
  if (attribute_ready.load(std::memory_order_relaxed)) return 0;
  int32_t attr = apxinf_gdn_bf16_fulldv_set_attr();
  if (attr != 0) return attr;
  attribute_ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_gdn_bf16_prefill_sm110(
    const void* zba, const void* conv_weight, void* new_conv_state,
    const float* A_log, const void* dt_bias_bf16,
    void* raw_qk_scratch, void* q, void* k, void* v, void* a, void* b,
    float* h_output_vmajor, float* final_state_kmajor,
    void* output_pad, int32_t* cu_seqlens, int32_t seq, int32_t seq_pad,
    cudaStream_t stream) {
  if (!attribute_ready.load(std::memory_order_acquire))
    return status(cudaErrorNotReady);
  if (!arguments_valid(zba, conv_weight, new_conv_state, A_log, dt_bias_bf16,
                       raw_qk_scratch, q, k, v, a, b, h_output_vmajor,
                       final_state_kmajor, output_pad, cu_seqlens, seq,
                       seq_pad))
    return status(cudaErrorInvalidValue);

  int multiprocessors = 0;
  cudaError_t error = cudaDeviceGetAttribute(
      &multiprocessors, cudaDevAttrMultiProcessorCount, 0);
  if (error != cudaSuccess) return status(error);
  // Full-DV: one tile per (sequence, o-head). The dv-split schedule's
  // `* 2` here is the single most expensive mistake available in this file.
  const int total_tiles = HV;
  const int grid_x = total_tiles < multiprocessors ? total_tiles : multiprocessors;
  if (static_cast<size_t>(grid_x) * kTensorMapWorkspacePerCta +
          kOptionalTailBytes >
      FLASH_WORKSPACE_BYTES)
    return status(cudaErrorInvalidValue);

  float* alpha = nullptr;
  float* beta = nullptr;
  void* descriptor_workspace = nullptr;
  int32_t prologue = apxinf_gdn_bf16_prefill_prologue(zba, conv_weight, new_conv_state, A_log,
                                  dt_bias_bf16, raw_qk_scratch, q, k, v, a, b,
                                  cu_seqlens, seq, seq_pad, &alpha, &beta,
                                  &descriptor_workspace, stream);
  if (prologue != 0) return prologue;

  CUtensorMap map_q{}, map_k{}, map_v{}, map_out{};
  if (encode(&map_q, q, seq_pad, H) != CUDA_SUCCESS ||
      encode(&map_k, k, seq_pad, H) != CUDA_SUCCESS ||
      encode(&map_v, v, seq_pad, HV) != CUDA_SUCCESS ||
      encode(&map_out, output_pad, seq_pad, HV) != CUDA_SUCCESS) {
    return status(cudaErrorInvalidValue);
  }

  auto* workspace = static_cast<unsigned char*>(descriptor_workspace);
  unsigned char* tail =
      workspace + static_cast<size_t>(grid_x) * kTensorMapWorkspacePerCta;
  if ((error = cudaMemsetAsync(tail, 0, kOptionalTailBytes, stream)) !=
      cudaSuccess)
    return status(error);
  auto* state_indices = reinterpret_cast<int*>(tail);
  auto* cu_checkpoints = reinterpret_cast<int*>(tail + 64);
  auto* checkpoint_state = reinterpret_cast<float*>(tail + 128);

  int32_t launch = apxinf_gdn_bf16_fulldv_launch(
      &map_q, &map_k, &map_v, &map_out, alpha, beta, cu_seqlens,
      h_output_vmajor, state_indices, checkpoint_state, cu_checkpoints,
      workspace, static_cast<long long>(HV) * D * D, SCALE, /*num_seqs=*/1, H,
      HV, total_tiles, grid_x, stream);
  if (launch != 0) return launch;
  return apxinf_gdn_bf16_prefill_state_transpose(h_output_vmajor,
                                                final_state_kmajor, stream);
}

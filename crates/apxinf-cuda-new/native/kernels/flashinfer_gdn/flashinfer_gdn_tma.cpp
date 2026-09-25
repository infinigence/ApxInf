// Copyright 2026 ApxInf contributors.
//
// TMA descriptor construction for the vendored FlashInfer Cake GDN prefill.
//
// Kept apart from the kernel translation unit: the generated kernel header
// declares its own CUtensorMap and <cuda.h> declares another, so including
// both in one file is a redefinition error. Descriptors cross the boundary as
// opaque bytes.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>

#include "flashinfer_gdn.h"

extern "C" int apxinf_flashinfer_gdn_launch(
    const void* map_q, const void* map_k, const void* map_v,
    const void* map_out, float* gate_log, float* beta, int* cu_seqlens,
    float* state, int* state_indices, float* checkpoint_state,
    int* cu_checkpoints, unsigned char* tensor_map_workspace,
    long long state_stride, float scale, int num_seqs, int q_heads,
    int v_heads, int total_tiles, int grid_x, cudaStream_t stream);

namespace apxinf::cuda::flashinfer_gdn {
namespace {

constexpr int kHeadDim = 128;
constexpr int kTensorMapWorkspacePerCta = 512;
// Tail of the workspace, for the three optional pointers the kernel still
// dereferences: state_indices (i32), cu_checkpoints (i32), checkpoint_state
// (f32). Passing nullptr for these yielded a state 1e27x too large at a
// single-chunk sequence while longer ones were correct.
constexpr size_t kOptionalTailBytes = 256;

int grid_for(int v_heads, int num_seqs, int* total_tiles_out) {
  // This variant splits the value dimension in two, so a tile is
  // (sequence, o-head, dv-half).
  const int total_tiles = num_seqs * v_heads * 2;
  if (total_tiles_out != nullptr) *total_tiles_out = total_tiles;
  int multiprocessors = 0;
  if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                             0) != cudaSuccess) {
    return 0;
  }
  return total_tiles < multiprocessors ? total_tiles : multiprocessors;
}

// A 3-D tiled descriptor over a [tokens, heads, 128] FP16 tensor.
//
// The kernel indexes (dim, token, head), so the global dimensions are reversed
// against the row-major shape, and cuTensorMapEncodeTiled takes strides for
// axes 1..rank-1 only -- the innermost is implicit.
CUresult encode(CUtensorMap* map, void* base, long long tokens,
                long long heads) {
  cuuint64_t global_dim[3] = {static_cast<cuuint64_t>(kHeadDim),
                              static_cast<cuuint64_t>(tokens),
                              static_cast<cuuint64_t>(heads)};
  cuuint64_t global_stride[2] = {
      static_cast<cuuint64_t>(heads * kHeadDim * sizeof(__half)),
      static_cast<cuuint64_t>(kHeadDim * sizeof(__half))};
  unsigned int box_dim[3] = {64u, 64u, 1u};
  unsigned int element_stride[3] = {1u, 1u, 1u};
  return cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, base, global_dim, global_stride,
      box_dim, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

}  // namespace

int prefill(const void* q, const void* k, const void* v, void* out,
            const void* gate_log, const void* beta, const void* cu_seqlens,
            void* state, void* tensor_map_workspace, int tokens, int q_heads,
            int v_heads, int num_seqs, float scale, cudaStream_t stream) {
  if (tokens <= 0 || q_heads <= 0 || v_heads <= 0 || num_seqs <= 0) return -1;
  if (v_heads % q_heads != 0) return -2;

  int total_tiles = 0;
  const int grid_x = grid_for(v_heads, num_seqs, &total_tiles);
  if (grid_x <= 0) return -3;

  CUtensorMap map_q{}, map_k{}, map_v{}, map_out{};
  if (encode(&map_q, const_cast<void*>(q), tokens, q_heads) != CUDA_SUCCESS ||
      encode(&map_k, const_cast<void*>(k), tokens, q_heads) != CUDA_SUCCESS ||
      encode(&map_v, const_cast<void*>(v), tokens, v_heads) != CUDA_SUCCESS ||
      encode(&map_out, out, tokens, v_heads) != CUDA_SUCCESS) {
    return -4;
  }

  const long long state_stride =
      static_cast<long long>(v_heads) * kHeadDim * kHeadDim;

  // The optional buffers live in the workspace tail and must read as zero.
  auto* workspace = static_cast<unsigned char*>(tensor_map_workspace);
  unsigned char* tail =
      workspace + static_cast<size_t>(grid_x) * kTensorMapWorkspacePerCta;
  if (cudaMemsetAsync(tail, 0, kOptionalTailBytes, stream) != cudaSuccess) {
    return -7;
  }
  auto* state_indices = reinterpret_cast<int*>(tail);
  auto* cu_checkpoints = reinterpret_cast<int*>(tail + 64);
  auto* checkpoint_state = reinterpret_cast<float*>(tail + 128);

  return apxinf_flashinfer_gdn_launch(
      &map_q, &map_k, &map_v, &map_out,
      const_cast<float*>(static_cast<const float*>(gate_log)),
      const_cast<float*>(static_cast<const float*>(beta)),
      const_cast<int*>(static_cast<const int*>(cu_seqlens)),
      static_cast<float*>(state), state_indices, checkpoint_state,
      cu_checkpoints, workspace, state_stride, scale, num_seqs, q_heads,
      v_heads, total_tiles, grid_x, stream);
}

size_t tensor_map_workspace_bytes(int v_heads, int num_seqs) {
  const int grid_x = grid_for(v_heads, num_seqs, nullptr);
  return grid_x <= 0 ? 0
                     : static_cast<size_t>(grid_x) * kTensorMapWorkspacePerCta +
                           kOptionalTailBytes;
}

}  // namespace apxinf::cuda::flashinfer_gdn

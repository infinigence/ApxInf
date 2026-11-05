// Copyright 2026 ApxInf contributors.
// Fixed family launcher; kernel body and launch parameters are unchanged.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {
#include "../../kernels/preprocess/planning_bf16.cuh"
}  // namespace

extern "C" cudaError_t apxinf_preprocess_concat7_cols_bf16(
    const void* s0, const void* s1, const void* s2, const void* s3,
    const void* s4, const void* s5, const void* s6, void* dst,
    int rows, int cols, int broadcast_mask, cudaStream_t stream) {
  if (s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr ||
      s4 == nullptr || s5 == nullptr || s6 == nullptr || dst == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  concat7_cols_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(s0),
      static_cast<const __nv_bfloat16*>(s1),
      static_cast<const __nv_bfloat16*>(s2),
      static_cast<const __nv_bfloat16*>(s3),
      static_cast<const __nv_bfloat16*>(s4),
      static_cast<const __nv_bfloat16*>(s5),
      static_cast<const __nv_bfloat16*>(s6),
      static_cast<__nv_bfloat16*>(dst), rows, cols, broadcast_mask);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_preprocess_flow_update_f32(
    void* w, const void* endpoint, float remaining, float step, int64_t count,
    cudaStream_t stream) {
  if (w == nullptr || endpoint == nullptr || count <= 0 ||
      !(remaining > 0.0f) || !std::isfinite(step)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  flow_update_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<float*>(w), static_cast<const float*>(endpoint),
      remaining, step, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_preprocess_vision_position_bf16(
    const void* table, void* embeddings, void* position_ids,
    int source_side, int height, int width, int hidden, int merge_size,
    cudaStream_t stream) {
  if (table == nullptr || embeddings == nullptr || position_ids == nullptr ||
      source_side <= 0 || height <= 0 || width <= 0 || hidden <= 0 ||
      merge_size <= 0 || height % merge_size != 0 ||
      width % merge_size != 0 ||
      height > std::numeric_limits<int>::max() / width) {
    return cudaErrorInvalidValue;
  }
  const int tokens = height * width;
  vision_position_kernel<<<tokens, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table),
      static_cast<__nv_bfloat16*>(embeddings),
      static_cast<int32_t*>(position_ids), source_side, height, width, hidden,
      merge_size);
  return cudaGetLastError();
}

#include "../../kernels/moe/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_moe_route(const void* x, const void* gate,
                                        const void* bias, void* out,
                                        int32_t tokens, int32_t experts,
                                        int32_t dim, int32_t top_k,
                                        float routed_scale,
                                        cudaStream_t stream) {
  if (!x || !gate || !bias || !out || tokens <= 0 || tokens > 4096 ||
      experts <= 0 || experts > kRouteMaxExperts || dim <= 0 || dim % 8 != 0 ||
      top_k <= 0 || top_k > kRouteMaxTopK || top_k > experts ||
      !(routed_scale == routed_scale) ||
      reinterpret_cast<uintptr_t>(x) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(gate) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(bias) % 4 != 0 ||
      reinterpret_cast<uintptr_t>(out) % 4 != 0) {
    return cudaErrorInvalidValue;
  }
  moe_route_kernel<<<tokens, kRouteThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(gate), static_cast<const float*>(bias),
      static_cast<float*>(out), experts, dim, top_k, routed_scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_moe_swiglu_scaled(
    const void* gate, const void* up, const void* weights, void* out,
    int32_t tokens, int32_t columns, int32_t experts, cudaStream_t stream) {
  if (!gate || !up || !weights || !out || tokens <= 0 || columns <= 0 ||
      experts <= 0 || columns % experts != 0 ||
      reinterpret_cast<uintptr_t>(weights) % 4 != 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t total = static_cast<int64_t>(tokens) * columns;
  const int block_columns = columns / experts;
  constexpr int threads = 256;
  const bool vector_path =
      columns % 8 == 0 && block_columns % 8 == 0 &&
      reinterpret_cast<uintptr_t>(gate) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(up) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(out) % 16 == 0;
  if (vector_path) {
    const int64_t vectors = total / 8;
    int blocks = static_cast<int>((vectors + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    moe_swiglu_scaled_vec8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint4*>(gate), static_cast<const uint4*>(up),
        static_cast<const float*>(weights), static_cast<uint4*>(out),
        columns / 8, experts, block_columns / 8, vectors);
    return cudaGetLastError();
  }
  int blocks = static_cast<int>((total + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  moe_swiglu_scaled_scalar_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<const float*>(weights), static_cast<__nv_bfloat16*>(out),
      columns, experts, block_columns, total);
  return cudaGetLastError();
}

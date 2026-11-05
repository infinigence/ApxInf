#include "../../kernels/activation/fixed_swiglu_bf16.cuh"

extern "C" cudaError_t apxinf_activation_gelu_exact_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  gelu_exact_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_activation_swiglu_bf16_rounded(
    const void* gate_up, void* output, int rows, int inner, cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535 ? 65535 : (count + 255) / 256);
  if (swiglu_vec8_ok(gate_up, output, inner)) {
    const int vblocks = static_cast<int>((count / 8 + 255) / 256 > 65535
                                             ? 65535
                                             : (count / 8 + 255) / 256);
    swiglu_bf16_vec8_kernel<true><<<vblocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
    return cudaGetLastError();
  }
  swiglu_bf16_kernel<true><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(output), rows, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_activation_swiglu_bf16_rounded_pair_interleaved(
    const void* gate_up, void* output, int rows, int inner, int pair,
    cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0 || pair <= 0 ||
      inner % pair != 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535
                                          ? 65535
                                          : (count + 255) / 256);
  swiglu_bf16_pair_interleaved_kernel<true><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up),
      static_cast<__nv_bfloat16*>(output), rows, inner, pair);
  return cudaGetLastError();
}


// Unrounded SwiGLU (the `RoundSilu=false` legacy form): the SiLU output feeds
// the multiply at full f32 precision. Distinct from the rounded variant above,
// which packed checkpoint path uses.
extern "C" cudaError_t apxinf_activation_swiglu_bf16(
    const void* gate_up, void* output, int rows, int inner, cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535 ? 65535 : (count + 255) / 256);
  if (swiglu_vec8_ok(gate_up, output, inner)) {
    const int vblocks = static_cast<int>((count / 8 + 255) / 256 > 65535
                                             ? 65535
                                             : (count / 8 + 255) / 256);
    swiglu_bf16_vec8_kernel<false><<<vblocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
    return cudaGetLastError();
  }
  swiglu_bf16_kernel<false><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(output), rows, inner);
  return cudaGetLastError();
}

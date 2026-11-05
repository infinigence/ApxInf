#include "../../kernels/rope/fixed_f16.cuh"

extern "C" cudaError_t apxinf_rope_half_split_f16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, cudaStream_t stream) {
  dim3 grid((head_dim / 2 + kThreads - 1) / kThreads, n_heads, seq_len);
  dim3 block(kThreads, 1, 1);
  rope_half_split_f16_kernel<<<grid, block, 0, stream>>>(
      (const half*)input, (half*)output,
      head_dim, n_heads, seq_len, rope_theta, pos_offset);
  return cudaGetLastError();
}

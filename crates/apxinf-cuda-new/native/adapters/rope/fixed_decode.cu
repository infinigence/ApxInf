#include "../../kernels/rope/fixed_decode.cuh"

extern "C" cudaError_t apxinf_rope_fixed_decode_decode_f32(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, n_heads, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    rope_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output,
        head_dim, n_heads, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_fixed_decode_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, n_heads, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    rope_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_fixed_decode_mrope_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, void* stream)
{
    dim3 grid((head_dim / 2 + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, n_heads, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    rope_mrope_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, theta,
        (const uint32_t*)pos_ids, sec_h, sec_w);
    return cudaGetLastError();
}

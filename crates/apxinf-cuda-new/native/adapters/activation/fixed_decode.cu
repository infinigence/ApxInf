#include "../../kernels/activation/fixed_decode.cuh"

extern "C" cudaError_t apxinf_activation_fixed_decode_silu_f32(
    const void* input, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    silu_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_activation_fixed_decode_silu_bf16(
    const void* input, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    silu_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_activation_fixed_decode_silu_mul_bf16(
    const void* gate_up, void* output, uint32_t inter, void* stream)
{
    dim3 grid((inter + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    silu_mul_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate_up, (__nv_bfloat16*)output, inter);
    return cudaGetLastError();
}

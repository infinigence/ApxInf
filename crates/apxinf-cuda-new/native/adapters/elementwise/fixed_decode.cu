#include "../../kernels/elementwise/fixed_decode.cuh"

extern "C" cudaError_t apxinf_elementwise_fixed_decode_add_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    add_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)a, (const float*)b, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_fixed_decode_add_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    // The legacy launcher's vec8 fast path is intentionally omitted: the
    // decode graph only adds [hidden]-sized vectors, where the scalar kernel
    // is launch-bound either way and the scalar path is the arithmetic the
    // fallback (f32-style) decode compares against.
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    add_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_fixed_decode_mul_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    mul_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)a, (const float*)b, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_fixed_decode_mul_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    mul_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_fixed_decode_scale_f32(
    const void* input, void* output, uint32_t count, float scale, void* stream)
{
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    scale_f32_kernel<<<grid, FIXED_DECODE_BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output, count, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_elementwise_fixed_decode_scale_bf16(
    const void* input, void* output, uint32_t count, float scale, void* stream)
{
    dim3 grid((count + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, 1, 1);
    scale_bf16_kernel<<<grid, FIXED_DECODE_BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, count, scale);
    return cudaGetLastError();
}

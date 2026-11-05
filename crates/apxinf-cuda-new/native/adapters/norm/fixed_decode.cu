#include "../../kernels/norm/fixed_decode.cuh"

extern "C" cudaError_t apxinf_norm_fixed_decode_rms_norm_f32(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    dim3 grid((cols + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, rows, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    rms_norm_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (const float*)weight, (float*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_fixed_decode_rms_norm_bf16(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    // One block per row. FIXED_DECODE_BLOCK_SIZE threads (256), strided over cols.
    dim3 grid(rows, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    size_t smem = cols * sizeof(float);
    rms_norm_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_fixed_decode_rms_norm_add_bf16(
    void* x_inout, const void* delta, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    dim3 grid(rows, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    size_t smem = cols * sizeof(float);
    rms_norm_add_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)x_inout, (const __nv_bfloat16*)delta,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

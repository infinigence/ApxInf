#include "../../kernels/embedding/fixed_decode.cuh"

extern "C" cudaError_t apxinf_embedding_fixed_decode_lookup_f32(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream)
{
    dim3 grid((embed_dim + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, seq_len, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    embedding_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)table, (const uint32_t*)ids, (float*)output,
        embed_dim, seq_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_embedding_fixed_decode_lookup_bf16(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream)
{
    dim3 grid((embed_dim + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, seq_len, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    embedding_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)table, (const uint32_t*)ids,
        (__nv_bfloat16*)output, embed_dim, seq_len);
    return cudaGetLastError();
}

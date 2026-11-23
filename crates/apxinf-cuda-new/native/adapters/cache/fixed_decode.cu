#include "../../kernels/cache/fixed_decode.cuh"

extern "C" cudaError_t apxinf_cache_fixed_decode_append_decode_f32(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    llmdec_kv_cache_append_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float*)cache, (const float*)new_data,
        n_kv_heads, head_dim, max_seq_len, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cache_fixed_decode_append_f32(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    uint32_t seq_len, uint32_t append_len, void* stream)
{
    dim3 grid((head_dim + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE,
              n_kv_heads, append_len);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    llmdec_kv_cache_append_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float*)cache, (const float*)new_data,
        n_kv_heads, head_dim, max_seq_len, seq_len, append_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cache_fixed_decode_append_bf16(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    uint32_t seq_len, uint32_t append_len, void* stream)
{
    dim3 grid((head_dim + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE,
              n_kv_heads, append_len);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    llmdec_kv_cache_append_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)cache, (const __nv_bfloat16*)new_data,
        n_kv_heads, head_dim, max_seq_len, seq_len, append_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cache_fixed_decode_append_decode_bf16(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    llmdec_kv_cache_append_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)cache, (const __nv_bfloat16*)new_data,
        n_kv_heads, head_dim, max_seq_len, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cache_fixed_decode_gather_bf16(
    const void* src, void* dst,
    uint32_t tokens, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_seq_len, uint32_t kv_offset, void* stream)
{
    uint32_t total = tokens * n_kv_heads * head_dim;
    uint32_t blocks = (total + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE;
    kv_cache_gather_bf16_kernel<<<blocks, FIXED_DECODE_BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst,
        tokens, n_kv_heads, head_dim, max_seq_len, kv_offset);
    return cudaGetLastError();
}

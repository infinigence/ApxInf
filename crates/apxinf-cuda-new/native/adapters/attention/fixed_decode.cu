#include "../../kernels/attention/fixed_decode.cuh"

extern "C" cudaError_t apxinf_attention_fixed_decode_softmax_f32(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset,
    uint32_t n_heads, void* stream)
{
    uint32_t n_col_blocks = (cols + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE;
    dim3 grid(n_col_blocks * rows, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    attention_softmax_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)scores, (float*)output, cols, rows, kv_offset, n_heads);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_fixed_decode_softmax_bf16(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset,
    uint32_t n_heads, void* stream)
{
    uint32_t n_col_blocks = (cols + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE;
    dim3 grid(n_col_blocks * rows, 1, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    attention_softmax_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
        cols, rows, kv_offset, n_heads);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_fixed_decode_softmax_decode_f32(
    const void* scores, void* output,
    uint32_t cols, uint32_t n_heads, const void* pos_ptr, void* stream)
{
    dim3 grid((cols + FIXED_DECODE_BLOCK_SIZE - 1) / FIXED_DECODE_BLOCK_SIZE, n_heads, 1);
    dim3 block(FIXED_DECODE_BLOCK_SIZE, 1, 1);
    attention_softmax_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)scores, (float*)output, cols, n_heads, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_fixed_decode_flash_attn_decode_bf16(
    const void* q, const void* k_cache, const void* v_cache, void* out,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t bucket_kv_len, uint32_t max_seq_len,
    float scale, const void* pos_ptr, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(n_heads, 1, 1);
    dim3 block(LLMDEC_SPLITK_WARPS * 32, 1, 1);
    if (head_dim == 64) {
        flash_attn_decode_bf16_splitk_kernel<64, LLMDEC_SPLITK_WARPS><<<grid, block, 0, s>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache, (__nv_bfloat16*)out,
            n_heads, n_kv_heads, bucket_kv_len, max_seq_len, scale,
            (const uint32_t*)pos_ptr);
    } else if (head_dim == 128) {
        flash_attn_decode_bf16_splitk_kernel<128, LLMDEC_SPLITK_WARPS><<<grid, block, 0, s>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache, (__nv_bfloat16*)out,
            n_heads, n_kv_heads, bucket_kv_len, max_seq_len, scale,
            (const uint32_t*)pos_ptr);
    } else {
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}

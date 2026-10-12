#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GR00T-family fused activation primitives. Copied verbatim from the legacy
   apxinf-cuda adapters with the apxinf_gr00t_ prefix, because the legacy crate
   links into the same binary and the legacy symbols are already present. */

cudaError_t apxinf_gr00t_silu_mul_separate_bf16(
    const void* gate, const void* up, void* output, uint32_t count,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_silu_mul_quant_bf16_e4m3(
    const void* gate, const void* up, void* output, int64_t count, float scale,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_gelu_quant_bf16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_gelu_bf16_packed8(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream);

/* Elementwise / row-selection primitives. Copied verbatim from the legacy
   static_bf16_adapter.cu with the apxinf_gr00t_ prefix. */

cudaError_t apxinf_gr00t_scatter_rows_bf16(
    const void* source, const void* rows, void* output, int row_count, int cols,
    int add, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_qkv_in_place_bf16(
    void* query, void* key, void* value, const void* query_bias,
    const void* key_bias, const void* value_bias, int rows, int cols,
    cudaStream_t stream);


cudaError_t apxinf_gr00t_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, uint32_t rows,
    uint32_t cols, float eps, cudaStream_t stream);

cudaError_t apxinf_gr00t_adaptive_layer_norm_quant_bf16_e4m3(
    const void* input, const void* modulation, void* output, void* quantized,
    uint32_t rows, uint32_t cols, float eps, float scale, cudaStream_t stream);

cudaError_t apxinf_gr00t_rms_norm_quant_bf16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream);

cudaError_t apxinf_gr00t_layer_norm_quant_bf16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream);


cudaError_t apxinf_gr00t_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_then_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);

cudaError_t
apxinf_gr00t_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* modulation, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_residual_layer_norm_quant_bf16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream);


cudaError_t apxinf_gr00t_rope_mrope_bf16(
    const void* input, void* output, uint32_t head_dim, uint32_t n_heads,
    uint32_t seq_len, float theta, const void* pos_ids, uint32_t sec_h,
    uint32_t sec_w, cudaStream_t stream);

cudaError_t apxinf_gr00t_build_vision_rotation_table_f32(
    const void* pos_ids, void* rotation_table, uint32_t head_dim,
    uint32_t seq_len, float theta, cudaStream_t stream);

cudaError_t apxinf_gr00t_rope_vision_2d_pair_bf16(
    const void* q, const void* k, void* q_out, void* k_out, uint32_t head_dim,
    uint32_t n_heads, uint32_t seq_len, float theta, const void* pos_ids,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_qk_rms_norm_mrope_bf16_with_threads(
    const void* query_input, const void* query_weight, void* query_output,
    const void* key_input, const void* key_weight, void* key_output,
    uint32_t head_dim, uint32_t query_heads, uint32_t key_heads,
    uint32_t seq_len, float eps, float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, uint32_t block_threads,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_qkv_split_bias_vision_rope_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len, float theta,
    const void* pos_ids, cudaStream_t stream);

cudaError_t apxinf_gr00t_qkv_split_bias_vision_rope_precomputed_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream);

cudaError_t apxinf_gr00t_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream);


cudaError_t apxinf_gr00t_split_strided_qkv_bf16(
    const void* qkv, void* q, void* k, void* v, int32_t tokens, int32_t hidden,
    cudaStream_t stream);


cudaError_t apxinf_gr00t_quantize_bf16_e4m3_packed8(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_concat_rows_quantize_bf16_e4m3(
    const void* first, const void* second, void* output, int32_t first_rows,
    int32_t second_rows, int32_t cols, float scale, cudaStream_t stream);


cudaError_t apxinf_gr00t_dequantize_int32_bf16(
    const void* accumulators, const void* row_scales, const void* column_scales,
    void* output, int32_t rows, int32_t cols, cudaStream_t stream);


cudaError_t apxinf_gr00t_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales, int32_t rows, int32_t cols,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_quantize_rows_bf16_int8_packed4(
    const void* input, void* output, void* scales, int32_t rows, int32_t cols,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_gelu_quantize_rows_bf16_int8(
    const void* input, const void* bias, void* output, void* scales,
    int32_t rows, int32_t cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_adaptive_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* modulation, void* output, void* quantized,
    void* scales, int32_t rows, int32_t cols, float eps, cudaStream_t stream);

cudaError_t apxinf_gr00t_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* weight, const void* bias, void* output,
    void* quantized, void* scales, int32_t rows, int32_t cols, float eps,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_silu_mul_quantize_rows_bf16_int8(
    const void* gate, const void* up, void* output, void* scales, int32_t rows,
    int32_t cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_silu_mul_quantize_rows_bf16_int8_packed4(
    const void* gate, const void* up, void* output, void* scales, int32_t rows,
    int32_t cols, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

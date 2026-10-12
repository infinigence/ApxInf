//! Raw bindings for the GR00T-family fused activation primitives, ported from
//! the legacy crate with the apxinf_gr00t_ prefix. Defined in
//! native/adapters/gr00t/ops.cu.

use std::ffi::c_void;

use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

unsafe extern "C" {
    pub(crate) fn apxinf_gr00t_silu_mul_separate_bf16(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_silu_mul_quant_bf16_e4m3(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        count: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_gelu_quant_bf16_e4m3(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_gelu_bf16_packed8(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_scatter_rows_bf16(
        source: *const c_void,
        rows: *const c_void,
        output: *mut c_void,
        row_count: i32,
        cols: i32,
        add: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_qkv_in_place_bf16(
        query: *mut c_void,
        key: *mut c_void,
        value: *mut c_void,
        query_bias: *const c_void,
        key_bias: *const c_void,
        value_bias: *const c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_adaptive_layer_norm_bf16(
        input: *const c_void,
        modulation: *const c_void,
        output: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_adaptive_layer_norm_quant_bf16_e4m3(
        input: *const c_void,
        modulation: *const c_void,
        output: *mut c_void,
        quantized: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_rms_norm_quant_bf16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_layer_norm_quant_bf16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_then_residual_bf16(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_residual_bf16_packed4(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_then_residual_bf16_packed4(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_residual_layer_norm_bf16_cached_1024(
        projection: *const c_void,
        projection_bias: *const c_void,
        residual: *const c_void,
        norm_weight: *const c_void,
        norm_bias: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_then_residual_layer_norm_bf16_cached_1536(
        projection: *const c_void,
        projection_bias: *const c_void,
        residual: *const c_void,
        norm_weight: *const c_void,
        norm_bias: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
        projection: *const c_void,
        projection_bias: *const c_void,
        residual: *const c_void,
        modulation: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_residual_layer_norm_quant_bf16_e4m3(
        projection: *const c_void,
        projection_bias: *const c_void,
        residual: *const c_void,
        norm_weight: *const c_void,
        norm_bias: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_rope_mrope_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        theta: f32,
        pos_ids: *const c_void,
        sec_h: u32,
        sec_w: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_build_vision_rotation_table_f32(
        pos_ids: *const c_void,
        rotation_table: *mut c_void,
        head_dim: u32,
        seq_len: u32,
        theta: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_rope_vision_2d_pair_bf16(
        q: *const c_void,
        k: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        theta: f32,
        pos_ids: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_qk_rms_norm_mrope_bf16_with_threads(
        query_input: *const c_void,
        query_weight: *const c_void,
        query_output: *mut c_void,
        key_input: *const c_void,
        key_weight: *const c_void,
        key_output: *mut c_void,
        head_dim: u32,
        query_heads: u32,
        key_heads: u32,
        seq_len: u32,
        eps: f32,
        theta: f32,
        pos_ids: *const c_void,
        sec_h: u32,
        sec_w: u32,
        block_threads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_qkv_split_bias_vision_rope_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        theta: f32,
        pos_ids: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_qkv_split_bias_vision_rope_precomputed_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rotation_table: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rotation_table: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_split_strided_qkv_bf16(
        qkv: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        hidden: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_quantize_bf16_e4m3_packed8(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_concat_rows_quantize_bf16_e4m3(
        first: *const c_void,
        second: *const c_void,
        output: *mut c_void,
        first_rows: i32,
        second_rows: i32,
        cols: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_dequantize_int32_bf16(
        accumulators: *const c_void,
        row_scales: *const c_void,
        column_scales: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_quantize_rows_bf16_int8(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_quantize_rows_bf16_int8_packed4(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_gelu_quantize_rows_bf16_int8(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_adaptive_layer_norm_quantize_rows_bf16_int8(
        input: *const c_void,
        modulation: *const c_void,
        output: *mut c_void,
        quantized: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_layer_norm_quantize_rows_bf16_int8(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        quantized: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_silu_mul_quantize_rows_bf16_int8(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_silu_mul_quantize_rows_bf16_int8_packed4(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

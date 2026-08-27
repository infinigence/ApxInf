//! Raw bindings for project-owned CUDA kernels and host adapters.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MarlinAwqU4G32V1Projection {
    pub marlin_qweight: *const c_void,
    pub scales_bf16: *const c_void,
    pub zero_points_u4: *const c_void,
    pub output: *mut c_void,
    pub padded_output: *mut c_void,
    pub logical_n: i32,
    pub padded_n: i32,
}

extern "C" {
    pub fn apxinf_static_evict_l2(
        buffer: *mut c_void,
        bytes: usize,
        seed: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_quantize_rows_bf16_int8(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;


    pub fn apxinf_static_dequantize_w4a16_asym_bf16(
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        dense: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_matmul_bf16_w4a16_asym(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        rows: i32,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Repack row-major AWQ U4 qweight into the standalone Marlin layout.
    pub fn apxinf_marlin_awq_u4_g32_v1_repack(
        awq_qweight: *const c_void,
        marlin_qweight: *mut c_void,
        padded_k: i32,
        padded_n: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Reconstruct one logical output-row tile in canonical compressed-tensors
    /// U4/BF16/U4 layout from the persistent Marlin representation.
    pub fn apxinf_marlin_awq_u4_g32_v1_inverse_raw_rows(
        marlin_qweight: *const c_void,
        scales_bf16: *const c_void,
        zero_points_u4: *const c_void,
        raw_qweight: *mut c_void,
        raw_scales_bf16: *mut c_void,
        raw_zero_points_u4: *mut c_void,
        logical_k: i32,
        padded_n: i32,
        padded_k: i32,
        source_n_offset: i32,
        row_start: i32,
        row_count: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Directly reconstruct logical row-major BF16 weights from Marlin layout.
    pub fn apxinf_marlin_awq_u4_g32_v1_dequant_bf16(
        marlin_qweight: *const c_void,
        scales_bf16: *const c_void,
        zero_points_u4: *const c_void,
        dense_bf16: *mut c_void,
        logical_n: i32,
        logical_k: i32,
        padded_n: i32,
        padded_k: i32,
        source_n_offset: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;


    /// BF16 x asymmetric U4 group-32 Marlin GEMM. The activation and output
    /// use physical strides `padded_k` and `padded_n`, respectively.
    pub fn apxinf_marlin_awq_u4_g32_v1_gemm_bf16(
        activation: *const c_void,
        marlin_qweight: *const c_void,
        scales_bf16: *const c_void,
        zero_points_u4: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        m: i32,
        logical_n: i32,
        logical_k: i32,
        padded_n: i32,
        padded_k: i32,
        sms: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Enqueue two to four independent Marlin projections sharing one physical
    /// activation and one sequentially reused lock workspace.
    pub fn apxinf_marlin_awq_u4_g32_v1_gemm_batch_bf16(
        activation: *const c_void,
        projections: *const MarlinAwqU4G32V1Projection,
        projection_count: i32,
        workspace: *mut c_void,
        m: i32,
        logical_k: i32,
        padded_k: i32,
        sms: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_conv_silu(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        state: *mut c_void,
        seq: i32,
        channels: i32,
        kernel: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_delta_norm_prepass(
        qkv: *const c_void,
        qk_out: *mut c_void,
        seq: i32,
        k_heads: i32,
        v_heads: i32,
        kdim: i32,
        vdim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_prepare_delta_step() -> cudaError_t;

    pub fn apxinf_qwen35_delta_step(
        qkv: *const c_void,
        qk_norm: *const c_void,
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        recurrent: *mut c_void,
        out: *mut c_void,
        seq: i32,
        k_heads: i32,
        v_heads: i32,
        kdim: i32,
        vdim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_prefill_delta_step(
        qkv: *const c_void,
        qk_norm: *const c_void,
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        recurrent: *mut c_void,
        out: *mut c_void,
        seq: i32,
        k_heads: i32,
        v_heads: i32,
        kdim: i32,
        vdim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_prepare_prefill_delta_step_4w(
        supported: *mut i32,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_prefill_delta_step_4w(
        qkv: *const c_void,
        qk_norm: *const c_void,
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        recurrent: *mut c_void,
        out: *mut c_void,
        seq: i32,
        k_heads: i32,
        v_heads: i32,
        kdim: i32,
        vdim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_prepare_prefill_delta_step_2w(
        supported: *mut i32,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_prefill_delta_step_2w(
        qkv: *const c_void,
        qk_norm: *const c_void,
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        recurrent: *mut c_void,
        out: *mut c_void,
        seq: i32,
        k_heads: i32,
        v_heads: i32,
        kdim: i32,
        vdim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_prepare_norm_delta_gated(supported: *mut i32) -> cudaError_t;

    pub fn apxinf_qwen35_norm_delta_gated(
        qkv: *const c_void,
        qk_out: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        z: *const c_void,
        norm_weight: *const c_void,
        recurrent: *mut c_void,
        delta_out: *mut c_void,
        out: *mut c_void,
        seq: i32,
        k_heads: i32,
        v_heads: i32,
        kdim: i32,
        vdim: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_prepare_packed_delta_gated(supported: *mut i32) -> cudaError_t;

    pub fn apxinf_qwen35_packed_delta_gated(
        qkv: *const c_void,
        conv_weight: *const c_void,
        conv_state: *mut c_void,
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        z: *const c_void,
        norm_weight: *const c_void,
        recurrent: *mut c_void,
        out: *mut c_void,
        seq: i32,
        k_heads: i32,
        v_heads: i32,
        kdim: i32,
        vdim: i32,
        conv_kernel: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gated_norm(
        input: *const c_void,
        z: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        seq: i32,
        v_heads: i32,
        vdim: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_q_split_norm_rope(
        q_gate: *const c_void,
        q_norm_w: *const c_void,
        q_out: *mut c_void,
        gate_out: *mut c_void,
        seq: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
        start_pos: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_k_norm_rope_append(
        k_in: *const c_void,
        k_norm_w: *const c_void,
        k_cache: *mut c_void,
        seq: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
        start_pos: u32,
        max_seq_len: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_qk_norm_rope_append(
        q_gate: *const c_void,
        q_norm_w: *const c_void,
        k_in: *const c_void,
        k_norm_w: *const c_void,
        q_out: *mut c_void,
        gate_out: *mut c_void,
        k_cache: *mut c_void,
        seq: i32,
        heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
        position: *const u32,
        max_seq_len: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_sigmoid_mul(
        gate: *const c_void,
        x: *const c_void,
        out: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_flash_prefill(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        out: *mut c_void,
        seq: i32,
        heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        scale: f32,
        start_pos: u32,
        max_seq_len: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_flash_decode_gated_256(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        scale: f32,
        position: *const u32,
        max_seq_len: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_dequant_w4a16_bf16_rows(
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        dense: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        row_start: i32,
        row_count: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_flash_decode_gated_256_split(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        partials: *mut c_void,
        scale: f32,
        position: *const u32,
        max_seq_len: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_flash_decode_gated_256_split_2w(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        partials: *mut c_void,
        scale: f32,
        position: *const u32,
        max_seq_len: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_flash_decode_gated_256_split_1w(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        partials: *mut c_void,
        scale: f32,
        position: *const u32,
        max_seq_len: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_flash_decode_gated_256_gqa(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        partials: *mut c_void,
        scale: f32,
        position: *const u32,
        max_seq_len: i32,
        group: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_dequant_w4a16_bf16_pair_rows(
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        dense0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        dense1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_silu_mul(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_dequantize_int32_bf16(
        accumulators: *const c_void,
        row_scales: *const c_void,
        column_scales: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_quantize_f16_e4m3(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Decode E4M3 values to real-range FP16 by applying the tensor scale.
    /// This avoids overflowing FP16 Tensor Core products on devices which
    /// emulate FP8 GEMM.
    pub fn apxinf_static_dequantize_e4m3_f16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_rgb_u8_to_patches_e4m3(
        images: *const c_void,
        patches: *mut c_void,
        views: i32,
        image_size: i32,
        patch_size: i32,
        layout: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_mqa_flash_f16(
        q: *const c_void,
        prefix_k: *const c_void,
        prefix_v: *const c_void,
        suffix_k: *const c_void,
        suffix_v: *const c_void,
        output: *mut c_void,
        suffix_tokens: i32,
        heads: i32,
        head_dim: i32,
        prefix_tokens: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_rms_norm_quant_f16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_layer_norm_quant_f16_e4m3(
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
    pub fn apxinf_static_bias_gelu_quant_f16_e4m3(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_silu_quant_f16_e4m3(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_silu_f16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_f16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_embedding_f16(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        tokens: i32,
        width: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_concat_rows_f16(
        first: *const c_void,
        second: *const c_void,
        output: *mut c_void,
        first_rows: i32,
        second_rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_euler_update_f16(
        state: *const c_void,
        velocity: *const c_void,
        output: *mut c_void,
        count: i64,
        dt: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_geglu_quant_f16_e4m3(
        gate_up: *const c_void,
        output: *mut c_void,
        rows: i32,
        inner: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_residual_f16(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_residual_rms_norm_quant_f16_e4m3(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_residual_layer_norm_quant_f16_e4m3(
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
    pub fn apxinf_static_ada_rms_norm_quant_f16_e4m3(
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_ada_gate_residual_f16(
        projection: *const c_void,
        residual: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_ada_gate_residual_rms_norm_quant_f16_e4m3(
        projection: *const c_void,
        residual: *const c_void,
        gate_style: *const c_void,
        norm_style: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_qkv_rope_f16(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        theta: f32,
        position_offset: i32,
        kv_output_offset: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_qkv_split_bias_f16(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        projection_width: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_mha_flash_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        tokens_per_batch: i32,
        batches: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_position_f16(
        projection: *const c_void,
        bias: *const c_void,
        position: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        tokens_per_view: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_static_rgb_u8_to_patches_bf16(
        images: *const c_void,
        patches: *mut c_void,
        views: i32,
        image_size: i32,
        patch_size: i32,
        layout: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    /// BF16 bias/activation epilogue. `activation`: 0=identity, 1=GELU-tanh,
    /// 2=SiLU.
    pub fn apxinf_static_bias_activation_bf16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        activation: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_embedding_bf16(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        tokens: i32,
        width: i32,
        vocab_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_concat_rows_bf16(
        first: *const c_void,
        second: *const c_void,
        output: *mut c_void,
        first_rows: i32,
        second_rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_euler_update_bf16(
        state: *const c_void,
        velocity: *const c_void,
        output: *mut c_void,
        count: i64,
        dt: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_geglu_bf16(
        gate_up: *const c_void,
        output: *mut c_void,
        rows: i32,
        inner: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_residual_bf16(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_rms_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_layer_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_residual_rms_norm_bf16(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_residual_layer_norm_bf16(
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
    pub fn apxinf_static_ada_rms_norm_bf16(
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_ada_gate_residual_bf16(
        projection: *const c_void,
        residual: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_ada_gate_residual_rms_norm_bf16(
        projection: *const c_void,
        residual: *const c_void,
        gate_style: *const c_void,
        norm_style: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_qkv_rope_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        theta: f32,
        position_offset: i32,
        kv_output_offset: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_qkv_split_bias_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        projection_width: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_mqa_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        query_tokens: i32,
        key_tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_mha_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        tokens_per_batch: i32,
        batches: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_bias_position_bf16(
        projection: *const c_void,
        bias: *const c_void,
        position: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        tokens_per_view: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rms_norm_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_silu_f32(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_silu_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_silu_mul_bf16(
        gate_up: *const c_void,
        output: *mut c_void,
        inter: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_softmax_f32(
        input: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_f32(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rope_theta: f32,
        pos_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_add_f32(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_mul_f32(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_embedding_f32(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        embed_dim: u32,
        seq_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_causal_mask_f32(
        input: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    // ── Async kernel launchers (no cudaStreamSynchronize) ────────────────

    pub fn apxinf_rope_batched_f32(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rope_theta: f32,
        pos_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_attention_softmax_f32(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        n_heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_kv_cache_append_f32(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        seq_len: u32,
        append_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_scale_f32(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    // ── Decode kernels reading pos from a device pointer (graph-safe) ──────

    pub fn apxinf_rope_decode_f32(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_attention_softmax_decode_f32(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        n_heads: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_kv_cache_append_decode_f32(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_rms_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rms_norm_add_bf16(
        x_inout: *mut c_void,
        delta: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_softmax_bf16(
        input: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rope_theta: f32,
        pos_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_add_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_mul_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_embedding_bf16(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        embed_dim: u32,
        seq_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_causal_mask_bf16(
        input: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_batched_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rope_theta: f32,
        pos_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_attention_softmax_bf16(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        n_heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_kv_cache_append_bf16(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        seq_len: u32,
        append_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_scale_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_decode_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_attention_softmax_decode_bf16(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        n_heads: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_kv_cache_append_decode_bf16(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_k_write_bf16(
        k_in: *const c_void,
        k_cache: *mut c_void,
        head_dim: u32,
        n_kv_heads: u32,
        max_seq_len: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_flash_attn_decode_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        out: *mut c_void,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        bucket_kv_len: u32,
        max_seq_len: u32,
        scale: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_mrope_bf16(
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

    /// Exact single-launch argmax over `[n]` bf16 logits.
    pub fn apxinf_argmax_bf16_single_launch(
        logits: *const c_void,
        n: u32,
        partials: *mut c_void,
        partial_capacity: u32,
        arrivals: *mut c_void,
        out: *mut c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Exact multi-block argmax over `[n]` bf16 logits.
    pub fn apxinf_argmax_bf16_parallel(
        logits: *const c_void,
        n: u32,
        partials: *mut c_void,
        partial_capacity: u32,
        out: *mut c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_mrope_decode_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        theta: f32,
        pos_ids: *const c_void,
        sec_h: u32,
        sec_w: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_layer_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_gelu_tanh_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_add_bias_bf16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_vision_2d_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        theta: f32,
        pos_ids: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_vision_sdpa_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_attention_softmax_rows(
        scores: *const c_void,
        l_out: *mut c_void,
        head_base: i32,
        seq: i32,
        heads: i32,
        visible: i32,
        row_stride: i32,
        start_pos: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_scale_out(
        pv: *const c_void,
        l: *const c_void,
        out: *mut c_void,
        seq: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_scale_out_gated(
        pv: *const c_void,
        l: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        seq: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_transpose_kt(
        k: *const c_void,
        kt: *mut c_void,
        visible: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_multi(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        weight_packed2: *const c_void,
        weight_scale2: *const c_void,
        weight_zero_point2: *const c_void,
        output2: *mut c_void,
        out_cols2: i32,
        weight_packed3: *const c_void,
        weight_scale3: *const c_void,
        weight_zero_point3: *const c_void,
        output3: *mut c_void,
        out_cols3: i32,
        projection_count: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_qkv_10240x5120(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_qkv_sched(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        mode: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_gate_up_silu(
        activation: *const c_void,
        gate_packed: *const c_void,
        gate_scale: *const c_void,
        gate_zero_point: *const c_void,
        up_packed: *const c_void,
        up_scale: *const c_void,
        up_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_cache_hint(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_store_alt_single(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_vector_mma(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;


    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_scale_epilogue(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_tile_alt(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_meta_shared(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_persistent(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_repacked_v1(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        padded_out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_w4_transform_cache(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        padded_out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_occupancy(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_store_alt(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_coarsen(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_warp(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_reg(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_vector_mma(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_meta(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_weight_stage(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_alt(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_6w(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_prefetch(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_act(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_shared(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_reuse(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_cache(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_tc_pair_2w(
        activation: *const c_void,
        weight_packed0: *const c_void,
        weight_scale0: *const c_void,
        weight_zero_point0: *const c_void,
        output0: *mut c_void,
        out_cols0: i32,
        weight_packed1: *const c_void,
        weight_scale1: *const c_void,
        weight_zero_point1: *const c_void,
        output1: *mut c_void,
        out_cols1: i32,
        in_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_prefill_fast_raw(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        rows: i32,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// Raw group-32 W4 prefill candidate. `rows` accepts 2..=2048; the CUDA
    /// adapter rejects unsupported geometry without changing the public ABI.
    pub fn apxinf_qwen35_gemm_w4a16_bf16_prefill_native(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        rows: i32,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_prefill_fast_repacked_v1(
        activation: *const c_void,
        weight_qwords: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        rows: i32,
        in_cols: i32,
        out_cols: i32,
        padded_out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_prefill_packed_raw(
        activation: *const c_void,
        weight_packed: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        rows: i32,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_prefill_packed_repacked_v1(
        activation: *const c_void,
        weight_qwords: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        rows: i32,
        in_cols: i32,
        out_cols: i32,
        padded_out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_gemm_w4a16_bf16_prefill_repacked_v1(
        activation: *const c_void,
        weight_qwords: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        output: *mut c_void,
        rows: i32,
        in_cols: i32,
        out_cols: i32,
        padded_out_cols: i32,
        groups: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_dequant_w4a16_bf16_rows_repacked_v1(
        weight_qwords: *const c_void,
        weight_scale: *const c_void,
        weight_zero_point: *const c_void,
        dense: *mut c_void,
        in_cols: i32,
        out_cols: i32,
        groups: i32,
        row_start: i32,
        row_count: i32,
        padded_out_cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_qwen35_v_to_f32(
        v: *const c_void,
        vf32: *mut c_void,
        visible: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

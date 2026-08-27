//! Raw bindings for project-owned CUDA kernels and host adapters.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    pub fn apxinf_static_evict_l2(
        buffer: *mut c_void,
        bytes: usize,
        seed: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_token_sampling_workspace_sizes(
        vocab_size: u32,
        sort_bytes: *mut usize,
        scan_bytes: *mut usize,
    ) -> cudaError_t;

    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_sample_token(
        logits: *const c_void,
        dtype: i32,
        vocab_size: u32,
        counts: *mut u32,
        repetition: f32,
        frequency: f32,
        presence: f32,
        selection: i32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
        sequence: u64,
        draw: u64,
        return_logprob: u32,
        adjusted: *mut f32,
        token_ids: *mut u32,
        sorted_logits: *mut f32,
        sorted_tokens: *mut u32,
        weights: *mut f32,
        cdf: *mut f32,
        partial_values: *mut f32,
        partial_tokens: *mut u32,
        partial_count: u32,
        sort_workspace: *mut c_void,
        sort_workspace_bytes: usize,
        scan_workspace: *mut c_void,
        scan_workspace_bytes: usize,
        output: *mut c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_fill_standard_normal(
        output: *mut c_void,
        dtype: i32,
        count: u64,
        seed: u64,
        sequence: u64,
        draw: u64,
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

    pub fn apxinf_embedding_scaled_bf16(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        tokens: i32,
        width: i32,
        vocab_size: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rms_norm_affine_bf16(
        input: *const c_void,
        raw_weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        weight_offset: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_precomputed_bf16(
        input: *const c_void,
        cosine: *const c_void,
        sine: *const c_void,
        output: *mut c_void,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_style_gate_mul_bf16(
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_sinusoidal_time_embedding_bf16(
        times: *const c_void,
        output: *mut c_void,
        steps: i32,
        dimension: i32,
        min_period: f32,
        max_period: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_rope_tables_bf16(
        positions: *const c_void,
        cosine: *mut c_void,
        sine: *mut c_void,
        tokens: i32,
        head_dim: i32,
        theta: f32,
        linear_factor: f32,
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

    /// Argmax over [n] bf16 logits → writes the winning index to `out` (u32).
    /// Typically `out` is a host-mapped u32 so the CPU reads it zero-copy.
    pub fn apxinf_argmax_bf16(
        logits: *const c_void,
        n: u32,
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
}

//! Raw bindings for the linear-attention / hybrid-recurrent custom operators.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    pub fn apxinf_static_cast_f32_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_cast_bf16_f32(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    // FIX (implement_r8): tiled block-per-row softmax (fp32 in, bf16 out) for the
    // route-3 composed vision attention; defined in adapters/custom_kernels.cu.
    pub fn apxinf_static_row_softmax_f32_bf16(
        input: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_causal_conv1d_silu_bf16(
        x: *const c_void,
        weight: *const c_void,
        state: *const c_void,
        out: *mut c_void,
        new_state: *mut c_void,
        channels: i32,
        seq: i32,
        kernel_size: i32,
        x_row_stride: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_gdn_qk_prep_bf16(
        conv_out: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        seq: i32,
        seq_pad: i32,
        conv_dim: i32,
        key_dim: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        scale: f32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_gdn_vb_prep_bf16(
        conv_out: *const c_void,
        b_proj: *const c_void,
        a_proj: *const c_void,
        dt_bias: *const c_void,
        a_log: *const c_void,
        v_out: *mut c_void,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        seq: i32,
        seq_pad: i32,
        conv_dim: i32,
        v_offset: i32,
        num_v_heads: i32,
        ba_row_stride: i32,
        head_v_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_gdn_cumsum_f32(
        g: *const c_void,
        g_cum: *mut c_void,
        seq_pad: i32,
        num_v_heads: i32,
        chunk_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_gdn_attn_raw_f32(
        q: *const c_void,
        k: *const c_void,
        beta: *const c_void,
        g_cum: *const c_void,
        a_out: *mut c_void,
        t_out: *mut c_void,
        seq_pad: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        chunk_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_gdn_tri_solve_f32(
        a: *mut c_void,
        matrices: i32,
        chunk_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_gdn_chunk_gemm_f32(
        a: *const c_void,
        v: *const c_void,
        k: *const c_void,
        beta: *const c_void,
        g_cum: *const c_void,
        vt_out: *mut c_void,
        kcd_out: *mut c_void,
        seq_pad: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        chunk_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_gdn_chunk_state_f32(
        q: *const c_void,
        k: *const c_void,
        g_cum: *const c_void,
        t_in: *const c_void,
        vt_in: *const c_void,
        kcd_in: *const c_void,
        state: *mut c_void,
        out: *mut c_void,
        seq: i32,
        seq_pad: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        chunk_size: i32,
        total_chunks: i32,
        out_row_width: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_gdn_recurrent_f32(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        beta: *const c_void,
        g: *const c_void,
        state: *mut c_void,
        out: *mut c_void,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_gated_rms_silu_bf16(
        x: *const c_void,
        z: *const c_void,
        weight: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
        z_heads: i32,
        z_row_stride: i64,
        z_col_offset: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_rms_norm_plus1_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_partial_rope_table_bf16(
        x: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        out: *mut c_void,
        rows: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_full_attn_prepare_bf16(
        fused: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        q_out: *mut c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        seq: i32,
        cache_offset: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        fused_width: i64,
        cache_width: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_sigmoid_gate_mul_bf16(
        attn: *mut c_void,
        fused: *const c_void,
        rows: i32,
        heads: i32,
        head_dim: i32,
        fused_width: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_adaln_rms_norm_bf16(
        x: *const c_void,
        weight: *const c_void,
        scale: *const c_void,
        shift: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_adaln_gate_residual_bf16(
        proj: *const c_void,
        residual: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        count: i64,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_expert_qkv_prepare_bf16(
        fused: *const c_void,
        q_norm_w: *const c_void,
        k_norm_w: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        q_out: *mut c_void,
        gate_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        seq: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        fused_width: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_expert_sigmoid_gate_mul_bf16(
        attn: *mut c_void,
        gate: *const c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_fourier_features_bf16(
        waypoints: *const c_void,
        freqs: *const c_void,
        out: *mut c_void,
        rows: i32,
        point_dim: i32,
        num_features: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_static_concat7_cols_bf16(
        s0: *const c_void,
        s1: *const c_void,
        s2: *const c_void,
        s3: *const c_void,
        s4: *const c_void,
        s5: *const c_void,
        s6: *const c_void,
        dst: *mut c_void,
        rows: i32,
        cols: i32,
        broadcast_mask: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_flow_update_f32(
        w: *mut c_void,
        endpoint: *const c_void,
        remaining: f32,
        step: f32,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_suppress_logits_bf16(
        logits: *mut c_void,
        ids: *const u32,
        count: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_gelu_exact_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_static_merge_rows_bf16(
        input: *const c_void,
        output: *mut c_void,
        out_count: i64,
        cols: i32,
        factor: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

//! Raw bindings for the linear-attention / hybrid-recurrent custom operators
//! (qwen_drive family), ported from the legacy crate with the apxinf_cn_ prefix.

use std::ffi::c_void;

use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};
use crate::kernels::gdn_policy::GdnLaunchPolicy;

unsafe extern "C" {
    pub(crate) fn apxinf_cn_broadcast_bf16_f32_rows(
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_cast_f32_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_cast_bf16_f32(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    // FIX (implement_r8): tiled block-per-row softmax (fp32 in, bf16 out) for the
    // route-3 composed vision attention; defined in adapters/custom_kernels.cu.
    pub(crate) fn apxinf_cn_row_softmax_f32_bf16(
        input: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    // FIX (implement_final_r20): tiled block-per-row CAUSAL fp32 softmax (fp32 in, fp32
    // out, in-place safe) for the Option A composed-causal budget repair; defined in
    // adapters/custom_kernels.cu with the attention.cuh row = seq_pos*n_heads + head
    // contract (valid_cols = min(seq_pos + kv_offset + 1, cols), masked cells exact 0.0f).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_row_softmax_causal_f32(
        input: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        n_heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_causal_conv1d_silu_bf16(
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
    pub(crate) fn apxinf_cn_gdn_qk_prep_bf16(
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
        recurrent: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_qk_prep_qk_bf16(
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
        recurrent: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_vb_prep_bf16(
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
    pub(crate) fn apxinf_cn_gdn_gate_prep_bf16(
        b_proj: *const c_void,
        a_proj: *const c_void,
        dt_bias: *const c_void,
        a_log: *const c_void,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        seq: i32,
        seq_pad: i32,
        num_v_heads: i32,
        ba_row_stride: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_gdn_cumsum_f32(
        g: *const c_void,
        g_cum: *mut c_void,
        seq_pad: i32,
        num_v_heads: i32,
        chunk_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_attn_raw_f32(
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
        policy: *const GdnLaunchPolicy,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_attn_raw_solve_f1_qk_bf16(
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_tri_solve_f32(
        a: *mut c_void,
        matrices: i32,
        chunk_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_chunk_gemm_f32(
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
        policy: *const GdnLaunchPolicy,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_chunk_gemm_tri_k_bf16_direct_v(
        a: *const c_void,
        conv_out: *const c_void,
        k: *const c_void,
        beta: *const c_void,
        g_cum: *const c_void,
        vt_out: *mut c_void,
        kcd_out: *mut c_void,
        seq: i32,
        seq_pad: i32,
        conv_dim: i32,
        v_offset: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        chunk_size: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_chunk_state_f32(
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
        policy: *const GdnLaunchPolicy,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gdn_chunk_state_qk_bf16(
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
    pub(crate) fn apxinf_cn_gdn_recurrent_f32(
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
        policy: *const GdnLaunchPolicy,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_gated_rms_silu_bf16(
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_add_rms_norm_plus1_bf16(
        a: *const c_void,
        b: *const c_void,
        weight: *const c_void,
        sum_out: *mut c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_rms_norm_plus1_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_full_attn_prepare_bf16(
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
    pub(crate) fn apxinf_cn_sigmoid_gate_mul_bf16(
        attn: *mut c_void,
        fused: *const c_void,
        rows: i32,
        heads: i32,
        head_dim: i32,
        fused_width: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_adaln_rms_norm_bf16(
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
    pub(crate) fn apxinf_cn_adaln_gate_residual_bf16(
        proj: *const c_void,
        residual: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        count: i64,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_expert_qkv_prepare_bf16(
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
    pub(crate) fn apxinf_cn_expert_sigmoid_gate_mul_bf16(
        attn: *mut c_void,
        gate: *const c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_fourier_features_bf16(
        waypoints: *const c_void,
        freqs: *const c_void,
        out: *mut c_void,
        rows: i32,
        point_dim: i32,
        num_features: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_cn_concat7_cols_bf16(
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
    pub(crate) fn apxinf_cn_flow_update_f32(
        w: *mut c_void,
        endpoint: *const c_void,
        remaining: f32,
        step: f32,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_suppress_logits_bf16(
        logits: *mut c_void,
        ids: *const u32,
        count: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_gelu_exact_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_sinusoidal_embedding_bf16(
        positions: *const c_void,
        output: *mut c_void,
        rows: i32,
        dim: i32,
        scale: f32,
        frequency_step: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_rgb_u8_to_temporal2_merge2_rect_bf16(
        rgb: *const c_void,
        patches: *mut c_void,
        lut: *const c_void,
        grid_h: i32,
        grid_w: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_pillow_bicubic_u8_axis(
        input: *const c_void,
        output: *mut c_void,
        input_offsets: *const c_void,
        output_offsets: *const c_void,
        bounds: *const c_void,
        weights: *const c_void,
        ksize: i32,
        in_w: i32,
        in_h: i32,
        out_w: i32,
        out_h: i32,
        batch: i32,
        horizontal: bool,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_adaln_gate_residual_rms_bf16(
        projection: *const c_void,
        residual: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        scale: *const c_void,
        shift: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cn_swiglu_bf16_rounded(
        gate_up: *const c_void,
        output: *mut c_void,
        rows: i32,
        inner: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

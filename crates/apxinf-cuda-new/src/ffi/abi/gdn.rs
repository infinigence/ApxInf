use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_gdn_recurrent_step(
        state: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        decay: *const c_void,
        beta: *const c_void,
        output: *mut c_void,
        v_heads: i64,
        k_heads: i64,
        v_dim: i64,
        k_dim: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_gated_norm(
        input: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_causal_conv_step(
        window: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        channels: i64,
        kernel_width: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_widen_f16_to_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_prepare_flashinfer(
        fused: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        g: *const c_void,
        alpha: *mut c_void,
        tokens: i64,
        row_width: i64,
        k_heads: i64,
        v_heads: i64,
        dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_flashinfer_gdn_prefill(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        gate_log: *const c_void,
        beta: *const c_void,
        cu_seqlens: *const c_void,
        state: *mut c_void,
        tensor_map_workspace: *mut c_void,
        tokens: i64,
        q_heads: i64,
        v_heads: i64,
        num_seqs: i64,
        scale: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_flashinfer_gdn_workspace_bytes(
        v_heads: i64,
        num_seqs: i64,
    ) -> i64;
    pub(crate) fn apxinf_gdn_causal_conv_forward(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        window: *mut c_void,
        tokens: i64,
        channels: i64,
        kernel_width: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_decay_and_beta_seq(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        decay: *mut c_void,
        beta: *mut c_void,
        tokens: i64,
        heads: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_gated_norm_seq(
        input: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_l2_normalize_heads(
        data: *mut c_void,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_decay_and_beta(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        decay: *mut c_void,
        beta: *mut c_void,
        heads: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_chunk_scan(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        out: *mut c_void,
        state: *mut c_void,
        seq_padded: i64,
        v_heads: i64,
        k_heads: i64,
        chunk_size: i64,
        k_dim: i64,
        num_chunks: i64,
        q_row_stride: i64,
        k_row_stride: i64,
        v_row_stride: i64,
        stream: CudaStream,
    ) -> i32;
}

use std::ffi::c_void;

use super::types::{CudaStream, Runtime};

pub(crate) const SPEC_VERSION: u32 = 1;

pub(crate) mod operation {
    pub const RECURRENT_STEP: u32 = 0;
    pub const GATED_NORM: u32 = 1;
    pub const CAUSAL_CONV: u32 = 2;
    pub const DECAY_AND_BETA: u32 = 3;
    pub const L2_NORMALIZE: u32 = 4;
    pub const CHUNK_SCAN: u32 = 5;
    pub const PREFILL: u32 = 6;
    pub const PREPARE_PREFILL: u32 = 7;
    pub const CONV_PREPARE: u32 = 8;
}

pub(crate) mod layout {
    pub const DENSE: u32 = 0;
    pub const STRIDED_GATE: u32 = 1;
    pub const SPLIT_QKV: u32 = 2;
    pub const INTERLEAVED_QKV: u32 = 3;
}

pub(crate) mod quantization {
    pub const NONE: u32 = 0;
    pub const STATIC_E4M3: u32 = 1;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Spec {
    pub version: u32,
    pub operation: u32,
    pub input_dtype: u32,
    pub output_dtype: u32,
    pub rank: u32,
    pub layout: u32,
    pub quantization: u32,
    pub has_window: u32,
    pub tokens: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub key_heads: i64,
    pub value_heads: i64,
    pub key_dim: i64,
    pub value_dim: i64,
    pub channels: i64,
    pub kernel_width: i64,
    pub row_width: i64,
    pub num_sequences: i64,
    pub chunk_size: i64,
    pub num_chunks: i64,
    pub q_row_stride: i64,
    pub k_row_stride: i64,
    pub v_row_stride: i64,
    pub gate_row_stride: i64,
    pub gate_column_offset: i64,
}

impl Spec {
    pub(crate) fn new(operation: u32, rank: u32) -> Self {
        Self {
            version: SPEC_VERSION,
            operation,
            input_dtype: 0,
            output_dtype: 0,
            rank,
            layout: layout::DENSE,
            quantization: quantization::NONE,
            has_window: 0,
            tokens: 0,
            heads: 0,
            head_dim: 0,
            key_heads: 0,
            value_heads: 0,
            key_dim: 0,
            value_dim: 0,
            channels: 0,
            kernel_width: 0,
            row_width: 0,
            num_sequences: 0,
            chunk_size: 0,
            num_chunks: 0,
            q_row_stride: 0,
            k_row_stride: 0,
            v_row_stride: 0,
            gate_row_stride: 0,
            gate_column_offset: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub gate: *const c_void,
    pub weight: *const c_void,
    pub q: *const c_void,
    pub k: *const c_void,
    pub v: *const c_void,
    pub decay: *mut c_void,
    pub beta: *mut c_void,
    pub a: *const c_void,
    pub b: *const c_void,
    pub a_log: *const c_void,
    pub dt_bias: *const c_void,
    pub gate_log: *const c_void,
    pub cu_seqlens: *const c_void,
    pub output: *mut c_void,
    pub state: *mut c_void,
    pub window: *mut c_void,
    pub q_output: *mut c_void,
    pub k_output: *mut c_void,
    pub v_output: *mut c_void,
    pub alpha: *mut c_void,
    pub quantized: *mut c_void,
    pub workspace: *mut c_void,
    pub stream: CudaStream,
    pub epsilon: f32,
    pub scale: f32,
    pub input_scale: f32,
}

impl Bindings {
    pub(crate) fn new(stream: CudaStream) -> Self {
        Self {
            input: std::ptr::null(),
            gate: std::ptr::null(),
            weight: std::ptr::null(),
            q: std::ptr::null(),
            k: std::ptr::null(),
            v: std::ptr::null(),
            decay: std::ptr::null_mut(),
            beta: std::ptr::null_mut(),
            a: std::ptr::null(),
            b: std::ptr::null(),
            a_log: std::ptr::null(),
            dt_bias: std::ptr::null(),
            gate_log: std::ptr::null(),
            cu_seqlens: std::ptr::null(),
            output: std::ptr::null_mut(),
            state: std::ptr::null_mut(),
            window: std::ptr::null_mut(),
            q_output: std::ptr::null_mut(),
            k_output: std::ptr::null_mut(),
            v_output: std::ptr::null_mut(),
            alpha: std::ptr::null_mut(),
            quantized: std::ptr::null_mut(),
            workspace: std::ptr::null_mut(),
            stream,
            epsilon: 0.0,
            scale: 0.0,
            input_scale: 0.0,
        }
    }
}

unsafe extern "C" {
    pub(crate) fn apxinf_gdn_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
    pub(crate) fn apxinf_gdn_prepare_fixed(
        device: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        chunk_size: i32,
    ) -> i32;
    pub(crate) fn apxinf_gdn_qk_bf16_supported(
        device: i32,
        recurrent: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        chunk_size: i32,
    ) -> i32;
    pub(crate) fn apxinf_gdn_chunk_scan_prepare(device: i32) -> i32;
    pub(crate) fn apxinf_gdn_prefill_workspace_bytes(v_heads: i64, num_seqs: i64) -> i64;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        #[cfg(apxinf_segmented_attention_sm110)]
        pub(crate) fn apxinf_gdn_bf16_prefill_sm110_init() -> i32;
        #[cfg(apxinf_segmented_attention_sm110)]
        pub(crate) fn apxinf_gdn_bf16_prefill_sm110(
            zba: *const c_void,
            conv_weight: *const c_void,
            new_conv_state: *mut c_void,
            a_log: *const f32,
            dt_bias_bf16: *const c_void,
            raw_qk_scratch: *mut c_void,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            a: *mut c_void,
            b: *mut c_void,
            h_output_vmajor: *mut f32,
            final_state_kmajor: *mut f32,
            output_pad: *mut c_void,
            cu_seqlens: *mut i32,
            seq: i32,
            seq_pad: i32,
            stream: cudaStream_t,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_causal_conv1d_silu_bf16(
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
        pub(crate) fn apxinf_gdn_qk_prep_bf16(
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
        pub(crate) fn apxinf_gdn_qk_prep_qk_bf16(
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
        pub(crate) fn apxinf_gdn_vb_prep_bf16(
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
        pub(crate) fn apxinf_gdn_gate_prep_bf16(
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
        pub(crate) fn apxinf_gdn_cumsum_f32(
            g: *const c_void,
            g_cum: *mut c_void,
            seq_pad: i32,
            num_v_heads: i32,
            chunk_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_attn_raw_f32(
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
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_attn_raw_solve_f1_qk_bf16(
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
        pub(crate) fn apxinf_gdn_tri_solve_f32(
            a: *mut c_void,
            matrices: i32,
            chunk_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_chunk_gemm_f32(
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
            device: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_chunk_gemm_tri_k_bf16_direct_v(
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
        pub(crate) fn apxinf_gdn_chunk_state_f32(
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
            device: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_chunk_state_qk_bf16(
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
        pub(crate) fn apxinf_gdn_recurrent_f32(
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
            device: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;

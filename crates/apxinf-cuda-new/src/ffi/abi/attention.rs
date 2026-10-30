use std::ffi::{c_char, c_void};

pub(crate) use super::types::Policy;
use super::types::{CudaStream, Runtime};
use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

pub(crate) const SPEC_VERSION: u32 = 6;
pub(crate) const SEMANTIC_DENSE: u32 = 0;
pub(crate) const SEMANTIC_KV_CACHE: u32 = 1;
pub(crate) const SEMANTIC_SEGMENTED: u32 = 2;

pub(crate) const QKV_LAYOUT_SEPARATE: u32 = 0;
pub(crate) const QKV_LAYOUT_PACKED_TOKEN: u32 = 1;

pub(crate) const GATE_SPEC_VERSION: u32 = 1;
pub(crate) const GATE_SEMANTIC_SPLIT_QUERY_GATE: u32 = 0;
pub(crate) const GATE_SEMANTIC_APPLY_OUTPUT_GATE: u32 = 1;
pub(crate) const GATE_SOURCE_TENSOR: u32 = 0;
pub(crate) const GATE_SOURCE_PROJECTION: u32 = 1;
pub(crate) const GATE_NUMERICS_FLEXIBLE: u32 = 0;
pub(crate) const GATE_NUMERICS_ROUNDED_SIGMOID: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub mask: u32,
    pub qkv_layout: u32,
    pub q_alignment: u32,
    pub k_alignment: u32,
    pub v_alignment: u32,
    pub output_alignment: u32,
    pub offsets_alignment: u32,
    pub batch: i64,
    pub query_tokens: i64,
    pub key_tokens: i64,
    pub key_capacity: i64,
    pub query_heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub query_start: i64,
    pub segments: i64,
    pub max_segment_tokens: i64,
    pub offsets_hash: u64,
    pub scale_is_default: u32,
    pub dynamic_decode: u32,
    pub numerics: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub query: *const c_void,
    pub key: *const c_void,
    pub value: *const c_void,
    pub offsets: *const c_void,
    pub decode_meta: *const u32,
    pub output: *mut c_void,
    pub stream: CudaStream,
    pub scale: f32,
    pub output_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GateSpec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub source: u32,
    pub numerics: u32,
    pub input_alignment: u32,
    pub output_alignment: u32,
    pub auxiliary_alignment: u32,
    pub tokens: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub elements: i64,
    pub source_row_width: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct GateBindings {
    pub input: *const c_void,
    pub output: *mut c_void,
    pub auxiliary: *mut c_void,
    pub stream: CudaStream,
}

pub(crate) type Execution = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_attention_gate_launch(
        runtime: Runtime,
        spec: *const GateSpec,
        bindings: *const GateBindings,
    ) -> i32;
    pub(crate) fn apxinf_decode_attention_workspace_bytes() -> i64;
    pub(crate) fn apxinf_decode_attention(
        runtime: Runtime,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: i64,
        key_tokens: i64,
        scale: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_attention_prepare(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        execution: *mut Execution,
    ) -> i32;
    pub(crate) fn apxinf_attention_enqueue(execution: Execution) -> i32;
    pub(crate) fn apxinf_attention_destroy(execution: Execution);
    #[cfg(test)]
    pub(crate) fn apxinf_attention_summary(execution: Execution) -> *const c_char;
    #[cfg(test)]
    pub(crate) fn apxinf_attention_test_validate_candidates(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        expected_output: *const f32,
        expected_output_len: u64,
    ) -> i32;
}

unsafe extern "C" {
    pub(crate) fn apxinf_attention_fixed_decode_softmax_decode_f32(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        n_heads: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_attention_fixed_decode_softmax_f32(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        n_heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_attention_fixed_decode_softmax_bf16(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        n_heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_attention_fixed_decode_flash_attn_decode_bf16(
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
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        // FIX (implement_r8): tiled block-per-row softmax (fp32 in, bf16 out) for the
        // route-3 composed vision attention; defined in adapters/custom_kernels.cu.
        pub(crate) fn apxinf_attention_row_softmax_f32_bf16(
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
        pub(crate) fn apxinf_attention_row_softmax_causal_f32(
            input: *const c_void,
            output: *mut c_void,
            cols: u32,
            rows: u32,
            kv_offset: u32,
            n_heads: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;

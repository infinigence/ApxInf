use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;
use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

pub(crate) const SPEC_VERSION: u32 = 6;

pub(crate) const SEMANTIC_SPLIT_QKV_MROPE_CACHE: u32 = 6;
pub(crate) const SEMANTIC_APPLY: u32 = 7;

pub(crate) const LAYOUT_PACKED_QKV: u32 = 0;
pub(crate) const LAYOUT_SEPARATE_QKV: u32 = 1;
pub(crate) const LAYOUT_IN_PLACE: u32 = 2;
pub(crate) const LAYOUT_SINGLE_TENSOR: u32 = 3;

pub(crate) const CACHE_WRITE_NONE: u32 = 0;
pub(crate) const CACHE_WRITE_TOKEN_MAJOR: u32 = 1;
pub(crate) const CACHE_WRITE_HEAD_MAJOR: u32 = 2;

pub(crate) const ROTATION_NONE: u32 = 0;
pub(crate) const ROTATION_POSITION_OFFSET: u32 = 1;
pub(crate) const ROTATION_DEVICE_SCALAR: u32 = 2;
pub(crate) const ROTATION_AXIAL_IDS: u32 = 3;
pub(crate) const ROTATION_MULTIMODAL_IDS: u32 = 4;
pub(crate) const ROTATION_PRECOMPUTED_TABLE: u32 = 5;

pub(crate) const CONVENTION_NONE: u32 = 0;
pub(crate) const CONVENTION_HALF_SPLIT: u32 = 1;
pub(crate) const CONVENTION_HALF_SPLIT_AXIAL: u32 = 2;
pub(crate) const CONVENTION_HALF_SPLIT_INTERLEAVED_AXES: u32 = 3;
pub(crate) const CONVENTION_HALF_SPLIT_AXIAL_ROUNDED_INPUT: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub has_bias: u32,
    pub layout: u32,
    pub cache_write: u32,
    pub rotation_source: u32,
    pub rotary_convention: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub section_t: u32,
    pub section_h: u32,
    pub section_w: u32,
    pub qkv_alignment: u32,
    pub bias_alignment: u32,
    pub q_alignment: u32,
    pub kv_alignment: u32,
    pub rotation_alignment: u32,
    pub tokens: i64,
    pub cache_capacity: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub qkv: *const c_void,
    pub bias: *const c_void,
    pub q: *mut c_void,
    pub k: *mut c_void,
    pub v: *mut c_void,
    pub key_input: *const c_void,
    pub value_input: *const c_void,
    pub position: *const u32,
    pub rotation_table: *const c_void,
    pub stream: CudaStream,
    pub theta: f32,
    pub position_offset: i32,
    pub kv_output_offset: i32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_rope_partial(
        data: *mut c_void,
        positions: *const c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        rotary_dim: i64,
        theta: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_rope_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}

unsafe extern "C" {
    pub(crate) fn apxinf_rope_fixed_decode_decode_f32(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_rope_fixed_decode_decode_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_rope_fixed_decode_mrope_decode_bf16(
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
}

mod fixed_launchers {
    use std::ffi::c_void;

    use super::Runtime;
    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_rope_build_vision_rotation_table_f32(
            pos_ids: *const c_void,
            rotation_table: *mut c_void,
            head_dim: u32,
            seq_len: u32,
            theta: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_vision_2d_pair_bf16(
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
        pub(crate) fn apxinf_rope_qk_rms_norm_mrope_bf16(
            runtime: Runtime,
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
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_split_strided_qkv_bf16(
            qkv: *const c_void,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            tokens: i32,
            hidden: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_rope_split_qkv_cache_bf16(
            projection: *const c_void,
            query_norm: *const c_void,
            key_norm: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            query: *mut c_void,
            key_cache: *mut c_void,
            value_cache: *mut c_void,
            seq: i32,
            cache_offset: i32,
            query_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            rotary_dim: i32,
            projection_width: i64,
            cache_width: i64,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_rope_split_grouped_qkv_bf16(
            projection: *const c_void,
            query_norm: *const c_void,
            key_norm: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            query: *mut c_void,
            gate: *mut c_void,
            key: *mut c_void,
            value: *mut c_void,
            seq: i32,
            query_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            rotary_dim: i32,
            projection_width: i64,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;

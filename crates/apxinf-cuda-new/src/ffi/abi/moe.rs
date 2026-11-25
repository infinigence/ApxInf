use std::ffi::c_void;

use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

unsafe extern "C" {
    pub(crate) fn apxinf_moe_route(
        x: *const c_void,
        gate: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        tokens: i32,
        experts: i32,
        dim: i32,
        top_k: i32,
        routed_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_moe_swiglu_scaled(
        gate: *const c_void,
        up: *const c_void,
        weights: *const c_void,
        out: *mut c_void,
        tokens: i32,
        columns: i32,
        experts: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

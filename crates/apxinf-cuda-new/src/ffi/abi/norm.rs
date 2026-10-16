use std::ffi::c_void;

use super::types::Runtime;
pub(crate) use super::types::CudaStream;

pub(crate) const SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub has_bias: u32,
    pub input_alignment: u32,
    pub weight_alignment: u32,
    pub bias_alignment: u32,
    pub residual_alignment: u32,
    pub style_alignment: u32,
    pub hidden_alignment: u32,
    pub normalized_alignment: u32,
    pub rows: i64,
    pub cols: i64,
    /// Structural predicate only: the scale value lives in [`Bindings`].
    pub output_scale_is_unit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub bias: *const c_void,
    pub residual: *const c_void,
    pub weight: *const c_void,
    pub norm_bias: *const c_void,
    pub norm_style: *const c_void,
    pub gate_style: *const c_void,
    pub hidden: *mut c_void,
    pub normalized: *mut c_void,
    pub stream: CudaStream,
    pub eps: f32,
    pub output_scale: f32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_norm_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}

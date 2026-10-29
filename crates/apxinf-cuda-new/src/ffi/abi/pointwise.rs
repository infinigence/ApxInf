use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const SPEC_VERSION: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub activation: u32,
    pub has_bias: u32,
    pub input_alignment: u32,
    pub secondary_alignment: u32,
    pub bias_alignment: u32,
    pub output_alignment: u32,
    pub rows: i64,
    pub cols: i64,
    pub input_cols: i64,
    pub logical_cols: i64,
    /// Structural predicate only: the scale value lives in [`Bindings`].
    pub output_scale_is_unit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub secondary: *const c_void,
    pub bias: *const c_void,
    pub output: *mut c_void,
    pub scales: *mut c_void,
    pub stream: CudaStream,
    pub dt: f32,
    pub output_scale: f32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_pointwise_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}

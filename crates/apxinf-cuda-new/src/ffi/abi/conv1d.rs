use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

/// Mirrors `apxinf::conv1d::Geometry` (`native/kernels/custom/conv1d.cuh`): twelve `i32` fields and the `f32` output scale, in this order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Geometry {
    pub batch: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub in_len: i32,
    pub out_len: i32,
    pub kernel: i32,
    pub stride: i32,
    pub dilation: i32,
    pub pad_left: i32,
    pub crop_left: i32,
    pub groups: i32,
    pub replicate: i32,
    pub scale: f32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_conv1d_f32(
        x: *const f32,
        w: *const f32,
        bias: *const f32,
        out: *mut f32,
        geometry: *const Geometry,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_conv_transpose1d_f32(
        x: *const f32,
        w: *const f32,
        bias: *const f32,
        out: *mut f32,
        geometry: *const Geometry,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_snake_beta_f32(
        x: *const f32,
        alpha: *const f32,
        beta: *const f32,
        out: *mut f32,
        channels: i32,
        len: i32,
        total: i64,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

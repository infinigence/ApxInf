use std::ffi::c_void;

use super::types::Runtime;
pub(crate) use super::types::CudaStream;

pub(crate) const SPEC_VERSION: u32 = 2;

/// Which outputs the row-wise pipeline writes. The stages themselves are
/// described by the orthogonal `norm_kind`/`affine_kind`/`has_gate`/
/// `bias_order`/`quant_kind` fields, not by this value.
pub(crate) const SEMANTIC_NORM: u32 = 0;
pub(crate) const SEMANTIC_FUSED_ADD_NORM: u32 = 1;
pub(crate) const SEMANTIC_RESIDUAL_ADD: u32 = 2;

pub(crate) const NORM_KIND_NONE: u32 = 0;
pub(crate) const NORM_KIND_RMS: u32 = 1;
pub(crate) const NORM_KIND_LAYER: u32 = 2;

pub(crate) const AFFINE_NONE: u32 = 0;
pub(crate) const AFFINE_WEIGHT: u32 = 1;
/// `(1 + weight)` scaling. Served by its own kernel in
/// `kernels/custom/norm_offset.cu`, not by a flag on the `norm.cuh` family:
/// its reduction order differs, so the two are not bit-comparable beyond the
/// `1 +`. Valid only with [`SEMANTIC_NORM`] + [`NORM_KIND_RMS`] + BF16 +
/// [`QUANT_NONE`]; the native adapter reports `UNSUPPORTED` for anything else
/// instead of silently using [`AFFINE_WEIGHT`].
pub(crate) const AFFINE_WEIGHT_OFFSET: u32 = 2;
pub(crate) const AFFINE_ADA_SCALE_SHIFT: u32 = 3;

/// One rounding of `input + bias + residual`.
pub(crate) const BIAS_ORDER_SINGLE_ROUND: u32 = 0;
/// `input + bias` is rounded before the residual is added and rounded again.
pub(crate) const BIAS_ORDER_ROUND_AFTER_BIAS: u32 = 1;

pub(crate) const QUANT_NONE: u32 = 0;
pub(crate) const QUANT_E4M3: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    /// `SEMANTIC_*`: which outputs are written.
    pub semantic: u32,
    /// `NORM_KIND_*`: which reduction, if any, runs over the row.
    pub norm_kind: u32,
    /// `AFFINE_*`: how the normalized row is rescaled.
    pub affine_kind: u32,
    /// Whether the residual combine multiplies the projection by a gate.
    pub has_gate: u32,
    /// `BIAS_ORDER_*`: rounding order of the bias/residual combine.
    pub bias_order: u32,
    /// `QUANT_*`: fused output quantization.
    pub quant_kind: u32,
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

unsafe extern "C" {
    /// Per-head RMSNorm in place. Not a [`Spec`] pipeline: it writes its input
    /// in place and reduces over the last axis of a three-rank activation.
    pub(crate) fn apxinf_norm_head_rms(
        data: *mut c_void,
        weight: *const c_void,
        rows: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
}

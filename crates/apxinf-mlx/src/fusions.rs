//! Safe, shape-bound device fusions with explicit rounding contracts.
pub use crate::ane_tiles::{AneTileHandoff, AneTileMerge};
pub use crate::causal_conv::CausalDepthwiseConv;
pub use crate::gate_swiglu::F16GateSwiglu;
pub use crate::gated_delta::{GatedDeltaLayout, GatedDeltaRecurrence};
pub use crate::half_split_rope::HalfSplitRope;
pub use crate::interleaved_qk_norm_rope::InterleavedQkNormRope;
pub use crate::mpp_f16_gemm::{MppF16Gemm, SplitMppF16Gemm};
pub use crate::mrope_rope::MropeHalfSplitRope;
pub use crate::mpp_gemm::MppBf16GemmNt;
pub use crate::q4_f16::{
    padded_rows, F16Gemm, PadRowsF16, Q4DecodeArithmetic, Q4RowDecodeF16, Q4TransposeDecode,
    SplitF16Gemm,
};
pub use crate::qk_norm_rope::QkNormRope;
pub use crate::swiglu_gemv::{SwigluMlpGemv, SwigluWeights};
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// BF16 residual sum and RMSNorm with the normalized value rounded before
/// BF16 weight multiplication. Returns both consumers in one owned allocation.
///
/// Fixed width 2048, one row, 256 threads. Other geometry is unsupported.
/// Adapted from EngineTailor, Copyright 2026 Haiyan Qin (MIT); see NOTICE.
pub struct PackedResidualRmsNorm {
    kernel: MetalKernel,
}

impl PackedResidualRmsNorm {
    pub fn new(stream: &Stream, epsilon: f32) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::Contract(
                "RMSNorm epsilon must be finite and positive",
            ));
        }
        let source = include_str!("../native/packed_residual_rmsnorm.metal")
            .replace("EPSILON", &format!("{epsilon:.12e}f"));
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                &format!(
                    "apxinf_packed_residual_rmsnorm_2048_tg256_eps_{:08x}",
                    epsilon.to_bits()
                ),
                &["x", "delta", "weight"],
                &["packed"],
                &source,
                "#pragma clang fp contract(off)\n",
            )?,
        })
    }

    pub fn call(&self, x: &Array, delta: &Array, weight: &Array) -> Result<(Array, Array)> {
        if x.shape() != [1, 1, 2048]
            || delta.shape() != x.shape()
            || weight.shape() != [2048]
            || [x, delta, weight]
                .iter()
                .any(|a| a.dtype() != MlxDType::BF16)
        {
            return Err(Error::Contract(
                "packed residual RMSNorm requires one BF16 row of width 2048",
            ));
        }
        // SAFETY: 256 threads each access exactly eight contiguous columns.
        // The guarded inputs contain 2048 elements each. The output has 4096
        // elements; the two stores are column and 2048 + column. The native
        // Metal facility guarantees row-contiguous inputs and checks streams.
        let output = unsafe {
            self.kernel.call(
                &[x.clone(), delta.clone(), weight.clone()],
                &[(&[2, 1, 2048], MlxDType::BF16)],
                [256, 1, 1],
                [256, 1, 1],
                &[("T", TemplateArg::DType(MlxDType::BF16))],
            )?
        };
        let packed = &output[0];
        Ok((packed.slice_axis(0, 0, 1)?, packed.slice_axis(0, 1, 2)?))
    }
}

/// Unweighted RMSNorm over the last axis plus a per-tensor scalar scale, in
/// one kernel, with the unfused path's two rounding points: the RMSNorm
/// output rounds to `T`, then `T * T` (the BF16-rounded scale) rounds to `T`.
/// This equals `x.fast_rms_norm_unweighted(eps)?.mul(scalar)` for the model's
/// BF16 activations, so the gated-delta q/k norm+scale saves one intermediate
/// allocation and one kernel launch without changing any value.
///
/// The axis must fit one threadgroup at four reads per thread (128 here).
pub struct RmsNormScale {
    kernel: MetalKernel,
}

impl RmsNormScale {
    pub fn new(stream: &Stream, epsilon: f32) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::Contract(
                "RMSNorm epsilon must be finite and positive",
            ));
        }
        let source = include_str!("../native/rms_norm_scale.metal")
            .replace("EPSILON", &format!("{epsilon:.12e}f"));
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                &format!(
                    "apxinf_rms_norm_scale_t_{:x}_eps_{:08x}",
                    128usize,
                    epsilon.to_bits()
                ),
                &["x", "scale"],
                &["dst"],
                &source,
                "",
            )?,
        })
    }

    pub fn call(&self, x: &Array, scale: &Array) -> Result<Array> {
        let axis = *x.shape().last().ok_or(Error::Contract("RMSNorm input is rank 0"))?;
        if axis != 128
            || x.shape().is_empty()
            || x.dtype() != MlxDType::BF16
            || scale.dtype() != MlxDType::BF16
            || scale.numel() != 1
        {
            return Err(Error::Contract(
                "RMSNormScale requires a BF16 input with last axis 128 and one BF16 scalar",
            ));
        }
        let rows = x.numel() / axis;
        // SAFETY: each of `rows` threadgroups handles one 128-element row;
        // 32 threads read four columns each, threads beyond 128 do not store,
        // and every store is to `dst[row * 128 + col]`. The guards bound the
        // input to BF16 with axis 128 and the scale to one BF16 scalar.
        // `grid.x = rows * 32` with `threadgroup.x = 32` yields exactly one
        // threadgroup per row, matching MLX's `dispatch_threads(n_rows * tg, tg)`.
        let out = unsafe {
            self.kernel.call(
                &[x.clone(), scale.clone()],
                &[(x.shape(), MlxDType::BF16)],
                [rows * 32, 1, 1],
                [32, 1, 1],
                &[
                    ("T", TemplateArg::DType(MlxDType::BF16)),
                    ("AXIS_SIZE", TemplateArg::Int(axis as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// Fusion #1 part A: split the gated-delta convolution output
/// `[1, L, 2*key_dim + value_dim]` into q `[1, L, hk, dk]`, k (same) and
/// v `[1, L, hv, dv]` in one kernel, preserving every BF16 bit. This replaces
/// three `slice_axis + reshape` pairs whose consumers each force a
/// row-contiguous gather; the outputs come out contiguous, so the composed
/// `RmsNormScale` and the recurrence read them without further copies.
pub struct GdnConvSplit {
    kernel: MetalKernel,
}

impl GdnConvSplit {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_gdn_conv_split",
                &["x", "length"],
                &["q", "k", "v"],
                include_str!("../native/gdn_conv_split.metal"),
                "",
            )?,
        })
    }

    /// `key_dim = hk * dk`, `value_dim = hv * dv`; `conv_dim = 2*key_dim + value_dim`.
    /// Returns q `[1, L, hk, dk]`, k (same) and v `[1, L, hv, dv]`, contiguous,
    /// so the recurrence and `RmsNormScale` read them with no further gather.
    pub fn call(
        &self,
        conv: &Array,
        hk: usize,
        dk: usize,
        hv: usize,
        dv: usize,
    ) -> Result<(Array, Array, Array)> {
        let [1, length, channels] = *conv.shape() else {
            return Err(Error::Contract(
                "gated-delta conv split input must be [1, L, C]",
            ));
        };
        let (key_dim, value_dim) = (hk * dk, hv * dv);
        let conv_dim = 2 * key_dim + value_dim;
        if conv.dtype() != MlxDType::BF16
            || channels != conv_dim
            || length == 0
            || key_dim == 0
            || value_dim == 0
            || channels > i32::MAX as usize
            || length > i32::MAX as usize
        {
            return Err(Error::Contract(
                "gated-delta conv split requires BF16 [1, L, 2*key+value] with positive extents",
            ));
        }
        let scalar = |v: usize| Array::from_i32(conv.stream(), &[], &[v as i32]);
        // SAFETY: `total = length * conv_dim` threads, each copying x[idx] to
        // exactly one output slot determined by `c = idx % conv_dim`; the
        // three outputs are contiguous BF16 buffers of `length * key_dim`,
        // `length * key_dim` and `length * value_dim` elements, and the
        // branch bounds keep every store inside its buffer. `length` is the
        // scalar equal to L.
        let mut out = unsafe {
            self.kernel.call(
                &[conv.clone(), scalar(length)?],
                &[
                    (&[1, length, hk, dk], MlxDType::BF16),
                    (&[1, length, hk, dk], MlxDType::BF16),
                    (&[1, length, hv, dv], MlxDType::BF16),
                ],
                [length * conv_dim, 1, 1],
                [256, 1, 1],
                &[
                    ("InT", TemplateArg::DType(MlxDType::BF16)),
                    ("KEY_DIM", TemplateArg::Int(key_dim as i32)),
                    ("VALUE_DIM", TemplateArg::Int(value_dim as i32)),
                    ("CONV_DIM", TemplateArg::Int(conv_dim as i32)),
                ],
            )?
        };
        let v = out.pop().unwrap();
        let k = out.pop().unwrap();
        Ok((out.pop().unwrap(), k, v))
    }
}

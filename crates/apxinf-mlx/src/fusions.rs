//! Safe, shape-bound device fusions with explicit rounding contracts.
pub use crate::bf16_qk_table_rope::Bf16QkTableRope;
pub use crate::gated_delta::GatedDeltaRecurrence;
pub use crate::half_split_rope::HalfSplitRope;
pub use crate::interleaved_qk_norm_rope::InterleavedQkNormRope;
pub use crate::mpp_gemm::MppBf16GemmNt;
pub use crate::q4_f16::{
    padded_rows, PadRowsF16, Q4DecodeArithmetic, Q4TransposeDecode, SplitF16Gemm,
};
pub use crate::qk_norm_rope::QkNormRope;
pub use crate::swiglu_gemv::{SwigluMlpGemv, SwigluWeights};
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// Where RMSNorm rounds the normalized row relative to its weight multiply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormRounding {
    /// `bf16(weight * bf16(x * rms⁻¹))`, the order of MLX's weighted
    /// `fast::rms_norm` and of the Transformers Llama norm.
    BeforeWeight,
    /// `bf16(x * rms⁻¹ * weight)`, rounded once after the weight multiply.
    AfterWeight,
}

/// BF16 residual sum `bf16(x + delta)` and its weighted RMSNorm in one launch.
/// Returns both consumers in one owned allocation.
///
/// Each row of `[1, rows, width]` is reduced by one 256-thread threadgroup in a
/// fixed order: every thread owns `ceil(width / 256)` contiguous columns.
/// Adapted from EngineTailor, Copyright 2026 Haiyan Qin (MIT); see NOTICE.
pub struct ResidualRmsNorm {
    kernel: MetalKernel,
    width: usize,
    rounding: NormRounding,
}

impl ResidualRmsNorm {
    /// Bounds the per-thread register row at 32 values.
    pub const MAX_WIDTH: usize = 8192;

    pub fn new(
        stream: &Stream,
        width: usize,
        epsilon: f32,
        rounding: NormRounding,
    ) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::Contract(
                "RMSNorm epsilon must be finite and positive",
            ));
        }
        if width == 0 || width > Self::MAX_WIDTH {
            return Err(Error::Contract(
                "residual RMSNorm width must be between 1 and 8192",
            ));
        }
        let source = include_str!("../native/residual_rmsnorm.metal")
            .replace("EPSILON", &format!("{epsilon:.12e}f"));
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                &format!(
                    "apxinf_residual_rmsnorm_tg256_eps_{:08x}",
                    epsilon.to_bits()
                ),
                &["x", "delta", "weight"],
                &["packed"],
                &source,
                "#pragma clang fp contract(off)\n",
            )?,
            width,
            rounding,
        })
    }

    /// `x` and `delta` are BF16 `[1, rows, width]`; `weight` is BF16 or F32
    /// `[width]`. Returns `(sum, normalized)`, both BF16 with the shape of `x`.
    pub fn call(&self, x: &Array, delta: &Array, weight: &Array) -> Result<(Array, Array)> {
        let shape = x.shape();
        if shape.len() != 3
            || shape[0] != 1
            || shape[1] == 0
            || shape[2] != self.width
            || delta.shape() != shape
            || [x, delta].iter().any(|a| a.dtype() != MlxDType::BF16)
            || weight.shape() != [self.width]
            || !matches!(weight.dtype(), MlxDType::BF16 | MlxDType::F32)
        {
            return Err(Error::Contract(
                "residual RMSNorm requires BF16 [1, rows, width] operands and a BF16 or F32 weight",
            ));
        }
        let rows = shape[1];
        // SAFETY: `rows` threadgroups of 256 threads; thread t of row r reads
        // and writes only columns t * ceil(width / 256) + i that are below
        // `width`. The guards give both inputs rows * width elements and the
        // weight width elements. The output has 2 * rows * width elements and
        // each column is stored once at r * width + column and once at
        // (rows + r) * width + column. The native Metal
        // facility guarantees row-contiguous inputs and checks streams.
        let output = unsafe {
            self.kernel.call(
                &[x.clone(), delta.clone(), weight.clone()],
                &[(&[2, rows, self.width], MlxDType::BF16)],
                [rows * 256, 1, 1],
                [256, 1, 1],
                &[
                    ("T", TemplateArg::DType(MlxDType::BF16)),
                    ("WIDTH", TemplateArg::Int(self.width as i32)),
                    (
                        "BEFORE_WEIGHT",
                        TemplateArg::Bool(self.rounding == NormRounding::BeforeWeight),
                    ),
                ],
            )?
        };
        let packed = &output[0];
        Ok((packed.slice_axis(0, 0, 1)?, packed.slice_axis(0, 1, 2)?))
    }
}

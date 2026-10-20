//! Safe, shape-bound device fusions with explicit rounding contracts.
pub use crate::qk_norm_rope::QkNormRope;
use crate::{Array, MetalKernel, MlxDType, Stream};
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
                Some(MlxDType::BF16),
            )?
        };
        let packed = &output[0];
        Ok((packed.slice_axis(0, 0, 1)?, packed.slice_axis(0, 1, 2)?))
    }
}

//! Partial rotary embedding with FP32 angles. Adapted from MLX-VLM 0.7.4
//! (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// Rotates the first `rotary_dim` features of `x` `[B, H, T, D]` in
/// half-split pairs `(i, i + rotary_dim / 2)` and copies the rest. Each angle
/// is `position[b, t] * inv_freq[i]` in FP32; each output is rounded once to
/// the input dtype.
pub struct HalfSplitRope {
    kernel: MetalKernel,
    rotary_dim: usize,
}

impl HalfSplitRope {
    pub fn new(stream: &Stream, rotary_dim: usize) -> Result<Self> {
        if rotary_dim == 0 || rotary_dim % 2 != 0 || i32::try_from(rotary_dim).is_err() {
            return Err(Error::Contract(
                "rotary dimension must be positive and even",
            ));
        }
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_rope_half_split",
                &["x", "position_ids", "inv_freq"],
                &["x_out"],
                include_str!("../native/rope_half_split.metal"),
                "",
            )?,
            rotary_dim,
        })
    }

    pub fn call(&self, x: &Array, positions: &Array, inv_freq: &Array) -> Result<Array> {
        let [batch, heads, length, width] = *x.shape() else {
            return Err(Error::Contract("rotary input must be [B, H, T, D]"));
        };
        let slots = width
            .checked_sub(self.rotary_dim / 2)
            .ok_or(Error::Contract(
                "rotary dimension exceeds the feature width",
            ))?;
        let work = [batch, heads, length, slots]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n > 0 && i32::try_from(n).is_ok())
            .ok_or(Error::Contract("rotary work size is empty or exceeds i32"))?;
        if self.rotary_dim > width
            || !matches!(x.dtype(), MlxDType::BF16 | MlxDType::F16 | MlxDType::F32)
            || positions.shape() != [batch, length]
            || positions.dtype() != MlxDType::I32
            || inv_freq.shape() != [self.rotary_dim / 2]
            || inv_freq.dtype() != MlxDType::F32
            || i32::try_from(x.numel()).is_err()
        {
            return Err(Error::Contract(
                "rotary embedding requires float [B, H, T, D], I32 [B, T] positions \
                 and F32 [rotary_dim / 2] inverse frequencies",
            ));
        }
        // SAFETY: one thread per (b, h, t, slot) with slot < D - rotary_dim / 2;
        // threads at or after the work size return. A rotary slot i reads and
        // writes features i and i + rotary_dim / 2 < D of its own row and reads
        // inv_freq[i] and position[b, t]; a pass-through slot copies one
        // feature >= rotary_dim. No two threads write the same feature.
        let out = unsafe {
            self.kernel.call(
                &[x.clone(), positions.clone(), inv_freq.clone()],
                &[(x.shape(), x.dtype())],
                [work, 1, 1],
                [256, 1, 1],
                &[
                    ("T", TemplateArg::DType(x.dtype())),
                    ("ROTARY_DIM", TemplateArg::Int(self.rotary_dim as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

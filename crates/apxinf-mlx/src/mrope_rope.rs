//! Interleaved 3-axis mRoPE with half-split pairing, as MLX-VLM applies it to
//! Qwen3.5 (`style="interleaved"`): frequency `i` takes its position from axis
//! `position_selector[i]`, angles are `position * inv_freq[i]` in FP32, and each
//! output feature is rounded once to the input dtype.
//!
//! The pointer arithmetic is MLX-VLM 0.7.4 `_mrope_apply_kernel`'s
//! (`position_ids[(axis * batch + b) * length + t]`); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// Rotates the first `rotary_dim` features of `x` `[B, H, T, D]` in half-split
/// pairs `(i, i + rotary_dim / 2)` and copies the rest. Each frequency's
/// position axis comes from `position_selector`.
pub struct MropeHalfSplitRope {
    kernel: MetalKernel,
    rotary_dim: usize,
}

impl MropeHalfSplitRope {
    pub fn new(stream: &Stream, rotary_dim: usize) -> Result<Self> {
        if rotary_dim == 0 || rotary_dim % 2 != 0 || i32::try_from(rotary_dim).is_err() {
            return Err(Error::Contract(
                "rotary dimension must be positive and even",
            ));
        }
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_rope_mrope_half_split",
                &["x", "position_ids", "inv_freq", "position_selector"],
                &["x_out"],
                include_str!("../native/mrope_half_split.metal"),
                "",
            )?,
            rotary_dim,
        })
    }

    /// `x` is `[B, H, T, D]`; `positions` is I32 `[3, B, T]`; `inv_freq` is F32
    /// `[rotary_dim / 2]`; `selector` is I32 `[rotary_dim / 2]` with every entry
    /// in `0..3`.
    pub fn call(
        &self,
        x: &Array,
        positions: &Array,
        inv_freq: &Array,
        selector: &Array,
    ) -> Result<Array> {
        let [batch, heads, length, width] = *x.shape() else {
            return Err(Error::Contract("rotary input must be [B, H, T, D]"));
        };
        let half = self.rotary_dim / 2;
        let slots = width
            .checked_sub(half)
            .ok_or(Error::Contract(
                "rotary dimension exceeds the feature width",
            ))?;
        if slots == 0 {
            return Err(Error::Contract("rotary work size is empty"));
        }
        let work = [batch, heads, length, slots]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .filter(|&n| n > 0 && i32::try_from(n).is_ok())
            .ok_or(Error::Contract("rotary work size is empty or exceeds i32"))?;
        if self.rotary_dim > width
            || !matches!(x.dtype(), MlxDType::BF16 | MlxDType::F16 | MlxDType::F32)
            || positions.shape() != [3, batch, length]
            || positions.dtype() != MlxDType::I32
            || inv_freq.shape() != [half]
            || inv_freq.dtype() != MlxDType::F32
            || selector.shape() != [half]
            || selector.dtype() != MlxDType::I32
            || i32::try_from(x.numel()).is_err()
        {
            return Err(Error::Contract(
                "mRoPE requires float [B, H, T, D], I32 [3, B, T] positions, F32 \
                 [rotary_dim / 2] inverse frequencies and I32 [rotary_dim / 2] axis selectors",
            ));
        }
        // SAFETY: one thread per (b, h, t, slot); threads at or after the work
        // size return. A rotary slot i reads and writes features i and
        // i + rotary_dim / 2 < D of its own row, reads inv_freq[i], selector[i]
        // and one of the three position rows at column t. A pass-through slot
        // copies one feature >= rotary_dim. No two threads write one feature.
        let out = unsafe {
            self.kernel.call(
                &[
                    x.clone(),
                    positions.clone(),
                    inv_freq.clone(),
                    selector.clone(),
                ],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_checked() {
        let stream = Stream::cpu().unwrap();
        assert!(MropeHalfSplitRope::new(&stream, 0).is_err());
        assert!(MropeHalfSplitRope::new(&stream, 3).is_err());
    }
}

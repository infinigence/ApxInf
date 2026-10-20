//! Non-affine LayerNorm (width 4096, eps 1e-6) followed by `* (1 + scale)`,
//! optionally after a gated residual, with the roundings of MLX `layer_norm`
//! followed by an elementwise modulation: two-pass FP32 mean and variance, the
//! normalized value, `1 + scale` and their product each rounded to BF16.
//! Portions adapted from EngineTailor (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream};
use apxinf_core::{Error, Result};

/// Row width of these kernels: 256 threads, four `uint2` words each.
pub const WIDTH: usize = 4096;

/// BF16 `[1, T, 4096]` rows with a nonzero T and BF16 `[1, T, >= min_width]`
/// modulation views.
fn rows(x: &Array, modulation: &[(&Array, usize)]) -> Result<usize> {
    let [1, t, WIDTH] = *x.shape() else {
        return Err(Error::Contract(
            "LayerNorm modulation input must be BF16 [1, T, 4096]",
        ));
    };
    let fits = |m: &Array, min_width: usize| {
        m.dtype() == MlxDType::BF16 && matches!(*m.shape(), [1, mt, w] if mt == t && w >= min_width)
    };
    if x.dtype() != MlxDType::BF16
        || t == 0
        || t.checked_mul(WIDTH).is_none_or(|n| n > i32::MAX as usize)
        || !modulation.iter().all(|&(m, width)| fits(m, width))
    {
        return Err(Error::Contract(
            "LayerNorm modulation needs BF16 [1, T, 4096] rows and [1, T, C] modulation views",
        ));
    }
    Ok(t)
}

/// `LayerNorm(x) * (1 + scale)` with `scale` the columns `0..4096` of
/// `modulation`, read through its strides so broadcast views are not copied.
pub struct LayerNormModulate {
    kernel: MetalKernel,
}

impl LayerNormModulate {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new_strided(
                stream,
                "apxinf_layer_norm_modulate_4096",
                &["x", "mod"],
                &["out"],
                concat!(
                    include_str!("../native/layer_norm_modulate_4096.metal"),
                    include_str!("../native/layer_norm_modulate_4096_tail.metal")
                ),
                "",
            )?,
        })
    }

    pub fn call(&self, x: &Array, modulation: &Array) -> Result<Array> {
        let t = rows(x, &[(modulation, WIDTH)])?;
        let x = x.contiguous()?;
        // SAFETY: `x` is row-contiguous BF16 `[1, T, 4096]` and T * 4096 fits
        // i32 (checked). Threadgroup `row < T` of 256 threads reads words
        // `tid + 256 c < 1024` of its row, `modulation[row, col < 4096]` through
        // its int64 strides (width >= 4096), and writes the same words of `out`
        // once.
        let out = unsafe {
            self.kernel.call(
                &[x, modulation.clone()],
                &[(&[1, t, WIDTH], MlxDType::BF16)],
                [256, t, 1],
                [256, 1, 1],
                &[],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

/// `x1 = x + tanh(gate) * a` with `gate` the columns `4096..8192` of
/// `gate_modulation` (tanh, product and sum each rounded to BF16), then
/// `LayerNorm(x1) * (1 + scale)` with `scale` the columns `0..4096` of
/// `modulation`. Returns `(x1, out)`.
pub struct GatedResidualLayerNormModulate {
    kernel: MetalKernel,
}

impl GatedResidualLayerNormModulate {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new_strided(
                stream,
                "apxinf_gated_residual_layer_norm_modulate_4096",
                &["x", "a", "gate_mod", "mod"],
                &["x1", "out"],
                concat!(
                    include_str!("../native/gated_residual_layer_norm_modulate_4096.metal"),
                    include_str!("../native/layer_norm_modulate_4096_tail.metal")
                ),
                "",
            )?,
        })
    }

    pub fn call(
        &self,
        x: &Array,
        a: &Array,
        gate_modulation: &Array,
        modulation: &Array,
    ) -> Result<(Array, Array)> {
        let t = rows(x, &[(gate_modulation, 2 * WIDTH), (modulation, WIDTH)])?;
        if a.dtype() != MlxDType::BF16 || a.shape() != x.shape() {
            return Err(Error::Contract("gated residual addend must match x"));
        }
        let (x, a) = (x.contiguous()?, a.contiguous()?);
        // SAFETY: `x` and `a` are row-contiguous BF16 `[1, T, 4096]`, T * 4096
        // fits i32 (checked). Threadgroup `row < T` reads words
        // `tid + 256 c < 1024` of its rows of `x` and `a`,
        // `gate_modulation[row, 4096 + col]` (width >= 8192) and
        // `modulation[row, col]` (width >= 4096) through their strides, and
        // writes the same words of `x1` and `out` once.
        let mut out = unsafe {
            self.kernel.call(
                &[x, a, gate_modulation.clone(), modulation.clone()],
                &[
                    (&[1, t, WIDTH], MlxDType::BF16),
                    (&[1, t, WIDTH], MlxDType::BF16),
                ],
                [256, t, 1],
                [256, 1, 1],
                &[],
            )?
        }
        .into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_checks_need_no_device() {
        let s = Stream::cpu().unwrap();
        let a = |shape: &[usize]| Array::zeros(&s, shape, MlxDType::BF16).unwrap();
        let x = a(&[1, 3, WIDTH]);
        let modulation = a(&[1, 1, 2 * WIDTH])
            .broadcast_to(&[1, 3, 2 * WIDTH])
            .unwrap();
        assert_eq!(rows(&x, &[(&modulation, 2 * WIDTH)]).unwrap(), 3);
        assert!(rows(&x, &[(&a(&[1, 3, WIDTH]), 2 * WIDTH)]).is_err());
        assert!(rows(&x, &[(&a(&[1, 2, 2 * WIDTH]), WIDTH)]).is_err());
        assert!(rows(&a(&[1, 3, 2048]), &[]).is_err());
        assert!(rows(&a(&[1, 0, WIDTH]), &[]).is_err());
    }
}

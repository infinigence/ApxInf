//! Device-ready FP8 E4M3 linear weights for π0-FAST.
//!
//! Weights use the standard absmax rule (`scale = amax / 448`) and stay resident
//! as E4M3; biases stay BF16 and are applied by the consumer, exactly as in the
//! BF16 path.
//!
//! The dispatch that consumes these matrices is `gemm::fp8_bf16`, whose output is
//! **BF16**. That choice is what keeps the change small: norms, RoPE, attention,
//! the KV cache, the argmax and the tied embedding lookup all keep operating on
//! BF16 and are reused verbatim from the BF16 module.

use apxinf_core::{Backend, Error, Result, Tensor};

#[cfg(feature = "cuda")]
use super::backend::{kernels, RuntimeBackend};
use super::{bf16_weights::concat_biases_bf16, device_weights::concat_host_2d, LinearWeights};

/// Largest finite NVIDIA/CUDA E4M3 value (`0x7e`).
///
/// Same constant and same absmax rule as the PI0.5 FP8 path, so a scale means
/// the same thing in both families.
pub const E4M3_MAX: f32 = 448.0;

#[derive(Debug)]
pub struct Fp8LinearWeights {
    /// Physical row-major `[input, output]` E4M3 matrix.
    pub weight: Tensor,
    /// Dequantization multiplier: `real = e4m3(value) * weight_scale`.
    pub weight_scale: f32,
    /// Bias stays BF16; the consumer applies it after the GEMM.
    pub bias: Option<Tensor>,
}

impl Fp8LinearWeights {
    #[cfg(feature = "cuda")]
    pub fn as_kernel_view(&self) -> kernels::gemm::Fp8WeightView<'_> {
        kernels::gemm::Fp8WeightView {
            values_e4m3: &self.weight,
            scale: self.weight_scale,
            // π0-FAST packs gate/up plainly, so the interleaved dual-GeGLU
            // layouts the π0.5 runtime uses never apply here.
            dual_geglu_interleaved: false,
            dual_geglu_auto_interleaved: None,
        }
    }

    pub fn from_host(linear: &LinearWeights, backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts(&[linear], backend)
    }

    /// Pack projections along the output dimension, then quantize the packed
    /// matrix under a single scale, so QKV and gate/up each stay one GEMM and
    /// one dequantization.
    pub fn from_host_parts(linears: &[&LinearWeights], backend: &dyn Backend) -> Result<Self> {
        if linears.is_empty() {
            return Err(Error::Other("cannot pack an empty FP8 linear group".into()));
        }
        let plain = concat_host_2d(
            &linears
                .iter()
                .map(|linear| &linear.weight)
                .collect::<Vec<_>>(),
        )?;
        let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
            None
        } else if linears.iter().all(|linear| linear.bias.is_some()) {
            Some(concat_biases_bf16(
                &linears
                    .iter()
                    .map(|linear| linear.bias.as_ref().expect("bias presence checked"))
                    .collect::<Vec<_>>(),
                backend,
            )?)
        } else {
            return Err(Error::Other(
                "cannot pack FP8 projections with mixed bias presence".into(),
            ));
        };
        let (weight, weight_scale) = quantize_e4m3_absmax_to_device(&plain, backend)?;
        Ok(Self {
            weight,
            weight_scale,
            bias,
        })
    }
}

/// Quantize a host matrix to a resident E4M3 tensor under its absmax scale.
///
/// The scalar CPU E4M3 encoder is deliberately not used: π0-FAST is a CUDA-only
/// family, and uploading FP16 once before letting the conversion kernel produce
/// the E4M3 matrix is far cheaper than an elementwise pass over the weights on
/// the host.
pub fn quantize_e4m3_absmax_to_device(
    host: &Tensor,
    backend: &dyn Backend,
) -> Result<(Tensor, f32)> {
    #[cfg(feature = "cuda")]
    {
        let cuda = backend
            .as_any()
            .downcast_ref::<RuntimeBackend>()
            .ok_or_else(|| Error::Other("π0-FAST FP8 weights require the CUDA backend".into()))?;
        let values = host.to_f32_vec()?;
        let amax = values.iter().fold(0.0f32, |m, value| m.max(value.abs()));
        // All-zero matrices keep scale 1 so the representation stays valid.
        let scale = if amax == 0.0 { 1.0 } else { amax / E4M3_MAX };
        let f16_host = Tensor::from_f16(
            host.shape().dims().to_vec(),
            &values
                .into_iter()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
        )?;
        let f16 = backend.to_device(&f16_host)?;
        let weight = kernels::quantization::quantize_f16_e4m3(cuda.context(), &f16, scale)?;
        Ok((weight, scale))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (host, backend);
        Err(Error::Other(
            "π0-FAST FP8 weights require the CUDA backend".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absmax_rule_matches_the_pi05_convention() {
        // 448 is the largest finite E4M3 value, so an amax of 448 is scale 1.
        assert_eq!(E4M3_MAX, 448.0);
        let host = Tensor::from_f32(vec![2, 2], &[1.0, -448.0, 0.0, 224.0]).unwrap();
        // The CUDA path needs a device; only the rule is checked here.
        let values = host.to_f32_vec().unwrap();
        let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert_eq!(amax / E4M3_MAX, 1.0);
    }
}

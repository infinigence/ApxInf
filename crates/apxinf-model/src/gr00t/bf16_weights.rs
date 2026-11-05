//! Native-BF16 GR00T device weights.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::action_weights::Gr00tLinearWeights;
use super::backend::{ops, RuntimeBackend};
use super::device_weights::DeviceLinearWeights;

#[derive(Debug)]
pub(super) struct Gr00tBf16LinearWeights {
    weights: Gr00tLinearWeights,
    calibration_name: String,
}

impl Gr00tBf16LinearWeights {
    pub(super) fn from_host(
        weights: Gr00tLinearWeights,
        calibration_name: impl Into<String>,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        Ok(Self {
            weights: Gr00tLinearWeights {
                weight: backend.to_device(&weights.weight)?,
                bias: backend.to_device(&weights.bias)?,
            },
            calibration_name: calibration_name.into(),
        })
    }
}

impl DeviceLinearWeights for Gr00tBf16LinearWeights {
    type ReusableInput = ();

    fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        super::calibration::observe(&self.calibration_name, input)?;
        let a = input.shape().dims();
        let b = self.weights.weight.shape().dims();
        if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
            return Err(Error::Other(format!(
                "GR00T BF16 GEMM shape mismatch: {a:?} @ {b:?}"
            )));
        }
        let mut output = backend
            .context()
            .allocate_output(Shape::new(vec![a[0], b[1]]), DType::BF16)?;
        ops::gemm::gemm(
            backend.context(),
            ops::gemm::GemmArgs::new(input, &self.weights.weight, &mut output),
        )?;
        Ok(output)
    }

    fn bias(&self) -> Option<&Tensor> {
        Some(&self.weights.bias)
    }

    fn bias_residual_adaptive_layer_norm(
        &self,
        projection: &Tensor,
        residual: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !matches!(backend.context().caps().sm, 87 | 110)
            || projection.shape().dims() != [41, 1536]
            || residual.shape() != projection.shape()
            || self.weights.bias.shape().dims() != [1536]
            || modulation.shape().dims() != [3072]
        {
            return Ok(None);
        }
        let hidden = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::BF16)?;
        let normalized = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::BF16)?;
        ops::fused_add_norm(
            backend.context(),
            ops::FusedAddNormArgs {
                input: projection,
                bias: Some(&self.weights.bias),
                residual,
                gate: None,
                hidden: &hidden,
                normalized: Some(&normalized),
                kind: ops::NormKind::Layer,
                affine: ops::Affine::Adaptive { style: modulation },
                quantized: None,
                numerics: ops::NormNumerics::Flexible,
                bias_order: ops::BiasOrder::BiasThenResidual,
                eps,
            },
        )?;
        Ok(Some((hidden, normalized)))
    }

    fn bias_residual_layer_norm(
        &self,
        projection: &Tensor,
        residual: &Tensor,
        norm_weight: &Tensor,
        norm_bias: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !matches!(backend.context().caps().sm, 87 | 110)
            || projection.shape().dims() != [41, 1536]
            || residual.shape() != projection.shape()
            || self.weights.bias.shape().dims() != [1536]
            || norm_weight.shape().dims() != [1536]
            || norm_bias.shape().dims() != [1536]
        {
            return Ok(None);
        }
        let hidden = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::BF16)?;
        let normalized = backend
            .context()
            .allocate_output(projection.shape().clone(), DType::BF16)?;
        ops::fused_add_norm(
            backend.context(),
            ops::FusedAddNormArgs {
                input: projection,
                bias: Some(&self.weights.bias),
                residual,
                gate: None,
                hidden: &hidden,
                normalized: Some(&normalized),
                kind: ops::NormKind::Layer,
                affine: ops::Affine::Weight {
                    weight: norm_weight,
                    bias: Some(norm_bias),
                    offset: false,
                },
                quantized: None,
                numerics: ops::NormNumerics::Flexible,
                bias_order: ops::BiasOrder::BiasThenResidual,
                eps,
            },
        )?;
        Ok(Some((hidden, normalized)))
    }

    fn supports_fused_self_qkv(&self) -> bool {
        true
    }
}

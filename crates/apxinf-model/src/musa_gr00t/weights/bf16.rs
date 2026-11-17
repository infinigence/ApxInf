//! Native-BF16 GR00T device weights.

use apxinf_core::{Result, Tensor};

use super::super::action_weights::Gr00tLinearWeights;
use super::super::backend::{kernels, RuntimeBackend};
use super::super::model::linear::DeviceLinearWeights;

#[derive(Debug)]
pub(crate) struct Gr00tBf16LinearWeights {
    weights: Gr00tLinearWeights,
    calibration_name: String,
}

impl Gr00tBf16LinearWeights {
    pub(super) fn from_device(weights: Gr00tLinearWeights, calibration_name: String) -> Self {
        Self { weights, calibration_name }
    }
    pub(crate) fn from_host(
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
        super::super::model::calibration::observe(&self.calibration_name, input)?;
        kernels::gemm::bf16(backend.context(), input, &self.weights.weight)
    }

    fn bias(&self) -> Option<&Tensor> {
        Some(&self.weights.bias)
    }

    fn packed8_bias(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Option<Tensor>> {
        super::super::backend::try_packed8_bias_activation(backend, input, &self.weights.bias, 0)
    }

    fn packed8_bias_gelu(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        super::super::backend::try_packed8_bias_activation(backend, input, &self.weights.bias, 1)
    }

    fn bias_residual_adaptive_layer_norm(
        &self,
        projection: &Tensor,
        residual: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor)>> {
        #[cfg(feature = "musa")]
        {
            if projection.shape().dims() != [41, 1536]
                || residual.shape() != projection.shape()
                || self.weights.bias.shape().dims() != [1536]
                || modulation.shape().dims() != [3072]
            {
                return Ok(None);
            }
            let fused = kernels::fused::bias_then_residual_adaptive_layer_bf16(
                backend.context(),
                projection,
                &self.weights.bias,
                residual,
                modulation,
                eps,
            )?;
            return Ok(Some((fused.hidden, fused.normalized)));
        }
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
        #[cfg(feature = "musa")]
        {
            if projection.shape().dims() != [41, 1536]
                || residual.shape() != projection.shape()
                || self.weights.bias.shape().dims() != [1536]
                || norm_weight.shape().dims() != [1536]
                || norm_bias.shape().dims() != [1536]
            {
                return Ok(None);
            }
            let fused = kernels::fused::bias_then_residual_layer_bf16(
                backend.context(),
                projection,
                &self.weights.bias,
                residual,
                norm_weight,
                norm_bias,
                eps,
            )?;
            return Ok(Some((fused.hidden, fused.normalized)));
        }
    }

    fn supports_fused_self_qkv(&self) -> bool {
        true
    }
}

//! Native-BF16 GR00T device weights.

use apxinf_core::{Backend, Result, Tensor};

use super::action_weights::Gr00tLinearWeights;
use super::backend::{kernels, RuntimeBackend};
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
        kernels::gemm::bf16(backend.context(), input, &self.weights.weight)
    }

    fn bias(&self) -> Option<&Tensor> {
        Some(&self.weights.bias)
    }

    fn packed8_bias(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Option<Tensor>> {
        super::backend::try_packed8_bias_activation(backend, input, &self.weights.bias, 0)
    }

    fn packed8_bias_gelu(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        super::backend::try_packed8_bias_activation(backend, input, &self.weights.bias, 1)
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
        let fused = kernels::fused::bias_then_residual_adaptive_layer_bf16_cached_1536(
            backend.context(),
            projection,
            &self.weights.bias,
            residual,
            modulation,
            eps,
        )?;
        Ok(Some((fused.hidden, fused.normalized)))
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
        let fused = kernels::fused::bias_then_residual_layer_bf16_cached_1536(
            backend.context(),
            projection,
            &self.weights.bias,
            residual,
            norm_weight,
            norm_bias,
            eps,
        )?;
        Ok(Some((fused.hidden, fused.normalized)))
    }

    fn supports_fused_self_qkv(&self) -> bool {
        true
    }
}

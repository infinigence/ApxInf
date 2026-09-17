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

    fn supports_fused_self_qkv(&self) -> bool {
        true
    }
}

//! Precision-neutral contracts for GR00T device linear weights.
//!
//! Concrete BF16, static-FP8, and W8A8 storage lives in the corresponding
//! precision module.  The executor is generic over these contracts, so one
//! precision is selected while the model is loaded and the hot path never
//! dispatches through a per-matrix precision enum.

use std::fmt::Debug;

use apxinf_core::{Error, Result, Tensor};

use super::backend::RuntimeBackend;

pub(super) trait DeviceLinearWeights: Debug {
    type ReusableInput;

    fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor>;

    fn bias(&self) -> Option<&Tensor>;

    fn activation_scale(&self) -> Option<f32> {
        None
    }

    fn can_share_quantized_input_with(&self, _other: &Self) -> bool {
        false
    }

    fn quantize_reusable_input(
        &self,
        _input: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(None)
    }

    fn quantize_bias_gelu_reusable_input(
        &self,
        _input: &Tensor,
        _bias: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(None)
    }

    fn forward_reusable_quantized(
        &self,
        _input: &Self::ReusableInput,
        _backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        Err(Error::Other(
            "this GR00T precision does not accept reusable quantized input".into(),
        ))
    }

    /// Tensor-valued quantization is used by static FP8 paths that feed
    /// model-neutral FP8 kernels directly.
    fn quantize_tensor_input(
        &self,
        _input: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn forward_quantized_tensor(
        &self,
        _input: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        Err(Error::Other(
            "this GR00T precision does not accept an FP8 tensor input".into(),
        ))
    }

    /// Return a fused precision-specific SiLU×up projection when available.
    fn fused_silu_mul(
        &self,
        _gate: &Tensor,
        _up: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Return a fused adaptive-normalization plus reusable quantization when
    /// this precision has an exact model-neutral kernel for that composition.
    fn adaptive_layer_norm_quantized(
        &self,
        _input: &Tensor,
        _modulation: &Tensor,
        _eps: f32,
        _backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        Ok(None)
    }

    fn rms_norm_quantized(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _eps: f32,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(None)
    }

    fn supports_fused_self_qkv(&self) -> bool {
        false
    }

    fn uses_quantized_output(&self) -> bool {
        false
    }
}

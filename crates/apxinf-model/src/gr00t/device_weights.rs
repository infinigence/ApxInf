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

    /// Optional GR00T-private BF16 packed8 bias path. Implementations return
    /// `None` when the precision, device, or exact profiled shape is not
    /// eligible, preserving the ordinary public operator as the fallback.
    fn packed8_bias(&self, _input: &Tensor, _backend: &RuntimeBackend) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Optional GR00T-private BF16 packed8 Bias+GELU path.
    fn packed8_bias_gelu(
        &self,
        _input: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

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

    /// Fuse an already-quantized FC1 input with projection, bias, GELU, and
    /// the next projection's reusable-input quantization when available.
    fn forward_reusable_quantized_bias_gelu(
        &self,
        _input: &Self::ReusableInput,
        _output_scale: f32,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(None)
    }

    /// Return an opt-in residual-add + LayerNorm + reusable quantization when
    /// the precision has an exact model-neutral implementation. The returned
    /// tensor is the BF16 residual boundary consumed by the rest of the block.
    #[allow(clippy::too_many_arguments)]
    fn residual_layer_norm_quantized(
        &self,
        _projection: &Tensor,
        _residual: &Tensor,
        _norm_weight: &Tensor,
        _norm_bias: &Tensor,
        _eps: f32,
        _backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
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

    /// Return a private exact-shape FC2 projection + bias + residual fusion
    /// when this precision provides one. All generic/default implementations
    /// retain the established projection followed by pointwise fallback.
    fn forward_reusable_quantized_bias_residual(
        &self,
        _input: &Self::ReusableInput,
        _residual: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        Ok(None)
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

    /// Return a precision-specific, bit-compatible packed bias-then-residual
    /// result when the output width satisfies that implementation's contract.
    fn packed_bias_then_residual(
        &self,
        _projection: &Tensor,
        _bias: Option<&Tensor>,
        _residual: &Tensor,
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

    /// Exact opt-in composition for a projection bias, residual add, and the
    /// following adaptive LayerNorm. Implementors return `None` unless their
    /// storage and kernel contract preserve the legacy BF16 boundaries.
    fn bias_residual_adaptive_layer_norm(
        &self,
        _projection: &Tensor,
        _residual: &Tensor,
        _modulation: &Tensor,
        _eps: f32,
        _backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor)>> {
        Ok(None)
    }

    /// Exact opt-in composition for a projection whose bias must be rounded
    /// to BF16 before the residual add and following fixed LayerNorm.
    fn bias_residual_layer_norm(
        &self,
        _projection: &Tensor,
        _residual: &Tensor,
        _norm_weight: &Tensor,
        _norm_bias: &Tensor,
        _eps: f32,
        _backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Tensor)>> {
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

    fn supports_fused_bias_gelu_quantization(&self) -> bool {
        false
    }

    fn forward_reusable_quantized_bias_gelu_quantized(
        &self,
        _input: &Self::ReusableInput,
        _bias: &Tensor,
        _backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        Ok(None)
    }

    fn forward_reusable_quantized_with_bias(
        &self,
        _input: &Self::ReusableInput,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn layer_norm_quantized(
        &self,
        _input: &Tensor,
        _weight: &Tensor,
        _bias: &Tensor,
        _eps: f32,
        _backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        Ok(None)
    }
}

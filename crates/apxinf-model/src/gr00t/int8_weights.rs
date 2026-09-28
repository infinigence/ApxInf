//! W8A8 INT8 GR00T device weights for Orin-class GPUs.

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::action_weights::Gr00tLinearWeights;
use super::backend::{kernels, DeviceBuffer, RuntimeBackend};
use super::device_weights::DeviceLinearWeights;

pub(super) struct Gr00tInt8LinearWeights {
    weight_output_major: DeviceBuffer,
    weight_scales: Tensor,
    bias: Option<Tensor>,
    input_dim: usize,
    output_dim: usize,
}

impl std::fmt::Debug for Gr00tInt8LinearWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gr00tInt8LinearWeights")
            .field("weight_bytes", &self.weight_output_major.len())
            .field("weight_scales", &self.weight_scales.shape().dims())
            .field("has_bias", &self.bias.is_some())
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .finish()
    }
}

impl Gr00tInt8LinearWeights {
    pub(super) fn from_host(
        weight: &Tensor,
        bias: Option<&Tensor>,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        let (quantized, scales, input_dim, output_dim) = quantize_output_channels(weight)?;
        let bytes = quantized
            .into_iter()
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        let weight_output_major =
            DeviceBuffer::alloc(bytes.len(), backend.device_id()).map_err(Error::Cuda)?;
        weight_output_major
            .copy_from_host(&bytes)
            .map_err(Error::Cuda)?;
        let weight_scales = backend.to_device(&Tensor::from_f32(vec![output_dim], &scales)?)?;
        let bias = bias
            .map(|tensor| {
                if tensor.shape().dims() != [output_dim] || tensor.dtype() == DType::F8E4M3 {
                    return Err(Error::Other(format!(
                        "GR00T INT8 bias must be a non-FP8 vector of width {output_dim}, got {} {:?}",
                        tensor.dtype(),
                        tensor.shape().dims()
                    )));
                }
                let values = tensor
                    .to_f32_vec()?
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>();
                backend.to_device(&Tensor::from_bf16(vec![output_dim], &values)?)
            })
            .transpose()?;
        Ok(Self {
            weight_output_major,
            weight_scales,
            bias,
            input_dim,
            output_dim,
        })
    }

    pub(super) fn linear(weights: Gr00tLinearWeights, backend: &RuntimeBackend) -> Result<Self> {
        Self::from_host(&weights.weight, Some(&weights.bias), backend)
    }

    pub(super) fn matrix(weight: &Tensor, backend: &RuntimeBackend) -> Result<Self> {
        Self::from_host(weight, None, backend)
    }

    fn as_kernel_view(&self) -> kernels::gemm::W8A8WeightView<'_> {
        kernels::gemm::W8A8WeightView {
            values_i8: &self.weight_output_major,
            scales_f32: &self.weight_scales,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            scale_mode: kernels::gemm::W8A8ScaleMode::DynamicRowPerOutputChannel,
            layout: kernels::gemm::W8A8Layout::OutputMajor,
        }
    }
}

impl DeviceLinearWeights for Gr00tInt8LinearWeights {
    type ReusableInput = kernels::gemm::W8A8Activation;

    fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        if backend.context().caps().sm == 87 && self.input_dim == 1536 && self.output_dim == 6144 {
            if let Some(output) = kernels::gemm::try_gemm_w8a8_m41_n6144_k1536(
                backend.context(),
                input,
                self.as_kernel_view(),
            )? {
                return Ok(output);
            }
        }
        kernels::gemm::w8a8(backend.context(), input, self.as_kernel_view())
    }

    fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    fn can_share_quantized_input_with(&self, other: &Self) -> bool {
        self.input_dim == other.input_dim
    }

    fn quantize_reusable_input(
        &self,
        input: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(Some(kernels::gemm::quantize_w8a8_activation(
            backend.context(),
            input,
        )?))
    }

    fn quantize_bias_gelu_reusable_input(
        &self,
        input: &Tensor,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::ReusableInput>> {
        Ok(Some(kernels::gemm::bias_gelu_quantize_w8a8_activation(
            backend.context(),
            input,
            bias,
        )?))
    }

    fn supports_fused_bias_gelu_quantization(&self) -> bool {
        true
    }

    fn forward_reusable_quantized(
        &self,
        input: &Self::ReusableInput,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        // Production M41 FC1 normally uses the fused bias-GELU producer.
        // Its decomposed/legacy path keeps the measured plain INT8 schedule
        // through an explicit model choice, without changing generic tactics.
        if backend.context().caps().sm == 87 && self.input_dim == 1536 && self.output_dim == 6144 {
            if let Some(output) = kernels::gemm::try_gemm_quantized_w8a8_m41_n6144_k1536(
                backend.context(),
                input,
                self.as_kernel_view(),
            )? {
                return Ok(output);
            }
        }
        kernels::gemm::gemm_quantized_w8a8(backend.context(), input, self.as_kernel_view())
    }

    fn forward_reusable_quantized_bias_gelu_quantized(
        &self,
        input: &Self::ReusableInput,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        if backend.context().caps().sm != 87 {
            return Ok(None);
        }
        if !w8a8_fc1_producer_fusion_enabled(
            std::env::var_os("APXINF_GR00T_W8A8_LEGACY_FC1_GELU_QUANT").is_some(),
        ) {
            return Ok(None);
        }
        kernels::gemm::try_gemm_quantized_w8a8_bias_gelu_quantized(
            backend.context(),
            input,
            self.as_kernel_view(),
            bias,
        )
    }

    fn forward_reusable_quantized_with_bias(
        &self,
        input: &Self::ReusableInput,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        if backend.context().caps().sm != 87 {
            return Ok(None);
        }
        let Some(bias) = self.bias.as_ref() else {
            return Ok(None);
        };
        if std::env::var_os("APXINF_GR00T_W8A8_LEGACY_QKV_GEMM_BIAS").is_some()
            || self.input_dim != 1536
            || self.output_dim != 4608
        {
            return Ok(None);
        }
        kernels::gemm::try_gemm_quantized_w8a8_bias(
            backend.context(),
            input,
            self.as_kernel_view(),
            bias,
        )
    }

    fn fused_silu_mul(
        &self,
        gate: &Tensor,
        up: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let activation = if backend.context().caps().sm == 87
            && gate.shape().dims() == [156, 6144]
            && std::env::var_os("APXINF_GR00T_W8A8_LEGACY_SILU_MUL_PACKED4").is_none()
        {
            kernels::gemm::quantize_w8a8_silu_mul_activation_packed4(backend.context(), gate, up)?
        } else {
            kernels::gemm::quantize_w8a8_silu_mul_activation(backend.context(), gate, up)?
        };
        Ok(Some(kernels::gemm::gemm_quantized_w8a8(
            backend.context(),
            &activation,
            self.as_kernel_view(),
        )?))
    }

    fn packed_bias_then_residual(
        &self,
        projection: &Tensor,
        bias: Option<&Tensor>,
        residual: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        if backend.context().caps().sm != 87 {
            return Ok(None);
        }
        if std::env::var_os("APXINF_GR00T_W8A8_LEGACY_BIAS_RESIDUAL").is_some()
            || self.output_dim % 4 != 0
        {
            return Ok(None);
        }
        Ok(Some(kernels::fused::bias_then_residual_bf16_packed4(
            backend.context(),
            projection,
            bias,
            residual,
        )?))
    }

    fn adaptive_layer_norm_quantized(
        &self,
        input: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        if backend.context().caps().sm != 87 {
            return Ok(None);
        }
        let (normalized, quantized) = kernels::gemm::adaptive_layer_norm_quantize_w8a8_activation(
            backend.context(),
            input,
            modulation,
            eps,
        )?;
        Ok(Some((normalized, quantized)))
    }

    fn layer_norm_quantized(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        if backend.context().caps().sm != 87 {
            return Ok(None);
        }
        if std::env::var_os("APXINF_GR00T_W8A8_LEGACY_LAYER_NORM_QUANT").is_some()
            || input.shape().dims() != [41, 1536]
        {
            return Ok(None);
        }
        let (normalized, quantized) = kernels::gemm::layer_norm_quantize_w8a8_activation(
            backend.context(),
            input,
            weight,
            bias,
            eps,
        )?;
        Ok(Some((normalized, quantized)))
    }

    fn supports_fused_self_qkv(&self) -> bool {
        true
    }

    fn uses_quantized_output(&self) -> bool {
        true
    }
}

fn w8a8_fc1_producer_fusion_enabled(legacy_requested: bool) -> bool {
    !legacy_requested
}

fn quantize_output_channels(tensor: &Tensor) -> Result<(Vec<i8>, Vec<f32>, usize, usize)> {
    if tensor.dtype() == DType::F8E4M3 {
        return Err(Error::Other(
            "cannot quantize a scale-less E4M3 matrix to INT8".into(),
        ));
    }
    let dims = tensor.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(format!(
            "GR00T INT8 weight must be a non-empty matrix, got {dims:?}"
        )));
    }
    let (input_dim, output_dim) = (dims[0], dims[1]);
    let values = tensor.to_f32_vec()?;
    let mut quantized = vec![0i8; input_dim * output_dim];
    let mut scales = vec![0.0f32; output_dim];
    for output in 0..output_dim {
        let maximum = (0..input_dim)
            .map(|input| values[input * output_dim + output].abs())
            .fold(0.0f32, f32::max);
        let scale = (maximum / 127.0).max(1.0e-12);
        scales[output] = scale;
        for input in 0..input_dim {
            let value = (values[input * output_dim + output] / scale)
                .round()
                .clamp(-128.0, 127.0);
            quantized[output * input_dim + input] = value as i8;
        }
    }
    Ok((quantized, scales, input_dim, output_dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_channel_quantization_round_trips_shape_and_scale() {
        let source = Tensor::from_f32(vec![2, 3], &[1.0, -2.0, 0.0, -1.0, 4.0, 0.5]).unwrap();
        let (values, scales, input, output) = quantize_output_channels(&source).unwrap();
        assert_eq!((input, output), (2, 3));
        assert_eq!(values.len(), 6);
        assert_eq!(scales.len(), 3);
        assert!(scales.iter().all(|scale| scale.is_finite() && *scale > 0.0));
    }

    #[test]
    fn legacy_fc1_producer_switch_disables_fusion() {
        assert!(w8a8_fc1_producer_fusion_enabled(false));
        assert!(!w8a8_fc1_producer_fusion_enabled(true));
    }
}

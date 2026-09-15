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

    fn forward_reusable_quantized(
        &self,
        input: &Self::ReusableInput,
        backend: &RuntimeBackend,
    ) -> Result<Tensor> {
        kernels::gemm::gemm_quantized_w8a8(backend.context(), input, self.as_kernel_view())
    }

    fn fused_silu_mul(
        &self,
        gate: &Tensor,
        up: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let activation =
            kernels::gemm::quantize_w8a8_silu_mul_activation(backend.context(), gate, up)?;
        Ok(Some(kernels::gemm::gemm_quantized_w8a8(
            backend.context(),
            &activation,
            self.as_kernel_view(),
        )?))
    }

    fn adaptive_layer_norm_quantized(
        &self,
        input: &Tensor,
        modulation: &Tensor,
        eps: f32,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        let (normalized, quantized) = kernels::gemm::adaptive_layer_norm_quantize_w8a8_activation(
            backend.context(),
            input,
            modulation,
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
}

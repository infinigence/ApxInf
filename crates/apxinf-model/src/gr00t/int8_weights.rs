//! W8A8 INT8 GR00T device weights for Orin-class GPUs.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::action_weights::Gr00tLinearWeights;
use super::backend::{ops, RuntimeBackend};
use super::device_weights::DeviceLinearWeights;

pub(super) struct Gr00tInt8LinearWeights {
    weight_output_major: Tensor,
    weight_scales: Tensor,
    bias: Option<Tensor>,
    input_dim: usize,
    output_dim: usize,
}

impl std::fmt::Debug for Gr00tInt8LinearWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gr00tInt8LinearWeights")
            .field("weight_shape", &self.weight_output_major.shape().dims())
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
        let weight_output_major =
            backend.to_device(&Tensor::from_i8(vec![output_dim, input_dim], &quantized)?)?;
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

    fn gemm_args<'a>(
        &'a self,
        input: &'a ops::quantization::DynamicI8Tensor,
        output: &'a mut Tensor,
    ) -> ops::gemm::GemmArgs<'a> {
        ops::gemm::GemmArgs::row_channel_quantized(
            &input.values,
            &input.scales,
            &self.weight_output_major,
            &self.weight_scales,
            output,
        )
        .with_output_major_weight()
        .with_immutable_weight(ops::gemm::WeightVersion::new(0))
    }

    fn output(&self, rows: usize, backend: &RuntimeBackend) -> Result<Tensor> {
        backend.context().allocate_output(
            apxinf_core::Shape::new(vec![rows, self.output_dim]),
            DType::BF16,
        )
    }
}

impl DeviceLinearWeights for Gr00tInt8LinearWeights {
    type ReusableInput = ops::quantization::DynamicI8Tensor;

    fn forward(&self, input: &Tensor, backend: &RuntimeBackend) -> Result<Tensor> {
        let quantized = ops::quantization::quantize_rows_i8(backend.context(), input)?;
        self.forward_reusable_quantized(&quantized, backend)
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
        Ok(Some(ops::quantization::quantize_rows_i8(
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
        Ok(Some(ops::pointwise::bias_gelu_quantize_rows(
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
        let rows = input.values.shape().dims().first().copied().unwrap_or(0);
        let mut output = self.output(rows, backend)?;
        ops::gemm::gemm(backend.context(), self.gemm_args(input, &mut output))?;
        Ok(output)
    }

    fn forward_reusable_quantized_bias_gelu_quantized(
        &self,
        input: &Self::ReusableInput,
        bias: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<(Tensor, Self::ReusableInput)>> {
        if !w8a8_fc1_producer_fusion_enabled(
            std::env::var_os("APXINF_GR00T_W8A8_LEGACY_FC1_GELU_QUANT").is_some(),
        ) {
            return Ok(None);
        }
        let rows = input.values.shape().dims().first().copied().unwrap_or(0);
        let mut activated = self.output(rows, backend)?;
        let gemm = self.gemm_args(input, &mut activated);
        ops::gemm::gemm_bias_gelu(
            backend.context(),
            ops::gemm::GemmBiasGeluArgs { gemm, bias },
        )?;
        let quantized = ops::quantization::quantize_rows_i8(backend.context(), &activated)?;
        Ok(Some((activated, quantized)))
    }

    fn forward_reusable_quantized_with_bias(
        &self,
        input: &Self::ReusableInput,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let Some(bias) = self.bias.as_ref() else {
            return Ok(None);
        };
        if std::env::var_os("APXINF_GR00T_W8A8_LEGACY_QKV_GEMM_BIAS").is_some() {
            return Ok(None);
        }
        let rows = input.values.shape().dims().first().copied().unwrap_or(0);
        let mut output = self.output(rows, backend)?;
        let gemm = self.gemm_args(input, &mut output);
        ops::gemm::gemm_bias(backend.context(), ops::gemm::GemmBiasArgs { gemm, bias })?;
        Ok(Some(output))
    }

    fn fused_silu_mul(
        &self,
        gate: &Tensor,
        up: &Tensor,
        backend: &RuntimeBackend,
    ) -> Result<Option<Tensor>> {
        let activation = ops::pointwise::silu_mul_quantize_rows(backend.context(), gate, up)?;
        Ok(Some(self.forward_reusable_quantized(&activation, backend)?))
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
        let dims = input.shape().dims();
        let rows = dims[0];
        let normalized = backend
            .context()
            .allocate_output(input.shape().clone(), DType::BF16)?;
        let values = backend
            .context()
            .allocate_output(input.shape().clone(), DType::I8)?;
        let scales = backend
            .context()
            .allocate_output(Shape::new(vec![rows]), DType::F32)?;
        ops::norm::norm(
            backend.context(),
            ops::norm::NormArgs {
                input,
                output: Some(&normalized),
                kind: ops::norm::NormKind::Layer,
                affine: ops::norm::Affine::Adaptive { style: modulation },
                quantized: Some(ops::norm::QuantOut::RowwiseI8 {
                    values: &values,
                    scales: &scales,
                }),
                numerics: ops::norm::NormNumerics::Flexible,
                eps,
            },
        )?;
        let quantized = ops::quantization::DynamicI8Tensor { values, scales };
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
        let dims = input.shape().dims();
        let rows = dims[0];
        let normalized = backend
            .context()
            .allocate_output(input.shape().clone(), DType::BF16)?;
        let values = backend
            .context()
            .allocate_output(input.shape().clone(), DType::I8)?;
        let scales = backend
            .context()
            .allocate_output(Shape::new(vec![rows]), DType::F32)?;
        ops::norm::norm(
            backend.context(),
            ops::norm::NormArgs {
                input,
                output: Some(&normalized),
                kind: ops::norm::NormKind::Layer,
                affine: ops::norm::Affine::Weight {
                    weight,
                    bias: Some(bias),
                    offset: false,
                },
                quantized: Some(ops::norm::QuantOut::RowwiseI8 {
                    values: &values,
                    scales: &scales,
                }),
                numerics: ops::norm::NormNumerics::Flexible,
                eps,
            },
        )?;
        let quantized = ops::quantization::DynamicI8Tensor { values, scales };
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

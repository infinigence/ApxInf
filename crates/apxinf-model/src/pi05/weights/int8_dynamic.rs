//! int8_dynamic device weights: model aggregates and linear storage.
mod linear {
    //! Output-channel-quantized W8A8 linear weights for π0.5.

    use apxinf_core::{Backend, DType, Error, Result, Tensor};

    use crate::pi05::backend::{kernels, Context, DeviceBuffer, RuntimeBackend};
    use crate::pi05::weights::packing::concat_host_2d;
    use crate::pi05::LinearWeights;
    use kernels::gemm::{w8a8, W8A8Layout, W8A8ScaleMode, W8A8WeightView};

    pub struct Int8DynamicLinearWeights {
        /// Physical output-major `[output,input]` INT8 bytes. This is also a
        /// zero-copy `[input,output]` column-major view for cuBLAS/CUTLASS.
        pub weight_output_major: DeviceBuffer,
        /// Dequantization multiplier for each output channel.
        pub weight_scales: Tensor,
        pub bias: Option<Tensor>,
        pub input_dim: usize,
        pub output_dim: usize,
    }

    impl Int8DynamicLinearWeights {
        pub fn from_host(linear: &LinearWeights, backend: &RuntimeBackend) -> Result<Self> {
            Self::from_host_parts(&[linear], backend)
        }

        /// Pack QKV or gate/up along the output dimension, then independently
        /// quantize every output channel across its complete input row.
        pub fn from_host_parts(
            linears: &[&LinearWeights],
            backend: &RuntimeBackend,
        ) -> Result<Self> {
            if linears.is_empty() {
                return Err(Error::Other(
                    "cannot pack an empty INT8 linear group".into(),
                ));
            }
            let packed = concat_host_2d(
                &linears
                    .iter()
                    .map(|linear| &linear.weight)
                    .collect::<Vec<_>>(),
            )?;
            let (quantized, scales, input_dim, output_dim) = quantize_output_channels(&packed)?;
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
            let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
                None
            } else if linears.iter().all(|linear| linear.bias.is_some()) {
                Some(concat_biases_bf16(
                    &linears
                        .iter()
                        .map(|linear| linear.bias.as_ref().unwrap())
                        .collect::<Vec<_>>(),
                    backend,
                )?)
            } else {
                return Err(Error::Other(
                    "cannot pack INT8 projections with mixed bias presence".into(),
                ));
            };
            Ok(Self {
                weight_output_major,
                weight_scales,
                bias,
                input_dim,
                output_dim,
            })
        }

        pub fn gemm(&self, ctx: &Context, activation: &Tensor) -> Result<Tensor> {
            w8a8(ctx, activation, self.as_kernel_view())
        }

        pub fn as_kernel_view(&self) -> W8A8WeightView<'_> {
            W8A8WeightView {
                values_i8: &self.weight_output_major,
                scales_f32: &self.weight_scales,
                input_dim: self.input_dim,
                output_dim: self.output_dim,
                scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
                layout: W8A8Layout::OutputMajor,
            }
        }
    }

    /// Convert ApxInf's physical `[input,output]` matrix into the kernel's physical
    /// `[output,input]` INT8 layout, with one `amax/127` scale per output.
    fn quantize_output_channels(tensor: &Tensor) -> Result<(Vec<i8>, Vec<f32>, usize, usize)> {
        if tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other(
                "cannot quantize a scale-less E4M3 matrix to INT8".into(),
            ));
        }
        let dims = tensor.shape().dims();
        if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
            return Err(Error::Other(format!(
                "INT8 weight must be a non-empty matrix, got {dims:?}"
            )));
        }
        let (input_dim, output_dim) = (dims[0], dims[1]);
        let values = tensor.to_f32_vec()?;
        let mut quantized = vec![0i8; input_dim * output_dim];
        let mut scales = vec![0.0f32; output_dim];
        for output in 0..output_dim {
            let mut maximum = 0.0f32;
            for input in 0..input_dim {
                maximum = maximum.max(values[input * output_dim + output].abs());
            }
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

    fn concat_biases_bf16(tensors: &[&Tensor], backend: &dyn Backend) -> Result<Tensor> {
        let mut values = Vec::new();
        for tensor in tensors {
            if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
                return Err(Error::Other(
                    "packed INT8 biases must be non-FP8 vectors".into(),
                ));
            }
            values.extend(tensor.to_f32_vec()?.into_iter().map(half::bf16::from_f32));
        }
        backend.to_device(&Tensor::from_bf16(vec![values.len()], &values)?)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn quantizes_each_output_channel_and_transposes_physically() {
            let weight = Tensor::from_f32(vec![3, 2], &[1.0, -10.0, 2.0, 0.0, 3.0, 10.0]).unwrap();
            let (quantized, scales, input, output) = quantize_output_channels(&weight).unwrap();
            assert_eq!((input, output), (3, 2));
            assert_eq!(quantized, vec![42, 85, 127, -127, 0, 127]);
            assert!((scales[0] - 3.0 / 127.0).abs() < 1.0e-7);
            assert!((scales[1] - 10.0 / 127.0).abs() < 1.0e-7);
        }

        #[test]
        fn zero_channel_uses_finite_minimum_scale() {
            let weight = Tensor::from_f32(vec![2, 1], &[0.0, 0.0]).unwrap();
            let (quantized, scales, _, _) = quantize_output_channels(&weight).unwrap();
            assert_eq!(quantized, vec![0, 0]);
            assert_eq!(scales, vec![1.0e-12]);
        }
    }
}
pub use linear::*;
// Fully materialized W8A8 π0.5 weights.

use apxinf_core::{Result, Tensor};

use crate::pi05::backend::RuntimeBackend;
use crate::pi05::{
    bf16_to_device, ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights,
    Pi05Weights, VisionBlockWeights,
};

pub struct Int8DynamicDeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct Int8DynamicDeviceVisionBlock {
    pub norm1: Int8DynamicDeviceLayerNorm,
    pub qkv: Int8DynamicLinearWeights,
    pub output: Int8DynamicLinearWeights,
    pub norm2: Int8DynamicDeviceLayerNorm,
    pub fc1: Int8DynamicLinearWeights,
    pub fc2: Int8DynamicLinearWeights,
}

pub struct Int8DynamicDeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Int8DynamicLinearWeights,
    pub output: Int8DynamicLinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Int8DynamicLinearWeights,
    pub down: Int8DynamicLinearWeights,
}

pub struct Int8DynamicDeviceActionLayer {
    pub input_style: Int8DynamicLinearWeights,
    pub qkv: Int8DynamicLinearWeights,
    pub output: Int8DynamicLinearWeights,
    pub post_attention_style: Int8DynamicLinearWeights,
    pub gate_up: Int8DynamicLinearWeights,
    pub down: Int8DynamicLinearWeights,
}

pub struct Int8DynamicWeights {
    pub patch_embedding: Int8DynamicLinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Int8DynamicDeviceVisionBlock>,
    pub vision_post_norm: Int8DynamicDeviceLayerNorm,
    pub multimodal_projector: Int8DynamicLinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Int8DynamicDeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Int8DynamicDeviceActionLayer>,
    pub action_final_style: Int8DynamicLinearWeights,
    pub action_in: Int8DynamicLinearWeights,
    pub action_out: Int8DynamicLinearWeights,
    pub time_mlp_in: Int8DynamicLinearWeights,
    pub time_mlp_out: Int8DynamicLinearWeights,
}

impl Int8DynamicWeights {
    pub fn from_host(weights: &Pi05Weights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            patch_embedding: Int8DynamicLinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Int8DynamicDeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Int8DynamicDeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Int8DynamicLinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Int8DynamicDeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Int8DynamicDeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_style: style_to_device(&weights.action_final_norm, backend)?,
            action_in: Int8DynamicLinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Int8DynamicLinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Int8DynamicLinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Int8DynamicLinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Int8DynamicDeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Int8DynamicDeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            norm1: Int8DynamicDeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Int8DynamicLinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: Int8DynamicLinearWeights::from_host(&weights.output, backend)?,
            norm2: Int8DynamicDeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Int8DynamicLinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Int8DynamicLinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Int8DynamicDeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Int8DynamicLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Int8DynamicLinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Int8DynamicLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Int8DynamicLinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl Int8DynamicDeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            input_style: style_to_device(&weights.input_norm, backend)?,
            qkv: Int8DynamicLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Int8DynamicLinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_style: style_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Int8DynamicLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Int8DynamicLinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn style_to_device(
    weights: &AdaRmsNormWeights,
    backend: &RuntimeBackend,
) -> Result<Int8DynamicLinearWeights> {
    Int8DynamicLinearWeights::from_host(&weights.style, backend)
}

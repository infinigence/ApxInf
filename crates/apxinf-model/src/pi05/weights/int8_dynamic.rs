//! int8_dynamic device weights: model aggregates and linear storage.
mod linear {
    //! Output-channel-quantized W8A8 linear weights for π0.5.

    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    use crate::pi05::backend::{self, ops, Context};
    use crate::pi05::weights::packing::concat_host_2d;
    use crate::pi05::LinearWeights;

    pub struct Int8DynamicLinearWeights {
        /// Canonical contiguous row-major `[input,output]` signed INT8 weight.
        pub weight: Tensor,
        /// Dequantization multiplier for each output channel.
        pub channel_scales: Tensor,
        pub bias: Option<Tensor>,
        pub input_dim: usize,
        pub output_dim: usize,
    }

    fn w8a8_gemm_policy(policy: &ops::GemmPolicy) -> ops::GemmPolicy {
        let mut policy = policy.clone();
        // W8A8 is defined as I8 x I8 with I32 accumulation.  Keep the
        // caller's tuning, workspace, graph-safety and cache choices, but do
        // not let a generic floating-point policy weaken that semantic
        // contract after `GemmArgs::w8a8` established it.
        policy.accumulation_dtype = DType::I32;
        policy
    }

    impl Int8DynamicLinearWeights {
        pub fn from_host(linear: &LinearWeights, backend: &Context) -> Result<Self> {
            Self::from_host_parts(&[linear], backend)
        }

        /// Pack QKV or gate/up along the output dimension, then independently
        /// quantize every output channel across its complete input row.
        pub fn from_host_parts(
            linears: &[&LinearWeights],
            backend: &Context,
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
            let weight = backend::to_device(backend, &Tensor::from_i8(
                vec![input_dim, output_dim],
                &quantized,
            )?)?;
            let channel_scales = backend::to_device(
                backend,
                &Tensor::from_f32(vec![output_dim], &scales)?,
            )?;
            let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
                None
            } else if linears.iter().all(|linear| linear.bias.is_some()) {
                Some(backend::to_device(backend, &concat_biases_bf16(
                    &linears
                        .iter()
                        .map(|linear| linear.bias.as_ref().unwrap())
                        .collect::<Vec<_>>(),
                )?)?)
            } else {
                return Err(Error::Other(
                    "cannot pack INT8 projections with mixed bias presence".into(),
                ));
            };
            Ok(Self {
                weight,
                channel_scales,
                bias,
                input_dim,
                output_dim,
            })
        }

        pub fn gemm_with_policies(
            &self,
            ctx: &Context,
            activation: &Tensor,
            gemm_policy: &ops::GemmPolicy,
        ) -> Result<Tensor> {
            let shape = activation.shape().dims();
            if activation.dtype() != DType::BF16
                || shape.len() != 2
                || shape[1] != self.input_dim
                || self.weight.dtype() != DType::I8
                || self.weight.shape().dims() != [self.input_dim, self.output_dim]
                || self.channel_scales.dtype() != DType::F32
                || self.channel_scales.shape().dims() != [self.output_dim]
            {
                return Err(Error::Other(format!(
                    "PI0.5 W8A8 GEMM contract mismatch: activation {} {:?}, weight {} {:?}, scales {} {:?}",
                    activation.dtype(),
                    shape,
                    self.weight.dtype(),
                    self.weight.shape().dims(),
                    self.channel_scales.dtype(),
                    self.channel_scales.shape().dims(),
                )));
            }
            let rows = shape[0];
            let mut quantized =
                ctx.allocate_output(Shape::new(vec![rows, self.input_dim]), DType::I8)?;
            let mut row_scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
            let mut quantization = ops::QuantizationArgs::new(
                ops::QuantizationSemantic::RowwiseI8,
                activation,
                &mut quantized,
            );
            quantization.scales = Some(&mut row_scales);
            ops::quantization(ctx, quantization)?;

            let mut output =
                ctx.allocate_output(Shape::new(vec![rows, self.output_dim]), DType::BF16)?;
            let mut args = ops::GemmArgs::w8a8(
                &quantized,
                &row_scales,
                &self.weight,
                &self.channel_scales,
                &mut output,
            )
            .with_immutable_weight(ops::WeightVersion::new(0));
            args.policy = w8a8_gemm_policy(gemm_policy);
            ops::gemm(ctx, args)?;
            Ok(output)
        }

        pub fn gemm(&self, ctx: &Context, activation: &Tensor) -> Result<Tensor> {
            self.gemm_with_policies(ctx, activation, &ops::GemmPolicy::default())
        }
    }

    /// Quantize canonical `[input,output]` weights with one `amax/127` scale
    /// per output channel. Provider-specific packing belongs to cuda-new.
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
                quantized[input * output_dim + output] = value as i8;
            }
        }
        Ok((quantized, scales, input_dim, output_dim))
    }

    fn concat_biases_bf16(tensors: &[&Tensor]) -> Result<Tensor> {
        let mut values = Vec::new();
        for tensor in tensors {
            if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
                return Err(Error::Other(
                    "packed INT8 biases must be non-FP8 vectors".into(),
                ));
            }
            values.extend(tensor.to_f32_vec()?.into_iter().map(half::bf16::from_f32));
        }
        Tensor::from_bf16(vec![values.len()], &values)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn quantizes_each_output_channel_in_canonical_layout() {
            let weight = Tensor::from_f32(vec![3, 2], &[1.0, -10.0, 2.0, 0.0, 3.0, 10.0]).unwrap();
            let (quantized, scales, input, output) = quantize_output_channels(&weight).unwrap();
            assert_eq!((input, output), (3, 2));
            assert_eq!(quantized, vec![42, -127, 85, 0, 127, 127]);
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

        #[test]
        fn w8a8_policy_preserves_tuning_choices_and_requires_i32_accumulation() {
            let input = ops::GemmPolicy {
                accumulation_dtype: DType::F32,
                workspace_limit: 1234,
                online_tune: false,
                allow_fallback: false,
                graph_safe: true,
                deterministic: true,
                cache_dir: Some("test-cache".into()),
            };
            let policy = w8a8_gemm_policy(&input);

            assert_eq!(policy.accumulation_dtype, DType::I32);
            assert_eq!(policy.workspace_limit, input.workspace_limit);
            assert_eq!(policy.online_tune, input.online_tune);
            assert_eq!(policy.allow_fallback, input.allow_fallback);
            assert_eq!(policy.graph_safe, input.graph_safe);
            assert_eq!(policy.deterministic, input.deterministic);
            assert_eq!(policy.cache_dir, input.cache_dir);
        }
    }
}
pub use linear::*;
// Fully materialized W8A8 π0.5 weights.

use apxinf_core::{Result, Tensor};

use crate::pi05::backend::Context;
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
    pub input_modulation: Int8DynamicLinearWeights,
    pub qkv: Int8DynamicLinearWeights,
    pub output: Int8DynamicLinearWeights,
    pub post_attention_modulation: Int8DynamicLinearWeights,
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
    pub action_final_modulation: Int8DynamicLinearWeights,
    pub action_in: Int8DynamicLinearWeights,
    pub action_out: Int8DynamicLinearWeights,
    pub time_mlp_in: Int8DynamicLinearWeights,
    pub time_mlp_out: Int8DynamicLinearWeights,
}

impl Int8DynamicWeights {
    pub fn from_host(weights: &Pi05Weights, backend: &Context) -> Result<Self> {
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
            action_final_modulation: modulation_to_device(&weights.action_final_norm, backend)?,
            action_in: Int8DynamicLinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Int8DynamicLinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Int8DynamicLinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Int8DynamicLinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Int8DynamicDeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Int8DynamicDeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &Context) -> Result<Self> {
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
    fn from_host(weights: &LanguageLayerWeights, backend: &Context) -> Result<Self> {
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
    fn from_host(weights: &ActionLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_modulation: modulation_to_device(&weights.input_norm, backend)?,
            qkv: Int8DynamicLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Int8DynamicLinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_modulation: modulation_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Int8DynamicLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Int8DynamicLinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn modulation_to_device(
    weights: &AdaRmsNormWeights,
    backend: &Context,
) -> Result<Int8DynamicLinearWeights> {
    Int8DynamicLinearWeights::from_host(&weights.modulation, backend)
}

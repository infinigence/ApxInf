//! fp8_static device weights: model aggregates and linear storage.
mod linear {
    //! Device-ready static-FP8 linear weights.

    use apxinf_core::{DType, Error, Result, Tensor};

    use crate::pi05::backend::{self, Context};
    use crate::pi05::weights::packing::concat_host_2d;
    use crate::pi05::{quantize_e4m3_absmax, LinearWeights};

    #[derive(Debug)]
    pub struct Fp8StaticLinearWeights {
        /// `[input, output]` CUDA E4M3 matrix.
        pub weight: Tensor,
        pub weight_scale: f32,
        /// Bias stays FP16 and is fused into the consumer kernel.
        pub bias: Option<Tensor>,
    }

    impl Fp8StaticLinearWeights {
        pub fn from_host(linear: &LinearWeights, context: &Context) -> Result<Self> {
            Self::from_host_parts(&[linear], context)
        }

        /// Concatenate projections along their output dimension before applying
        /// one absmax quantization scale. This produces graph-ready QKV and
        /// gate/up matrices without runtime concatenation or mixed descales.
        pub fn from_host_parts(linears: &[&LinearWeights], context: &Context) -> Result<Self> {
            if linears.is_empty() {
                return Err(Error::Other("cannot pack an empty FP8 linear group".into()));
            }
            let (quantized, bias) = pack_host_parts(linears)?;
            let weight_scale = quantized.scale;
            let weight = backend::to_device(context, &quantized.values)?;
            let bias = bias
                .as_ref()
                .map(|bias| backend::to_device(context, bias))
                .transpose()?;
            Ok(Self {
                weight,
                weight_scale,
                bias,
            })
        }
    }

    fn pack_host_parts(
        linears: &[&LinearWeights],
    ) -> Result<(crate::pi05::Fp8Tensor, Option<Tensor>)> {
        let weight_host =
            concat_host_2d(&linears.iter().map(|x| &x.weight).collect::<Vec<_>>())?;
        let quantized = quantize_e4m3_absmax(&weight_host)?;
        let bias = if linears.iter().all(|x| x.bias.is_none()) {
            None
        } else if linears.iter().all(|x| x.bias.is_some()) {
            let biases = linears
                .iter()
                .map(|x| x.bias.as_ref().unwrap())
                .collect::<Vec<_>>();
            Some(concat_host_1d_f16(&biases)?)
        } else {
            return Err(Error::Other(
                "cannot pack projections with a mixture of present and absent biases".into(),
            ));
        };
        Ok((quantized, bias))
    }

    pub fn fp16_to_device(tensor: &Tensor, context: &Context) -> Result<Tensor> {
        let values = tensor.to_f32_vec()?;
        let values = values
            .iter()
            .map(|value| half::f16::from_f32(*value))
            .collect::<Vec<_>>();
        backend::to_device(
            context,
            &Tensor::from_f16(tensor.shape().dims().to_vec(), &values)?,
        )
    }

    fn concat_host_1d_f16(tensors: &[&Tensor]) -> Result<Tensor> {
        let mut output = Vec::new();
        for tensor in tensors {
            if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
                return Err(Error::Other("packed biases must be non-FP8 vectors".into()));
            }
            output.extend(tensor.to_f32_vec()?.into_iter().map(half::f16::from_f32));
        }
        Tensor::from_f16(vec![output.len()], &output)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn linear(weight: &[f32], shape: [usize; 2], bias: Option<&[f32]>) -> LinearWeights {
            LinearWeights {
                weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
                bias: bias.map(|x| Tensor::from_f32(vec![x.len()], x).unwrap()),
            }
        }

        #[test]
        fn packs_qkv_before_quantization() {
            let q = linear(&[1., 2., 3., 4.], [2, 2], Some(&[1., 2.]));
            let k = linear(&[5., 6.], [2, 1], Some(&[3.]));
            let v = linear(&[7., 8.], [2, 1], Some(&[4.]));
            let (packed, bias) = pack_host_parts(&[&q, &k, &v]).unwrap();
            assert_eq!(packed.values.shape().dims(), &[2, 4]);
            assert_eq!(packed.values.dtype(), DType::F8E4M3);
            let bias = bias.unwrap();
            assert_eq!(bias.dtype(), DType::F16);
            assert_eq!(bias.to_f32_vec().unwrap(), vec![1., 2., 3., 4.]);
        }

    }
}
pub use linear::*;
// Fully materialized static-FP8 π0.5 weights.

use apxinf_core::{Result, Tensor};
use crate::pi05::backend::Context;

use crate::pi05::{
    ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights, Pi05Weights,
    VisionBlockWeights,
};

#[derive(Debug)]
pub struct Fp8StaticDeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Fp8StaticDeviceVisionBlock {
    pub norm1: Fp8StaticDeviceLayerNorm,
    pub qkv: Fp8StaticLinearWeights,
    pub output: Fp8StaticLinearWeights,
    pub norm2: Fp8StaticDeviceLayerNorm,
    pub fc1: Fp8StaticLinearWeights,
    pub fc2: Fp8StaticLinearWeights,
}

#[derive(Debug)]
pub struct Fp8StaticDeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Fp8StaticLinearWeights,
    pub output: Fp8StaticLinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Fp8StaticLinearWeights,
    pub down: Fp8StaticLinearWeights,
}

#[derive(Debug)]
pub struct Fp8StaticDeviceActionLayer {
    pub input_modulation: Fp8StaticLinearWeights,
    pub qkv: Fp8StaticLinearWeights,
    pub output: Fp8StaticLinearWeights,
    pub post_attention_modulation: Fp8StaticLinearWeights,
    pub gate_up: Fp8StaticLinearWeights,
    pub down: Fp8StaticLinearWeights,
}

#[derive(Debug)]
pub struct Fp8StaticWeights {
    pub patch_embedding: Fp8StaticLinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Fp8StaticDeviceVisionBlock>,
    pub vision_post_norm: Fp8StaticDeviceLayerNorm,
    pub multimodal_projector: Fp8StaticLinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Fp8StaticDeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Fp8StaticDeviceActionLayer>,
    pub action_final_modulation: Fp8StaticLinearWeights,
    pub action_in: Fp8StaticLinearWeights,
    pub action_out: Fp8StaticLinearWeights,
    pub time_mlp_in: Fp8StaticLinearWeights,
    pub time_mlp_out: Fp8StaticLinearWeights,
}

impl Fp8StaticWeights {
    pub fn from_host(weights: &Pi05Weights, backend: &Context) -> Result<Self> {
        let vision_layers = weights
            .vision
            .blocks
            .iter()
            .map(|layer| Fp8StaticDeviceVisionBlock::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        let language_layers = weights
            .language_layers
            .iter()
            .map(|layer| Fp8StaticDeviceLanguageLayer::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        let action_layers = weights
            .action_layers
            .iter()
            .map(|layer| Fp8StaticDeviceActionLayer::from_host(layer, backend))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding: Fp8StaticLinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: fp16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers,
            vision_post_norm: Fp8StaticDeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Fp8StaticLinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: fp16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers,
            language_final_norm_scale: fp16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers,
            action_final_modulation: modulation_to_device(&weights.action_final_norm, backend)?,
            action_in: Fp8StaticLinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Fp8StaticLinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Fp8StaticLinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Fp8StaticLinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Fp8StaticDeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            weight: fp16_to_device(&weights.weight, backend)?,
            bias: fp16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Fp8StaticDeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            norm1: Fp8StaticDeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Fp8StaticLinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: Fp8StaticLinearWeights::from_host(&weights.output, backend)?,
            norm2: Fp8StaticDeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Fp8StaticLinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Fp8StaticLinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Fp8StaticDeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_norm_scale: fp16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Fp8StaticLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp8StaticLinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: fp16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Fp8StaticLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Fp8StaticLinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl Fp8StaticDeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_modulation: modulation_to_device(&weights.input_norm, backend)?,
            qkv: Fp8StaticLinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Fp8StaticLinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_modulation: modulation_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Fp8StaticLinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Fp8StaticLinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn modulation_to_device(
    weights: &AdaRmsNormWeights,
    backend: &Context,
) -> Result<Fp8StaticLinearWeights> {
    Fp8StaticLinearWeights::from_host(&weights.modulation, backend)
}

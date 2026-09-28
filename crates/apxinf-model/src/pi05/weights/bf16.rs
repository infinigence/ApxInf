//! bf16 device weights: model aggregates and linear storage.
mod linear {
    //! Device-ready BF16 linear weights for π0.5.

    use apxinf_core::{DType, Error, Result, Tensor};

    use crate::pi05::backend::{self, Context};
    use crate::pi05::weights::packing::concat_host_2d;
    use crate::pi05::LinearWeights;

    #[derive(Debug)]
    pub struct Bf16LinearWeights {
        /// Canonical row-major `[input, output]` matrix. Candidate-specific
        /// preparation belongs to cuda-new and is never stored in model state.
        pub weight: Tensor,
        pub bias: Option<Tensor>,
    }

    impl Bf16LinearWeights {
        pub fn from_host(linear: &LinearWeights, context: &Context) -> Result<Self> {
            Self::from_host_parts(&[linear], context)
        }

        /// Pack projections along the output dimension so QKV and gate/up each
        /// remain one tensor-core GEMM, matching the static FP8 computation schedule.
        pub fn from_host_parts(linears: &[&LinearWeights], context: &Context) -> Result<Self> {
            if linears.is_empty() {
                return Err(Error::Other(
                    "cannot pack an empty BF16 linear group".into(),
                ));
            }
            let plain_weight = concat_host_2d(
                &linears
                    .iter()
                    .map(|linear| &linear.weight)
                    .collect::<Vec<_>>(),
            )?;
            let weight = bf16_to_device(&plain_weight, context)?;
            let bias = if linears.iter().all(|linear| linear.bias.is_none()) {
                None
            } else if linears.iter().all(|linear| linear.bias.is_some()) {
                Some(backend::to_device(context, &concat_biases_bf16(
                    &linears
                        .iter()
                        .map(|linear| linear.bias.as_ref().unwrap())
                        .collect::<Vec<_>>(),
                )?)?)
            } else {
                return Err(Error::Other(
                    "cannot pack BF16 projections with mixed bias presence".into(),
                ));
            };
            Ok(Self {
                weight,
                bias,
            })
        }
    }

    fn to_bf16_host(tensor: &Tensor) -> Result<Tensor> {
        if tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other(
                "cannot convert scale-less E4M3 data to BF16".into(),
            ));
        }
        let values = tensor
            .to_f32_vec()?
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
        Tensor::from_bf16(tensor.shape().dims().to_vec(), &values)
    }

    pub fn bf16_to_device(tensor: &Tensor, context: &Context) -> Result<Tensor> {
        backend::to_device(context, &to_bf16_host(tensor)?)
    }

    fn concat_biases_bf16(tensors: &[&Tensor]) -> Result<Tensor> {
        let mut values = Vec::new();
        for tensor in tensors {
            if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
                return Err(Error::Other(
                    "packed BF16 biases must be non-FP8 vectors".into(),
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
        fn packs_qkv_as_native_bf16() {
            let linear = |weight: &[f32], shape: [usize; 2], bias: &[f32]| LinearWeights {
                weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
                bias: Some(Tensor::from_f32(vec![bias.len()], bias).unwrap()),
            };
            let q = linear(&[1., 2., 3., 4.], [2, 2], &[1., 2.]);
            let k = linear(&[5., 6.], [2, 1], &[3.]);
            let v = linear(&[7., 8.], [2, 1], &[4.]);
            let plain = concat_host_2d(&[&q.weight, &k.weight, &v.weight]).unwrap();
            let packed = to_bf16_host(&plain).unwrap();
            assert_eq!(packed.shape().dims(), &[2, 4]);
            assert_eq!(packed.dtype(), DType::BF16);
            let biases = concat_biases_bf16(&[
                q.bias.as_ref().unwrap(),
                k.bias.as_ref().unwrap(),
                v.bias.as_ref().unwrap(),
            ])
            .unwrap();
            assert_eq!(
                biases.to_f32_vec().unwrap(),
                vec![1., 2., 3., 4.]
            );
        }
    }
}
pub use linear::*;
// Fully materialized native-BF16 π0.5 weights.

use apxinf_core::{Result, Tensor};
use crate::pi05::backend::Context;

use crate::pi05::{
    ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights, Pi05Weights,
    VisionBlockWeights,
};

#[derive(Debug)]
pub struct Bf16DeviceLayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

#[derive(Debug)]
pub struct Bf16DeviceVisionBlock {
    pub norm1: Bf16DeviceLayerNorm,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub norm2: Bf16DeviceLayerNorm,
    pub fc1: Bf16LinearWeights,
    pub fc2: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16DeviceLanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Bf16LinearWeights,
    pub down: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16DeviceActionLayer {
    pub input_modulation: Bf16LinearWeights,
    pub qkv: Bf16LinearWeights,
    pub output: Bf16LinearWeights,
    pub post_attention_modulation: Bf16LinearWeights,
    pub gate_up: Bf16LinearWeights,
    pub down: Bf16LinearWeights,
}

#[derive(Debug)]
pub struct Bf16Weights {
    pub patch_embedding: Bf16LinearWeights,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<Bf16DeviceVisionBlock>,
    pub vision_post_norm: Bf16DeviceLayerNorm,
    pub multimodal_projector: Bf16LinearWeights,
    pub token_embedding: Tensor,
    pub language_layers: Vec<Bf16DeviceLanguageLayer>,
    pub language_final_norm_scale: Tensor,
    pub action_layers: Vec<Bf16DeviceActionLayer>,
    pub action_final_modulation: Bf16LinearWeights,
    pub action_in: Bf16LinearWeights,
    pub action_out: Bf16LinearWeights,
    pub time_mlp_in: Bf16LinearWeights,
    pub time_mlp_out: Bf16LinearWeights,
}

impl Bf16Weights {
    pub fn from_host(weights: &Pi05Weights, backend: &Context) -> Result<Self> {
        Ok(Self {
            patch_embedding: Bf16LinearWeights::from_host(
                &weights.vision.patch_embedding,
                backend,
            )?,
            position_embedding: bf16_to_device(&weights.vision.position_embedding, backend)?,
            vision_layers: weights
                .vision
                .blocks
                .iter()
                .map(|layer| Bf16DeviceVisionBlock::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            vision_post_norm: Bf16DeviceLayerNorm::from_host(
                &weights.vision.post_layer_norm,
                backend,
            )?,
            multimodal_projector: Bf16LinearWeights::from_host(
                &weights.vision.multimodal_projector,
                backend,
            )?,
            token_embedding: bf16_to_device(&weights.vision.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| Bf16DeviceLanguageLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            language_final_norm_scale: bf16_to_device(&weights.language_final_norm_scale, backend)?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| Bf16DeviceActionLayer::from_host(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            action_final_modulation: modulation_to_device(&weights.action_final_norm, backend)?,
            action_in: Bf16LinearWeights::from_host(&weights.action_in, backend)?,
            action_out: Bf16LinearWeights::from_host(&weights.action_out, backend)?,
            time_mlp_in: Bf16LinearWeights::from_host(&weights.time_mlp_in, backend)?,
            time_mlp_out: Bf16LinearWeights::from_host(&weights.time_mlp_out, backend)?,
        })
    }
}

impl Bf16DeviceLayerNorm {
    fn from_host(weights: &LayerNormWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            weight: bf16_to_device(&weights.weight, backend)?,
            bias: bf16_to_device(&weights.bias, backend)?,
        })
    }
}

impl Bf16DeviceVisionBlock {
    fn from_host(weights: &VisionBlockWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            norm1: Bf16DeviceLayerNorm::from_host(&weights.norm1, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[&weights.q, &weights.k, &weights.v],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.output, backend)?,
            norm2: Bf16DeviceLayerNorm::from_host(&weights.norm2, backend)?,
            fc1: Bf16LinearWeights::from_host(&weights.fc1, backend)?,
            fc2: Bf16LinearWeights::from_host(&weights.fc2, backend)?,
        })
    }
}

impl Bf16DeviceLanguageLayer {
    fn from_host(weights: &LanguageLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_norm_scale: bf16_to_device(&weights.input_norm_scale, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_norm_scale: bf16_to_device(&weights.post_attention_norm_scale, backend)?,
            gate_up: Bf16LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Bf16LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

impl Bf16DeviceActionLayer {
    fn from_host(weights: &ActionLayerWeights, backend: &Context) -> Result<Self> {
        Ok(Self {
            input_modulation: modulation_to_device(&weights.input_norm, backend)?,
            qkv: Bf16LinearWeights::from_host_parts(
                &[
                    &weights.attention.q,
                    &weights.attention.k,
                    &weights.attention.v,
                ],
                backend,
            )?,
            output: Bf16LinearWeights::from_host(&weights.attention.output, backend)?,
            post_attention_modulation: modulation_to_device(&weights.post_attention_norm, backend)?,
            gate_up: Bf16LinearWeights::from_host_parts(
                &[&weights.mlp.gate, &weights.mlp.up],
                backend,
            )?,
            down: Bf16LinearWeights::from_host(&weights.mlp.down, backend)?,
        })
    }
}

fn modulation_to_device(
    weights: &AdaRmsNormWeights,
    backend: &Context,
) -> Result<Bf16LinearWeights> {
    Bf16LinearWeights::from_host(&weights.modulation, backend)
}

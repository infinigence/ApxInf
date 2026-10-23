//! π0.5 weights on a HIP device, in BF16.
//!
//! Built from this port's [`Checkpoint`]: Q/K/V and gate/up are concatenated
//! along the output dimension so each stays one GEMM, the packing the CUDA BF16
//! path also uses, and every matrix keeps the row-major `[in, out]` the
//! checkpoint reader produced.

use apxinf_core::{Error, Result, Tensor};
use apxinf_hip::HipContext;
use half::bf16;

use super::checkpoint::{self as host, Checkpoint};

/// A row-major `[in, out]` projection and its optional bias.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct VisionLayer {
    pub norm1: LayerNorm,
    pub qkv: Linear,
    pub output: Linear,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct LanguageLayer {
    pub input_norm_scale: Tensor,
    pub qkv: Linear,
    pub output: Linear,
    pub post_attention_norm_scale: Tensor,
    pub gate_up: Linear,
    pub down: Linear,
}

pub struct ActionLayer {
    /// Conditioning → `[3 * width]` scale/shift/gate for the attention input.
    pub input_modulation: Linear,
    pub qkv: Linear,
    pub output: Linear,
    /// Conditioning → `[3 * width]` scale/shift/gate for the MLP input.
    pub post_attention_modulation: Linear,
    pub gate_up: Linear,
    pub down: Linear,
}

pub struct HipPi05Weights {
    pub patch_embedding: Linear,
    pub position_embedding: Tensor,
    pub vision_layers: Vec<VisionLayer>,
    pub vision_post_norm: LayerNorm,
    pub multimodal_projector: Linear,
    pub token_embedding: Tensor,
    pub language_layers: Vec<LanguageLayer>,
    pub action_layers: Vec<ActionLayer>,
    pub action_final_modulation: Linear,
    pub action_in: Linear,
    pub action_out: Linear,
    pub time_mlp_in: Linear,
    pub time_mlp_out: Linear,
}

impl HipPi05Weights {
    pub fn from_checkpoint(checkpoint: &Checkpoint, ctx: &HipContext) -> Result<Self> {
        let c = checkpoint;
        Ok(Self {
            patch_embedding: linear(&[&c.patch_embedding], ctx)?,
            position_embedding: bf16_to_device(&c.position_embedding, ctx)?,
            vision_layers: c
                .vision_blocks
                .iter()
                .map(|layer| vision_layer(layer, ctx))
                .collect::<Result<_>>()?,
            vision_post_norm: layer_norm(&c.vision_post_norm, ctx)?,
            multimodal_projector: linear(&[&c.multimodal_projector], ctx)?,
            token_embedding: bf16_to_device(&c.token_embedding, ctx)?,
            language_layers: c
                .language_layers
                .iter()
                .map(|layer| language_layer(layer, ctx))
                .collect::<Result<_>>()?,
            action_layers: c
                .action_layers
                .iter()
                .map(|layer| action_layer(layer, ctx))
                .collect::<Result<_>>()?,
            action_final_modulation: linear(&[&c.action_final_modulation], ctx)?,
            action_in: linear(&[&c.action_in], ctx)?,
            action_out: linear(&[&c.action_out], ctx)?,
            time_mlp_in: linear(&[&c.time_mlp_in], ctx)?,
            time_mlp_out: linear(&[&c.time_mlp_out], ctx)?,
        })
    }
}

fn vision_layer(layer: &host::VisionBlock, ctx: &HipContext) -> Result<VisionLayer> {
    Ok(VisionLayer {
        norm1: layer_norm(&layer.norm1, ctx)?,
        qkv: linear(&[&layer.q, &layer.k, &layer.v], ctx)?,
        output: linear(&[&layer.output], ctx)?,
        norm2: layer_norm(&layer.norm2, ctx)?,
        fc1: linear(&[&layer.fc1], ctx)?,
        fc2: linear(&[&layer.fc2], ctx)?,
    })
}

fn language_layer(layer: &host::LanguageLayer, ctx: &HipContext) -> Result<LanguageLayer> {
    let attention = &layer.attention;
    Ok(LanguageLayer {
        input_norm_scale: bf16_to_device(&layer.input_norm_scale, ctx)?,
        qkv: linear(&[&attention.q, &attention.k, &attention.v], ctx)?,
        output: linear(&[&attention.output], ctx)?,
        post_attention_norm_scale: bf16_to_device(&layer.post_attention_norm_scale, ctx)?,
        gate_up: linear(&[&layer.mlp.gate, &layer.mlp.up], ctx)?,
        down: linear(&[&layer.mlp.down], ctx)?,
    })
}

fn action_layer(layer: &host::ActionLayer, ctx: &HipContext) -> Result<ActionLayer> {
    let attention = &layer.attention;
    Ok(ActionLayer {
        input_modulation: linear(&[&layer.input_modulation], ctx)?,
        qkv: linear(&[&attention.q, &attention.k, &attention.v], ctx)?,
        output: linear(&[&attention.output], ctx)?,
        post_attention_modulation: linear(&[&layer.post_attention_modulation], ctx)?,
        gate_up: linear(&[&layer.mlp.gate, &layer.mlp.up], ctx)?,
        down: linear(&[&layer.mlp.down], ctx)?,
    })
}

fn layer_norm(norm: &host::LayerNorm, ctx: &HipContext) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: bf16_to_device(&norm.weight, ctx)?,
        bias: bf16_to_device(&norm.bias, ctx)?,
    })
}

/// Pack projections along the output dimension; biases follow the same order.
fn linear(parts: &[&host::Linear], ctx: &HipContext) -> Result<Linear> {
    let weight = concat_columns(&parts.iter().map(|part| &part.weight).collect::<Vec<_>>())?;
    let bias = if parts.iter().all(|part| part.bias.is_none()) {
        None
    } else if parts.iter().all(|part| part.bias.is_some()) {
        let mut values = Vec::new();
        for part in parts {
            values.extend(part.bias.as_ref().unwrap().to_f32_vec()?);
        }
        Some(bf16_to_device(
            &Tensor::from_f32(vec![values.len()], &values)?,
            ctx,
        )?)
    } else {
        return Err(Error::Other(
            "π0.5 HIP: cannot pack projections with mixed bias presence".into(),
        ));
    };
    Ok(Linear {
        weight: bf16_to_device(&weight, ctx)?,
        bias,
    })
}

/// `[rows, a] ++ [rows, b] ++ … → [rows, a + b + …]`, in F32.
fn concat_columns(tensors: &[&Tensor]) -> Result<Tensor> {
    let rows = match tensors.first().map(|t| t.shape().dims()) {
        Some([rows, _]) => *rows,
        _ => return Err(Error::Other("π0.5 HIP: packing needs 2-D weights".into())),
    };
    let mut widths = Vec::with_capacity(tensors.len());
    let mut sources = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        match tensor.shape().dims() {
            [r, width] if *r == rows => widths.push(*width),
            _ => {
                return Err(Error::Other(
                    "π0.5 HIP: packed projections differ in input width".into(),
                ))
            }
        }
        sources.push(tensor.to_f32_vec()?);
    }
    let total: usize = widths.iter().sum();
    let mut out = Vec::with_capacity(rows * total);
    for row in 0..rows {
        for (source, &width) in sources.iter().zip(&widths) {
            out.extend_from_slice(&source[row * width..(row + 1) * width]);
        }
    }
    Tensor::from_f32(vec![rows, total], &out)
}

/// Round to BF16 on the host (round-to-nearest-even) and upload.
fn bf16_to_device(tensor: &Tensor, ctx: &HipContext) -> Result<Tensor> {
    let values: Vec<bf16> = tensor
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    ctx.to_device(&Tensor::from_bf16(tensor.shape().dims().to_vec(), &values)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_projections_column_wise() {
        let q = Tensor::from_f32(vec![2, 2], &[1., 2., 3., 4.]).unwrap();
        let k = Tensor::from_f32(vec![2, 1], &[5., 6.]).unwrap();
        let packed = concat_columns(&[&q, &k]).unwrap();
        assert_eq!(packed.shape().dims(), &[2, 3]);
        assert_eq!(packed.to_f32_vec().unwrap(), vec![1., 2., 5., 3., 4., 6.]);
        let tall = Tensor::from_f32(vec![3, 1], &[0.; 3]).unwrap();
        assert!(concat_columns(&[&q, &tall]).is_err());
    }
}

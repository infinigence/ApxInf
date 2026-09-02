//! GR00T-owned Cosmos/Qwen3-VL backbone weights.

use super::weights::{take, transpose_2d, upload_linear};
use super::{LayerNormWeights, LinearWeights};
use apxinf_core::{Backend, DType, Error, Result, Tensor};
use std::collections::HashMap;

pub struct GrootLanguageLayer {
    pub input_norm: Tensor,
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub post_norm: Tensor,
    pub gate: LinearWeights,
    pub up: LinearWeights,
    pub down: LinearWeights,
}

pub struct GrootVisionBlock {
    pub norm1: LayerNormWeights,
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
    pub norm2: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

pub struct GrootVisionMerger {
    pub norm: LayerNormWeights,
    pub fc1: LinearWeights,
    pub fc2: LinearWeights,
}

pub struct GrootVisionWeights {
    pub patch: LinearWeights,
    pub position_one_view: Tensor,
    pub position_two_view: Tensor,
    pub blocks: Vec<GrootVisionBlock>,
    pub primary_merger: GrootVisionMerger,
    pub deepstack_mergers: Vec<GrootVisionMerger>,
}

pub struct GrootN17BackboneWeights {
    pub embedding: Tensor,
    pub language_layers: Vec<GrootLanguageLayer>,
    pub output_norm: Tensor,
    pub vision: GrootVisionWeights,
}

impl GrootN17BackboneWeights {
    pub fn from_map(tensors: &mut HashMap<String, Tensor>) -> Result<Self> {
        const ROOT: &str = "backbone.model.model";
        let mut language_layers = Vec::with_capacity(16);
        for index in 0..16 {
            let prefix = format!("{ROOT}.language_model.layers.{index}");
            language_layers.push(GrootLanguageLayer {
                input_norm: take(tensors, &format!("{prefix}.input_layernorm.weight"))?,
                q: take_linear(tensors, &format!("{prefix}.self_attn.q_proj"), false)?,
                k: take_linear(tensors, &format!("{prefix}.self_attn.k_proj"), false)?,
                v: take_linear(tensors, &format!("{prefix}.self_attn.v_proj"), false)?,
                output: take_linear(tensors, &format!("{prefix}.self_attn.o_proj"), false)?,
                q_norm: take(tensors, &format!("{prefix}.self_attn.q_norm.weight"))?,
                k_norm: take(tensors, &format!("{prefix}.self_attn.k_norm.weight"))?,
                post_norm: take(
                    tensors,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                gate: take_linear(tensors, &format!("{prefix}.mlp.gate_proj"), false)?,
                up: take_linear(tensors, &format!("{prefix}.mlp.up_proj"), false)?,
                down: take_linear(tensors, &format!("{prefix}.mlp.down_proj"), false)?,
            });
        }
        let mut blocks = Vec::with_capacity(24);
        for index in 0..24 {
            let prefix = format!("{ROOT}.visual.blocks.{index}");
            let qkv_weight = take(tensors, &format!("{prefix}.attn.qkv.weight"))?;
            let qkv_bias = take(tensors, &format!("{prefix}.attn.qkv.bias"))?;
            blocks.push(GrootVisionBlock {
                norm1: take_norm(tensors, &format!("{prefix}.norm1"))?,
                q: split_linear(&qkv_weight, &qkv_bias, 0)?,
                k: split_linear(&qkv_weight, &qkv_bias, 1)?,
                v: split_linear(&qkv_weight, &qkv_bias, 2)?,
                output: take_linear(tensors, &format!("{prefix}.attn.proj"), true)?,
                norm2: take_norm(tensors, &format!("{prefix}.norm2"))?,
                fc1: take_linear(tensors, &format!("{prefix}.mlp.linear_fc1"), true)?,
                fc2: take_linear(tensors, &format!("{prefix}.mlp.linear_fc2"), true)?,
            });
        }
        let patch_raw = take(tensors, &format!("{ROOT}.visual.patch_embed.proj.weight"))?;
        let patch_weight = flatten_transpose(&patch_raw, 1024, 1536)?;
        let patch_bias = take(tensors, &format!("{ROOT}.visual.patch_embed.proj.bias"))?;
        Ok(Self {
            embedding: take(
                tensors,
                &format!("{ROOT}.language_model.embed_tokens.weight"),
            )?,
            language_layers,
            output_norm: take(tensors, &format!("{ROOT}.language_model.norm.weight"))?,
            vision: GrootVisionWeights {
                patch: LinearWeights {
                    weight: patch_weight,
                    bias: Some(patch_bias),
                },
                position_one_view: take(tensors, &format!("{ROOT}.visual.pos_embed.weight"))?,
                position_two_view: Tensor::zeros(vec![1], DType::BF16),
                blocks,
                primary_merger: take_merger(tensors, &format!("{ROOT}.visual.merger"))?,
                deepstack_mergers: (0..3)
                    .map(|i| {
                        take_merger(tensors, &format!("{ROOT}.visual.deepstack_merger_list.{i}"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            },
        })
    }

    pub fn to_device(mut self, backend: &dyn Backend) -> Result<Self> {
        self.embedding = backend.to_device(&self.embedding)?;
        self.output_norm = backend.to_device(&self.output_norm)?;
        for layer in &mut self.language_layers {
            layer.input_norm = backend.to_device(&layer.input_norm)?;
            layer.q_norm = backend.to_device(&layer.q_norm)?;
            layer.k_norm = backend.to_device(&layer.k_norm)?;
            layer.post_norm = backend.to_device(&layer.post_norm)?;
            for linear in [
                &mut layer.q,
                &mut layer.k,
                &mut layer.v,
                &mut layer.output,
                &mut layer.gate,
                &mut layer.up,
                &mut layer.down,
            ] {
                upload_linear(linear, backend)?;
            }
        }
        upload_linear(&mut self.vision.patch, backend)?;
        let (one_view, two_view) = fixed_vision_positions(&self.vision.position_one_view)?;
        self.vision.position_one_view = backend.to_device(&one_view)?;
        self.vision.position_two_view = backend.to_device(&two_view)?;
        for block in &mut self.vision.blocks {
            upload_norm(&mut block.norm1, backend)?;
            upload_norm(&mut block.norm2, backend)?;
            for linear in [
                &mut block.q,
                &mut block.k,
                &mut block.v,
                &mut block.output,
                &mut block.fc1,
                &mut block.fc2,
            ] {
                upload_linear(linear, backend)?;
            }
        }
        upload_merger(&mut self.vision.primary_merger, backend)?;
        for merger in &mut self.vision.deepstack_mergers {
            upload_merger(merger, backend)?;
        }
        Ok(self)
    }
}

fn take_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
    bias: bool,
) -> Result<LinearWeights> {
    Ok(LinearWeights {
        weight: transpose_2d(&take(tensors, &format!("{prefix}.weight"))?)?,
        bias: bias
            .then(|| take(tensors, &format!("{prefix}.bias")))
            .transpose()?,
    })
}

fn take_norm(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<LayerNormWeights> {
    Ok(LayerNormWeights {
        weight: take(tensors, &format!("{prefix}.weight"))?,
        bias: take(tensors, &format!("{prefix}.bias"))?,
    })
}

fn take_merger(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<GrootVisionMerger> {
    Ok(GrootVisionMerger {
        norm: take_norm(tensors, &format!("{prefix}.norm"))?,
        fc1: take_linear(tensors, &format!("{prefix}.linear_fc1"), true)?,
        fc2: take_linear(tensors, &format!("{prefix}.linear_fc2"), true)?,
    })
}

fn upload_norm(value: &mut LayerNormWeights, backend: &dyn Backend) -> Result<()> {
    value.weight = backend.to_device(&value.weight)?;
    value.bias = backend.to_device(&value.bias)?;
    Ok(())
}

fn upload_merger(value: &mut GrootVisionMerger, backend: &dyn Backend) -> Result<()> {
    upload_norm(&mut value.norm, backend)?;
    upload_linear(&mut value.fc1, backend)?;
    upload_linear(&mut value.fc2, backend)
}

fn split_linear(weight: &Tensor, bias: &Tensor, part: usize) -> Result<LinearWeights> {
    let width = weight.shape().dims()[0] / 3;
    Ok(LinearWeights {
        weight: transpose_2d(&slice_rows(weight, part * width, width)?)?,
        bias: Some(slice_rows(bias, part * width, width)?),
    })
}

fn slice_rows(tensor: &Tensor, start: usize, rows: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    let stride: usize = dims.get(1..).unwrap_or(&[]).iter().product();
    let mut shape = dims.to_vec();
    shape[0] = rows;
    match tensor.dtype() {
        DType::BF16 => Tensor::from_bf16(
            shape,
            &tensor.as_bf16()?[start * stride..(start + rows) * stride],
        ),
        DType::F32 => Tensor::from_f32(
            shape,
            &tensor.as_f32()?[start * stride..(start + rows) * stride],
        ),
        dtype => Err(Error::Other(format!("unsupported backbone dtype {dtype}"))),
    }
}

fn flatten_transpose(tensor: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let reshaped = match tensor.dtype() {
        DType::BF16 => Tensor::from_bf16(vec![rows, cols], tensor.as_bf16()?)?,
        DType::F32 => Tensor::from_f32(vec![rows, cols], tensor.as_f32()?)?,
        dtype => return Err(Error::Other(format!("unsupported patch dtype {dtype}"))),
    };
    transpose_2d(&reshaped)
}

fn fixed_vision_positions(table: &Tensor) -> Result<(Tensor, Tensor)> {
    const SOURCE: usize = 48;
    const TARGET: usize = 16;
    const WIDTH: usize = 1024;
    let values = table.to_f32_vec()?;
    let mut one_view = vec![0.0f32; TARGET * TARGET * WIDTH];
    for row in 0..TARGET {
        let yf = row as f32 * (SOURCE - 1) as f32 / (TARGET - 1) as f32;
        let y0 = yf.floor() as usize;
        let y1 = (y0 + 1).min(SOURCE - 1);
        let dy = yf - y0 as f32;
        for col in 0..TARGET {
            let xf = col as f32 * (SOURCE - 1) as f32 / (TARGET - 1) as f32;
            let x0 = xf.floor() as usize;
            let x1 = (x0 + 1).min(SOURCE - 1);
            let dx = xf - x0 as f32;
            // Transformers casts the interpolation weights to the embedding
            // table's BF16 dtype before multiplying, then performs the four
            // additions in BF16. Preserve those rounding boundaries: doing
            // the complete bilinear expression in f32 measurably changes the
            // frozen vision prefix after 24 residual blocks.
            let round = |value: f32| half::bf16::from_f32(value).to_f32();
            let weights = [
                round((1.0 - dy) * (1.0 - dx)),
                round((1.0 - dy) * dx),
                round(dy * (1.0 - dx)),
                round(dy * dx),
            ];
            for channel in 0..WIDTH {
                let at = |y, x| values[(y * SOURCE + x) * WIDTH + channel];
                let products = [
                    round(at(y0, x0) * weights[0]),
                    round(at(y0, x1) * weights[1]),
                    round(at(y1, x0) * weights[2]),
                    round(at(y1, x1) * weights[3]),
                ];
                let sum =
                    round(round(round(products[0] + products[1]) + products[2]) + products[3]);
                one_view[(row * TARGET + col) * WIDTH + channel] = sum;
            }
        }
    }
    // Processor patch order already groups each 2x2 merge tile contiguously.
    let mut permuted = Vec::with_capacity(one_view.len());
    for macro_row in 0..TARGET / 2 {
        for macro_col in 0..TARGET / 2 {
            for inner_row in 0..2 {
                for inner_col in 0..2 {
                    let offset =
                        ((macro_row * 2 + inner_row) * TARGET + macro_col * 2 + inner_col) * WIDTH;
                    permuted.extend_from_slice(&one_view[offset..offset + WIDTH]);
                }
            }
        }
    }
    let one_values = permuted
        .into_iter()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    let mut two_values = Vec::with_capacity(one_values.len() * 2);
    two_values.extend_from_slice(&one_values);
    two_values.extend_from_slice(&one_values);
    Ok((
        Tensor::from_bf16(vec![256, WIDTH], &one_values)?,
        Tensor::from_bf16(vec![512, WIDTH], &two_values)?,
    ))
}

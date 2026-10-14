//! BF16 Qwen3.5 vision-tower weights from the same 27B checkpoint.
//!
//! The language stack is quantized, but every `model.visual.*` tensor in the
//! shipped checkpoint is BF16. Stable transposes happen once during load.

use std::collections::HashMap;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext};

use super::config::*;

pub(crate) struct VisionBlockWeights {
    pub(crate) norm1_w: Tensor,
    pub(crate) norm1_b: Tensor,
    pub(crate) qkv_w: Tensor,
    pub(crate) qkv_b: Tensor,
    pub(crate) proj_w: Tensor,
    pub(crate) proj_b: Tensor,
    pub(crate) norm2_w: Tensor,
    pub(crate) norm2_b: Tensor,
    pub(crate) fc1_w: Tensor,
    pub(crate) fc1_b: Tensor,
    pub(crate) fc2_w: Tensor,
    pub(crate) fc2_b: Tensor,
}

pub(crate) struct VisionWeights {
    /// Flattened Conv3d `[patch_width, hidden]`.
    pub(crate) patch_w: Tensor,
    pub(crate) patch_b: Tensor,
    pub(crate) pos_embed: Tensor,
    pub(crate) blocks: Vec<VisionBlockWeights>,
    pub(crate) merger_norm_w: Tensor,
    pub(crate) merger_norm_b: Tensor,
    pub(crate) merger_fc1_w: Tensor,
    pub(crate) merger_fc1_b: Tensor,
    pub(crate) merger_fc2_w: Tensor,
    pub(crate) merger_fc2_b: Tensor,
}

fn get<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| Error::Other(format!("qwen38 vision weights: missing {name}")))
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "qwen38 vision transpose expected rank 2, got {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::BF16 => {
            let source = tensor.as_bf16()?;
            let mut destination = vec![half::bf16::from_f32(0.0); rows * cols];
            for row in 0..rows {
                for column in 0..cols {
                    destination[column * rows + row] = source[row * cols + column];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &destination)
        }
        DType::F32 => {
            let source = tensor.as_f32()?;
            let mut destination = vec![0.0f32; rows * cols];
            for row in 0..rows {
                for column in 0..cols {
                    destination[column * rows + row] = source[row * cols + column];
                }
            }
            Tensor::from_f32(vec![cols, rows], &destination)
        }
        dtype => Err(Error::Other(format!(
            "qwen38 vision transpose does not support {dtype}"
        ))),
    }
}

fn upload(ctx: &CudaContext, tensor: &Tensor) -> Result<Tensor> {
    transfers::to_cuda(tensor, ctx.device_id())
}

fn upload_named(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    name: &str,
) -> Result<Tensor> {
    upload(ctx, get(tensors, name)?)
}

fn upload_transposed(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    name: &str,
) -> Result<Tensor> {
    upload(ctx, &transpose_2d(get(tensors, name)?)?)
}

impl VisionWeights {
    pub(crate) fn load(tensors: &HashMap<String, Tensor>, ctx: &CudaContext) -> Result<Self> {
        let patch = get(tensors, "model.visual.patch_embed.proj.weight")?;
        let patch_dims = patch.shape().dims();
        let patch_width =
            VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
        if patch_dims.len() != 5
            || patch_dims[0] != VISION_HIDDEN
            || patch_dims.iter().product::<usize>() != VISION_HIDDEN * patch_width
        {
            return Err(Error::Other(format!(
                "qwen38 vision patch weight has shape {patch_dims:?}"
            )));
        }
        if patch.dtype() != DType::BF16 {
            return Err(Error::Other(format!(
                "qwen38 vision patch weight must be BF16, got {}",
                patch.dtype()
            )));
        }
        let patch_2d = patch.reshape(vec![VISION_HIDDEN, patch_width])?;
        let patch_w = upload(ctx, &transpose_2d(&patch_2d)?)?;

        let mut blocks = Vec::with_capacity(VISION_DEPTH);
        for index in 0..VISION_DEPTH {
            let prefix = format!("model.visual.blocks.{index}");
            blocks.push(VisionBlockWeights {
                norm1_w: upload_named(ctx, tensors, &format!("{prefix}.norm1.weight"))?,
                norm1_b: upload_named(ctx, tensors, &format!("{prefix}.norm1.bias"))?,
                qkv_w: upload_transposed(ctx, tensors, &format!("{prefix}.attn.qkv.weight"))?,
                qkv_b: upload_named(ctx, tensors, &format!("{prefix}.attn.qkv.bias"))?,
                proj_w: upload_transposed(ctx, tensors, &format!("{prefix}.attn.proj.weight"))?,
                proj_b: upload_named(ctx, tensors, &format!("{prefix}.attn.proj.bias"))?,
                norm2_w: upload_named(ctx, tensors, &format!("{prefix}.norm2.weight"))?,
                norm2_b: upload_named(ctx, tensors, &format!("{prefix}.norm2.bias"))?,
                fc1_w: upload_transposed(
                    ctx,
                    tensors,
                    &format!("{prefix}.mlp.linear_fc1.weight"),
                )?,
                fc1_b: upload_named(ctx, tensors, &format!("{prefix}.mlp.linear_fc1.bias"))?,
                fc2_w: upload_transposed(
                    ctx,
                    tensors,
                    &format!("{prefix}.mlp.linear_fc2.weight"),
                )?,
                fc2_b: upload_named(ctx, tensors, &format!("{prefix}.mlp.linear_fc2.bias"))?,
            });
        }

        let merged = VISION_HIDDEN * VISION_MERGE * VISION_MERGE;
        let merger_fc1 = transpose_2d(get(tensors, "model.visual.merger.linear_fc1.weight")?)?;
        let merger_fc2 = transpose_2d(get(tensors, "model.visual.merger.linear_fc2.weight")?)?;
        if merger_fc1.shape().dims() != [merged, merged]
            || merger_fc2.shape().dims() != [merged, VISION_OUT_HIDDEN]
        {
            return Err(Error::Other(format!(
                "qwen38 vision merger shape mismatch: {:?}, {:?}",
                merger_fc1.shape().dims(),
                merger_fc2.shape().dims()
            )));
        }

        Ok(Self {
            patch_w,
            patch_b: upload_named(ctx, tensors, "model.visual.patch_embed.proj.bias")?,
            pos_embed: upload_named(ctx, tensors, "model.visual.pos_embed.weight")?,
            blocks,
            merger_norm_w: upload_named(ctx, tensors, "model.visual.merger.norm.weight")?,
            merger_norm_b: upload_named(ctx, tensors, "model.visual.merger.norm.bias")?,
            merger_fc1_w: upload(ctx, &merger_fc1)?,
            merger_fc1_b: upload_named(ctx, tensors, "model.visual.merger.linear_fc1.bias")?,
            merger_fc2_w: upload(ctx, &merger_fc2)?,
            merger_fc2_b: upload_named(ctx, tensors, "model.visual.merger.linear_fc2.bias")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_vision_geometry_matches_the_27b_checkpoint() {
        assert_eq!(VISION_HEAD_DIM, 72);
        assert_eq!(VISION_HIDDEN * VISION_MERGE * VISION_MERGE, 4608);
        assert_eq!(VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * 16 * 16, 1536);
    }
}

use super::super::backend::{transfers, Context};
use apxinf_core::{DType, Error, Result, Tensor};
use std::collections::HashMap;

pub(in crate::dreamzero) const DIM: usize = 1280;
pub(in crate::dreamzero) const TOKENS: usize = 257;
pub(in crate::dreamzero) const HEADS: usize = 16;

pub(in crate::dreamzero) struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}
pub(in crate::dreamzero) struct Norm {
    pub weight: Tensor,
    pub bias: Tensor,
}
pub(in crate::dreamzero) struct ClipBlock {
    pub norm1: Norm,
    pub qkv: Linear,
    pub o: Linear,
    pub norm2: Norm,
    pub up: Linear,
    pub down: Linear,
}
pub(in crate::dreamzero) struct ClipWeights {
    pub patch: Tensor,
    pub class: Tensor,
    pub position: Tensor,
    pub pre_norm: Norm,
    pub blocks: Vec<ClipBlock>,
}

fn take(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let name = format!("action_head.image_encoder.model.{name}");
    let x = map
        .remove(&name)
        .ok_or_else(|| Error::Other(format!("missing DreamZero {name}")))?;
    if x.dtype() != DType::BF16 || x.shape().dims() != shape {
        return Err(Error::Other(format!(
            "DreamZero {name} must be BF16 {shape:?}, got {} {:?}",
            x.dtype(),
            x.shape().dims()
        )));
    }
    Ok(x)
}
fn upload(ctx: &Context, x: Tensor) -> Result<Tensor> {
    transfers::to_cuda(&x, ctx.device_id())
}
fn norm(ctx: &Context, map: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Norm> {
    Ok(Norm {
        weight: upload(ctx, take(map, &format!("{prefix}.weight"), &[DIM])?)?,
        bias: upload(ctx, take(map, &format!("{prefix}.bias"), &[DIM])?)?,
    })
}
fn linear(
    ctx: &Context,
    map: &mut HashMap<String, Tensor>,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<Linear> {
    let weight = take(map, &format!("{prefix}.weight"), &[output, input])?;
    Ok(Linear {
        weight: upload(ctx, weight)?,
        bias: upload(ctx, take(map, &format!("{prefix}.bias"), &[output])?)?,
    })
}
impl ClipWeights {
    pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
        let p = "visual";
        let kernel = take(map, "visual.patch_embedding.weight", &[DIM, 3, 14, 14])?;
        let src = kernel.as_bf16()?;
        // im2col columns are (height,width,channel), checkpoint OIHW.
        // Align channels to 16 to preserve the cuDNN reduction grouping.
        let mut patch = vec![half::bf16::ZERO; 16 * 14 * 14 * DIM];
        for o in 0..DIM {
            for h in 0..14 {
                for w in 0..14 {
                    for c in 0..3 {
                        patch[((h * 14 + w) * 16 + c) * DIM + o] =
                            src[((o * 3 + c) * 14 + h) * 14 + w];
                    }
                }
            }
        }
        let patch = upload(ctx, Tensor::from_bf16(vec![14 * 14 * 16, DIM], &patch)?)?;
        let class = upload(
            ctx,
            take(map, "visual.cls_embedding", &[1, 1, DIM])?.reshape(vec![1, DIM])?,
        )?;
        let position = upload(
            ctx,
            take(map, "visual.pos_embedding", &[1, TOKENS, DIM])?.reshape(vec![TOKENS, DIM])?,
        )?;
        let pre_norm = norm(ctx, map, &format!("{p}.pre_norm"))?;
        let mut blocks = Vec::with_capacity(31);
        for i in 0..32 {
            let prefix = format!("{p}.transformer.{i}");
            // Official use_31_block=True deliberately omits the final block.
            // Validate and consume its weights without uploading unused state.
            if i == 31 {
                for n in ["norm1", "norm2"] {
                    for part in ["weight", "bias"] {
                        take(map, &format!("{prefix}.{n}.{part}"), &[DIM])?;
                    }
                }
                for (n, input, output) in [
                    ("attn.to_qkv", DIM, 3 * DIM),
                    ("attn.proj", DIM, DIM),
                    ("mlp.0", DIM, 4 * DIM),
                    ("mlp.2", 4 * DIM, DIM),
                ] {
                    take(map, &format!("{prefix}.{n}.weight"), &[output, input])?;
                    take(map, &format!("{prefix}.{n}.bias"), &[output])?;
                }
                continue;
            }
            blocks.push(ClipBlock {
                norm1: norm(ctx, map, &format!("{prefix}.norm1"))?,
                qkv: linear(ctx, map, &format!("{prefix}.attn.to_qkv"), DIM, 3 * DIM)?,
                o: linear(ctx, map, &format!("{prefix}.attn.proj"), DIM, DIM)?,
                norm2: norm(ctx, map, &format!("{prefix}.norm2"))?,
                up: linear(ctx, map, &format!("{prefix}.mlp.0"), DIM, 4 * DIM)?,
                down: linear(ctx, map, &format!("{prefix}.mlp.2"), 4 * DIM, DIM)?,
            });
        }
        take(map, "log_scale", &[])?;
        take(map, "visual.head", &[DIM, 1024])?;
        take(map, "visual.post_norm.weight", &[DIM])?;
        take(map, "visual.post_norm.bias", &[DIM])?;
        Ok(Self {
            patch,
            class,
            position,
            pre_norm,
            blocks,
        })
    }
}

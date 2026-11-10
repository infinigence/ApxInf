//! Qwen3-VL vision-tower weights.
//!
//! Consumes the `model.visual.*` keys from the HF weight map. Each block
//! has: norm1 (LayerNorm w+bias), qkv (Linear w+bias), proj (Linear w+bias),
//! norm2 (LayerNorm w+bias), fc1 (Linear w+bias), fc2 (Linear w+bias).
//! The primary merger and 3 deepstack mergers are LayerNorm + 2 Linears
//! each (operating on 4096-d = 1024 * spatial_merge^2 shuffled input).

use std::collections::HashMap;

use apxinf_core::{Error, Result, Tensor};

use super::config::Qwen3VLConfig;

pub struct Qwen3VLVisionWeights {
    /// `[embed_dim, in_channels*t_p*p*p]` = `[1024, 1536]` transposed for
    /// matmul (HF stores `[1024, 3, 2, 16, 16]`; we flatten to `[1024, 1536]`).
    pub patch_embed_weight: Tensor,
    pub patch_embed_bias: Tensor,
    /// `[num_position_embeddings, hidden_size]` = `[2304, 1024]`.
    pub pos_embed: Tensor,
    pub blocks: Vec<Qwen3VLVisionBlock>,
    pub merger: Qwen3VLMerger,
    pub deepstack_mergers: Vec<Qwen3VLMerger>,
}

pub struct Qwen3VLVisionBlock {
    pub norm1_w: Tensor,
    pub norm1_b: Tensor,
    /// `[hidden, 3*hidden]` = `[1024, 3072]` transposed qkv.
    pub qkv_w: Tensor,
    pub qkv_b: Tensor,
    /// `[3*hidden, hidden]` = `[3072, 1024]`... actually `[hidden, hidden]`.
    pub proj_w: Tensor,
    pub proj_b: Tensor,
    pub norm2_w: Tensor,
    pub norm2_b: Tensor,
    /// `[hidden, inter]` = `[1024, 4096]` transposed fc1.
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    /// `[inter, hidden]` = `[4096, 1024]`... actually `[hidden, inter]` reversed.
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
}

pub struct Qwen3VLMerger {
    /// LayerNorm weight. For primary merger: `[1024]`. For deepstack: `[4096]`.
    pub norm_w: Tensor,
    pub norm_b: Tensor,
    /// `[hidden, hidden]` = `[4096, 4096]` transposed fc1.
    pub fc1_w: Tensor,
    pub fc1_b: Tensor,
    /// `[hidden, out_hidden]` = `[4096, 2048]` transposed fc2.
    pub fc2_w: Tensor,
    pub fc2_b: Tensor,
}

impl Qwen3VLVisionWeights {
    pub fn from_map(cfg: &Qwen3VLConfig, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let vc = &cfg.vision;
        let depth = vc.depth;
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let p = format!("model.visual.blocks.{i}");
            let take = |name: &str, m: &mut HashMap<String, Tensor>| -> Result<Tensor> {
                m.remove(name)
                    .ok_or_else(|| Error::Other(format!("missing {name}")))
            };
            blocks.push(Qwen3VLVisionBlock {
                norm1_w: take(&format!("{p}.norm1.weight"), &mut tensors)?,
                norm1_b: take(&format!("{p}.norm1.bias"), &mut tensors)?,
                qkv_w: reshape_linear_weight(&take(
                    &format!("{p}.attn.qkv.weight"),
                    &mut tensors,
                )?)?,
                qkv_b: take(&format!("{p}.attn.qkv.bias"), &mut tensors)?,
                proj_w: reshape_linear_weight(&take(
                    &format!("{p}.attn.proj.weight"),
                    &mut tensors,
                )?)?,
                proj_b: take(&format!("{p}.attn.proj.bias"), &mut tensors)?,
                norm2_w: take(&format!("{p}.norm2.weight"), &mut tensors)?,
                norm2_b: take(&format!("{p}.norm2.bias"), &mut tensors)?,
                fc1_w: reshape_linear_weight(&take(
                    &format!("{p}.mlp.linear_fc1.weight"),
                    &mut tensors,
                )?)?,
                fc1_b: take(&format!("{p}.mlp.linear_fc1.bias"), &mut tensors)?,
                fc2_w: reshape_linear_weight(&take(
                    &format!("{p}.mlp.linear_fc2.weight"),
                    &mut tensors,
                )?)?,
                fc2_b: take(&format!("{p}.mlp.linear_fc2.bias"), &mut tensors)?,
            });
        }

        let patch_embed_weight = {
            let raw = tensors
                .remove("model.visual.patch_embed.proj.weight")
                .ok_or_else(|| Error::Other("missing patch_embed.proj.weight".into()))?;
            // Shape [1024, 3, 2, 16, 16] → reshape to [1024, 1536] then
            // transpose to [1536, 1024] for matmul (cuBLAS row-major).
            let flattened = reshape_5d_to_2d(&raw, 1024, 1536)?;
            transpose_2d(&flattened)?
        };
        let patch_embed_bias = tensors
            .remove("model.visual.patch_embed.proj.bias")
            .ok_or_else(|| Error::Other("missing patch_embed.proj.bias".into()))?;
        let pos_embed = tensors
            .remove("model.visual.pos_embed.weight")
            .ok_or_else(|| Error::Other("missing pos_embed.weight".into()))?;

        let merger = load_merger("model.visual.merger", &mut tensors, false)?;
        let deepstack_mergers = (0..3)
            .map(|i| {
                load_merger(
                    &format!("model.visual.deepstack_merger_list.{i}"),
                    &mut tensors,
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            patch_embed_weight,
            patch_embed_bias,
            pos_embed,
            blocks,
            merger,
            deepstack_mergers,
        })
    }
}

fn load_merger(
    prefix: &str,
    m: &mut HashMap<String, Tensor>,
    _deepstack: bool,
) -> Result<Qwen3VLMerger> {
    let take = |name: &str, m: &mut HashMap<String, Tensor>| -> Result<Tensor> {
        m.remove(name)
            .ok_or_else(|| Error::Other(format!("missing {name}")))
    };
    Ok(Qwen3VLMerger {
        norm_w: take(&format!("{prefix}.norm.weight"), m)?,
        norm_b: take(&format!("{prefix}.norm.bias"), m)?,
        fc1_w: reshape_linear_weight(&take(&format!("{prefix}.linear_fc1.weight"), m)?)?,
        fc1_b: take(&format!("{prefix}.linear_fc1.bias"), m)?,
        fc2_w: reshape_linear_weight(&take(&format!("{prefix}.linear_fc2.weight"), m)?)?,
        fc2_b: take(&format!("{prefix}.linear_fc2.bias"), m)?,
    })
}

/// Reshape a 4D/5D HF projection weight `[out, ...in_dims]` to 2D `[out, in]`.
/// Does NOT transpose — the matmul-ready transpose happens in `transpose_2d`.
fn reshape_5d_to_2d(t: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let dims = t.shape().dims();
    let expected: usize = dims.iter().product();
    if expected != rows * cols {
        return Err(Error::Other(format!(
            "reshape_5d_to_2d: {} elements != {}x{}",
            expected, rows, cols
        )));
    }
    match t.dtype() {
        apxinf_core::DType::F32 => {
            let data = t.as_f32()?;
            Tensor::from_f32(vec![rows, cols], &data)
        }
        apxinf_core::DType::BF16 => {
            let data = t.as_bf16()?;
            Tensor::from_bf16(vec![rows, cols], &data)
        }
        dtype => Err(Error::Other(format!(
            "Qwen3-VL vision reshape does not support {dtype}"
        ))),
    }
}

/// Transpose a 2D HF Linear weight `[out, in]` → `[in, out]` for cuBLAS
/// row-major matmul. Same logic as `llama::transpose_weight`.
fn reshape_linear_weight(t: &Tensor) -> Result<Tensor> {
    transpose_2d(t)
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "transpose_2d expected 2D, got {}D",
            dims.len()
        )));
    }
    let [rows, cols] = [dims[0], dims[1]];
    match tensor.dtype() {
        apxinf_core::DType::F32 => {
            let data = tensor.as_f32()?;
            let mut out = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_f32(vec![cols, rows], &out)
        }
        apxinf_core::DType::BF16 => {
            let data = tensor.as_bf16()?;
            let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &out)
        }
        dtype => Err(Error::Other(format!(
            "Qwen3-VL vision transpose does not support {dtype}"
        ))),
    }
}

/// Transfer all vision weights to the backend's device.
pub fn transfer_vision_weights(
    w: &Qwen3VLVisionWeights,
    backend: &dyn apxinf_core::Backend,
) -> Result<Qwen3VLVisionWeights> {
    let transfer_block = |b: &Qwen3VLVisionBlock| -> Result<Qwen3VLVisionBlock> {
        Ok(Qwen3VLVisionBlock {
            norm1_w: backend.to_device(&b.norm1_w)?,
            norm1_b: backend.to_device(&b.norm1_b)?,
            qkv_w: backend.to_device(&b.qkv_w)?,
            qkv_b: backend.to_device(&b.qkv_b)?,
            proj_w: backend.to_device(&b.proj_w)?,
            proj_b: backend.to_device(&b.proj_b)?,
            norm2_w: backend.to_device(&b.norm2_w)?,
            norm2_b: backend.to_device(&b.norm2_b)?,
            fc1_w: backend.to_device(&b.fc1_w)?,
            fc1_b: backend.to_device(&b.fc1_b)?,
            fc2_w: backend.to_device(&b.fc2_w)?,
            fc2_b: backend.to_device(&b.fc2_b)?,
        })
    };
    let transfer_merger = |m: &Qwen3VLMerger| -> Result<Qwen3VLMerger> {
        Ok(Qwen3VLMerger {
            norm_w: backend.to_device(&m.norm_w)?,
            norm_b: backend.to_device(&m.norm_b)?,
            fc1_w: backend.to_device(&m.fc1_w)?,
            fc1_b: backend.to_device(&m.fc1_b)?,
            fc2_w: backend.to_device(&m.fc2_w)?,
            fc2_b: backend.to_device(&m.fc2_b)?,
        })
    };
    let blocks = w
        .blocks
        .iter()
        .map(transfer_block)
        .collect::<Result<Vec<_>>>()?;
    let deepstack_mergers = w
        .deepstack_mergers
        .iter()
        .map(transfer_merger)
        .collect::<Result<Vec<_>>>()?;
    Ok(Qwen3VLVisionWeights {
        patch_embed_weight: backend.to_device(&w.patch_embed_weight)?,
        patch_embed_bias: backend.to_device(&w.patch_embed_bias)?,
        pos_embed: backend.to_device(&w.pos_embed)?,
        blocks,
        merger: transfer_merger(&w.merger)?,
        deepstack_mergers,
    })
}

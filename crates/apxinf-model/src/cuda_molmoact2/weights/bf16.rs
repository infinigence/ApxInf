//! Device weights. Linear weights are transposed once at load from the checkpoint's `[out, in]`
//! to the canonical GEMM `[in, out]` (K, N) layout. Two checkpoint tensors are re-expressed at
//! load time (and nowhere else):
//! * `att_proj` `[q|k|v]` rows and `ff_proj` `[up|gate]` rows are split into separate blocks;
//! * the patch-embedding weight's input axis is permuted from the processor's `(dy, dx, c)` patch
//!   order to the `(c, dy, dx)` order of the existing GPU patchify kernel (pure K permutation);
//! * the two embedding tables are concatenated into one `[vocab + additional, hidden]` table.
use super::host::{self, Collected};
use crate::cuda_molmoact2::config::MolmoAct2Config;
use apxinf_core::{DType, Error, Result, Tensor};
use std::{collections::BTreeMap, path::Path};

pub(in crate::cuda_molmoact2) struct Linear {
    /// `[in, out]` BF16.
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

pub(in crate::cuda_molmoact2) struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub(in crate::cuda_molmoact2) struct VitBlock {
    pub attention_norm: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: LayerNorm,
    pub w1: Linear,
    pub w2: Linear,
}

pub(in crate::cuda_molmoact2) struct Pooling {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
}

/// `w2(silu(w1 x) * w3 x)`, all bias-free.
pub(in crate::cuda_molmoact2) struct Projector {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

pub(in crate::cuda_molmoact2) struct LmBlock {
    pub attn_norm: Tensor,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub attn_out: Linear,
    pub ff_norm: Tensor,
    /// First half of `ff_proj` (multiplied by the activated gate).
    pub up: Linear,
    /// Second half of `ff_proj` (activated).
    pub gate: Linear,
    pub down: Linear,
}

pub(in crate::cuda_molmoact2) struct ExpertBlock {
    /// Self-attention Q/K/V projections: the checkpoint's fused `qkv` rows split into thirds.
    pub self_q: Linear,
    pub self_k: Linear,
    pub self_v: Linear,
    pub self_out: Linear,
    pub cross_q: Linear,
    pub cross_out: Linear,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    /// `[hidden, 9 * hidden]`: (shift, scale, gate) for the self, cross and MLP sub-layers.
    pub modulation: Linear,
}

/// Action expert. Its RMS norms are non-affine (no checkpoint tensors); `context_k` / `context_v` are
/// expert-level projections shared by all 36 layers.
pub(in crate::cuda_molmoact2) struct Expert {
    pub time_in: Linear,
    pub time_out: Linear,
    pub action_embed: Linear,
    pub context_k: Linear,
    pub context_v: Linear,
    pub blocks: Vec<ExpertBlock>,
    /// `[hidden, 2 * hidden]`: (shift, scale) of the final layer.
    pub final_modulation: Linear,
    pub final_out: Linear,
}

pub(in crate::cuda_molmoact2) struct Bf16Weights {
    pub patch_embedding: Linear,
    /// `[num_pos, hidden]`.
    pub positional_embedding: Tensor,
    pub vit: Vec<VitBlock>,
    pub pooling: Pooling,
    pub projector: Projector,
    /// `[vocab + additional, hidden]`.
    pub wte: Tensor,
    pub blocks: Vec<LmBlock>,
    pub expert: Expert,
    /// FP32 `inv_freq` buffer of the checkpoint (the official RoPE source).
    pub inv_freq: Vec<f32>,
}

fn bad(what: impl std::fmt::Display) -> Error {
    Error::Other(format!("MolmoAct2 weights: {what}"))
}

/// Host tensor -> CUDA tensor through the backend's own transfer helper.
pub(in crate::cuda_molmoact2) fn upload(tensor: &Tensor, device: usize) -> Result<Tensor> {
    apxinf_cuda_new::transfers::to_cuda(tensor, device)
}

/// Upload a checkpoint `[out, in]` linear weight as `[in, out]`.
fn put_linear(c: &mut Collected, key: String, host_weight: Tensor, device: usize) -> Result<()> {
    c.tensors.insert(key, upload(&host::transpose(&host_weight)?, device)?);
    Ok(())
}

fn ingest(c: &mut Collected, name: &str, tensor: &Tensor, cfg: &MolmoAct2Config, device: usize) -> Result<()> {
    if host::is_skipped(name) {
        return Ok(());
    }
    if name == "model.transformer.rotary_emb.inv_freq" {
        if tensor.dtype() != DType::F32 {
            return Err(bad("rotary inv_freq must be FP32"));
        }
        c.inv_freq = Some(tensor.as_f32()?.to_vec());
        return Ok(());
    }
    let b = host::to_bf16(name, tensor)?;
    let t = &cfg.text;
    if let Some(prefix) = name.strip_suffix("att_proj.weight") {
        let (q, kv) = (t.heads * t.head_dim, t.kv_heads * t.head_dim);
        if b.shape().dims()[0] != q + 2 * kv {
            return Err(bad(format!("{name}: fused QKV rows do not match the head layout")));
        }
        put_linear(c, format!("{prefix}att_q.weight"), host::row_slice(&b, 0, q)?, device)?;
        put_linear(c, format!("{prefix}att_k.weight"), host::row_slice(&b, q, q + kv)?, device)?;
        put_linear(c, format!("{prefix}att_v.weight"), host::row_slice(&b, q + kv, q + 2 * kv)?, device)?;
    } else if let Some(prefix) = name.strip_suffix("ff_proj.weight") {
        let i = t.intermediate;
        if b.shape().dims()[0] != 2 * i {
            return Err(bad(format!("{name}: expected {} rows", 2 * i)));
        }
        // `x, gate = ff_proj(h).chunk(2)`: the first half is multiplied by act(second half).
        put_linear(c, format!("{prefix}ff_up.weight"), host::row_slice(&b, 0, i)?, device)?;
        put_linear(c, format!("{prefix}ff_gate.weight"), host::row_slice(&b, i, 2 * i)?, device)?;
    } else if let Some(prefix) = name.strip_suffix("self_attn.qkv.weight") {
        // Expert self-attention: `qkv(x).view(B, T, 3, heads, head_dim)`, i.e. rows are [Q | K | V].
        let h = cfg.expert.hidden;
        if b.shape().dims() != [3 * h, h] {
            return Err(bad(format!("{name}: expected [{}, {h}]", 3 * h)));
        }
        for (i, part) in ["q", "k", "v"].into_iter().enumerate() {
            put_linear(c, format!("{prefix}self_attn.{part}.weight"), host::row_slice(&b, i * h, (i + 1) * h)?, device)?;
        }
    } else if let Some(prefix) = name.strip_suffix("self_attn.qkv.bias") {
        let h = cfg.expert.hidden;
        if b.shape().dims() != [3 * h] {
            return Err(bad(format!("{name}: expected [{}]", 3 * h)));
        }
        for (i, part) in ["q", "k", "v"].into_iter().enumerate() {
            c.tensors.insert(format!("{prefix}self_attn.{part}.bias"), upload(&host::slice_1d(&b, i * h, (i + 1) * h)?, device)?);
        }
    } else if name == "model.vision_backbone.image_vit.patch_embedding.weight" {
        let permuted = host::patch_columns_to_channel_major(&b, cfg.vit.patch_size)?;
        put_linear(c, name.to_owned(), permuted, device)?;
    } else if name == "model.transformer.wte.embedding" || name == "model.transformer.wte.new_embedding" {
        c.tensors.insert(name.to_owned(), b); // stays on the host until the tables are joined
    } else if name.ends_with(".weight") && b.shape().ndim() == 2 {
        put_linear(c, name.to_owned(), b, device)?;
    } else {
        c.tensors.insert(name.to_owned(), upload(&b, device)?);
    }
    Ok(())
}

impl Bf16Weights {
    pub fn load(dir: &Path, cfg: &MolmoAct2Config, device: usize) -> Result<Self> {
        let mut c = Collected {
            tensors: BTreeMap::new(),
            inv_freq: None,
        };
        for shard in host::shard_files(dir)? {
            let (tensors, _) = apxinf_loader::safetensors::load_native(&shard)
                .map_err(|e| bad(format!("load {}: {e}", shard.display())))?;
            for (name, tensor) in &tensors {
                ingest(&mut c, name, tensor, cfg, device)?;
            }
        }
        let inv_freq = c.inv_freq.take().ok_or_else(|| bad("missing rotary inv_freq"))?;
        if inv_freq.len() != cfg.text.head_dim / 2 {
            return Err(bad("rotary inv_freq length does not match head_dim / 2"));
        }
        let (v, a) = (&cfg.vit, &cfg.adapter);
        let t = &cfg.text;

        // Device linear weights are `[in, out]`; `in`/`out` below are the checkpoint's `[out, in]`.
        let linear = |c: &mut Collected, prefix: &str, out: usize, inp: usize, bias: bool| -> Result<Linear> {
            Ok(Linear {
                weight: c.take(&format!("{prefix}.weight"), &[inp, out])?,
                bias: if bias { Some(c.take(&format!("{prefix}.bias"), &[out])?) } else { None },
            })
        };
        let layer_norm = |c: &mut Collected, prefix: &str, n: usize| -> Result<LayerNorm> {
            Ok(LayerNorm {
                weight: c.take(&format!("{prefix}.weight"), &[n])?,
                bias: c.take(&format!("{prefix}.bias"), &[n])?,
            })
        };

        let vb = "model.vision_backbone";
        let patch_dim = v.patch_size * v.patch_size * 3;
        let patch_embedding = linear(&mut c, &format!("{vb}.image_vit.patch_embedding"), v.hidden, patch_dim, true)?;
        let positional_embedding = c.take(&format!("{vb}.image_vit.positional_embedding"), &[v.num_pos, v.hidden])?;
        let mut vit = Vec::with_capacity(v.executed_layers);
        for i in 0..v.executed_layers {
            let p = format!("{vb}.image_vit.transformer.resblocks.{i}");
            vit.push(VitBlock {
                attention_norm: layer_norm(&mut c, &format!("{p}.attention_norm"), v.hidden)?,
                wq: linear(&mut c, &format!("{p}.attention.wq"), v.hidden, v.hidden, true)?,
                wk: linear(&mut c, &format!("{p}.attention.wk"), v.hidden, v.hidden, true)?,
                wv: linear(&mut c, &format!("{p}.attention.wv"), v.hidden, v.hidden, true)?,
                wo: linear(&mut c, &format!("{p}.attention.wo"), v.hidden, v.hidden, true)?,
                ffn_norm: layer_norm(&mut c, &format!("{p}.ffn_norm"), v.hidden)?,
                w1: linear(&mut c, &format!("{p}.feed_forward.w1"), v.intermediate, v.hidden, true)?,
                w2: linear(&mut c, &format!("{p}.feed_forward.w2"), v.hidden, v.intermediate, true)?,
            });
        }
        let pool_in = v.hidden * a.vit_layers.len();
        let pooling = Pooling {
            wq: linear(&mut c, &format!("{vb}.image_pooling_2d.wq"), a.hidden, pool_in, true)?,
            wk: linear(&mut c, &format!("{vb}.image_pooling_2d.wk"), a.hidden, pool_in, true)?,
            wv: linear(&mut c, &format!("{vb}.image_pooling_2d.wv"), a.hidden, pool_in, true)?,
            wo: linear(&mut c, &format!("{vb}.image_pooling_2d.wo"), a.hidden, a.hidden, true)?,
        };
        let projector = Projector {
            w1: linear(&mut c, &format!("{vb}.image_projector.w1"), a.intermediate, a.hidden, false)?,
            w2: linear(&mut c, &format!("{vb}.image_projector.w2"), a.text_hidden, a.intermediate, false)?,
            w3: linear(&mut c, &format!("{vb}.image_projector.w3"), a.intermediate, a.hidden, false)?,
        };

        let table = host::concat_rows(
            &c.take("model.transformer.wte.embedding", &[t.vocab, t.hidden])?,
            &c.take("model.transformer.wte.new_embedding", &[t.additional_vocab, t.hidden])?,
        )?;
        let wte = upload(&table, device)?;
        drop(table);

        let mut blocks = Vec::with_capacity(t.layers);
        for i in 0..t.layers {
            let p = format!("model.transformer.blocks.{i}");
            blocks.push(LmBlock {
                attn_norm: c.take(&format!("{p}.attn_norm.weight"), &[t.hidden])?,
                q: linear(&mut c, &format!("{p}.self_attn.att_q"), t.heads * t.head_dim, t.hidden, false)?,
                k: linear(&mut c, &format!("{p}.self_attn.att_k"), t.kv_heads * t.head_dim, t.hidden, false)?,
                v: linear(&mut c, &format!("{p}.self_attn.att_v"), t.kv_heads * t.head_dim, t.hidden, false)?,
                q_norm: c.take(&format!("{p}.self_attn.q_norm.weight"), &[t.head_dim])?,
                k_norm: c.take(&format!("{p}.self_attn.k_norm.weight"), &[t.head_dim])?,
                attn_out: linear(&mut c, &format!("{p}.self_attn.attn_out"), t.hidden, t.heads * t.head_dim, false)?,
                ff_norm: c.take(&format!("{p}.ff_norm.weight"), &[t.hidden])?,
                up: linear(&mut c, &format!("{p}.mlp.ff_up"), t.intermediate, t.hidden, false)?,
                gate: linear(&mut c, &format!("{p}.mlp.ff_gate"), t.intermediate, t.hidden, false)?,
                down: linear(&mut c, &format!("{p}.mlp.ff_out"), t.hidden, t.intermediate, false)?,
            });
        }
        // The final norm is validated and consumed but not kept: this profile only reads the per-layer K/V
        // cache, which does not depend on it.
        c.take("model.transformer.ln_f.weight", &[t.hidden])?;

        let (ex, ep) = (&cfg.expert, "model.action_expert");
        let kv_width = t.kv_heads * t.head_dim;
        let mut expert_blocks = Vec::with_capacity(ex.layers);
        for i in 0..ex.layers {
            let p = format!("{ep}.blocks.{i}");
            let (h, inner) = (ex.hidden, ex.mlp_inner);
            expert_blocks.push(ExpertBlock {
                self_q: linear(&mut c, &format!("{p}.self_attn.q"), h, h, true)?,
                self_k: linear(&mut c, &format!("{p}.self_attn.k"), h, h, true)?,
                self_v: linear(&mut c, &format!("{p}.self_attn.v"), h, h, true)?,
                self_out: linear(&mut c, &format!("{p}.self_attn.out_proj"), h, h, true)?,
                cross_q: linear(&mut c, &format!("{p}.cross_attn.q_proj"), h, h, true)?,
                cross_out: linear(&mut c, &format!("{p}.cross_attn.out_proj"), h, h, true)?,
                gate: linear(&mut c, &format!("{p}.mlp.gate_proj"), inner, h, true)?,
                up: linear(&mut c, &format!("{p}.mlp.up_proj"), inner, h, true)?,
                down: linear(&mut c, &format!("{p}.mlp.down_proj"), h, inner, true)?,
                modulation: linear(&mut c, &format!("{p}.modulation.linear"), 9 * h, h, true)?,
            });
        }
        let expert = Expert {
            time_in: linear(&mut c, &format!("{ep}.time_embed.1"), ex.hidden, ex.time_embed_dim, true)?,
            time_out: linear(&mut c, &format!("{ep}.time_embed.3"), ex.hidden, ex.hidden, true)?,
            action_embed: linear(&mut c, &format!("{ep}.action_embed"), ex.hidden, ex.max_action_dim, true)?,
            context_k: linear(&mut c, &format!("{ep}.context_k_proj"), ex.hidden, kv_width, false)?,
            context_v: linear(&mut c, &format!("{ep}.context_v_proj"), ex.hidden, kv_width, false)?,
            blocks: expert_blocks,
            final_modulation: linear(&mut c, &format!("{ep}.final_layer.modulation.linear"), 2 * ex.hidden, ex.hidden, true)?,
            final_out: linear(&mut c, &format!("{ep}.final_layer.linear"), ex.max_action_dim, ex.hidden, true)?,
        };
        c.finish()?;
        Ok(Self {
            patch_embedding,
            positional_embedding,
            vit,
            pooling,
            projector,
            wte,
            blocks,
            expert,
            inv_freq,
        })
    }
}

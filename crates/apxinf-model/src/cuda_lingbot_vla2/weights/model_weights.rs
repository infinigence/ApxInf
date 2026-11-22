//! Device weights of the whole model, quantised/folded once at load.

use super::device::*;
use super::host::Checkpoint;
use super::quant::{concat_rows, quantize_rows, QuantLinear};
use crate::cuda_lingbot_vla2::backend::Context;
use crate::cuda_lingbot_vla2::backend::DeviceBuffer;
use crate::cuda_lingbot_vla2::config::*;
use crate::cuda_lingbot_vla2::math;
use apxinf_core::{Error, Result, Tensor};
use half::bf16;

const LM: &str = "model.qwenvl_with_expert.qwenvl.model.language_model";
const VIS: &str = "model.qwenvl_with_expert.qwenvl.model.visual";
const EXP: &str = "model.qwenvl_with_expert.qwen_expert.model";

pub(crate) struct VitBlock {
    pub norm1_w: Tensor,
    pub norm1_b: Tensor,
    pub qkv: Int8Linear,
    pub qkv_bias: Tensor,
    pub proj: Int8Linear,
    pub proj_bias: Tensor,
    pub norm2_w: Tensor,
    pub norm2_b: Tensor,
    pub fc1: Int8Linear,
    pub fc1_bias: Tensor,
    pub fc2: Int8Linear,
    pub fc2_bias: Tensor,
}

pub(crate) struct Merger {
    pub norm_w: Tensor,
    pub norm_b: Tensor,
    pub fc1: Int8Linear,
    pub fc1_bias: Tensor,
    pub fc2: Int8Linear,
    pub fc2_bias: Tensor,
}

pub(crate) struct VisionWeights {
    /// Patch embedding as a dense GEMM `[1536, 1024]` (Conv3d with kernel == stride is a matmul).
    pub patch: DenseLinear,
    pub blocks: Vec<VitBlock>,
    pub merger: Merger,
    /// One per tap in `DEEPSTACK_TAPS`; the norm runs on the merged `[*, 4096]` rows.
    pub deepstack: Vec<Merger>,
    /// Interpolated position embeddings for the fixed 3 x [1,16,16] grids, `[768, 1024]`.
    pub pos_embeds: Tensor,
    /// `[tokens, 2]` (row, col) pairs for the axial RoPE, merge-block-major, i32.
    pub axial_positions: Tensor,
    /// Offsets of the three 256-row segments for `segmented_attention` (host copy and device copy).
    pub segment_offsets: Vec<u32>,
    pub segment_offsets_dev: DeviceBuffer,
}

pub(crate) struct VlmLayer {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    /// Separate projections: `rms_norm_apply_mrope_qk` takes q and k as flat `[tokens * heads, 128]` rows.
    pub q: Int8Linear,
    pub k: Int8Linear,
    pub v: Int8Linear,
    pub o: Int8Linear,
    pub gate: Int8Linear,
    pub up: Int8Linear,
    pub down: Int8Linear,
}

/// Token embedding table kept in host memory: a request reads only its prompt rows and the two vision special tokens
/// (about 90 KB), so the 742 MiB table does not need to occupy the 8 GB card.
pub(crate) struct HostEmbedding {
    table: Vec<bf16>,
}

impl HostEmbedding {
    /// Rows for `ids`, concatenated `[ids.len(), 2560]`.
    pub fn gather(&self, ids: &[i32]) -> Result<Vec<bf16>> {
        let mut out = Vec::with_capacity(ids.len() * VLM_HIDDEN);
        for &id in ids {
            let row = usize::try_from(id)
                .ok()
                .filter(|r| *r < VLM_VOCAB)
                .ok_or_else(|| {
                    Error::Other(format!(
                        "LingBot-VLA 2.0: token id {id} is outside the vocabulary of {VLM_VOCAB}"
                    ))
                })?;
            out.extend_from_slice(&self.table[row * VLM_HIDDEN..(row + 1) * VLM_HIDDEN]);
        }
        Ok(out)
    }
}

pub(crate) struct VlmWeights {
    pub embed: HostEmbedding,
    pub layers: Vec<VlmLayer>,
}

pub(crate) struct MoeLayer {
    /// Router `[32, 768]` BF16 (kept BF16: the route op accumulates in FP32).
    pub gate_router: Tensor,
    /// F32 `[32]`, used for expert selection only. Zero in the published checkpoint.
    pub router_bias: Tensor,
    /// Gate projections of all experts stacked: `[32 * 512, 768]`.
    pub gate: Int8Linear,
    /// Up projections of all experts stacked: `[32 * 512, 768]`.
    pub up: Int8Linear,
    /// Down projections concatenated along K: `[768, 32 * 512]`.
    pub down: Int8Linear,
    pub shared_gate: DenseLinear,
    pub shared_up: DenseLinear,
    pub shared_down: DenseLinear,
}

pub(crate) struct ExpertLayer {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    pub qkv: Int8Linear,
    pub qkv_bias: Tensor,
    pub o: Int8Linear,
    pub moe: MoeLayer,
}

pub(crate) struct ExpertWeights {
    pub layers: Vec<ExpertLayer>,
    pub final_norm: Tensor,
    pub state_proj: DenseLinear,
    pub action_in: DenseLinear,
    pub action_time_in_action: Tensor,
    pub action_time_out: DenseLinear,
    pub action_out: DenseLinear,
    /// `[steps, 36, 2 (input, post), 2 (scale, shift), 768]` BF16.
    pub ada: Tensor,
    /// `[steps, 768]` BF16: the time half of `action_time_mlp_in` (folded with its bias), per step.
    pub time_bias: Tensor,
    pub times: Vec<f32>,
    pub dt: f32,
}

pub(crate) struct QueryTokens {
    /// `[16, 2560]` BF16: 8 current-query rows then 8 future-query rows, already projected.
    pub rows: Tensor,
}

pub(crate) struct LingbotWeights {
    pub vision: VisionWeights,
    pub vlm: VlmWeights,
    pub expert: ExpertWeights,
    pub queries: QueryTokens,
}

fn quant_stack_rows(ctx: &Context, rows: Vec<f32>, n: usize, k: usize) -> Result<Int8Linear> {
    let q: QuantLinear = quantize_rows(&rows, n, k);
    upload_int8(ctx, &q)
}

fn merger(ctx: &Context, ck: &Checkpoint, prefix: &str, norm_width: usize) -> Result<Merger> {
    let hidden = VIT_HIDDEN * VIT_MERGE * VIT_MERGE;
    Ok(Merger {
        norm_w: vector_bf16(ctx, ck, &format!("{prefix}.norm.weight"), norm_width)?,
        norm_b: vector_bf16(ctx, ck, &format!("{prefix}.norm.bias"), norm_width)?,
        fc1: int8_from_checkpoint(
            ctx,
            ck,
            &format!("{prefix}.linear_fc1.weight"),
            hidden,
            hidden,
        )?,
        fc1_bias: bias_bf16(ctx, ck, &format!("{prefix}.linear_fc1.bias"), hidden)?,
        fc2: int8_from_checkpoint(
            ctx,
            ck,
            &format!("{prefix}.linear_fc2.weight"),
            VLM_HIDDEN,
            hidden,
        )?,
        fc2_bias: bias_bf16(ctx, ck, &format!("{prefix}.linear_fc2.bias"), VLM_HIDDEN)?,
    })
}

pub(crate) fn load_vision(ctx: &Context, ck: &Checkpoint) -> Result<VisionWeights> {
    // Conv3d(3, 1024, [2, 16, 16]) with stride == kernel is a matmul over the 1536-wide patch rows.
    let conv = ck.bf16(
        &format!("{VIS}.patch_embed.proj.weight"),
        &[VIT_HIDDEN, 3, VIT_TEMPORAL_PATCH, VIT_PATCH, VIT_PATCH],
    )?;
    let mut patch_w = vec![bf16::ZERO; VIT_PATCH_DIM * VIT_HIDDEN];
    for o in 0..VIT_HIDDEN {
        for i in 0..VIT_PATCH_DIM {
            patch_w[i * VIT_HIDDEN + o] = conv[o * VIT_PATCH_DIM + i];
        }
    }
    let patch = DenseLinear {
        weight: bf16_tensor(ctx, &patch_w, vec![VIT_PATCH_DIM, VIT_HIDDEN])?,
        bias: Some(bias_bf16(
            ctx,
            ck,
            &format!("{VIS}.patch_embed.proj.bias"),
            VIT_HIDDEN,
        )?),
    };
    let mut blocks = Vec::with_capacity(VIT_LAYERS);
    for i in 0..VIT_LAYERS {
        let p = format!("{VIS}.blocks.{i}");
        blocks.push(VitBlock {
            norm1_w: vector_bf16(ctx, ck, &format!("{p}.norm1.weight"), VIT_HIDDEN)?,
            norm1_b: vector_bf16(ctx, ck, &format!("{p}.norm1.bias"), VIT_HIDDEN)?,
            qkv: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.attn.qkv.weight"),
                3 * VIT_HIDDEN,
                VIT_HIDDEN,
            )?,
            qkv_bias: bias_bf16(ctx, ck, &format!("{p}.attn.qkv.bias"), 3 * VIT_HIDDEN)?,
            proj: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.attn.proj.weight"),
                VIT_HIDDEN,
                VIT_HIDDEN,
            )?,
            proj_bias: bias_bf16(ctx, ck, &format!("{p}.attn.proj.bias"), VIT_HIDDEN)?,
            norm2_w: vector_bf16(ctx, ck, &format!("{p}.norm2.weight"), VIT_HIDDEN)?,
            norm2_b: vector_bf16(ctx, ck, &format!("{p}.norm2.bias"), VIT_HIDDEN)?,
            fc1: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.mlp.linear_fc1.weight"),
                VIT_FFN,
                VIT_HIDDEN,
            )?,
            fc1_bias: bias_bf16(ctx, ck, &format!("{p}.mlp.linear_fc1.bias"), VIT_FFN)?,
            fc2: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.mlp.linear_fc2.weight"),
                VIT_HIDDEN,
                VIT_FFN,
            )?,
            fc2_bias: bias_bf16(ctx, ck, &format!("{p}.mlp.linear_fc2.bias"), VIT_HIDDEN)?,
        });
    }
    // Final merger normalises each ViT row (1024) before the 2x2 merge; the deepstack mergers use the
    // post-shuffle layout and normalise the merged 4096-wide rows.
    let merger_final = merger(ctx, ck, &format!("{VIS}.merger"), VIT_HIDDEN)?;
    let mut deepstack = Vec::new();
    for i in 0..DEEPSTACK_TAPS.len() {
        deepstack.push(merger(
            ctx,
            ck,
            &format!("{VIS}.deepstack_merger_list.{i}"),
            VIT_HIDDEN * 4,
        )?);
    }
    let table = ck.bf16(
        &format!("{VIS}.pos_embed.weight"),
        &[VIT_POS_GRID * VIT_POS_GRID, VIT_HIDDEN],
    )?;
    let mut pos = Vec::new();
    let mut axial = Vec::new();
    for _ in 0..NUM_VIEWS {
        pos.extend(math::vit_pos_embeds(&table, [1, GRID, GRID]));
        axial.extend(math::vit_axial_positions([1, GRID, GRID]));
    }
    let tokens = NUM_VIEWS * PATCHES_PER_VIEW;
    check_len("vit position embeddings", pos.len(), tokens * VIT_HIDDEN)?;
    let segment_offsets: Vec<u32> = (0..=NUM_VIEWS as u32)
        .map(|v| v * PATCHES_PER_VIEW as u32)
        .collect();
    Ok(VisionWeights {
        patch,
        blocks,
        merger: merger_final,
        deepstack,
        pos_embeds: bf16_tensor(ctx, &pos, vec![tokens, VIT_HIDDEN])?,
        axial_positions: u32_tensor(ctx, &axial, vec![tokens, 2])?,
        segment_offsets_dev: upload_u32_buffer(ctx, &segment_offsets)?,
        segment_offsets,
    })
}

pub(crate) fn load_vlm(ctx: &Context, ck: &Checkpoint) -> Result<VlmWeights> {
    let embed = HostEmbedding {
        table: ck.bf16(
            &format!("{LM}.embed_tokens.weight"),
            &[VLM_VOCAB, VLM_HIDDEN],
        )?,
    };
    let mut layers = Vec::with_capacity(VLM_LAYERS);
    for i in 0..VLM_LAYERS {
        let p = format!("{LM}.layers.{i}");
        layers.push(VlmLayer {
            input_norm: vector_bf16(ctx, ck, &format!("{p}.input_layernorm.weight"), VLM_HIDDEN)?,
            post_norm: vector_bf16(
                ctx,
                ck,
                &format!("{p}.post_attention_layernorm.weight"),
                VLM_HIDDEN,
            )?,
            q_norm: vector_bf16(ctx, ck, &format!("{p}.self_attn.q_norm.weight"), HEAD_DIM)?,
            k_norm: vector_bf16(ctx, ck, &format!("{p}.self_attn.k_norm.weight"), HEAD_DIM)?,
            q: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.self_attn.q_proj.weight"),
                VLM_HEADS * HEAD_DIM,
                VLM_HIDDEN,
            )?,
            k: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.self_attn.k_proj.weight"),
                VLM_KV_HEADS * HEAD_DIM,
                VLM_HIDDEN,
            )?,
            v: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.self_attn.v_proj.weight"),
                VLM_KV_HEADS * HEAD_DIM,
                VLM_HIDDEN,
            )?,
            o: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.self_attn.o_proj.weight"),
                VLM_HIDDEN,
                VLM_HEADS * HEAD_DIM,
            )?,
            gate: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.mlp.gate_proj.weight"),
                VLM_FFN,
                VLM_HIDDEN,
            )?,
            up: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.mlp.up_proj.weight"),
                VLM_FFN,
                VLM_HIDDEN,
            )?,
            down: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.mlp.down_proj.weight"),
                VLM_HIDDEN,
                VLM_FFN,
            )?,
        });
    }
    // The VLM's final norm only feeds the (removed) language head; the expert reads the K/V caches, so
    // `norm.weight` is intentionally not loaded.
    Ok(VlmWeights { embed, layers })
}

fn load_moe(ctx: &Context, ck: &Checkpoint, p: &str) -> Result<MoeLayer> {
    let e = MOE_EXPERTS;
    let i = MOE_FFN;
    let gate_all = ck.f32(
        &format!("{p}.mlp.experts.gate_proj"),
        &[e, i, EXPERT_HIDDEN],
    )?;
    let up_all = ck.f32(&format!("{p}.mlp.experts.up_proj"), &[e, i, EXPERT_HIDDEN])?;
    // [E, 768, I] -> [768, E * I]: column e * I + c holds expert e's channel c, the order the
    // routing-weighted hidden is produced in, so one K = E * I GEMM sums the experts.
    let down_all = ck.f32(
        &format!("{p}.mlp.experts.down_proj"),
        &[e, EXPERT_HIDDEN, i],
    )?;
    let mut down = vec![0f32; EXPERT_HIDDEN * e * i];
    for ex in 0..e {
        for o in 0..EXPERT_HIDDEN {
            let src = &down_all[(ex * EXPERT_HIDDEN + o) * i..(ex * EXPERT_HIDDEN + o + 1) * i];
            down[o * e * i + ex * i..o * e * i + (ex + 1) * i].copy_from_slice(src);
        }
    }
    let router_bias = ck.f32(&format!("{p}.mlp.e_score_correction_bias"), &[e])?;
    Ok(MoeLayer {
        gate_router: bf16_tensor(
            ctx,
            &ck.bf16(&format!("{p}.mlp.gate.weight"), &[e, EXPERT_HIDDEN])?,
            vec![e, EXPERT_HIDDEN],
        )?,
        router_bias: f32_tensor(ctx, &router_bias, vec![e])?,
        gate: quant_stack_rows(ctx, gate_all, e * i, EXPERT_HIDDEN)?,
        up: quant_stack_rows(ctx, up_all, e * i, EXPERT_HIDDEN)?,
        down: quant_stack_rows(ctx, down, EXPERT_HIDDEN, e * i)?,
        shared_gate: dense_from_checkpoint(
            ctx,
            ck,
            &format!("{p}.mlp.shared_expert.gate_proj.weight"),
            MOE_SHARED_FFN,
            EXPERT_HIDDEN,
            None,
        )?,
        shared_up: dense_from_checkpoint(
            ctx,
            ck,
            &format!("{p}.mlp.shared_expert.up_proj.weight"),
            MOE_SHARED_FFN,
            EXPERT_HIDDEN,
            None,
        )?,
        shared_down: dense_from_checkpoint(
            ctx,
            ck,
            &format!("{p}.mlp.shared_expert.down_proj.weight"),
            EXPERT_HIDDEN,
            MOE_SHARED_FFN,
            None,
        )?,
    })
}

/// AdaRMSNorm conditioning of every layer and flow step: `gamma(t)` and `beta(t)` are `Linear(768, 768)`
/// of the BF16 time embedding, so for the fixed schedule they are constants. Layout
/// `[step, layer, norm (input, post), part (scale, shift), 768]`.
fn ada_tables(ck: &Checkpoint, time_embeddings: &[Vec<f32>]) -> Result<Vec<bf16>> {
    let d = EXPERT_HIDDEN;
    let mut out = vec![bf16::ZERO; time_embeddings.len() * VLM_LAYERS * 2 * 2 * d];
    for layer in 0..VLM_LAYERS {
        for (norm_index, norm) in ["input_layernorm", "post_attention_layernorm"]
            .into_iter()
            .enumerate()
        {
            let p = format!("{EXP}.layers.{layer}.{norm}");
            let gamma_w = ck.f32(&format!("{p}.gamma.weight"), &[d, d])?;
            let gamma_b = ck.f32(&format!("{p}.gamma.bias"), &[d])?;
            let beta_w = ck.f32(&format!("{p}.beta.weight"), &[d, d])?;
            let beta_b = ck.f32(&format!("{p}.beta.bias"), &[d])?;
            for (step, t) in time_embeddings.iter().enumerate() {
                let scale = math::matvec_f32(&gamma_w, d, d, t, Some(&gamma_b));
                let shift = math::matvec_f32(&beta_w, d, d, t, Some(&beta_b));
                let base = (((step * VLM_LAYERS + layer) * 2 + norm_index) * 2) * d;
                for c in 0..d {
                    out[base + c] = bf16::from_f32(scale[c]);
                    out[base + d + c] = bf16::from_f32(shift[c]);
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn load_expert(ctx: &Context, ck: &Checkpoint, steps: usize) -> Result<ExpertWeights> {
    let d = EXPERT_HIDDEN;
    let (times, dt) = math::denoise_schedule(steps);
    let time_embeddings: Vec<Vec<f32>> = times
        .iter()
        .map(|t| {
            math::bf16_round(&math::sinusoidal_time_embedding(
                *t,
                d,
                TIME_MIN_PERIOD,
                TIME_MAX_PERIOD,
            ))
        })
        .collect();

    let mut layers = Vec::with_capacity(VLM_LAYERS);
    for i in 0..VLM_LAYERS {
        let p = format!("{EXP}.layers.{i}");
        let (q, k, v) = (
            format!("{p}.self_attn.q_proj"),
            format!("{p}.self_attn.k_proj"),
            format!("{p}.self_attn.v_proj"),
        );
        let qkv = {
            let wq = ck.f32(&format!("{q}.weight"), &[EXPERT_HEADS * HEAD_DIM, d])?;
            let wk = ck.f32(&format!("{k}.weight"), &[EXPERT_KV_HEADS * HEAD_DIM, d])?;
            let wv = ck.f32(&format!("{v}.weight"), &[EXPERT_KV_HEADS * HEAD_DIM, d])?;
            let all = concat_rows(&[&wq, &wk, &wv]);
            quant_stack_rows(ctx, all, (EXPERT_HEADS + 2 * EXPERT_KV_HEADS) * HEAD_DIM, d)?
        };
        let mut bias = ck.bf16(&format!("{q}.bias"), &[EXPERT_HEADS * HEAD_DIM])?;
        bias.extend(ck.bf16(&format!("{k}.bias"), &[EXPERT_KV_HEADS * HEAD_DIM])?);
        bias.extend(ck.bf16(&format!("{v}.bias"), &[EXPERT_KV_HEADS * HEAD_DIM])?);
        layers.push(ExpertLayer {
            input_norm: vector_bf16(ctx, ck, &format!("{p}.input_layernorm.weight"), d)?,
            post_norm: vector_bf16(ctx, ck, &format!("{p}.post_attention_layernorm.weight"), d)?,
            qkv,
            qkv_bias: bf16_tensor(
                ctx,
                &bias,
                vec![(EXPERT_HEADS + 2 * EXPERT_KV_HEADS) * HEAD_DIM],
            )?,
            o: int8_from_checkpoint(
                ctx,
                ck,
                &format!("{p}.self_attn.o_proj.weight"),
                d,
                EXPERT_HEADS * HEAD_DIM,
            )?,
            moe: load_moe(ctx, ck, &p)?,
        });
    }
    let ada = ada_tables(ck, &time_embeddings)?;

    // action_time_mlp_in sees cat([action_emb, time_emb]): split its [768, 1536] weight into the action half
    // (a runtime GEMM) and the time half, which is constant per step and folds with the bias.
    let w_in = ck.f32("model.action_time_mlp_in.weight", &[d, 2 * d])?;
    let b_in = ck.f32("model.action_time_mlp_in.bias", &[d])?;
    let mut w_action = vec![0f32; d * d];
    let mut w_time = vec![0f32; d * d];
    for o in 0..d {
        w_action[o * d..(o + 1) * d].copy_from_slice(&w_in[o * 2 * d..o * 2 * d + d]);
        w_time[o * d..(o + 1) * d].copy_from_slice(&w_in[o * 2 * d + d..(o + 1) * 2 * d]);
    }
    let mut time_bias = Vec::with_capacity(steps * d);
    for t in &time_embeddings {
        time_bias.extend(
            math::matvec_f32(&w_time, d, d, t, Some(&b_in))
                .into_iter()
                .map(bf16::from_f32),
        );
    }
    let to_bf16_t = |w: &[f32], rows: usize, cols: usize| -> Vec<bf16> {
        // [rows, cols] -> [cols, rows] BF16
        let mut t = vec![bf16::ZERO; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                t[c * rows + r] = bf16::from_f32(w[r * cols + c]);
            }
        }
        t
    };
    let action_time_in_action = bf16_tensor(ctx, &to_bf16_t(&w_action, d, d), vec![d, d])?;

    // 55-wide state/action vectors are zero-padded to 64 so every GEMM sees an aligned K or N; the padded
    // columns of the noise are zero and stay zero (padded rows of action_out are zero).
    let padded_in = |name: &str, bias: &str| -> Result<DenseLinear> {
        let w = ck.f32(name, &[d, STATE_DIM])?;
        let mut p = vec![0f32; d * PADDED_DIM];
        for o in 0..d {
            p[o * PADDED_DIM..o * PADDED_DIM + STATE_DIM]
                .copy_from_slice(&w[o * STATE_DIM..(o + 1) * STATE_DIM]);
        }
        Ok(DenseLinear {
            weight: bf16_tensor(ctx, &to_bf16_t(&p, d, PADDED_DIM), vec![PADDED_DIM, d])?,
            bias: Some(bias_bf16(ctx, ck, bias, d)?),
        })
    };
    let state_proj = padded_in("model.state_proj.weight", "model.state_proj.bias")?;
    let action_in = padded_in("model.action_in_proj.weight", "model.action_in_proj.bias")?;
    let out_w = ck.f32("model.action_out_proj.weight", &[ACTION_DIM, d])?;
    let mut out_padded = vec![0f32; PADDED_DIM * d];
    out_padded[..ACTION_DIM * d].copy_from_slice(&out_w);
    let mut out_bias = ck.bf16("model.action_out_proj.bias", &[ACTION_DIM])?;
    out_bias.resize(PADDED_DIM, bf16::ZERO);
    let action_out = DenseLinear {
        weight: bf16_tensor(
            ctx,
            &to_bf16_t(&out_padded, PADDED_DIM, d),
            vec![d, PADDED_DIM],
        )?,
        bias: Some(bf16_tensor(ctx, &out_bias, vec![PADDED_DIM])?),
    };
    Ok(ExpertWeights {
        layers,
        final_norm: vector_bf16(ctx, ck, &format!("{EXP}.norm.weight"), d)?,
        state_proj,
        action_in,
        action_time_in_action,
        action_time_out: dense_from_checkpoint(
            ctx,
            ck,
            "model.action_time_mlp_out.weight",
            d,
            d,
            Some("model.action_time_mlp_out.bias"),
        )?,
        action_out,
        ada: bf16_tensor(ctx, &ada, vec![steps, VLM_LAYERS, 2, 2, d])?,
        time_bias: bf16_tensor(ctx, &time_bias, vec![steps, d])?,
        times,
        dt,
    })
}

/// The 16 learned query tokens appended to the prefix. Upstream averages each group of 32 consecutive rows of
/// a 256-row table into 8 tokens, then mixes the depth and video tables through a shared `Linear(5120, 2560)`.
pub(crate) fn load_queries(ctx: &Context, ck: &Checkpoint) -> Result<QueryTokens> {
    let h = VLM_HIDDEN;
    let group = QUERY_TABLE_ROWS / TASK_TOKENS;
    let mean_tokens = |name: &str| -> Result<Vec<f32>> {
        let t = ck.f32(name, &[QUERY_TABLE_ROWS, h])?;
        let mut out = vec![0f32; TASK_TOKENS * h];
        for token in 0..TASK_TOKENS {
            for r in 0..group {
                for c in 0..h {
                    out[token * h + c] += t[(token * group + r) * h + c];
                }
            }
            for c in 0..h {
                out[token * h + c] /= group as f32;
            }
        }
        Ok(out)
    };
    let project = |proj: &str, a: Vec<f32>, b: Vec<f32>| -> Result<Vec<f32>> {
        let w = ck.f32(&format!("{proj}.weight"), &[h, 2 * h])?;
        let bias = ck.f32(&format!("{proj}.bias"), &[h])?;
        let mut out = Vec::with_capacity(TASK_TOKENS * h);
        for token in 0..TASK_TOKENS {
            let mut x = Vec::with_capacity(2 * h);
            x.extend_from_slice(&a[token * h..(token + 1) * h]);
            x.extend_from_slice(&b[token * h..(token + 1) * h]);
            out.extend(math::matvec_f32(&w, h, 2 * h, &x, Some(&bias)));
        }
        Ok(out)
    };
    let current = project(
        "model.current_shared_task_proj",
        mean_tokens("model.depth_align_embs")?,
        mean_tokens("model.current_video_align_embs")?,
    )?;
    let future = project(
        "model.future_shared_task_proj",
        mean_tokens("model.future_depth_align_embs")?,
        mean_tokens("model.future_video_align_embs")?,
    )?;
    let rows: Vec<bf16> = current
        .into_iter()
        .chain(future)
        .map(bf16::from_f32)
        .collect();
    Ok(QueryTokens {
        rows: bf16_tensor(ctx, &rows, vec![PREFIX_QUERY_TOKENS, h])?,
    })
}

impl LingbotWeights {
    pub fn load(ctx: &Context, dir: &std::path::Path, steps: usize) -> Result<Self> {
        let ck = Checkpoint::open(dir)?;
        validate_checkpoint(&ck)?;
        let t0 = std::time::Instant::now();
        let vision = load_vision(ctx, &ck)?;
        let vlm = load_vlm(ctx, &ck)?;
        let expert = load_expert(ctx, &ck, steps)?;
        let queries = load_queries(ctx, &ck)?;
        eprintln!(
            "[apxinf] LingBot-VLA 2.0 weights prepared in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        Ok(Self {
            vision,
            vlm,
            expert,
            queries,
        })
    }
}

/// Fail before any weight is converted when the checkpoint is not the architecture this module implements.
pub(crate) fn validate_checkpoint(ck: &Checkpoint) -> Result<()> {
    let l0 = format!("{LM}.layers.0");
    let e0 = format!("{EXP}.layers.0");
    ck.expect(
        &format!("{l0}.self_attn.q_proj.weight"),
        &[VLM_HEADS * HEAD_DIM, VLM_HIDDEN],
    )?;
    ck.expect(
        &format!("{l0}.mlp.gate_proj.weight"),
        &[VLM_FFN, VLM_HIDDEN],
    )?;
    ck.expect(
        &format!("{e0}.mlp.experts.gate_proj"),
        &[MOE_EXPERTS, MOE_FFN, EXPERT_HIDDEN],
    )?;
    ck.expect(
        &format!("{e0}.mlp.shared_expert.gate_proj.weight"),
        &[MOE_SHARED_FFN, EXPERT_HIDDEN],
    )?;
    ck.expect("model.action_in_proj.weight", &[EXPERT_HIDDEN, STATE_DIM])?;
    ck.expect(
        &format!("{VIS}.blocks.0.attn.qkv.weight"),
        &[3 * VIT_HIDDEN, VIT_HIDDEN],
    )?;
    for (layers, prefix, what) in [
        (VLM_LAYERS, format!("{LM}.layers"), "VLM"),
        (VLM_LAYERS, format!("{EXP}.layers"), "expert"),
    ] {
        if ck.contains(&format!("{prefix}.{layers}.input_layernorm.weight")) {
            return Err(Error::Other(format!(
                "LingBot-VLA 2.0: checkpoint has more than {layers} {what} layers"
            )));
        }
    }
    if ck.contains(&format!("{VIS}.blocks.{VIT_LAYERS}.norm1.weight")) {
        return Err(Error::Other(format!(
            "LingBot-VLA 2.0: checkpoint has more than {VIT_LAYERS} ViT blocks"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> HostEmbedding {
        // row r holds the value r in every column, so a gathered row identifies itself
        let mut t = Vec::with_capacity(VLM_VOCAB * VLM_HIDDEN);
        for r in 0..VLM_VOCAB {
            t.extend(std::iter::repeat(bf16::from_f32((r % 251) as f32)).take(VLM_HIDDEN));
        }
        HostEmbedding { table: t }
    }

    #[test]
    fn gather_returns_the_requested_rows_in_order() {
        let out = table().gather(&[3, 0, VISION_START_ID as i32, 2]).unwrap();
        assert_eq!(out.len(), 4 * VLM_HIDDEN);
        for (k, want) in [3.0, 0.0, (VISION_START_ID % 251) as f32, 2.0]
            .into_iter()
            .enumerate()
        {
            assert!(
                out[k * VLM_HIDDEN..(k + 1) * VLM_HIDDEN]
                    .iter()
                    .all(|v| v.to_f32() == want),
                "row {k}"
            );
        }
    }

    #[test]
    fn gather_rejects_ids_outside_the_vocabulary() {
        let t = table();
        assert!(t.gather(&[-1]).is_err());
        assert!(t.gather(&[VLM_VOCAB as i32]).is_err());
        assert!(t.gather(&[]).unwrap().is_empty());
    }

    #[test]
    fn vision_special_tokens_are_inside_the_vocabulary() {
        assert!((VISION_START_ID as usize) < VLM_VOCAB && (VISION_END_ID as usize) < VLM_VOCAB);
    }
}

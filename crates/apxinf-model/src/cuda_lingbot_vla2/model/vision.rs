//! Qwen3-VL vision tower (24 blocks, W8A8 ConvRot) plus the final and deepstack mergers.
//!
//! The three camera views are three independent 256-row segments of one packed sequence, so the
//! tower is one batched pass: patch embedding, 24 blocks, then the 2x2 spatial merge.

use super::util::*;
use crate::cuda_lingbot_vla2::backend::{ops, Context};
use crate::cuda_lingbot_vla2::config::*;
use crate::cuda_lingbot_vla2::weights::model_weights::{Merger, VisionWeights, VitBlock};
use apxinf_core::{DType, Result, Tensor};

pub(crate) struct VisionOutput {
    /// `[3 * 64, 2560]` merged visual tokens, view-major.
    pub tokens: Tensor,
    /// One `[3 * 64, 2560]` tensor per deepstack tap, in the same token order.
    pub deepstack: Vec<Tensor>,
}

fn block(
    ctx: &Context,
    w: &VisionWeights,
    b: &VitBlock,
    x: &Tensor,
    policy: &ops::GemmPolicy,
    attention: &ops::AttentionPolicy,
) -> Result<Tensor> {
    let tokens = x.shape().dims()[0];
    let normed = layer_norm(ctx, x, &b.norm1_w, &b.norm1_b, VIT_EPS)?;
    let qkv = int8_linear(ctx, &rotate_quantize(ctx, &normed)?, &b.qkv, policy)?;
    let shape = [tokens, VIT_HEADS, VIT_HIDDEN / VIT_HEADS];
    let mut q = alloc(ctx, &shape, DType::BF16)?;
    let mut k = alloc(ctx, &shape, DType::BF16)?;
    let mut v = alloc(ctx, &shape, DType::BF16)?;
    // The qkv bias is added inside the split kernel; RoundedInput matches a bf16 Linear followed by RoPE.
    ops::split_qkv_axial_rope(
        ctx,
        ops::AxialRopeArgs {
            qkv: &qkv,
            bias: Some(&b.qkv_bias),
            positions: &w.axial_positions,
            query: &mut q,
            key: &mut k,
            value: &mut v,
            theta: VIT_ROPE_THETA,
            numerics: ops::AxialRopeNumerics::RoundedInput,
        },
    )?;
    let mut attended = alloc(ctx, &shape, DType::BF16)?;
    let mut args = ops::SegmentedAttentionArgs::new(
        &q,
        &k,
        &v,
        &mut attended,
        &w.segment_offsets_dev,
        &w.segment_offsets,
    );
    args.policy = attention.clone();
    ops::segmented_attention(ctx, args)?;
    let attended = attended.reshape(vec![tokens, VIT_HIDDEN])?;
    let proj = add_bias(
        ctx,
        &int8_linear(ctx, &rotate_quantize(ctx, &attended)?, &b.proj, policy)?,
        &b.proj_bias,
    )?;
    let x = {
        let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
        ops::elementwise_add(ctx, x, &proj, &out)?;
        out
    };
    let normed = layer_norm(ctx, &x, &b.norm2_w, &b.norm2_b, VIT_EPS)?;
    let h = add_bias(
        ctx,
        &int8_linear(ctx, &rotate_quantize(ctx, &normed)?, &b.fc1, policy)?,
        &b.fc1_bias,
    )?;
    let activated = alloc(ctx, h.shape().dims(), DType::BF16)?;
    ops::elementwise_activation(ctx, &h, &activated, ops::ElementwiseActivation::GeluTanh)?;
    let h2 = add_bias(
        ctx,
        &int8_linear(ctx, &rotate_quantize(ctx, &activated)?, &b.fc2, policy)?,
        &b.fc2_bias,
    )?;
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::elementwise_add(ctx, &x, &h2, &out)?;
    Ok(out)
}

/// Merge each 2x2 patch block (4 consecutive rows of the merge-block-major order) and project to 2560.
/// `post_shuffle_norm`: the deepstack mergers normalise the merged 4096-wide rows; the final merger
/// normalises each 1024-wide ViT row before merging.
fn merge(
    ctx: &Context,
    m: &Merger,
    x: &Tensor,
    post_shuffle_norm: bool,
    policy: &ops::GemmPolicy,
) -> Result<Tensor> {
    let rows = x.shape().dims()[0];
    let merged_width = VIT_HIDDEN * VIT_MERGE * VIT_MERGE;
    let merged = if post_shuffle_norm {
        let flat = x.reshape(vec![rows / 4, merged_width])?;
        layer_norm(ctx, &flat, &m.norm_w, &m.norm_b, VIT_EPS)?
    } else {
        let normed = layer_norm(ctx, x, &m.norm_w, &m.norm_b, VIT_EPS)?;
        normed.reshape(vec![rows / 4, merged_width])?
    };
    let h = add_bias(
        ctx,
        &int8_linear(ctx, &rotate_quantize(ctx, &merged)?, &m.fc1, policy)?,
        &m.fc1_bias,
    )?;
    let g = ops::pointwise::gelu_exact(ctx, &h)?;
    add_bias(
        ctx,
        &int8_linear(ctx, &rotate_quantize(ctx, &g)?, &m.fc2, policy)?,
        &m.fc2_bias,
    )
}

/// `patches`: BF16 `[3 * 256, 1536]` Qwen3-VL patch rows of the three views.
pub(crate) fn forward(
    ctx: &Context,
    w: &VisionWeights,
    patches: &Tensor,
    policy: &ops::GemmPolicy,
    attention: &ops::AttentionPolicy,
) -> Result<VisionOutput> {
    let embedded = dense(ctx, patches, &w.patch, policy)?;
    let mut x = alloc(ctx, embedded.shape().dims(), DType::BF16)?;
    ops::elementwise_add(ctx, &embedded, &w.pos_embeds, &x)?;
    let mut taps: Vec<Tensor> = Vec::with_capacity(DEEPSTACK_TAPS.len());
    for (index, b) in w.blocks.iter().enumerate() {
        x = block(ctx, w, b, &x, policy, attention)?;
        if let Some(slot) = DEEPSTACK_TAPS.iter().position(|tap| *tap == index) {
            taps.push(merge(ctx, &w.deepstack[slot], &x, true, policy)?);
        }
    }
    let tokens = merge(ctx, &w.merger, &x, false, policy)?;
    Ok(VisionOutput {
        tokens,
        deepstack: taps,
    })
}

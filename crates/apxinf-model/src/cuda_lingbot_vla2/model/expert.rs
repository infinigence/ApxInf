//! Action expert: 36 layers, each a joint-attention step against the VLM's cached K/V, then a token-choice MoE.
//!
//! One flow step embeds `[state | 50 noisy actions]` (51 tokens), runs the 36 layers, and reads the 50
//! action rows through `action_out_proj`. The prefix K/V are fixed for the whole request; each layer
//! writes its 51 suffix K/V rows into the cache behind the prefix and the next step overwrites them.

use super::language::KvCaches;
use super::util::*;
use crate::cuda_lingbot_vla2::backend::{ops, Context};
use crate::cuda_lingbot_vla2::config::*;
use crate::cuda_lingbot_vla2::weights::model_weights::{ExpertLayer, ExpertWeights, MoeLayer};
use apxinf_core::{DType, Result, Tensor};

/// Sparse MoE as dense-masked GEMMs: every expert's gate/up are one stacked GEMM, each expert's block of
/// the SwiGLU output is scaled by its routing weight (0 when unselected), and one K = 32 * 512 down GEMM sums
/// the experts. Equal to sparse dispatch because an unselected expert contributes exactly zero.
fn moe(ctx: &Context, w: &MoeLayer, x: &Tensor, policy: &ops::GemmPolicy) -> Result<Tensor> {
    let routing = ops::moe_route(
        ctx,
        x,
        &w.gate_router,
        &w.router_bias,
        MOE_TOP_K,
        MOE_ROUTED_SCALE,
    )?;
    let rotated = rotate_quantize(ctx, x)?;
    let gate = int8_linear(ctx, &rotated, &w.gate, policy)?;
    let up = int8_linear(ctx, &rotated, &w.up, policy)?;
    let hidden = ops::moe_swiglu_scaled(ctx, &gate, &up, &routing)?;
    let routed = int8_linear(ctx, &rotate_quantize(ctx, &hidden)?, &w.down, policy)?;
    // Shared expert (always active, no gate): plain BF16 SwiGLU MLP.
    let sg = dense(ctx, x, &w.shared_gate, policy)?;
    let su = dense(ctx, x, &w.shared_up, policy)?;
    let act = ops::pointwise::silu_mul(ctx, &sg, &su)?;
    let shared = dense(ctx, &act, &w.shared_down, policy)?;
    let out = alloc(ctx, routed.shape().dims(), DType::BF16)?;
    ops::elementwise_add(ctx, &routed, &shared, &out)?;
    Ok(out)
}

fn adaptive_rms(
    ctx: &Context,
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
) -> Result<Tensor> {
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::norm::norm(
        ctx,
        ops::NormArgs {
            input: x,
            output: Some(&out),
            kind: ops::NormKind::Rms,
            affine: ops::Affine::WeightedAdaptive {
                weight,
                scale,
                shift,
            },
            quantized: None,
            numerics: ops::NormNumerics::Flexible,
            eps: EXPERT_EPS,
        },
    )?;
    Ok(out)
}

/// The `[scale, shift]` pair of `ada[step, layer, which]`, as two `[768]` views.
fn ada_pair(ada: &Tensor, step: usize, layer: usize, which: usize) -> Result<(Tensor, Tensor)> {
    let base = ((step * VLM_LAYERS + layer) * 2 + which) * 2 * EXPERT_HIDDEN;
    Ok((
        byte_view(ada, base, EXPERT_HIDDEN, vec![EXPERT_HIDDEN])?,
        byte_view(
            ada,
            base + EXPERT_HIDDEN,
            EXPERT_HIDDEN,
            vec![EXPERT_HIDDEN],
        )?,
    ))
}

fn layer(
    ctx: &Context,
    w: &ExpertLayer,
    ew: &ExpertWeights,
    index: usize,
    step: usize,
    x: &Tensor,
    positions: &Tensor,
    caches: &KvCaches,
    policy: &ops::GemmPolicy,
    attention: &ops::AttentionPolicy,
) -> Result<Tensor> {
    let prefix = caches.prefix;
    let (scale, shift) = ada_pair(&ew.ada, step, index, 0)?;
    let normed = adaptive_rms(ctx, x, &w.input_norm, &scale, &shift)?;
    let fused = int8_linear(ctx, &rotate_quantize(ctx, &normed)?, &w.qkv, policy)?;
    let mut q = alloc(ctx, &[SUFFIX_TOKENS, EXPERT_HEADS, HEAD_DIM], DType::BF16)?;
    // Tensor clones share device storage, so writing through a clone fills the layer's cache itself.
    let (mut k_cache, mut v_cache) = (caches.k[index].clone(), caches.v[index].clone());
    // Q/K get the VLM's rotary embedding (theta 5e6) at the suffix positions; the bias is added in the split.
    ops::split_qkv_mrope_cache(
        ctx,
        ops::MropeCacheArgs {
            qkv: &fused,
            bias: Some(&w.qkv_bias),
            positions,
            query: &mut q,
            key_cache: &mut k_cache,
            value_cache: &mut v_cache,
            theta: VLM_ROPE_THETA as f32,
            sections: MROPE_SECTIONS,
            cache_offset: prefix,
        },
    )?;
    // State token (row 0): sees the prefix and itself. Action tokens (rows 1..): see the prefix, the state
    // token and all action tokens (bidirectional), so no mask is needed for them.
    let keys = |count: usize| {
        byte_view(
            &caches.k[index],
            0,
            count * VLM_KV_HEADS * HEAD_DIM,
            vec![1, count, VLM_KV_HEADS, HEAD_DIM],
        )
    };
    let values = |count: usize| {
        byte_view(
            &caches.v[index],
            0,
            count * VLM_KV_HEADS * HEAD_DIM,
            vec![1, count, VLM_KV_HEADS, HEAD_DIM],
        )
    };
    let q_state = byte_view(
        &q,
        0,
        EXPERT_HEADS * HEAD_DIM,
        vec![1, 1, EXPERT_HEADS, HEAD_DIM],
    )?;
    let q_action = byte_view(
        &q,
        EXPERT_HEADS * HEAD_DIM,
        ACTION_HORIZON * EXPERT_HEADS * HEAD_DIM,
        vec![1, ACTION_HORIZON, EXPERT_HEADS, HEAD_DIM],
    )?;
    let (k_state, v_state) = (keys(prefix + 1)?, values(prefix + 1)?);
    let (k_all, v_all) = (
        keys(prefix + SUFFIX_TOKENS)?,
        values(prefix + SUFFIX_TOKENS)?,
    );
    let mut a_state = alloc(ctx, &[1, 1, EXPERT_HEADS, HEAD_DIM], DType::BF16)?;
    let mut args = ops::AttentionArgs::new(&q_state, &k_state, &v_state, &mut a_state);
    args.policy = attention.clone();
    ops::attention(ctx, args)?;
    let mut a_action = alloc(
        ctx,
        &[1, ACTION_HORIZON, EXPERT_HEADS, HEAD_DIM],
        DType::BF16,
    )?;
    let mut args = ops::AttentionArgs::new(&q_action, &k_all, &v_all, &mut a_action);
    args.policy = attention.clone();
    ops::attention(ctx, args)?;
    let attended = ops::concat_rows(
        ctx,
        &a_state.reshape(vec![1, EXPERT_HEADS * HEAD_DIM])?,
        &a_action.reshape(vec![ACTION_HORIZON, EXPERT_HEADS * HEAD_DIM])?,
    )?;
    let branch = int8_linear(ctx, &rotate_quantize(ctx, &attended)?, &w.o, policy)?;
    let x = {
        let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
        ops::elementwise_add(ctx, x, &branch, &out)?;
        out
    };
    let (scale, shift) = ada_pair(&ew.ada, step, index, 1)?;
    let normed = adaptive_rms(ctx, &x, &w.post_norm, &scale, &shift)?;
    let mlp = moe(ctx, &w.moe, &normed, policy)?;
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::elementwise_add(ctx, &x, &mlp, &out)?;
    Ok(out)
}

/// One velocity prediction: `[state | noisy actions]` embedding, 36 layers, final norm, `action_out_proj`.
/// `state_row` is the already projected state token `[1, 768]` (constant for the request);
/// `actions` is BF16 `[50, 64]` (zero-padded). Returns `[50, 64]` BF16 velocities.
pub(crate) fn predict_velocity(
    ctx: &Context,
    w: &ExpertWeights,
    step: usize,
    state_row: &Tensor,
    actions: &Tensor,
    positions: &Tensor,
    caches: &KvCaches,
    policy: &ops::GemmPolicy,
    attention: &ops::AttentionPolicy,
) -> Result<Tensor> {
    let embedded = dense(ctx, actions, &w.action_in, policy)?; // [50, 768], bias included
    let mixed = dense_no_bias(ctx, &embedded, &w.action_time_in_action, policy)?;
    let time_bias = byte_view(
        &w.time_bias,
        step * EXPERT_HIDDEN,
        EXPERT_HIDDEN,
        vec![EXPERT_HIDDEN],
    )?;
    let mixed = add_bias(ctx, &mixed, &time_bias)?;
    let activated = alloc(ctx, mixed.shape().dims(), DType::BF16)?;
    ops::elementwise_activation(ctx, &mixed, &activated, ops::ElementwiseActivation::Silu)?;
    let action_tokens = dense(ctx, &activated, &w.action_time_out, policy)?; // [50, 768]
    let mut x = ops::concat_rows(ctx, state_row, &action_tokens)?; // [51, 768]
    for (index, lw) in w.layers.iter().enumerate() {
        x = layer(
            ctx, lw, w, index, step, &x, positions, caches, policy, attention,
        )?;
    }
    let normed = rms(ctx, &x, &w.final_norm, EXPERT_EPS)?;
    let action_rows = rows(&normed, 1, ACTION_HORIZON, EXPERT_HIDDEN)?;
    dense(ctx, &action_rows, &w.action_out, policy)
}

fn dense_no_bias(
    ctx: &Context,
    x: &Tensor,
    w: &Tensor,
    policy: &ops::GemmPolicy,
) -> Result<Tensor> {
    let m = x.shape().dims()[0];
    let n = w.shape().dims()[1];
    let mut out = alloc(ctx, &[m, n], DType::BF16)?;
    let mut args = ops::GemmArgs::new(x, w, &mut out);
    args.policy = policy.clone();
    args.weight_version = Some(ops::WeightVersion::new(0));
    ops::gemm(ctx, args)?;
    Ok(out)
}

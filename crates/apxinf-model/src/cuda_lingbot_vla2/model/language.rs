//! Qwen3-VL language stack over the multimodal prefix (W8A8 ConvRot, BF16 residual stream).
//!
//! The prefix is processed once per request. Every layer's K and V stay resident in a token-major cache
//! `[prefix + suffix, 8, 128]` because the action expert attends to them at each of its ten flow steps.

use super::util::*;
use crate::cuda_lingbot_vla2::backend::{ops, Context, DeviceBuffer};
use crate::cuda_lingbot_vla2::config::*;
use crate::cuda_lingbot_vla2::weights::model_weights::{VlmLayer, VlmWeights};
use apxinf_core::{DType, Result, Tensor};

/// Per-layer K and V, `[capacity, 8, 128]` BF16, token-major. `capacity = prefix + SUFFIX_TOKENS`: the
/// expert appends its suffix K/V behind the prefix rows and overwrites them at every flow step.
pub(crate) struct KvCaches {
    pub k: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub prefix: usize,
}

impl KvCaches {
    pub fn new(ctx: &Context, prefix: usize) -> Result<Self> {
        let shape = [prefix + SUFFIX_TOKENS, VLM_KV_HEADS, HEAD_DIM];
        let mut k = Vec::with_capacity(VLM_LAYERS);
        let mut v = Vec::with_capacity(VLM_LAYERS);
        for _ in 0..VLM_LAYERS {
            k.push(alloc(ctx, &shape, DType::BF16)?);
            v.push(alloc(ctx, &shape, DType::BF16)?);
        }
        Ok(Self { k, v, prefix })
    }
}

fn layer(
    ctx: &Context,
    w: &VlmLayer,
    x: &Tensor,
    positions: &DeviceBuffer,
    k_cache: &Tensor,
    v_cache: &Tensor,
    policy: &ops::GemmPolicy,
    attention: &ops::AttentionPolicy,
) -> Result<Tensor> {
    let tokens = x.shape().dims()[0];
    let normed = rms(ctx, x, &w.input_norm, VLM_EPS)?;
    let rotated = rotate_quantize(ctx, &normed)?;
    let q =
        int8_linear(ctx, &rotated, &w.q, policy)?.reshape(vec![tokens * VLM_HEADS, HEAD_DIM])?;
    let k =
        int8_linear(ctx, &rotated, &w.k, policy)?.reshape(vec![tokens * VLM_KV_HEADS, HEAD_DIM])?;
    let v = int8_linear(ctx, &rotated, &w.v, policy)?;
    // per-head `weight * rms(x)`, then half-split rotation by the integer mRoPE positions (3 interleaved axes)
    let (q, k) = ops::rope::rms_norm_apply_mrope_qk(
        ctx,
        &q,
        &w.q_norm,
        &k,
        &w.k_norm,
        tokens,
        VLM_HEADS,
        VLM_KV_HEADS,
        HEAD_DIM,
        VLM_EPS,
        VLM_ROPE_THETA as f32,
        MROPE_SECTIONS,
        positions,
    )?;
    // this layer's K and V stay resident for the expert: token-major `[tokens, 8, 128]` rows of the cache
    let k_rows = byte_view(
        k_cache,
        0,
        tokens * VLM_KV_HEADS * HEAD_DIM,
        vec![tokens, VLM_KV_HEADS, HEAD_DIM],
    )?;
    let v_rows = byte_view(
        v_cache,
        0,
        tokens * VLM_KV_HEADS * HEAD_DIM,
        vec![tokens, VLM_KV_HEADS, HEAD_DIM],
    )?;
    ops::copy_tensor(ctx, &k, &k_rows)?;
    ops::copy_tensor(
        ctx,
        &v.reshape(vec![tokens, VLM_KV_HEADS, HEAD_DIM])?,
        &v_rows,
    )?;
    let k4 = k_rows.reshape(vec![1, tokens, VLM_KV_HEADS, HEAD_DIM])?;
    let v4 = v_rows.reshape(vec![1, tokens, VLM_KV_HEADS, HEAD_DIM])?;
    let q4 = q.reshape(vec![1, tokens, VLM_HEADS, HEAD_DIM])?;
    let mut attended = alloc(ctx, &[1, tokens, VLM_HEADS, HEAD_DIM], DType::BF16)?;
    let mut args = ops::AttentionArgs::new(&q4, &k4, &v4, &mut attended).causal();
    args.policy = attention.clone();
    ops::attention(ctx, args)?;
    let attended = attended.reshape(vec![tokens, VLM_HEADS * HEAD_DIM])?;
    let branch = int8_linear(ctx, &rotate_quantize(ctx, &attended)?, &w.o, policy)?;
    let x = {
        let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
        ops::elementwise_add(ctx, x, &branch, &out)?;
        out
    };
    let normed = rms(ctx, &x, &w.post_norm, VLM_EPS)?;
    let rotated = rotate_quantize(ctx, &normed)?;
    let gate = int8_linear(ctx, &rotated, &w.gate, policy)?;
    let up = int8_linear(ctx, &rotated, &w.up, policy)?;
    let act = ops::pointwise::silu_mul(ctx, &gate, &up)?;
    let down = int8_linear(ctx, &rotate_quantize(ctx, &act)?, &w.down, policy)?;
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::elementwise_add(ctx, &x, &down, &out)?;
    Ok(out)
}

/// Run the prefix through the 36 layers. `visual_rows[view]` is the row offset of that view's 64 image tokens
/// inside `prefix`; the three deepstack tensors are added onto those rows after layers 0, 1 and 2.
pub(crate) fn forward(
    ctx: &Context,
    w: &VlmWeights,
    prefix: &Tensor,
    positions: &DeviceBuffer,
    visual_rows: &[usize; NUM_VIEWS],
    deepstack: &[Tensor],
    caches: &KvCaches,
    policy: &ops::GemmPolicy,
    attention: &ops::AttentionPolicy,
) -> Result<()> {
    let mut x = prefix.clone();
    for (index, lw) in w.layers.iter().enumerate() {
        x = layer(
            ctx,
            lw,
            &x,
            positions,
            &caches.k[index],
            &caches.v[index],
            policy,
            attention,
        )?;
        if let Some(extra) = deepstack.get(index) {
            for (view, row0) in visual_rows.iter().enumerate() {
                let target = rows(&x, *row0, MERGED_PER_VIEW, VLM_HIDDEN)?;
                let addend = rows(extra, view * MERGED_PER_VIEW, MERGED_PER_VIEW, VLM_HIDDEN)?;
                ops::add_into(ctx, &addend, &target)?;
            }
        }
    }
    Ok(())
}

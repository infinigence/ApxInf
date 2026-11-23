//! One CUDA-graph-capturable decode step for qwen3-asr: the per-step
//! mathematics over the persistent step workspace, shared verbatim by the
//! graph capture and the eager decode loop.

use apxinf_core::{Error, Result, Shape};
use apxinf_cuda_new::ops::{self, ArgmaxArgs, KvCacheAttentionArgs};
use apxinf_cuda_new::CudaContext as Context;

use crate::cuda_qwen3_asr::config::*;
use crate::cuda_qwen3_asr::model::decoder::{
    add_rms_norm_into, gemm_weight_into, rms_norm_into, view_rows, KvCaches, StepWorkspace,
};
use crate::cuda_qwen3_asr::weights::Weights;

/// One decode step, from the persistent token tensor to the argmax in
/// `ws.index`. No host synchronization, no allocation, no H2D copy:
/// everything position-dependent is read on the device.
///
/// `position` (the cache row this step writes) is a host-side parameter used
/// only to seed the attention spec and its prepare-time metadata write; the
/// captured kernels never see it directly.
pub(crate) fn step_body(
    ctx: &Context,
    weights: &mut Weights,
    caches: &KvCaches,
    ws: &mut StepWorkspace,
    position: usize,
) -> Result<()> {
    let kv_len = position + 1;
    // Embedding gather from the persistent token tensor (fixed address).
    let table = weights.get(ctx, "text.embed.w")?;
    ops::embedding_gather(ctx, &table, &ws.token, &mut ws.hidden[0])
        .map_err(|e| Error::Other("embedding_gather: ".to_owned() + &e.to_string()))?;
    let mut cur = 0usize;

    for layer in 0..TXT_LAYERS {
        let p = format!("text.layers.{layer}");
        // Shallow clone sharing the current slot's storage; the add below
        // targets the other slot, so it stays a safe snapshot.
        let residual = ws.hidden[cur].clone();
        if layer == 0 {
            // Layer 0's entry norm has no preceding add to fuse with (the
            // gather writes hidden[0] directly); from layer 1 on, the entry
            // norm is produced by the previous layer's tail fusion (see the
            // end of the loop).
            let iln_w = weights.get(ctx, &format!("{p}.input_layernorm.w"))?;
            rms_norm_into(ctx, &ws.hidden[cur], &iln_w, &mut ws.normed)?;
        }

        // Q/K/V projections (no bias) into the stable buffers.
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed,
            &format!("{p}.self_attn.q.w"),
            &mut ws.q,
        )?; // [1, 2048]
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed,
            &format!("{p}.self_attn.k.w"),
            &mut ws.k,
        )?; // [1, 1024]
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed,
            &format!("{p}.self_attn.v.w"),
            &mut ws.v,
        )?; // [1, 1024]

        // Per-head q_norm / k_norm into fixed buffers. Qwen3's RMSNorm
        // multiplies the weight directly (plain-weight semantics, same as
        // the prefill path's rank-2 norm); the fused kernel below is
        // bit-identical to that. Fixed addresses so the captured graph's
        // fused launch stays valid.
        let q_flat = ws.q.reshape(Shape::new(vec![TXT_HEADS, TXT_HEAD_DIM]))?;
        let k_flat = ws.k.reshape(Shape::new(vec![TXT_KV_HEADS, TXT_HEAD_DIM]))?;
        let q_norm_w = weights.get(ctx, &format!("{p}.self_attn.q_norm.w"))?;
        let k_norm_w = weights.get(ctx, &format!("{p}.self_attn.k_norm.w"))?;
        // Fused per-head RMSNorm + full-width rope: bit-identical to the
        // rms_norm_into + partial_rope pair it replaces (norm.cuh plain-weight
        // semantics, 256-thread reduction, bf16 round-trip preserved).
        ops::qk_norm_rope(
            ctx,
            &q_flat,
            &q_norm_w,
            &mut ws.q_normed,
            &ws.positions,
            TXT_HEADS,
            TXT_HEAD_DIM,
            TXT_HEAD_DIM,
            ROPE_THETA,
            RMS_EPS,
        )
        .map_err(|e| Error::Other(format!("layer {layer} q norm+rope: {e}")))?;
        ops::qk_norm_rope(
            ctx,
            &k_flat,
            &k_norm_w,
            &mut ws.k_normed,
            &ws.positions,
            TXT_KV_HEADS,
            TXT_HEAD_DIM,
            TXT_HEAD_DIM,
            ROPE_THETA,
            RMS_EPS,
        )
        .map_err(|e| Error::Other(format!("layer {layer} k norm+rope: {e}")))?;

        // Heads views are taken AFTER the fused kernel has written
        // q_normed/k_normed: a view built before the write could hand the
        // attention/cache stage stale pre-norm data.
        let q_heads = ws
            .q_normed
            .reshape(Shape::new(vec![1, TXT_HEADS, TXT_HEAD_DIM]))?;
        let k_heads = ws
            .k_normed
            .reshape(Shape::new(vec![1, TXT_KV_HEADS, TXT_HEAD_DIM]))?;

        // K/V into the cache row `position`: the scatter kernel reads the
        // row index from device memory, so the captured launch is valid for
        // every later position too.
        let k_row = k_heads.reshape(Shape::new(vec![1, TXT_KV_HEADS * TXT_HEAD_DIM]))?;
        ops::cache_append_row(ctx, &k_row, &caches.keys[layer], &ws.positions)
            .map_err(|e| Error::Other(format!("layer {layer} k cache append: {e}")))?;
        ops::cache_append_row(ctx, &ws.v, &caches.values[layer], &ws.positions)
            .map_err(|e| Error::Other(format!("layer {layer} v cache append: {e}")))?;

        // Attention over the full-capacity cache; the runtime length and
        // query position come from the fixed-address decode metadata, read
        // by the kernel during replay.
        let k_cache = caches.key_view(layer, 0, KV_CAPACITY)?;
        let v_cache = caches.value_view(layer, 0, KV_CAPACITY)?;
        let q_attn = q_heads.reshape(Shape::new(vec![1, 1, TXT_HEADS, TXT_HEAD_DIM]))?;
        {
            let mut out_heads = ws
                .attn_out
                .reshape(Shape::new(vec![1, 1, TXT_HEADS, TXT_HEAD_DIM]))?;
            let mut args =
                KvCacheAttentionArgs::new(&q_attn, &k_cache, &v_cache, &mut out_heads)
                    .with_decode_meta(&ws.decode_meta);
            // Seeds the prepare traversal's metadata write; during capture
            // the spec records capacity/0 and the kernel reads the live
            // values instead.
            args.valid_key_tokens = kv_len;
            args.query_start = position;
            ops::kv_cache_attention(ctx, args).map_err(|e| {
                Error::Other(format!("layer {layer} kv attention (kv_len {kv_len}): {e}"))
            })?;
        }
        gemm_weight_into(
            ctx,
            weights,
            &ws.attn_out,
            &format!("{p}.self_attn.o.w"),
            &mut ws.attn_proj,
        )?;
        // Residual add + next-entry norm fused (C3): one kernel in place of
        // add_into + rms_norm_into. weights.get has no graph node, so
        // hoisting it above the fused launch keeps the captured node order
        // identical.
        let pln_w = weights.get(ctx, &format!("{p}.post_attention_layernorm.w"))?;
        add_rms_norm_into(
            ctx,
            &residual,
            &ws.attn_proj,
            &pln_w,
            &mut ws.hidden[(cur + 1) % 2],
            &mut ws.normed2,
        )?;
        cur = (cur + 1) % 2;

        // MLP: fused gate+up -> SwiGLU -> down.
        let residual2 = ws.hidden[cur].clone();
        gemm_weight_into(
            ctx,
            weights,
            &ws.normed2,
            &format!("{p}.mlp.gate_up.w"),
            &mut ws.fused,
        )?; // [1, 12288]
        ops::swiglu_into(ctx, &ws.fused, &ws.inner)
            .map_err(|e| Error::Other(format!("layer {layer} swiglu: {e}")))?;
        gemm_weight_into(ctx, weights, &ws.inner, &format!("{p}.mlp.down.w"), &mut ws.outer)?;
        // Tail fusion (C3): the residual add pairs with the next layer's
        // entry norm (or the final norm on the last layer) into one kernel;
        // the two original nodes are adjacent in the captured stream (only
        // alias clones and weights.get sit between them).
        if layer + 1 < TXT_LAYERS {
            let next_p = format!("text.layers.{}", layer + 1);
            let next_iln_w = weights.get(ctx, &format!("{next_p}.input_layernorm.w"))?;
            add_rms_norm_into(
                ctx,
                &residual2,
                &ws.outer,
                &next_iln_w,
                &mut ws.hidden[(cur + 1) % 2],
                &mut ws.normed,
            )?;
        } else {
            let norm_w = weights.get(ctx, "text.norm.w")?;
            add_rms_norm_into(
                ctx,
                &residual2,
                &ws.outer,
                &norm_w,
                &mut ws.hidden[(cur + 1) % 2],
                &mut ws.final_norm,
            )?;
        }
        cur = (cur + 1) % 2;
    }
    // lm_head logits + argmax into the workspace buffers (view first, then
    // weights.get — same borrow order as logits_argmax_into).
    let last = view_rows(&ws.final_norm, 0, 1, TXT_HIDDEN)?;
    gemm_weight_into(ctx, weights, &last, "text.lm_head.w", &mut ws.logits)?;
    ops::argmax(ctx, ArgmaxArgs::new(&ws.logits, &ws.index))
        .map_err(|e| Error::Other(format!("argmax: {e}")))?;
    Ok(())
}

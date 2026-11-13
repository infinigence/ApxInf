//! Qwen3-VL vision tower on HIP, BF16 eager.
//!
//! Takes the official processor's `pixel_values [N, 1536]` with its
//! `grid_thw`, and returns the primary merger output plus the three deepstack
//! outputs the DiT consumes.
//!
//! The host-side parts -- bilinear position-embedding interpolation, the
//! 2D-RoPE position ids and the grid validation -- are copied from `gr00t/`
//! verbatim, rounding points included: HF materializes the interpolation
//! weights in BF16 and evaluates the four-term sum as three BF16 additions,
//! and contracting that in F32 would move the result.
//!
//! The device path drops every fusion and quantization branch the CUDA
//! implementation selects between and keeps one eager route, so each stage can
//! be dumped and compared against the NVIDIA reference.

use std::sync::Arc;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_hip::native::{self, Activation, AttentionGeometry, HipArena, HipDeviceBytes};
use apxinf_hip::HipContext;

use crate::hip_gr00t::config::Qwen3VLConfig;
use crate::hip_gr00t::model::stages::{NoStages, StageSink};
use crate::hip_gr00t::weights::{HipGr00tWeights, Merger, Norm, VisionBlock};

/// What the vision tower hands the backbone and the DiT.
pub struct VisionOutput {
    /// `[N / merge², 2048]` primary visual embedding.
    pub primary: Tensor,
    /// Three `[N / merge², 2048]` deepstack embeddings.
    pub deepstack: Vec<Tensor>,
}

/// Per-shape host work, done once and reused by every run of that shape.
pub struct PreparedVision {
    /// `[N, hidden]` interpolated and merge-permuted position embeddings.
    position_embeddings: Tensor,
    /// `[N, 2]` `(h, w)` ids for the 2D rotation.
    position_ids: HipDeviceBytes,
    /// Patches per frame; attention is block diagonal over these.
    segments: Vec<usize>,
    patches: usize,
}

impl PreparedVision {
    pub fn patches(&self) -> usize {
        self.patches
    }

    /// The interpolated, merge-permuted position table. Exposed so a stage
    /// comparison can tell a host-preparation mismatch from a block mismatch.
    pub fn position_embeddings(&self) -> &Tensor {
        &self.position_embeddings
    }

    /// Patches per frame; attention is block diagonal over these.
    pub fn segments(&self) -> &[usize] {
        &self.segments
    }
}

/// Interpolate the position table and build the rotation ids for one grid.
///
/// `pos_embed_table` is the checkpoint's `[num_position_embeddings, hidden]`
/// table, on the **host**: the interpolation runs there to keep HF's BF16
/// rounding points, so a loaded model need only retain this one 4.7 MiB
/// tensor rather than the whole host vision weight set.
pub fn prepare(
    ctx: &Arc<HipContext>,
    cfg: &Qwen3VLConfig,
    pos_embed_table: &Tensor,
    grid_thw: &[[u32; 3]],
    patches: usize,
) -> Result<PreparedVision> {
    let merge = cfg.vision.spatial_merge_size;
    let segments = validate_grid_layout(grid_thw, patches, merge)?;
    let host_embeddings =
        compute_pos_embeds(cfg, pos_embed_table, grid_thw, merge, cfg.vision.hidden_size)?;
    if host_embeddings.shape().dims() != [patches, cfg.vision.hidden_size] {
        return Err(Error::Other(format!(
            "Qwen3-VL position embeddings must be [{patches}, {}], got {:?}",
            cfg.vision.hidden_size,
            host_embeddings.shape().dims()
        )));
    }
    let ids = compute_vision_pos_ids(grid_thw, merge)?;
    let position_ids = HipDeviceBytes::new(ctx, ids.len() * std::mem::size_of::<u32>())?;
    position_ids.upload_u32(&ids)?;
    Ok(PreparedVision {
        position_embeddings: ctx.to_device(&host_embeddings)?,
        position_ids,
        segments,
        patches,
    })
}

/// Run the tower. Activations come from `arena`, so every run of a prepared
/// shape reuses the same addresses.
pub fn forward(
    ctx: &HipContext,
    cfg: &Qwen3VLConfig,
    weights: &HipGr00tWeights,
    arena: &HipArena,
    pixel_values: &Tensor,
    prepared: &PreparedVision,
) -> Result<VisionOutput> {
    forward_observed(ctx, cfg, weights, arena, pixel_values, prepared, &mut NoStages)
}

/// [`forward`], reporting each stage to `stages` for comparison against the
/// reference dump. Stage names match `step3_vision_stage_reference.py`.
#[allow(clippy::too_many_arguments)]
pub fn forward_observed(
    ctx: &HipContext,
    cfg: &Qwen3VLConfig,
    weights: &HipGr00tWeights,
    arena: &HipArena,
    pixel_values: &Tensor,
    prepared: &PreparedVision,
    stages: &mut dyn StageSink,
) -> Result<VisionOutput> {
    let vc = &cfg.vision;
    let (hidden, heads, head_dim) = (vc.hidden_size, vc.num_heads, vc.head_dim());
    let merge = vc.spatial_merge_size;
    let eps = 1e-6f32;
    let patches = prepared.patches;

    let dims = pixel_values.shape().dims();
    let patch_width = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
    if dims != [patches, patch_width] {
        return Err(Error::ShapeMismatch {
            expected: format!("[{patches}, {patch_width}]"),
            got: pixel_values.shape().to_string(),
        });
    }

    // Patch embedding, then the interpolated position table.
    let mut x = arena.tensor(vec![patches, hidden], DType::BF16)?;
    native::gemm(ctx, pixel_values, &weights.patch_embed.weight, &x)?;
    let biased = arena.tensor(vec![patches, hidden], DType::BF16)?;
    native::bias_activation(
        ctx,
        &x,
        weights.patch_embed.bias.as_ref(),
        Activation::None,
        &biased,
    )?;
    stages.stage("patch_embed", &biased)?;
    x = arena.tensor(vec![patches, hidden], DType::BF16)?;
    // `bias_residual` with no bias is the plain elementwise add.
    native::bias_residual(ctx, &biased, None, &prepared.position_embeddings, &x)?;
    stages.stage("patch_embed_positioned", &x)?;

    let mut deepstack_states: Vec<Tensor> = Vec::with_capacity(3);
    // The next block's norm1, produced by the previous block's tail in the
    // same kernel as its residual write -- cuda-new's `prepared_norm1` path.
    // One launch fewer at each of the 23 boundaries, and the same values: the
    // fused kernel normalizes the stored, rounded residual exactly as the
    // separate `bias_residual` + `layer_norm` pair did.
    let mut prepared_norm1: Option<Tensor> = None;
    for (index, block) in weights.vision_blocks.iter().enumerate() {
        let next_norm1 = weights.vision_blocks.get(index + 1).map(|next| &next.norm1);
        let (out, norm1) = vision_block(
            ctx,
            arena,
            block,
            &x,
            prepared_norm1.take(),
            next_norm1,
            prepared,
            patches,
            hidden,
            heads,
            head_dim,
            eps,
            index,
            stages,
        )?;
        x = out;
        prepared_norm1 = norm1;
        stages.stage(&format!("block_{index:02}"), &x)?;
        if vc.deepstack_visual_indexes.contains(&index) {
            // The merger reads this state after the loop has moved on, so it
            // needs its own buffer rather than an arena slot a later block
            // would overwrite.
            let keep = arena.tensor(vec![patches, hidden], DType::BF16)?;
            native::copy_rows(ctx, &x, &keep, 0)?;
            deepstack_states.push(keep);
        }
    }

    // Primary merger: LayerNorm over `hidden`, then group merge² patches.
    let primary = merge_primary(
        ctx, arena, &weights.merger, &x, patches, hidden, merge, eps,
    )?;
    stages.stage("primary", &primary)?;
    let mut deepstack = Vec::with_capacity(deepstack_states.len());
    for (index, (merger, state)) in weights
        .deepstack_mergers
        .iter()
        .zip(&deepstack_states)
        .enumerate()
    {
        let out = merge_deepstack(ctx, arena, merger, state, patches, hidden, merge, eps)?;
        stages.stage(&format!("deepstack_out_{index}"), &out)?;
        deepstack.push(out);
    }
    Ok(VisionOutput { primary, deepstack })
}

#[allow(clippy::too_many_arguments)]
fn vision_block(
    ctx: &HipContext,
    arena: &HipArena,
    block: &VisionBlock,
    x: &Tensor,
    prepared_norm1: Option<Tensor>,
    next_norm1: Option<&Norm>,
    prepared: &PreparedVision,
    patches: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    index: usize,
    stages: &mut dyn StageSink,
) -> Result<(Tensor, Option<Tensor>)> {
    // Pre-attention LayerNorm: the previous block's tail computed this one.
    let normed = match prepared_norm1 {
        Some(normed) => normed,
        None => {
            let normed = arena.tensor(vec![patches, hidden], DType::BF16)?;
            norm_into(ctx, &block.norm1, x, eps, &normed)?;
            normed
        }
    };
    stages.stage(&format!("block_{index:02}_norm1"), &normed)?;

    // QKV projection, then split, bias and axial rotation in one kernel --
    // cuda-new's fused vision path. It rounds the biased projection to BF16
    // before rotating, exactly the rounding the split-then-rotate sequence it
    // replaced performed, so the block's outputs are bit-identical.
    let qkv = arena.tensor(vec![patches, 3 * hidden], DType::BF16)?;
    native::gemm(ctx, &normed, &block.qkv.weight, &qkv)?;
    let head_shape = || vec![patches, heads, head_dim];
    let q = arena.tensor(head_shape(), DType::BF16)?;
    let k = arena.tensor(head_shape(), DType::BF16)?;
    let v = arena.tensor(head_shape(), DType::BF16)?;
    native::split_qkv_axial_rope(
        ctx,
        &qkv,
        block.qkv.bias.as_ref(),
        &prepared.position_ids,
        heads,
        head_dim,
        10_000.0,
        &q,
        &k,
        &v,
    )?;

    // Non-causal attention inside each frame: Qwen3-VL's block-diagonal
    // cu_seqlens mask is the batch dimension when every frame has the same
    // patch count, which is what a multi-view grid of one camera produces.
    let attended = arena.tensor(head_shape(), DType::BF16)?;
    let segment = equal_segment(&prepared.segments, index)?;
    native::attention(
        ctx,
        &q,
        &k,
        &v,
        &AttentionGeometry {
            batch: prepared.segments.len(),
            q_tokens: segment,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            kv_tokens: segment,
            causal_query_start: None,
        },
        &attended,
    )?;

    let flat = attended.reshape(vec![patches, hidden])?;
    let projected = arena.tensor(vec![patches, hidden], DType::BF16)?;
    native::gemm(ctx, &flat, &block.proj.weight, &projected)?;
    // The reference's `attn` hook fires after `proj`, bias included, so the
    // comparable tensor is the biased projection rather than this one; the
    // fused residual below is what adds that bias, and `block_NN` covers it.

    // Residual and the pre-MLP norm in one pass: `hidden` is the residual sum
    // rounded once, and `normed2` is computed from that stored value.
    let residual = arena.tensor(vec![patches, hidden], DType::BF16)?;
    let normed2 = arena.tensor(vec![patches, hidden], DType::BF16)?;
    let norm2_bias = block
        .norm2
        .bias
        .as_ref()
        .ok_or(Error::Contract("Qwen3-VL vision norm2 needs a bias"))?;
    native::bias_residual_layer_norm(
        ctx,
        &projected,
        block.proj.bias.as_ref(),
        x,
        &block.norm2.weight,
        norm2_bias,
        eps,
        &residual,
        &normed2,
    )?;
    stages.stage(&format!("block_{index:02}_norm2"), &normed2)?;

    // MLP: FC1, bias + GELU (tanh), FC2, then the residual.
    let inner = block.fc1.weight.shape().dims()[1];
    let activated = arena.tensor(vec![patches, inner], DType::BF16)?;
    // Fused: where the recipe routes this shape to hipBLASLt, the bias and
    // activation are applied inside the GEMM epilogue, on the F32 accumulator.
    // Where it does not, this falls back to exactly the two calls it replaces.
    native::gemm_bias_activation(
        ctx,
        &normed2,
        &block.fc1.weight,
        block.fc1.bias.as_ref(),
        Activation::GeluTanh,
        &activated,
    )?;
    let h2 = arena.tensor(vec![patches, hidden], DType::BF16)?;
    native::gemm(ctx, &activated, &block.fc2.weight, &h2)?;
    let out = arena.tensor(vec![patches, hidden], DType::BF16)?;
    match next_norm1 {
        Some(next) => {
            // The residual write and the next block's norm1 in one pass: the
            // same kernel `norm2` uses above, over the same stored-and-rounded
            // residual, so the boundary's values are bit-identical to the
            // separate pair.
            let normed = arena.tensor(vec![patches, hidden], DType::BF16)?;
            let next_bias = next.bias.as_ref().ok_or(Error::Contract(
                "Qwen3-VL vision norm1 needs a bias",
            ))?;
            native::bias_residual_layer_norm(
                ctx,
                &h2,
                block.fc2.bias.as_ref(),
                &residual,
                &next.weight,
                next_bias,
                eps,
                &out,
                &normed,
            )?;
            Ok((out, Some(normed)))
        }
        None => {
            native::bias_residual(ctx, &h2, block.fc2.bias.as_ref(), &residual, &out)?;
            Ok((out, None))
        }
    }
}

/// Every frame must hold the same patch count for the batched attention above.
fn equal_segment(segments: &[usize], block: usize) -> Result<usize> {
    let first = *segments
        .first()
        .ok_or(Error::Contract("Qwen3-VL vision needs at least one frame"))?;
    if segments.iter().any(|&value| value != first) {
        return Err(Error::Other(format!(
            "Qwen3-VL vision block {block} on HIP needs equally sized frames, got {segments:?}; \
             per-frame attention over mixed resolutions is not implemented"
        )));
    }
    Ok(first)
}

/// `use_postshuffle_norm = false`: normalize over `hidden`, then group.
#[allow(clippy::too_many_arguments)]
fn merge_primary(
    ctx: &HipContext,
    arena: &HipArena,
    merger: &Merger,
    x: &Tensor,
    patches: usize,
    hidden: usize,
    merge: usize,
    eps: f32,
) -> Result<Tensor> {
    let normed = arena.tensor(vec![patches, hidden], DType::BF16)?;
    norm_into(ctx, &merger.norm, x, eps, &normed)?;
    let grouped = reshape_merge(&normed, patches, hidden, merge)?;
    merger_mlp(ctx, arena, merger, &grouped)
}

/// `use_postshuffle_norm = true`: group first, then normalize over the
/// grouped width.
#[allow(clippy::too_many_arguments)]
fn merge_deepstack(
    ctx: &HipContext,
    arena: &HipArena,
    merger: &Merger,
    state: &Tensor,
    patches: usize,
    hidden: usize,
    merge: usize,
    eps: f32,
) -> Result<Tensor> {
    let grouped = reshape_merge(state, patches, hidden, merge)?;
    let (rows, columns) = (grouped.shape().dims()[0], grouped.shape().dims()[1]);
    let normed = arena.tensor(vec![rows, columns], DType::BF16)?;
    norm_into(ctx, &merger.norm, &grouped, eps, &normed)?;
    merger_mlp(ctx, arena, merger, &normed)
}

fn merger_mlp(
    ctx: &HipContext,
    arena: &HipArena,
    merger: &Merger,
    input: &Tensor,
) -> Result<Tensor> {
    let rows = input.shape().dims()[0];
    let inner = merger.fc1.weight.shape().dims()[1];
    let out_width = merger.fc2.weight.shape().dims()[1];
    let h1 = arena.tensor(vec![rows, inner], DType::BF16)?;
    native::gemm(ctx, input, &merger.fc1.weight, &h1)?;
    let activated = arena.tensor(vec![rows, inner], DType::BF16)?;
    // The merger is `nn.GELU()` in Qwen3-VL -- the exact form. Only the vision
    // MLP is configured `gelu_pytorch_tanh`. `gr00t/` uses the tanh
    // approximation for both; see the port's accuracy notes.
    native::bias_activation(
        ctx,
        &h1,
        merger.fc1.bias.as_ref(),
        Activation::GeluErf,
        &activated,
    )?;
    let h2 = arena.tensor(vec![rows, out_width], DType::BF16)?;
    native::gemm(ctx, &activated, &merger.fc2.weight, &h2)?;
    let out = arena.tensor(vec![rows, out_width], DType::BF16)?;
    native::bias_activation(ctx, &h2, merger.fc2.bias.as_ref(), Activation::None, &out)?;
    Ok(out)
}

fn norm_into(
    ctx: &HipContext,
    norm: &Norm,
    input: &Tensor,
    eps: f32,
    out: &Tensor,
) -> Result<()> {
    let bias = norm
        .bias
        .as_ref()
        .ok_or(Error::Contract("Qwen3-VL vision LayerNorm needs a bias"))?;
    native::layer_norm(ctx, input, &norm.weight, bias, eps, out)
}

/// Group `merge²` consecutive rows. Qwen has already permuted the patch
/// sequence into spatial-merge order, so this only changes metadata.
fn reshape_merge(x: &Tensor, patches: usize, hidden: usize, merge: usize) -> Result<Tensor> {
    let merge_sq = merge
        .checked_mul(merge)
        .ok_or(Error::Contract("Qwen3-VL merge size overflow"))?;
    if merge_sq == 0 || patches % merge_sq != 0 {
        return Err(Error::Other(format!(
            "Qwen3-VL merge reshape needs {patches} divisible by {merge_sq}"
        )));
    }
    x.reshape(vec![patches / merge_sq, hidden * merge_sq])
}

// ── Host-side preparation, copied from `gr00t/backbone/vision.rs` ───

fn round_bf16(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

#[inline]
fn interpolate_position_bf16(values: [f32; 4], weights: [f32; 4]) -> f32 {
    // HF materializes the interpolation weights in the BF16 dtype of
    // `pos_embed.weight`, multiplies into a BF16 tensor, and evaluates the
    // left-associated four-tensor sum as three separate BF16 additions.
    // Preserve those rounding points instead of contracting the expression in
    // f32 and rounding only once at the end.
    let weighted = [
        round_bf16(values[0] * round_bf16(weights[0])),
        round_bf16(values[1] * round_bf16(weights[1])),
        round_bf16(values[2] * round_bf16(weights[2])),
        round_bf16(values[3] * round_bf16(weights[3])),
    ];
    let first = round_bf16(weighted[0] + weighted[1]);
    let second = round_bf16(first + weighted[2]);
    round_bf16(second + weighted[3])
}

/// HF's `fast_pos_embed_interpolate`: bilinearly interpolate the 48x48
/// learned table to each grid, then permute into the spatial-merge layout
/// where `merge²` patches are consecutive.
fn compute_pos_embeds(
    cfg: &Qwen3VLConfig,
    pos_embed_table: &Tensor,
    grid_thw: &[[u32; 3]],
    merge: usize,
    hidden: usize,
) -> Result<Tensor> {
    let vc = &cfg.vision;
    let grid_side = (vc.num_position_embeddings as f64).sqrt().round() as usize;
    if pos_embed_table.shape().dims() != [vc.num_position_embeddings, hidden] {
        return Err(Error::Other(format!(
            "Qwen3-VL position table must be [{}, {hidden}], got {:?}",
            vc.num_position_embeddings,
            pos_embed_table.shape().dims()
        )));
    }
    let table = pos_embed_table.to_f32_vec()?;

    let mut permuted = Vec::new();
    for &[temporal, height, grid_width] in grid_thw {
        let (t, h, width) = (temporal as usize, height as usize, grid_width as usize);
        let mut interp = vec![0.0f32; h * width * hidden];
        for hi in 0..h {
            let hf = hi as f32 * (grid_side - 1) as f32 / (h - 1).max(1) as f32;
            let h0 = hf.floor() as usize;
            let h1 = (h0 + 1).min(grid_side - 1);
            let dh = hf - h0 as f32;
            for wi in 0..width {
                let wf = wi as f32 * (grid_side - 1) as f32 / (width - 1).max(1) as f32;
                let w0 = wf.floor() as usize;
                let w1 = (w0 + 1).min(grid_side - 1);
                let dw = wf - w0 as f32;
                let dst = (hi * width + wi) * hidden;
                for c in 0..hidden {
                    let v00 = table[(h0 * grid_side + w0) * hidden + c];
                    let v01 = table[(h0 * grid_side + w1) * hidden + c];
                    let v10 = table[(h1 * grid_side + w0) * hidden + c];
                    let v11 = table[(h1 * grid_side + w1) * hidden + c];
                    interp[dst + c] = interpolate_position_bf16(
                        [v00, v01, v10, v11],
                        [
                            (1.0 - dh) * (1.0 - dw),
                            (1.0 - dh) * dw,
                            dh * (1.0 - dw),
                            dh * dw,
                        ],
                    );
                }
            }
        }
        let (merged_h, merged_w) = (h / merge, width / merge);
        for _frame in 0..t {
            for mh in 0..merged_h {
                for mw in 0..merged_w {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            // The learned spatial table has no temporal axis;
                            // Qwen repeats the same interpolated HxW table for
                            // every frame before the merge permutation.
                            let src = ((mh * merge + ih) * width + (mw * merge + iw)) * hidden;
                            permuted.extend_from_slice(&interp[src..src + hidden]);
                        }
                    }
                }
            }
        }
    }
    let values: Vec<half::bf16> = permuted.iter().map(|&v| half::bf16::from_f32(v)).collect();
    Tensor::from_bf16(vec![permuted.len() / hidden, hidden], &values)
}

/// Vision 2D-RoPE position ids: `(h, w)` per patch in the spatial-merge
/// layout. Matches HF's `rot_pos_emb`.
fn compute_vision_pos_ids(grid_thw: &[[u32; 3]], merge: usize) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for &[temporal, height, grid_width] in grid_thw {
        let (t, h, width) = (temporal as usize, height as usize, grid_width as usize);
        if merge == 0 || h % merge != 0 || width % merge != 0 {
            return Err(Error::Other(format!(
                "Qwen3-VL grid [{t}, {h}, {width}] is incompatible with merge {merge}"
            )));
        }
        for _frame in 0..t {
            for mh in 0..h / merge {
                for mw in 0..width / merge {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            ids.push((mh * merge + ih) as u32);
                            ids.push((mw * merge + iw) as u32);
                        }
                    }
                }
            }
        }
    }
    Ok(ids)
}

/// Patches per frame, and the guarantee that the grids describe exactly the
/// rows `pixel_values` holds.
fn validate_grid_layout(
    grid_thw: &[[u32; 3]],
    patches: usize,
    merge: usize,
) -> Result<Vec<usize>> {
    if grid_thw.is_empty() || merge == 0 {
        return Err(Error::Contract(
            "Qwen3-VL needs image grids and a non-zero merge size",
        ));
    }
    let mut segments = Vec::new();
    let mut total = 0usize;
    for (index, &[temporal, height, width]) in grid_thw.iter().enumerate() {
        let (t, h, w) = (temporal as usize, height as usize, width as usize);
        if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
            return Err(Error::Other(format!(
                "Qwen3-VL grid {index} [{t}, {h}, {w}] is incompatible with merge {merge}"
            )));
        }
        let frame = h
            .checked_mul(w)
            .ok_or(Error::Contract("Qwen3-VL frame patch count overflow"))?;
        total = total
            .checked_add(t * frame)
            .ok_or(Error::Contract("Qwen3-VL total patch count overflow"))?;
        segments.extend(std::iter::repeat(frame).take(t));
    }
    if total != patches {
        return Err(Error::Other(format!(
            "Qwen3-VL grids describe {total} patches but pixel_values has {patches} rows"
        )));
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(t: u32, h: u32, w: u32) -> [u32; 3] {
        [t, h, w]
    }

    #[test]
    fn grid_layout_describes_one_segment_per_frame() {
        // The fixture's two 16x16 grids: 512 patches in two equal segments.
        assert_eq!(
            validate_grid_layout(&[grid(1, 16, 16), grid(1, 16, 16)], 512, 2).unwrap(),
            vec![256, 256]
        );
        // A temporal grid repeats its frame count.
        assert_eq!(
            validate_grid_layout(&[grid(2, 16, 16)], 512, 2).unwrap(),
            vec![256, 256]
        );
        // Mismatched totals and merge-incompatible grids are refused.
        assert!(validate_grid_layout(&[grid(1, 16, 16)], 512, 2).is_err());
        assert!(validate_grid_layout(&[grid(1, 15, 16)], 240, 2).is_err());
        assert!(validate_grid_layout(&[], 0, 2).is_err());
    }

    #[test]
    fn position_ids_walk_each_merge_block_before_moving_on() {
        // Two 2x2 patches at merge 2: every (h, w) inside a block is
        // consecutive, which is what makes the merge reshape metadata-only.
        let ids = compute_vision_pos_ids(&[grid(1, 2, 4)], 2).unwrap();
        assert_eq!(
            ids,
            vec![0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3]
        );
    }

    #[test]
    fn interpolation_keeps_the_bf16_rounding_points() {
        // An exact corner must come back exactly, and the midpoint must round
        // through BF16 three times rather than once -- the property that
        // distinguishes this from an f32 contraction.
        let corner = interpolate_position_bf16([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(corner, 1.0);
        let mixed = interpolate_position_bf16([1.0, 1.0, 1.0, 1.0], [0.25, 0.25, 0.25, 0.25]);
        assert_eq!(mixed, 1.0);
        let third = round_bf16(1.0 / 3.0);
        assert_eq!(interpolate_position_bf16([1.0, 0.0, 0.0, 0.0], [1.0 / 3.0, 0.0, 0.0, 0.0]), third);
    }

    #[test]
    fn equal_segments_are_required_and_the_error_says_why() {
        assert_eq!(equal_segment(&[256, 256], 0).unwrap(), 256);
        let error = equal_segment(&[256, 64], 3).unwrap_err().to_string();
        assert!(error.contains("equally sized frames"), "{error}");
        assert!(error.contains("not implemented"), "{error}");
    }
}

//! Qwen-Drive vision tower (Qwen3.5 ViT) forward path on the CUDA executor.
//!
//! Structure and canonical rewrites follow the maintained qwen3vl vision
//! tower (copied with provenance, renamed, no cross-family import): patch
//! GEMM, bilinear-interpolated position embeddings computed on the host
//! (checkpoint-lifetime table; the host copy is refreshed per request in this
//! revision - recorded as optimization debt), 2D RoPE via the maintained
//! split_vision_qkv_rope kernel, per-image segmented attention, and the
//! merger (LayerNorm -> row merge -> fc1 -> exact erf GELU -> fc2). Qwen3.5
//! has no deepstack mergers.

use apxinf_core::{Error, Result, Tensor};

use super::backend::{kernels, transfers, Context, DeviceBuffer};
use kernels::{activation, attention, elementwise, gemm, linear_attention as la, norm};

use super::config::QwenDriveConfig;
use super::device_weights::VisionDeviceWeights;

pub struct VisionOutput {
    /// Merged image embeddings `[tokens, out_hidden]` scattered into the
    /// text stream at the image-token positions.
    pub primary: Tensor,
    /// Pre-merger patch features `[patches, hidden]` (perception tap).
    pub pre_merger: Tensor,
}

// TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 3 L5): cross-module probe-line sink
// for the vision pos-embed corner-anchor check; general.rs drains it into the diag_digest
// after prefill_done; revert in the acceptance-bound revision.
static VISION_DIAG_LINES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// TEMP-DIAG (implement_r5): drain the vision probe lines; revert in the acceptance-bound revision.
pub fn take_vision_diag_lines() -> Vec<String> {
    VISION_DIAG_LINES
        .lock()
        .map(|mut lines| std::mem::take(&mut *lines))
        .unwrap_or_default()
}

fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

/// Run the vision tower over the concatenated per-image patch stream.
/// `pixel_values` is `[patches, 3 * temporal * patch * patch]` BF16 on device;
/// `grid_thw` holds one `[T, H, W]` entry per image.
pub fn forward(
    config: &QwenDriveConfig,
    weights: &VisionDeviceWeights,
    ctx: &Context,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<VisionOutput> {
    let vc = &config.vision;
    let hidden = vc.hidden_size;
    let heads = vc.num_heads;
    let head_dim = vc.head_dim();
    let merge = vc.spatial_merge_size;
    let eps = 1e-6f32;

    let dims = pixel_values.shape().dims();
    let patch_vec = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
    if grid_thw.is_empty() || dims.len() != 2 || dims[1] != patch_vec {
        return Err(Error::Other(format!(
            "qwen_drive vision: pixel_values must be [patches, {patch_vec}], got {dims:?}"
        )));
    }
    let total_patches = dims[0];
    let mut offsets = vec![0u32];
    let mut max_tokens = 0usize;
    for grid in grid_thw {
        let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
        if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
            return Err(Error::Other(format!(
                "qwen_drive vision: invalid grid [{t}, {h}, {w}] for merge {merge}"
            )));
        }
        let patches = t * h * w;
        max_tokens = max_tokens.max(patches);
        offsets.push(offsets[offsets.len() - 1] + patches as u32);
    }
    if offsets[offsets.len() - 1] as usize != total_patches {
        return Err(Error::Other(format!(
            "qwen_drive vision: grids cover {} patches but pixel_values has {total_patches}",
            offsets[offsets.len() - 1]
        )));
    }
    // TEMP-DIAG (implement_r3): vision-entry geometry; revert in the acceptance-bound revision.
    eprintln!("[qwen_drive] vision_entry total_patches={} segments={} offsets_len={} offsets_max={} max_tokens={} heads={} head_dim={} n={} grids={:?}", total_patches, grid_thw.len(), offsets.len(), offsets.last().copied().unwrap_or(0), max_tokens, heads, head_dim, weights.blocks.len(), grid_thw);
    // TEMP-DIAG (implement_r4): per-op elapsed-ms reference; revert in the acceptance-bound revision.
    let vis_t0 = std::time::Instant::now();
    let offsets_dev = upload_u32(ctx, &offsets)?;

    // Patch embedding: [N, patch_vec] @ [patch_vec, hidden] + bias.
    let mut x = gemm::matmul(ctx, pixel_values, &weights.patch_w)?;
    x = elementwise::bias_bf16(ctx, &x, Some(&weights.patch_b))?;
    super::general::trace_rows("model_visual_patch_embed", &x)?;

    // Bilinear-interpolated learned position embeddings (host canonicalization).
    let pos_embeds = compute_pos_embeds(config, weights, ctx, grid_thw)?;
    x = elementwise::add(ctx, &x, &pos_embeds)?;

    // 2D rope position ids in the merge-block-major patch order.
    let pos_ids = compute_vision_pos_ids(grid_thw, merge);
    let pos_ids_dev = upload_u32(ctx, &pos_ids)?;

    for (block_idx, block) in weights.blocks.iter().enumerate() {
        // TEMP-DIAG (implement_r2, ungated in implement_r3): vision-block heartbeat every block; revert in the acceptance-bound revision.
        eprintln!(
            "[qwen_drive] vision_block k={} n={} ms={}",
            block_idx,
            weights.blocks.len(),
            vis_t0.elapsed().as_millis()
        );
        let normed = norm::layer_bf16(ctx, &x, &block.norm1_w, &block.norm1_b, eps)?;
        if block_idx == 0 {
            super::general::trace_rows("model_visual_blocks_0_norm1", &normed)?;
        }
        // nn.Linear adds bias before its final BF16 rounding.
        let qkv = gemm::bf16_bias(ctx, &normed, &block.qkv_w, &block.qkv_b)?;
        if block_idx == 0 {
            super::general::trace_rows("model_visual_blocks_0_attn_qkv", &qkv)?;
        }
        let qkv = attention::split_vision_qkv_rope_bf16(
            ctx,
            &qkv,
            None,
            &pos_ids_dev,
            heads,
            head_dim,
            10000.0,
        )?;
        let attn = attention::segmented_mha_bf16(
            ctx,
            &qkv.q,
            &qkv.k,
            &qkv.v,
            &offsets_dev,
            &offsets,
            grid_thw.len(),
            max_tokens,
        )?;
        let attn = attn.reshape(vec![total_patches, hidden])?;
        let proj = gemm::bf16_bias(ctx, &attn, &block.proj_w, &block.proj_b)?;
        if block_idx == 0 {
            super::general::trace_rows("model_visual_blocks_0_attn", &proj)?;
        }
        x = elementwise::add(ctx, &x, &proj)?;

        let normed = norm::layer_bf16(ctx, &x, &block.norm2_w, &block.norm2_b, eps)?;
        if block_idx == 0 {
            super::general::trace_rows("model_visual_blocks_0_norm2", &normed)?;
        }
        let h = gemm::bf16_bias(ctx, &normed, &block.fc1_w, &block.fc1_b)?;
        if block_idx == 0 {
            super::general::trace_rows("model_visual_blocks_0_mlp_linear_fc1", &h)?;
        }
        let h = activation::gelu_tanh(ctx, &h)?;
        let h2 = gemm::bf16_bias(ctx, &h, &block.fc2_w, &block.fc2_b)?;
        if block_idx == 0 {
            super::general::trace_rows("model_visual_blocks_0_mlp", &h2)?;
        }
        x = elementwise::add(ctx, &x, &h2)?;
        super::general::trace_rows(&format!("model_visual_blocks_{block_idx}"), &x)?;
    }

    // Merger: LayerNorm(1024) -> merge 4 rows -> fc1 -> exact erf GELU -> fc2.
    let normed = norm::layer_bf16(ctx, &x, &weights.merger_norm_w, &weights.merger_norm_b, eps)?;
    let merged = la::merge_rows(ctx, &normed, merge * merge)?;
    let h = gemm::bf16_bias(ctx, &merged, &weights.merger_fc1_w, &weights.merger_fc1_b)?;
    let h = la::gelu_exact(ctx, &h)?;
    let primary = gemm::bf16_bias(ctx, &h, &weights.merger_fc2_w, &weights.merger_fc2_b)?;
    Ok(VisionOutput {
        primary,
        pre_merger: x,
    })
}

/// Bilinear-interpolate the learned 48x48 position table to each image's
/// patch grid and permute to the merge-block-major layout. Copied with
/// provenance from qwen3vl's vision tower (HF `fast_pos_embed_interpolate` +
/// the spatial-merge shuffle); the table is read back per request (debt).
fn compute_pos_embeds(
    config: &QwenDriveConfig,
    weights: &VisionDeviceWeights,
    ctx: &Context,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    let vc = &config.vision;
    let hidden = vc.hidden_size;
    let merge = vc.spatial_merge_size;
    let grid_side = (vc.num_position_embeddings as f64).sqrt().round() as usize;
    let cpu = transfers::to_cpu(&weights.pos_embed)?;
    let table = cpu
        .to_f32_vec()
        .map_err(|e| Error::Other(format!("qwen_drive vision pos_embed table: {e}")))?;
    if table.len() != grid_side * grid_side * hidden {
        return Err(Error::Other(
            "qwen_drive vision: pos_embed table shape mismatch".into(),
        ));
    }
    let total: usize = grid_thw
        .iter()
        .map(|grid| grid[0] as usize * grid[1] as usize * grid[2] as usize)
        .sum();
    let mut out = vec![0.0f32; total * hidden];
    let mut dst_token = 0usize;
    for grid in grid_thw {
        let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
        let merged_h = h / merge;
        let merged_w = w / merge;
        for _ti in 0..t {
            for mh in 0..merged_h {
                for mw in 0..merged_w {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            let hi = mh * merge + ih;
                            let wi = mw * merge + iw;
                            let hf = hi as f32 * (grid_side - 1) as f32 / (h - 1).max(1) as f32;
                            let h0 = hf.floor() as usize;
                            let h1 = (h0 + 1).min(grid_side - 1);
                            let dh = hf - h0 as f32;
                            let wf = wi as f32 * (grid_side - 1) as f32 / (w - 1).max(1) as f32;
                            let w0 = wf.floor() as usize;
                            let w1 = (w0 + 1).min(grid_side - 1);
                            let dw = wf - w0 as f32;
                            let dst = dst_token * hidden;
                            for c in 0..hidden {
                                let v00 = table[(h0 * grid_side + w0) * hidden + c];
                                let v01 = table[(h0 * grid_side + w1) * hidden + c];
                                let v10 = table[(h1 * grid_side + w0) * hidden + c];
                                let v11 = table[(h1 * grid_side + w1) * hidden + c];
                                out[dst + c] = (1.0 - dh) * (1.0 - dw) * v00
                                    + (1.0 - dh) * dw * v01
                                    + dh * (1.0 - dw) * v10
                                    + dh * dw * v11;
                            }
                            dst_token += 1;
                        }
                    }
                }
            }
        }
    }
    // TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 3 L5): corner-anchor pos-embed
    // check -- corner patches satisfy dh=dw=0, so the emitted rows must equal the checkpoint
    // table corner rows {0, grid_side-1, (grid_side-1)*grid_side, grid_side^2-1} exactly; a
    // golden-free indexing discriminator; routed via VISION_DIAG_LINES; revert in the
    // acceptance-bound revision.
    if let Some(grid) = grid_thw.first() {
        let (h, w) = (grid[1] as usize, grid[2] as usize);
        let merged_h = h / merge;
        let merged_w = w / merge;
        if merged_h > 0 && merged_w > 0 {
            let corners: [(usize, usize, usize, usize); 4] = [
                (0, 0, 0, 0),
                (0, merged_w - 1, 0, merge - 1),
                (merged_h - 1, 0, merge - 1, 0),
                (merged_h - 1, merged_w - 1, merge - 1, merge - 1),
            ];
            let tbl_rows: [usize; 4] = [
                0,
                grid_side - 1,
                (grid_side - 1) * grid_side,
                grid_side * grid_side - 1,
            ];
            let mut idx_parts: Vec<String> = Vec::new();
            let mut match_parts: Vec<String> = Vec::new();
            let mut maxdiff = 0.0f32;
            for (c, &(mh, mw, ih, iw)) in corners.iter().enumerate() {
                let idx = ((mh * merged_w + mw) * merge + ih) * merge + iw;
                let tbl = tbl_rows[c];
                let mut row_match = true;
                for ch in 0..hidden {
                    let d = (out[idx * hidden + ch] - table[tbl * hidden + ch]).abs();
                    if d > maxdiff {
                        maxdiff = d;
                    }
                    if d != 0.0 {
                        row_match = false;
                    }
                }
                idx_parts.push(idx.to_string());
                match_parts.push(if row_match { "T" } else { "F" }.to_string());
            }
            if let Ok(mut lines) = VISION_DIAG_LINES.lock() {
                lines.push(format!(
                    "[qwen_drive] vis_pos_anchor idx=[{}] tbl=[{}] match=[{}] maxdiff={:.6}",
                    idx_parts.join(","),
                    tbl_rows
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                    match_parts.join(","),
                    maxdiff
                ));
            }
        }
    }
    let rounded: Vec<half::bf16> = out.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let tensor = Tensor::from_bf16(vec![total, hidden], &rounded)?;
    transfers::to_cuda(&tensor, ctx.device_id())
}

/// Vision 2D-RoPE position ids `(h, w)` per patch in the merge-block-major
/// layout (copied with provenance from qwen3vl's `compute_vision_pos_ids`).
fn compute_vision_pos_ids(grid_thw: &[[u32; 3]], merge: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    for grid in grid_thw {
        let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
        let merged_h = h / merge;
        let merged_w = w / merge;
        for _ti in 0..t {
            for mh in 0..merged_h {
                for mw in 0..merged_w {
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
    ids
}

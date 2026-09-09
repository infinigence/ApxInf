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

fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_ne_bytes()).collect();
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

    // Bilinear-interpolated learned position embeddings (host canonicalization).
    let pos_embeds = compute_pos_embeds(config, weights, ctx, grid_thw)?;
    x = elementwise::add(ctx, &x, &pos_embeds)?;

    // 2D rope position ids in the merge-block-major patch order.
    let pos_ids = compute_vision_pos_ids(grid_thw, merge);
    let pos_ids_dev = upload_u32(ctx, &pos_ids)?;

    for (block_idx, block) in weights.blocks.iter().enumerate() {
        // TEMP-DIAG (implement_r2, ungated in implement_r3): vision-block heartbeat every block; revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] vision_block k={} n={} ms={}", block_idx, weights.blocks.len(), vis_t0.elapsed().as_millis());
        let normed = norm::layer_bf16(ctx, &x, &block.norm1_w, &block.norm1_b, eps)?;
        let qkv = gemm::matmul(ctx, &normed, &block.qkv_w)?;
        let qkv = attention::split_vision_qkv_rope_bf16(
            ctx,
            &qkv,
            Some(&block.qkv_b),
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
        let proj = gemm::matmul(ctx, &attn, &block.proj_w)?;
        let proj = elementwise::bias_bf16(ctx, &proj, Some(&block.proj_b))?;
        x = elementwise::add(ctx, &x, &proj)?;

        let normed = norm::layer_bf16(ctx, &x, &block.norm2_w, &block.norm2_b, eps)?;
        let h = gemm::matmul(ctx, &normed, &block.fc1_w)?;
        let h = activation::bias_gelu_bf16(ctx, &h, Some(&block.fc1_b))?;
        let h2 = gemm::matmul(ctx, &h, &block.fc2_w)?;
        let h2 = elementwise::bias_bf16(ctx, &h2, Some(&block.fc2_b))?;
        x = elementwise::add(ctx, &x, &h2)?;
    }

    // Merger: LayerNorm(1024) -> merge 4 rows -> fc1 -> exact erf GELU -> fc2.
    let normed = norm::layer_bf16(ctx, &x, &weights.merger_norm_w, &weights.merger_norm_b, eps)?;
    let merged = la::merge_rows(ctx, &normed, merge * merge)?;
    let h = gemm::matmul(ctx, &merged, &weights.merger_fc1_w)?;
    let h = elementwise::bias_bf16(ctx, &h, Some(&weights.merger_fc1_b))?;
    let h = la::gelu_exact(ctx, &h)?;
    let out = gemm::matmul(ctx, &h, &weights.merger_fc2_w)?;
    let primary = elementwise::bias_bf16(ctx, &out, Some(&weights.merger_fc2_b))?;
    Ok(VisionOutput { primary, pre_merger: x })
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
        return Err(Error::Other("qwen_drive vision: pos_embed table shape mismatch".into()));
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

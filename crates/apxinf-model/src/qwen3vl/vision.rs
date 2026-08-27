//! Qwen3-VL vision tower forward path.
//!
//! Takes preprocessed `pixel_values [N, 1536]` and `grid_thw [1, 3]`,
//! produces the primary visual embedding `[N/4, 2048]` and 3 deepstack
//! embeddings `[N/4, 2048]` for injection into the LLM.
//!
//! All compute via `dyn Backend` primitives (matmul, layer_norm, gelu_tanh,
//! add_bias, rope_vision_2d, vision_sdpa). No KV cache — vision attention
//! is a single non-causal forward over the full patch sequence.

use apxinf_core::{Backend, Error, Result, Tensor};

use super::config::Qwen3VLConfig;
use super::vision_weights::Qwen3VLVisionWeights;
pub struct VisionOutput {
    /// Primary embedding `[N/4, out_hidden]` injected at the image_pad
    /// positions in the LLM input embedding stream.
    pub primary: Tensor,
    /// 3 deepstack embeddings, each `[N/4, out_hidden]`, injected at the
    /// 3 deepstack layer depths of the LLM.
    pub deepstack: Vec<Tensor>,
}

/// Run the vision tower. `pixel_values` is `[N, 1536]` bf16 on device;
/// `grid_thw` is `[[T, H, W]]` (one image assumed for now).
pub fn forward(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<VisionOutput> {
    forward_impl(cfg, w, b, pixel_values, grid_thw, None)
}

/// Debug variant that dumps intermediate states to the given directory.
pub fn forward_debug(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
    dump_prefix: &str,
) -> Result<VisionOutput> {
    forward_impl(cfg, w, b, pixel_values, grid_thw, Some(dump_prefix))
}

fn forward_impl(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
    dump: Option<&str>,
) -> Result<VisionOutput> {
    let _vision_range = crate::profiling::trace::range("vision_encoder");
    let vc = &cfg.vision;
    let hidden = vc.hidden_size; // 1024
    let n_heads = vc.num_heads; // 16
    let head_dim = vc.head_dim(); // 64
    let merge = vc.spatial_merge_size; // 2
    let eps = 1e-6f32;

    let dims = pixel_values.shape().dims();
    let n_patches = dims[0]; // e.g. 400
    let (t, h, width) = (
        grid_thw[0][0] as usize,
        grid_thw[0][1] as usize,
        grid_thw[0][2] as usize,
    );

    // ── Patch embedding: pixel_values @ W^T + bias → [N, 1024] ──────
    // patch_embed_weight is [1536, 1024] (already transposed). matmul
    // of [N, 1536] @ [1536, 1024] → [N, 1024].
    let mut x = b.matmul(pixel_values, &w.patch_embed_weight)?;
    x = b.add_bias(&x, &w.patch_embed_bias)?;
    if let Some(d) = dump {
        dump_tensor(b, &x, &format!("{d}_post_patch_embed"))?;
    }

    // ── Positional embedding (bilinear-interpolated, permuted) ──────
    let pos_embeds = compute_pos_embeds(cfg, b, &w.pos_embed, t, h, width, merge, hidden)?;
    if let Some(d) = dump {
        dump_tensor(b, &pos_embeds, &format!("{d}_pos_embeds"))?;
    }
    x = b.add(&x, &pos_embeds)?;
    if let Some(d) = dump {
        dump_tensor(b, &x, &format!("{d}_post_add_pos"))?;
    }

    // ── Vision 2D-RoPE position IDs ─────────────────────────────────
    let pos_ids = compute_vision_pos_ids(t, h, width, merge);

    // ── 24 vision blocks ────────────────────────────────────────────
    let mut deepstack_hidden_states: Vec<Tensor> = Vec::with_capacity(3);
    for (i, blk) in w.blocks.iter().enumerate() {
        let _blk_range = crate::profiling::trace::range(&format!("vision_block_{i}"));
        // pre-attn LayerNorm
        let normed = b.layer_norm(&x, &blk.norm1_w, &blk.norm1_b, eps)?;
        // QKV: [N, 1024] @ [1024, 3072] → [N, 3072]
        let qkv = b.matmul(&normed, &blk.qkv_w)?;
        let qkv = b.add_bias(&qkv, &blk.qkv_b)?;
        // Split Q/K/V: qkv is [N, 3072] = [N, 3, 1024] interleaved as
        // (q[N,0:1024], k[N,1024:2048], v[N,2048:3072]). Reshape to
        // [N, 3, n_heads, head_dim] then split.
        // HF does: qkv.reshape(seq, 3, n_heads, head_dim).permute(1,0,2,3).unbind(0)
        // So q = qkv[:, 0, :, :], k = qkv[:, 1, :, :], v = qkv[:, 2, :, :].
        // In our flat [N, 3072] layout with head-major order:
        //   qkv[n, :] = [q[n,h,d] for h,d] ++ [k[n,h,d] for h,d] ++ [v[n,h,d] for h,d]
        // We need to extract Q = qkv[:, 0:1024], K = qkv[:, 1024:2048],
        // V = qkv[:, 2048:3072] as [N, n_heads, head_dim] tensors.
        let q = slice_and_reshape(b, &qkv, 0, hidden, n_patches, n_heads, head_dim)?;
        let k = slice_and_reshape(b, &qkv, hidden, hidden, n_patches, n_heads, head_dim)?;
        let v = slice_and_reshape(b, &qkv, 2 * hidden, hidden, n_patches, n_heads, head_dim)?;

        // Vision 2D-RoPE on Q and K
        let q = b.rope_vision_2d(&q, n_heads, head_dim, 10000.0, &pos_ids)?;
        let k = b.rope_vision_2d(&k, n_heads, head_dim, 10000.0, &pos_ids)?;

        // Non-causal full attention
        let attn_out = b.vision_sdpa(&q, &k, &v, n_patches, n_heads, head_dim)?;
        // Output projection + residual
        let attn_out = b.matmul(&attn_out, &blk.proj_w)?;
        let attn_out = b.add_bias(&attn_out, &blk.proj_b)?;
        x = b.add(&x, &attn_out)?;

        // pre-MLP LayerNorm
        let normed = b.layer_norm(&x, &blk.norm2_w, &blk.norm2_b, eps)?;
        // FC1 + GELU
        let h1 = b.matmul(&normed, &blk.fc1_w)?;
        let h1 = b.add_bias(&h1, &blk.fc1_b)?;
        let h1 = b.gelu_tanh(&h1)?;
        // FC2 + residual
        let h2 = b.matmul(&h1, &blk.fc2_w)?;
        let h2 = b.add_bias(&h2, &blk.fc2_b)?;
        x = b.add(&x, &h2)?;

        // Capture deepstack states at the configured block indexes.
        if vc.deepstack_visual_indexes.contains(&i) {
            deepstack_hidden_states.push(x.clone());
        }
        if let Some(d) = dump {
            if i < 2 || i == vc.depth - 1 {
                dump_tensor(b, &x, &format!("{d}_post_block_{i}"))?;
            }
        }
    }

    // ── Primary merger ──────────────────────────────────────────────
    // use_postshuffle_norm=False: LayerNorm(1024) on x, then reshape
    // [N, 1024] → [N/4, 4096] (concatenate 4 consecutive patches).
    let primary = merge_primary(b, &w.merger, &x, n_patches, hidden, merge, eps)?;

    // ── Deepstack mergers ───────────────────────────────────────────
    // use_postshuffle_norm=True: reshape [N, 1024] → [N/4, 4096] first,
    // then LayerNorm(4096), then fc1 → GELU → fc2.
    let mut deepstack = Vec::with_capacity(3);
    for (merger, hs) in w
        .deepstack_mergers
        .iter()
        .zip(deepstack_hidden_states.iter())
    {
        let out = merge_deepstack(b, merger, hs, n_patches, hidden, merge, eps)?;
        deepstack.push(out);
    }

    Ok(VisionOutput { primary, deepstack })
}

/// Primary merger: LayerNorm(1024) → reshape [N,1024]→[N/4,4096] → fc1 → GELU → fc2.
fn merge_primary(
    b: &dyn Backend,
    m: &super::vision_weights::Qwen3VLMerger,
    x: &Tensor,
    n_patches: usize,
    hidden: usize,
    merge: usize,
    eps: f32,
) -> Result<Tensor> {
    let normed = b.layer_norm(x, &m.norm_w, &m.norm_b, eps)?;
    // Reshape [N, 1024] → [N/4, 4096]: concatenate 4 consecutive rows.
    let merged = reshape_merge(b, &normed, n_patches, hidden, merge)?;
    let h = b.matmul(&merged, &m.fc1_w)?;
    let h = b.add_bias(&h, &m.fc1_b)?;
    let h = b.gelu_tanh(&h)?;
    let out = b.matmul(&h, &m.fc2_w)?;
    b.add_bias(&out, &m.fc2_b)
}

/// Deepstack merger: reshape [N,1024]→[N/4,4096] → LayerNorm(4096) → fc1 → GELU → fc2.
fn merge_deepstack(
    b: &dyn Backend,
    m: &super::vision_weights::Qwen3VLMerger,
    x: &Tensor,
    n_patches: usize,
    hidden: usize,
    merge: usize,
    eps: f32,
) -> Result<Tensor> {
    let merged = reshape_merge(b, x, n_patches, hidden, merge)?;
    let normed = b.layer_norm(&merged, &m.norm_w, &m.norm_b, eps)?;
    let h = b.matmul(&normed, &m.fc1_w)?;
    let h = b.add_bias(&h, &m.fc1_b)?;
    let h = b.gelu_tanh(&h)?;
    let out = b.matmul(&h, &m.fc2_w)?;
    b.add_bias(&out, &m.fc2_b)
}

/// Reshape [N, hidden] → [N/merge², hidden*merge²] by concatenating
/// merge² consecutive rows. This is the spatial-merge shuffle. Done on
/// CPU (round-trip through to_cpu/to_device) since it's a one-time copy
/// per image and the Backend trait has no strided-reshape op.
fn reshape_merge(
    b: &dyn Backend,
    x: &Tensor,
    n_patches: usize,
    hidden: usize,
    merge: usize,
) -> Result<Tensor> {
    let merge_sq = merge * merge; // 4
    let out_rows = n_patches / merge_sq;
    let out_cols = hidden * merge_sq; // 4096
    let cpu = b.to_cpu(x)?;
    let out = match cpu.dtype() {
        apxinf_core::DType::F32 => {
            let data = cpu.as_f32()?;
            let mut o = vec![0.0f32; out_rows * out_cols];
            for r in 0..out_rows {
                for s in 0..merge_sq {
                    for c in 0..hidden {
                        o[r * out_cols + s * hidden + c] =
                            data[r * merge_sq * hidden + s * hidden + c];
                    }
                }
            }
            Tensor::from_f32(vec![out_rows, out_cols], &o)?
        }
        apxinf_core::DType::BF16 => {
            let data = cpu.as_bf16()?;
            let mut o = vec![half::bf16::from_f32(0.0); out_rows * out_cols];
            for r in 0..out_rows {
                for s in 0..merge_sq {
                    for c in 0..hidden {
                        o[r * out_cols + s * hidden + c] =
                            data[r * merge_sq * hidden + s * hidden + c];
                    }
                }
            }
            Tensor::from_bf16(vec![out_rows, out_cols], &o)?
        }
        dtype => {
            return Err(Error::Other(format!(
                "Qwen3-VL merge reshape does not support {dtype}"
            )))
        }
    };
    b.to_device(&out)
}

/// Extract a contiguous slice `qkv[:, col_start..col_start+width]` and
/// reshape to `[N, n_heads, head_dim]`. The qkv tensor is `[N, 3*hidden]`
/// in row-major; slicing columns is a strided gather. Done on CPU for now.
fn slice_and_reshape(
    b: &dyn Backend,
    qkv: &Tensor,
    col_start: usize,
    width: usize,
    n_patches: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let cpu = b.to_cpu(qkv)?;
    let total_cols = 3 * width; // 3 * hidden
    let out = match cpu.dtype() {
        apxinf_core::DType::F32 => {
            let data = cpu.as_f32()?;
            let mut o = vec![0.0f32; n_patches * width];
            for n in 0..n_patches {
                for c in 0..width {
                    o[n * width + c] = data[n * total_cols + col_start + c];
                }
            }
            Tensor::from_f32(vec![n_patches, n_heads, head_dim], &o)?
        }
        apxinf_core::DType::BF16 => {
            let data = cpu.as_bf16()?;
            let mut o = vec![half::bf16::from_f32(0.0); n_patches * width];
            for n in 0..n_patches {
                for c in 0..width {
                    o[n * width + c] = data[n * total_cols + col_start + c];
                }
            }
            Tensor::from_bf16(vec![n_patches, n_heads, head_dim], &o)?
        }
        dtype => {
            return Err(Error::Other(format!(
                "Qwen3-VL vision slice does not support {dtype}"
            )))
        }
    };
    b.to_device(&out)
}

/// Compute the bilinear-interpolated, permuted positional embeddings.
/// HF's `fast_pos_embed_interpolate`: take the 48×48 learned pos_embed
/// table, bilinearly interpolate to (H, W), then permute to the
/// spatial-merge layout where 2×2 patches are consecutive.
fn compute_pos_embeds(
    cfg: &Qwen3VLConfig,
    b: &dyn Backend,
    pos_embed_table: &Tensor,
    t: usize,
    h: usize,
    width: usize,
    merge: usize,
    hidden: usize,
) -> Result<Tensor> {
    let vc = &cfg.vision;
    let num_pos = vc.num_position_embeddings; // 2304
    let grid_side = (num_pos as f64).sqrt().round() as usize; // 48

    let cpu = b.to_cpu(pos_embed_table)?;
    let table = cpu
        .to_f32_vec()
        .map_err(|e| apxinf_core::Error::Other(format!("pos_embed table: {e}")))?;
    // table is [2304, 1024] = [48*48, hidden].

    // Bilinear-interpolate to (h, width) → [h*width, hidden].
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
                interp[dst + c] = (1.0 - dh) * (1.0 - dw) * v00
                    + (1.0 - dh) * dw * v01
                    + dh * (1.0 - dw) * v10
                    + dh * dw * v11;
            }
        }
    }

    // Permute to spatial-merge layout: view as
    // [t, h/merge, merge, width/merge, merge, hidden] → permute(0,1,3,2,4,5)
    // → flatten(0,4) → [t * (h/merge) * (width/merge) * merge * merge, hidden].
    let merged_h = h / merge;
    let merged_w = width / merge;
    let n_tokens = t * merged_h * merged_w * merge * merge; // = t * h * width
    let mut permuted = vec![0.0f32; n_tokens * hidden];
    let mut idx = 0usize;
    for ti in 0..t {
        for mh in 0..merged_h {
            for mw in 0..merged_w {
                for ih in 0..merge {
                    for iw in 0..merge {
                        let src_hi = mh * merge + ih;
                        let src_wi = mw * merge + iw;
                        let src = (ti * h * width + src_hi * width + src_wi) * hidden;
                        for c in 0..hidden {
                            permuted[idx * hidden + c] = interp[src + c];
                        }
                        idx += 1;
                    }
                }
            }
        }
    }

    // Cast to bf16 (to match x's dtype) and upload.
    let bf16: Vec<half::bf16> = permuted.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let tensor = Tensor::from_bf16(vec![n_tokens, hidden], &bf16)?;
    b.to_device(&tensor)
}

/// Vision 2D-RoPE position IDs: for each patch in the spatial-merge
/// layout, (h, w) coordinates. Matches HF's `rot_pos_emb`.
fn compute_vision_pos_ids(t: usize, h: usize, width: usize, merge: usize) -> Vec<u32> {
    let merged_h = h / merge;
    let merged_w = width / merge;
    let mut ids = Vec::with_capacity(t * merged_h * merged_w * merge * merge * 2);
    for _ti in 0..t {
        for mh in 0..merged_h {
            for mw in 0..merged_w {
                for ih in 0..merge {
                    for iw in 0..merge {
                        let row = mh * merge + ih;
                        let col = mw * merge + iw;
                        ids.push(row as u32);
                        ids.push(col as u32);
                    }
                }
            }
        }
    }
    ids
}

/// Dump a GPU tensor to a .npy file (f32) for debugging. Downloads to
/// CPU, converts to f32, then writes as f32 .npy.
fn dump_tensor(b: &dyn Backend, t: &Tensor, path: &str) -> Result<()> {
    let cpu = b.to_cpu(t)?;
    let f32_data = cpu.to_f32_vec()?;
    let dims = t.shape().dims().to_vec();
    let shape_str = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}");
    let pad = (64 - ((header.len() + 10) % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(f32_data.iter().flat_map(|&v| v.to_le_bytes()));
    std::fs::write(format!("{path}.npy"), &out)
        .map_err(|e| apxinf_core::Error::Other(format!("dump {path}: {e}")))?;
    eprintln!("dumped {path}.npy shape={:?}", dims);
    Ok(())
}

//! Qwen3-VL vision tower forward path.
//!
//! Takes preprocessed `pixel_values [N, 1536]` and `grid_thw [images, 3]`,
//! produces the primary visual embedding `[N/4, 2048]` and 3 deepstack
//! embeddings `[N/4, 2048]` for injection into the LLM.
//!
//! The portable path uses `dyn Backend` primitives (matmul, layer_norm,
//! gelu_tanh, add_bias, rope_vision_2d, vision_sdpa). CUDA selects existing
//! model-neutral fused operators where their contracts match. No KV cache —
//! vision attention is a single non-causal forward over the full patch
//! sequence.

#[cfg(feature = "cuda")]
use std::sync::LazyLock;

use apxinf_core::{Backend, Error, Result, Tensor};

use super::config::Qwen3VLConfig;
use super::vision_weights::Qwen3VLVisionWeights;

#[cfg(feature = "cuda")]
static USE_FUSED_VISION_QKV_ROPE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FUSED_VISION_QKV_ROPE").map_or(true, |value| value != "0")
});
#[cfg(feature = "cuda")]
static USE_PRECOMPUTED_VISION_ROPE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_PRECOMPUTED_VISION_ROPE").is_ok_and(|value| value == "1")
});
pub struct VisionOutput {
    /// Primary embedding `[N/4, out_hidden]` injected at the image_pad
    /// positions in the LLM input embedding stream.
    pub primary: Tensor,
    /// 3 deepstack embeddings, each `[N/4, out_hidden]`, injected at the
    /// 3 deepstack layer depths of the LLM.
    pub deepstack: Vec<Tensor>,
}

pub(crate) type VisionMatmul<'a> = dyn Fn(&str, &Tensor, &Tensor) -> Result<Tensor> + 'a;
pub(crate) type VisionNormMatmul<'a> =
    dyn Fn(&str, &Tensor, &Tensor, &Tensor, f32, &Tensor) -> Result<Tensor> + 'a;
pub(crate) type VisionBiasGelu<'a> = dyn Fn(&str, &Tensor, &Tensor) -> Result<Tensor> + 'a;

/// Fixed-grid CUDA inputs that must keep stable addresses across graph replay.
/// The portable variant retains only the host position IDs.
pub(crate) struct PreparedVisionPositions {
    embeddings: Tensor,
    position_ids: Vec<u32>,
    device_position_ids: PreparedVisionPositionBuffer,
}

/// Run the vision tower. `pixel_values` is `[N, 1536]` bf16 on device;
/// `grid_thw` is `[[T, H, W]]`. Attention is segmented per temporal image,
/// matching Qwen3-VL's `cu_seqlens`: patches from different camera images or
/// temporal frames never attend to one another inside the vision tower.
pub fn forward(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<VisionOutput> {
    forward_impl(
        cfg,
        w,
        b,
        pixel_values,
        grid_thw,
        None,
        None,
        None,
        None,
        None,
        false,
        None,
    )
}

/// Prepare both interpolated embeddings and uploaded RoPE IDs for a fixed
/// image grid. The returned buffers are reusable by eager runs and retain the
/// stable addresses required by CUDA Graph replay.
pub(crate) fn prepare_positions(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    grid_thw: &[[u32; 3]],
) -> Result<PreparedVisionPositions> {
    let embeddings = prepare_position_embeddings(cfg, w, b, grid_thw)?;
    let position_ids = compute_vision_pos_ids(grid_thw, cfg.vision.spatial_merge_size)?;
    let device_position_ids = prepare_vision_position_ids(b, &position_ids)?;
    Ok(PreparedVisionPositions {
        embeddings,
        position_ids,
        device_position_ids,
    })
}

pub(crate) fn forward_with_prepared_positions_and_matmul(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
    positions: &PreparedVisionPositions,
    matmul: &VisionMatmul<'_>,
    norm_matmul: Option<&VisionNormMatmul<'_>>,
    bias_gelu: Option<&VisionBiasGelu<'_>>,
) -> Result<VisionOutput> {
    forward_impl(
        cfg,
        w,
        b,
        pixel_values,
        grid_thw,
        Some(&positions.embeddings),
        Some((&positions.position_ids, &positions.device_position_ids)),
        Some(matmul),
        norm_matmul,
        bias_gelu,
        true,
        None,
    )
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
    forward_impl(
        cfg,
        w,
        b,
        pixel_values,
        grid_thw,
        None,
        None,
        None,
        None,
        None,
        false,
        Some(dump_prefix),
    )
}

fn forward_impl(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
    prepared_position_embeddings: Option<&Tensor>,
    prepared_position_ids: Option<(&[u32], &PreparedVisionPositionBuffer)>,
    matmul: Option<&VisionMatmul<'_>>,
    norm_matmul: Option<&VisionNormMatmul<'_>>,
    bias_gelu: Option<&VisionBiasGelu<'_>>,
    allow_gr00t_fused_qkv_rope: bool,
    dump: Option<&str>,
) -> Result<VisionOutput> {
    #[cfg(not(feature = "cuda"))]
    let _ = allow_gr00t_fused_qkv_rope;
    let _vision_range = crate::profiling::trace::range("vision_encoder");
    let vc = &cfg.vision;
    let hidden = vc.hidden_size; // 1024
    let n_heads = vc.num_heads; // 16
    let head_dim = vc.head_dim(); // 64
    let merge = vc.spatial_merge_size; // 2
    let eps = 1e-6f32;

    let dims = pixel_values.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "Qwen3-VL pixel_values must be [patches, patch_width], got {dims:?}"
        )));
    }
    let n_patches = dims[0];
    let expected_patch_width = vc
        .in_channels
        .checked_mul(vc.temporal_patch_size)
        .and_then(|value| value.checked_mul(vc.patch_size))
        .and_then(|value| value.checked_mul(vc.patch_size))
        .ok_or_else(|| Error::Other("Qwen3-VL patch width overflow".into()))?;
    if dims[1] != expected_patch_width {
        return Err(Error::Other(format!(
            "Qwen3-VL pixel_values width must be {expected_patch_width}, got {}",
            dims[1]
        )));
    }
    let attention_segments = validate_grid_layout(grid_thw, n_patches, merge)?;

    // ── Patch embedding: pixel_values @ W^T + bias → [N, 1024] ──────
    // patch_embed_weight is [1536, 1024] (already transposed). matmul
    // of [N, 1536] @ [1536, 1024] → [N, 1024].
    let mut x = vision_matmul(
        matmul,
        b,
        "backbone.vision.patch_embed",
        pixel_values,
        &w.patch_embed_weight,
    )?;
    x = b.add_bias(&x, &w.patch_embed_bias)?;
    if let Some(d) = dump {
        dump_tensor(b, &x, &format!("{d}_post_patch_embed"))?;
    }

    // ── Positional embedding (bilinear-interpolated, permuted) ──────
    let computed_position_embeddings;
    let pos_embeds = match prepared_position_embeddings {
        Some(position_embeddings) => position_embeddings,
        None => {
            computed_position_embeddings =
                compute_pos_embeds(cfg, b, &w.pos_embed, grid_thw, merge, hidden)?;
            &computed_position_embeddings
        }
    };
    if pos_embeds.shape().dims() != [n_patches, hidden]
        || pos_embeds.dtype() != x.dtype()
        || pos_embeds.device() != x.device()
    {
        return Err(Error::Other(format!(
            "Qwen3-VL position embeddings must match {} {:?} on {}, got {} {:?} on {}",
            x.dtype(),
            [n_patches, hidden],
            x.device(),
            pos_embeds.dtype(),
            pos_embeds.shape().dims(),
            pos_embeds.device()
        )));
    }
    if let Some(d) = dump {
        dump_tensor(b, &pos_embeds, &format!("{d}_pos_embeds"))?;
    }
    x = b.add(&x, &pos_embeds)?;
    if let Some(d) = dump {
        dump_tensor(b, &x, &format!("{d}_post_add_pos"))?;
    }

    // ── Vision 2D-RoPE position IDs ─────────────────────────────────
    let computed_pos_ids;
    let computed_device_pos_ids;
    let (pos_ids, prepared_pos_ids) = match prepared_position_ids {
        Some(prepared) => prepared,
        None => {
            computed_pos_ids = compute_vision_pos_ids(grid_thw, merge)?;
            computed_device_pos_ids = prepare_vision_position_ids(b, &computed_pos_ids)?;
            (computed_pos_ids.as_slice(), &computed_device_pos_ids)
        }
    };

    #[cfg(feature = "cuda")]
    let precomputed_vision_rope = if allow_gr00t_fused_qkv_rope
        && *USE_FUSED_VISION_QKV_ROPE
        && *USE_PRECOMPUTED_VISION_ROPE
    {
        if let PreparedVisionPositionBuffer::Cuda(positions) = prepared_pos_ids {
            let cuda = b
                .as_any()
                .downcast_ref::<apxinf_cuda::CudaBackend>()
                .ok_or_else(|| Error::Other("Qwen3-VL CUDA position/backend mismatch".into()))?;
            Some(apxinf_cuda::kernels::rope::prepare_vision_rope_cos_sin(
                cuda.context(),
                n_patches,
                head_dim,
                10000.0,
                positions,
            )?)
        } else {
            None
        }
    } else {
        None
    };

    // ── 24 vision blocks ────────────────────────────────────────────
    let mut deepstack_hidden_states: Vec<Tensor> = Vec::with_capacity(3);
    for (i, blk) in w.blocks.iter().enumerate() {
        let _blk_range = crate::profiling::trace::range(&format!("vision_block_{i}"));
        // pre-attn LayerNorm
        let qkv_name = format!("backbone.vision.blocks.{i}.qkv");
        // QKV: [N, 1024] @ [1024, 3072] → [N, 3072]. Quantized runtimes may
        // fuse this single-use LayerNorm directly into their activation type.
        let qkv = if let Some(norm_matmul) = norm_matmul {
            norm_matmul(&qkv_name, &x, &blk.norm1_w, &blk.norm1_b, eps, &blk.qkv_w)?
        } else {
            let normed = b.layer_norm(&x, &blk.norm1_w, &blk.norm1_b, eps)?;
            vision_matmul(matmul, b, &qkv_name, &normed, &blk.qkv_w)?
        };
        // Split Q/K/V: qkv is [N, 3072] = [N, 3, 1024] interleaved as
        // (q[N,0:1024], k[N,1024:2048], v[N,2048:3072]). Reshape to
        // [N, 3, n_heads, head_dim] then split.
        // HF does: qkv.reshape(seq, 3, n_heads, head_dim).permute(1,0,2,3).unbind(0)
        // So q = qkv[:, 0, :, :], k = qkv[:, 1, :, :], v = qkv[:, 2, :, :].
        // In our flat [N, 3072] layout with head-major order:
        //   qkv[n, :] = [q[n,h,d] for h,d] ++ [k[n,h,d] for h,d] ++ [v[n,h,d] for h,d]
        // We need to extract Q = qkv[:, 0:1024], K = qkv[:, 1024:2048],
        // V = qkv[:, 2048:3072] as [N, n_heads, head_dim] tensors.
        let fused_qkv_rope: Option<(Tensor, Tensor, Tensor)> = {
            #[cfg(feature = "cuda")]
            {
                if allow_gr00t_fused_qkv_rope && *USE_FUSED_VISION_QKV_ROPE {
                    if let PreparedVisionPositionBuffer::Cuda(positions) = prepared_pos_ids {
                        let cuda = b
                            .as_any()
                            .downcast_ref::<apxinf_cuda::CudaBackend>()
                            .ok_or_else(|| {
                                Error::Other("Qwen3-VL CUDA position/backend mismatch".into())
                            })?;
                        let split = if let Some(table) = precomputed_vision_rope.as_ref() {
                            apxinf_cuda::kernels::rope::split_qkv_bias_apply_vision_2d_precomputed(
                                cuda.context(),
                                &qkv,
                                &blk.qkv_b,
                                n_heads,
                                head_dim,
                                positions,
                                table,
                            )?
                        } else {
                            apxinf_cuda::kernels::rope::split_qkv_bias_apply_vision_2d(
                                cuda.context(),
                                &qkv,
                                &blk.qkv_b,
                                n_heads,
                                head_dim,
                                10000.0,
                                positions,
                            )?
                        };
                        Some((split.q, split.k, split.v))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        };
        let (q, k, v) = if let Some(qkv) = fused_qkv_rope {
            qkv
        } else {
            let (q, k, v) = split_vision_qkv(b, &qkv, &blk.qkv_b, n_patches, n_heads, head_dim)?;
            let (q, k) = apply_vision_rope_pair(
                b,
                &q,
                &k,
                n_heads,
                head_dim,
                10000.0,
                &pos_ids,
                &prepared_pos_ids,
            )?;
            (q, k, v)
        };

        // Non-causal attention within each image/frame segment. This is
        // equivalent to Qwen3-VL's cu_seqlens-based block-diagonal mask.
        let attn_out =
            segmented_vision_sdpa(b, &q, &k, &v, &attention_segments, n_heads, head_dim)?;
        // Output projection + residual + pre-MLP LayerNorm
        let attn_out = vision_matmul(
            matmul,
            b,
            &format!("backbone.vision.blocks.{i}.output"),
            &attn_out,
            &blk.proj_w,
        )?;
        let (hidden, normed) = apply_bias_residual_layer_norm(
            b,
            &attn_out,
            &blk.proj_b,
            &x,
            &blk.norm2_w,
            &blk.norm2_b,
            eps,
        )?;
        x = hidden;
        // FC1 + GELU
        let h1 = vision_matmul(
            matmul,
            b,
            &format!("backbone.vision.blocks.{i}.fc1"),
            &normed,
            &blk.fc1_w,
        )?;
        let fc1_name = format!("backbone.vision.blocks.{i}.fc1");
        let h1 = vision_bias_gelu(bias_gelu, b, &fc1_name, &h1, &blk.fc1_b)?;
        // FC2 + residual
        let h2 = vision_matmul(
            matmul,
            b,
            &format!("backbone.vision.blocks.{i}.fc2"),
            &h1,
            &blk.fc2_w,
        )?;
        x = apply_bias_residual(b, &h2, &blk.fc2_b, &x)?;

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
    let primary = merge_primary(
        b, &w.merger, &x, n_patches, hidden, merge, eps, matmul, bias_gelu,
    )?;

    // ── Deepstack mergers ───────────────────────────────────────────
    // use_postshuffle_norm=True: reshape [N, 1024] → [N/4, 4096] first,
    // then LayerNorm(4096), then fc1 → GELU → fc2.
    let mut deepstack = Vec::with_capacity(3);
    for (index, (merger, hs)) in w
        .deepstack_mergers
        .iter()
        .zip(deepstack_hidden_states.iter())
        .enumerate()
    {
        let out = merge_deepstack(
            b, merger, hs, n_patches, hidden, merge, eps, index, matmul, bias_gelu,
        )?;
        deepstack.push(out);
    }

    Ok(VisionOutput { primary, deepstack })
}

fn vision_matmul(
    hook: Option<&VisionMatmul<'_>>,
    backend: &dyn Backend,
    name: &str,
    input: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    match hook {
        Some(hook) => hook(name, input, weight),
        None => backend.matmul(input, weight),
    }
}

pub(crate) fn prepare_position_embeddings(
    cfg: &Qwen3VLConfig,
    w: &Qwen3VLVisionWeights,
    b: &dyn Backend,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    compute_pos_embeds(
        cfg,
        b,
        &w.pos_embed,
        grid_thw,
        cfg.vision.spatial_merge_size,
        cfg.vision.hidden_size,
    )
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
    matmul: Option<&VisionMatmul<'_>>,
    bias_gelu: Option<&VisionBiasGelu<'_>>,
) -> Result<Tensor> {
    let normed = b.layer_norm(x, &m.norm_w, &m.norm_b, eps)?;
    // Reshape [N, 1024] → [N/4, 4096]: concatenate 4 consecutive rows.
    let merged = reshape_merge(&normed, n_patches, hidden, merge)?;
    let h = vision_matmul(matmul, b, "backbone.vision.merger.fc1", &merged, &m.fc1_w)?;
    let h = vision_bias_gelu(bias_gelu, b, "backbone.vision.merger.fc1", &h, &m.fc1_b)?;
    let out = vision_matmul(matmul, b, "backbone.vision.merger.fc2", &h, &m.fc2_w)?;
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
    index: usize,
    matmul: Option<&VisionMatmul<'_>>,
    bias_gelu: Option<&VisionBiasGelu<'_>>,
) -> Result<Tensor> {
    let merged = reshape_merge(x, n_patches, hidden, merge)?;
    let normed = b.layer_norm(&merged, &m.norm_w, &m.norm_b, eps)?;
    let h = vision_matmul(
        matmul,
        b,
        &format!("backbone.vision.deepstack.{index}.fc1"),
        &normed,
        &m.fc1_w,
    )?;
    let h = vision_bias_gelu(
        bias_gelu,
        b,
        &format!("backbone.vision.deepstack.{index}.fc1"),
        &h,
        &m.fc1_b,
    )?;
    let out = vision_matmul(
        matmul,
        b,
        &format!("backbone.vision.deepstack.{index}.fc2"),
        &h,
        &m.fc2_w,
    )?;
    b.add_bias(&out, &m.fc2_b)
}

fn apply_bias_gelu(b: &dyn Backend, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(cuda) = b.as_any().downcast_ref::<apxinf_cuda::CudaBackend>() {
        return apxinf_cuda::kernels::activation::bias_gelu_bf16(cuda.context(), input, Some(bias));
    }

    let biased = b.add_bias(input, bias)?;
    b.gelu_tanh(&biased)
}

fn vision_bias_gelu(
    callback: Option<&VisionBiasGelu<'_>>,
    b: &dyn Backend,
    name: &str,
    input: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    match callback {
        Some(callback) => callback(name, input, bias),
        None => apply_bias_gelu(b, input, bias),
    }
}

fn apply_bias_residual(
    b: &dyn Backend,
    projection: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(cuda) = b.as_any().downcast_ref::<apxinf_cuda::CudaBackend>() {
        return apxinf_cuda::kernels::fused::bias_residual_bf16(
            cuda.context(),
            projection,
            Some(bias),
            residual,
        );
    }

    let projection = b.add_bias(projection, bias)?;
    b.add(residual, &projection)
}

#[allow(clippy::too_many_arguments)]
fn apply_bias_residual_layer_norm(
    b: &dyn Backend,
    projection: &Tensor,
    projection_bias: &Tensor,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if let Some(cuda) = b.as_any().downcast_ref::<apxinf_cuda::CudaBackend>() {
        let fused = apxinf_cuda::kernels::fused::bias_residual_layer_bf16(
            cuda.context(),
            projection,
            Some(projection_bias),
            residual,
            norm_weight,
            norm_bias,
            eps,
        )?;
        return Ok((fused.hidden, fused.normalized));
    }

    let projection = b.add_bias(projection, projection_bias)?;
    let hidden = b.add(residual, &projection)?;
    let normalized = b.layer_norm(&hidden, norm_weight, norm_bias, eps)?;
    Ok((hidden, normalized))
}

/// Reshape `[N, hidden]` into `[N/merge², hidden*merge²]` by grouping
/// `merge²` consecutive rows. Qwen has already permuted the patch sequence
/// into spatial-merge order, so this operation only changes tensor metadata;
/// no element shuffle or device transfer is required.
fn reshape_merge(x: &Tensor, n_patches: usize, hidden: usize, merge: usize) -> Result<Tensor> {
    let merge_sq = merge
        .checked_mul(merge)
        .ok_or_else(|| Error::Other("Qwen3-VL merge size overflow".into()))?;
    if merge_sq == 0 || n_patches % merge_sq != 0 || x.shape().dims() != [n_patches, hidden] {
        return Err(Error::Other(format!(
            "Qwen3-VL merge reshape expected [{n_patches}, {hidden}] with patch count divisible by {merge_sq}, got {:?}",
            x.shape().dims()
        )));
    }
    let out_rows = n_patches / merge_sq;
    let out_cols = hidden
        .checked_mul(merge_sq)
        .ok_or_else(|| Error::Other("Qwen3-VL merged width overflow".into()))?;
    x.reshape(vec![out_rows, out_cols])
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

fn split_vision_qkv(
    b: &dyn Backend,
    qkv: &Tensor,
    bias: &Tensor,
    n_patches: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if let Some(cuda) = b.as_any().downcast_ref::<apxinf_cuda::CudaBackend>() {
        let split = apxinf_cuda::kernels::attention::split_qkv_bias_bf16(
            cuda.context(),
            qkv,
            Some(bias),
            n_heads,
            head_dim,
        )?;
        return Ok((split.q, split.k, split.v));
    }

    let width = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("Qwen3-VL QKV width overflow".into()))?;
    let qkv = b.add_bias(qkv, bias)?;
    let q = slice_and_reshape(b, &qkv, 0, width, n_patches, n_heads, head_dim)?;
    let k = slice_and_reshape(b, &qkv, width, width, n_patches, n_heads, head_dim)?;
    let v = slice_and_reshape(b, &qkv, 2 * width, width, n_patches, n_heads, head_dim)?;
    Ok((q, k, v))
}

enum PreparedVisionPositionBuffer {
    Portable,
    #[cfg(feature = "cuda")]
    Cuda(apxinf_cuda::CudaBuffer),
}

fn prepare_vision_position_ids(
    b: &dyn Backend,
    position_ids: &[u32],
) -> Result<PreparedVisionPositionBuffer> {
    #[cfg(not(feature = "cuda"))]
    let _ = (b, position_ids);
    #[cfg(feature = "cuda")]
    if let Some(cuda) = b.as_any().downcast_ref::<apxinf_cuda::CudaBackend>() {
        let bytes = position_ids
            .iter()
            .flat_map(|position| position.to_ne_bytes())
            .collect::<Vec<_>>();
        let positions =
            apxinf_cuda::CudaBuffer::alloc(bytes.len(), cuda.device_id()).map_err(Error::Cuda)?;
        positions.copy_from_host(&bytes).map_err(Error::Cuda)?;
        return Ok(PreparedVisionPositionBuffer::Cuda(positions));
    }
    Ok(PreparedVisionPositionBuffer::Portable)
}

#[allow(clippy::too_many_arguments)]
fn apply_vision_rope_pair(
    b: &dyn Backend,
    q: &Tensor,
    k: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    position_ids: &[u32],
    prepared: &PreparedVisionPositionBuffer,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "cuda")]
    if let PreparedVisionPositionBuffer::Cuda(positions) = prepared {
        let cuda = b
            .as_any()
            .downcast_ref::<apxinf_cuda::CudaBackend>()
            .ok_or_else(|| Error::Other("Qwen3-VL CUDA position/backend mismatch".into()))?;
        return apxinf_cuda::kernels::rope::apply_vision_2d_pair(
            cuda.context(),
            q,
            k,
            n_heads,
            head_dim,
            theta,
            positions,
        );
    }

    let _ = prepared;
    Ok((
        b.rope_vision_2d(q, n_heads, head_dim, theta, position_ids)?,
        b.rope_vision_2d(k, n_heads, head_dim, theta, position_ids)?,
    ))
}

/// Execute Qwen vision attention independently for every temporal image.
/// Qwen's CUDA implementation expresses the same partition through
/// `cu_seqlens`. CUDA uses zero-copy row views plus device concatenation;
/// other backends retain the materialized correctness path. Both keep the
/// portable `Backend` trait unchanged.
fn segmented_vision_sdpa(
    b: &dyn Backend,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    segment_lengths: &[usize],
    n_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if segment_lengths.is_empty() {
        return Err(Error::Other(
            "Qwen3-VL attention requires at least one segment".into(),
        ));
    }
    if segment_lengths.len() == 1 {
        return b.vision_sdpa(q, k, v, segment_lengths[0], n_heads, head_dim);
    }
    let row_width = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("Qwen3-VL attention row width overflow".into()))?;
    let total_rows = segment_lengths.iter().try_fold(0usize, |total, &length| {
        total
            .checked_add(length)
            .ok_or_else(|| Error::Other("Qwen3-VL attention segment length overflow".into()))
    })?;
    let total_values = total_rows
        .checked_mul(row_width)
        .ok_or_else(|| Error::Other("Qwen3-VL attention tensor size overflow".into()))?;
    let expected_shape = [total_rows, n_heads, head_dim];
    if q.dtype() != apxinf_core::DType::BF16
        || k.dtype() != apxinf_core::DType::BF16
        || v.dtype() != apxinf_core::DType::BF16
        || q.shape().dims() != expected_shape
        || k.shape().dims() != expected_shape
        || v.shape().dims() != expected_shape
    {
        return Err(Error::Other(format!(
            "Qwen3-VL segmented BF16 attention expected {expected_shape:?}, got {} {:?}, {} {:?}, {} {:?}",
            q.dtype(),
            q.shape().dims(),
            k.dtype(),
            k.shape().dims(),
            v.dtype(),
            v.shape().dims()
        )));
    }

    #[cfg(feature = "cuda")]
    if let Some(cuda) = b.as_any().downcast_ref::<apxinf_cuda::CudaBackend>() {
        let q = q.reshape(vec![total_rows, row_width])?;
        let k = k.reshape(vec![total_rows, row_width])?;
        let v = v.reshape(vec![total_rows, row_width])?;
        let mut segments = Vec::with_capacity(segment_lengths.len());
        let mut first_row = 0usize;
        for &row_count in segment_lengths {
            let shape = vec![row_count, n_heads, head_dim];
            let q_segment = apxinf_cuda::kernels::elementwise::contiguous_rows(
                cuda.context(),
                &q,
                first_row,
                row_count,
            )?
            .reshape(shape.clone())?;
            let k_segment = apxinf_cuda::kernels::elementwise::contiguous_rows(
                cuda.context(),
                &k,
                first_row,
                row_count,
            )?
            .reshape(shape.clone())?;
            let v_segment = apxinf_cuda::kernels::elementwise::contiguous_rows(
                cuda.context(),
                &v,
                first_row,
                row_count,
            )?
            .reshape(shape)?;
            segments.push(b.vision_sdpa(
                &q_segment, &k_segment, &v_segment, row_count, n_heads, head_dim,
            )?);
            first_row = first_row
                .checked_add(row_count)
                .ok_or_else(|| Error::Other("Qwen3-VL attention row range overflow".into()))?;
        }
        let mut output = segments.remove(0);
        for segment in segments {
            output = apxinf_cuda::kernels::elementwise::concat_rows_bf16(
                cuda.context(),
                &output,
                &segment,
            )?;
        }
        return Ok(output);
    }

    let q = b.to_cpu(q)?;
    let k = b.to_cpu(k)?;
    let v = b.to_cpu(v)?;
    let q_values = q.as_bf16()?;
    let k_values = k.as_bf16()?;
    let v_values = v.as_bf16()?;
    debug_assert_eq!(q_values.len(), total_values);
    debug_assert_eq!(k_values.len(), total_values);
    debug_assert_eq!(v_values.len(), total_values);

    let mut output = Vec::with_capacity(total_values);
    let mut first_row = 0usize;
    for &row_count in segment_lengths {
        let first = first_row
            .checked_mul(row_width)
            .ok_or_else(|| Error::Other("Qwen3-VL attention slice overflow".into()))?;
        let last_row = first_row
            .checked_add(row_count)
            .ok_or_else(|| Error::Other("Qwen3-VL attention row range overflow".into()))?;
        let last = last_row
            .checked_mul(row_width)
            .ok_or_else(|| Error::Other("Qwen3-VL attention slice overflow".into()))?;
        let shape = vec![row_count, n_heads, head_dim];
        let q_segment = b.to_device(&Tensor::from_bf16(shape.clone(), &q_values[first..last])?)?;
        let k_segment = b.to_device(&Tensor::from_bf16(shape.clone(), &k_values[first..last])?)?;
        let v_segment = b.to_device(&Tensor::from_bf16(shape, &v_values[first..last])?)?;
        let segment = b.vision_sdpa(
            &q_segment, &k_segment, &v_segment, row_count, n_heads, head_dim,
        )?;
        let segment = b.to_cpu(&segment)?;
        output.extend_from_slice(segment.as_bf16()?);
        first_row = last_row;
    }
    b.to_device(&Tensor::from_bf16(vec![total_rows, row_width], &output)?)
}

/// Validate grids and return the cu_seqlens-equivalent segment sizes. Qwen3-VL
/// creates one independent attention segment for each temporal frame.
fn validate_grid_layout(
    grid_thw: &[[u32; 3]],
    n_patches: usize,
    merge: usize,
) -> Result<Vec<usize>> {
    if grid_thw.is_empty() || merge == 0 {
        return Err(Error::Other(
            "Qwen3-VL requires image grids and a non-zero merge size".into(),
        ));
    }
    let mut segments = Vec::new();
    let mut total = 0usize;
    for (index, &[temporal, height, width]) in grid_thw.iter().enumerate() {
        let temporal = usize::try_from(temporal)
            .map_err(|_| Error::Other(format!("Qwen3-VL grid {index} temporal overflow")))?;
        let height = usize::try_from(height)
            .map_err(|_| Error::Other(format!("Qwen3-VL grid {index} height overflow")))?;
        let width = usize::try_from(width)
            .map_err(|_| Error::Other(format!("Qwen3-VL grid {index} width overflow")))?;
        if temporal == 0 || height == 0 || width == 0 || height % merge != 0 || width % merge != 0 {
            return Err(Error::Other(format!(
                "Qwen3-VL grid {index} [{temporal}, {height}, {width}] is incompatible with merge {merge}"
            )));
        }
        let frame_patches = height
            .checked_mul(width)
            .ok_or_else(|| Error::Other("Qwen3-VL frame patch count overflow".into()))?;
        total = total
            .checked_add(
                temporal
                    .checked_mul(frame_patches)
                    .ok_or_else(|| Error::Other("Qwen3-VL grid patch count overflow".into()))?,
            )
            .ok_or_else(|| Error::Other("Qwen3-VL total patch count overflow".into()))?;
        segments.extend(std::iter::repeat(frame_patches).take(temporal));
    }
    if total != n_patches {
        return Err(Error::Other(format!(
            "Qwen3-VL grids describe {total} patches but pixel_values has {n_patches} rows"
        )));
    }
    Ok(segments)
}

/// Compute the bilinear-interpolated, permuted positional embeddings.
/// HF's `fast_pos_embed_interpolate`: take the 48×48 learned pos_embed
/// table, bilinearly interpolate to (H, W), then permute to the
/// spatial-merge layout where 2×2 patches are consecutive.
fn compute_pos_embeds(
    cfg: &Qwen3VLConfig,
    b: &dyn Backend,
    pos_embed_table: &Tensor,
    grid_thw: &[[u32; 3]],
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

    let mut permuted = Vec::new();
    for &[temporal, height, grid_width] in grid_thw {
        let t = temporal as usize;
        let h = height as usize;
        let width = grid_width as usize;

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

        // Permute each image independently to spatial-merge layout.
        let merged_h = h / merge;
        let merged_w = width / merge;
        for _ti in 0..t {
            for mh in 0..merged_h {
                for mw in 0..merged_w {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            let src_hi = mh * merge + ih;
                            let src_wi = mw * merge + iw;
                            // The learned spatial table has no temporal axis;
                            // Qwen repeats the same interpolated HxW table for
                            // every frame before applying the merge permutation.
                            let src = (src_hi * width + src_wi) * hidden;
                            permuted.extend_from_slice(&interp[src..src + hidden]);
                        }
                    }
                }
            }
        }
    }

    // Cast to bf16 (to match x's dtype) and upload.
    let bf16: Vec<half::bf16> = permuted.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let tensor = Tensor::from_bf16(vec![permuted.len() / hidden, hidden], &bf16)?;
    b.to_device(&tensor)
}

/// Vision 2D-RoPE position IDs: for each patch in the spatial-merge
/// layout, (h, w) coordinates. Matches HF's `rot_pos_emb`.
fn compute_vision_pos_ids(grid_thw: &[[u32; 3]], merge: usize) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for &[temporal, height, grid_width] in grid_thw {
        let t = temporal as usize;
        let h = height as usize;
        let width = grid_width as usize;
        let merged_h = h / merge;
        let merged_w = width / merge;
        for _ti in 0..t {
            for mh in 0..merged_h {
                for mw in 0..merged_w {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            let row = u32::try_from(mh * merge + ih).map_err(|_| {
                                Error::Other("Qwen3-VL rotary row index overflow".into())
                            })?;
                            let column = u32::try_from(mw * merge + iw).map_err(|_| {
                                Error::Other("Qwen3-VL rotary column index overflow".into())
                            })?;
                            ids.push(row);
                            ids.push(column);
                        }
                    }
                }
            }
        }
    }
    Ok(ids)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_layout_segments_multiple_views_and_temporal_frames() {
        let segments = validate_grid_layout(&[[2, 2, 2], [1, 2, 4]], 16, 2).unwrap();
        assert_eq!(segments, vec![4, 4, 8]);
        assert!(validate_grid_layout(&[[1, 3, 2]], 6, 2).is_err());
        assert!(validate_grid_layout(&[[1, 2, 2]], 5, 2).is_err());
    }

    #[test]
    fn rotary_positions_restart_for_each_view_and_repeat_per_frame() {
        let ids = compute_vision_pos_ids(&[[2, 2, 2], [1, 2, 4]], 2).unwrap();
        let first_frame = vec![0, 0, 0, 1, 1, 0, 1, 1];
        assert_eq!(&ids[..8], first_frame.as_slice());
        assert_eq!(&ids[8..16], first_frame.as_slice());
        assert_eq!(&ids[16..24], first_frame.as_slice());
        assert_eq!(&ids[24..32], &[0, 2, 0, 3, 1, 2, 1, 3]);
    }

    #[test]
    fn spatial_merge_is_a_metadata_only_contiguous_reshape() {
        let values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let input = Tensor::from_f32(vec![8, 3], &values).unwrap();
        let output = reshape_merge(&input, 8, 3, 2).unwrap();

        assert_eq!(output.shape().dims(), &[2, 12]);
        assert_eq!(output.as_f32().unwrap(), values.as_slice());
        assert!(reshape_merge(&input, 7, 3, 2).is_err());
    }
}

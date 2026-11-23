//! Native BF16 Qwen3.5 vision tower and image-embedding injection.
//!
//! Scope is one image per request. Processor output enters as canonical
//! `[patches, 1536]` F32/BF16 plus `grid_thw`, whose `t` must be 1. Several
//! images, and a video's `t > 1`, both need Attention segmented per frame and
//! are rejected rather than fused into one span; see `prepare`.
//!
//! All per-request tower computation remains on CUDA. One fused kernel
//! interpolates the learned 48x48 position table, applies spatial-merge order,
//! and emits the matching axial IDs on every request. Vision and language
//! computation share the same cuda-new context and stream.

use std::path::Path;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};

use super::config::*;
use super::vision_weights::VisionWeights;

struct PreparedVision {
    height: usize,
    width: usize,
    total_patches: usize,
}

fn upload_u32(ctx: &CudaContext, values: &[u32]) -> Result<CudaBuffer> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

fn upload_i32(ctx: &CudaContext, dims: Vec<usize>, values: &[i32]) -> Result<Tensor> {
    transfers::to_cuda(&Tensor::from_i32(dims, values)?, ctx.device_id())
}

fn device_tensor(ctx: &CudaContext, dims: &[usize], dtype: DType) -> Result<Tensor> {
    let bytes = dims
        .iter()
        .product::<usize>()
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen38 vision tensor size overflow".into()))?;
    CudaBuffer::alloc(bytes.max(1), ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(dims.to_vec()), dtype)
        .map_err(Error::Cuda)
}

fn dump_vision_stage(stage: &str, tensor: &Tensor) -> Result<()> {
    let Ok(directory) = std::env::var("APXINF_QWEN38_VISION_DUMP_DIR") else {
        return Ok(());
    };
    let directory = Path::new(&directory);
    std::fs::create_dir_all(directory).map_err(|error| {
        Error::Other(format!(
            "create qwen38 vision dump directory {}: {error}",
            directory.display()
        ))
    })?;
    let values = transfers::to_cpu(tensor)?.to_f32_vec()?;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let data_path = directory.join(format!("{stage}.f32"));
    std::fs::write(&data_path, bytes).map_err(|error| {
        Error::Other(format!(
            "write qwen38 vision stage {}: {error}",
            data_path.display()
        ))
    })?;
    let shape = tensor
        .shape()
        .dims()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let shape_path = directory.join(format!("{stage}.shape"));
    std::fs::write(&shape_path, format!("{shape}\n")).map_err(|error| {
        Error::Other(format!(
            "write qwen38 vision stage shape {}: {error}",
            shape_path.display()
        ))
    })?;
    Ok(())
}

pub(crate) fn upload_pixels(ctx: &CudaContext, pixels: &Tensor) -> Result<Tensor> {
    let on_device = if pixels.device() != Device::Cuda(ctx.device_id()) {
        transfers::to_cuda(pixels, ctx.device_id())?
    } else {
        pixels.clone()
    };
    match on_device.dtype() {
        DType::BF16 => Ok(on_device),
        DType::F32 => {
            let dims = on_device.shape().dims().to_vec();
            let mut out = device_tensor(ctx, &dims, DType::BF16)?;
            ops::quantization(
                ctx,
                ops::QuantizationArgs::new(
                    ops::QuantizationSemantic::CastF32ToBf16,
                    &on_device,
                    &mut out,
                ),
            )?;
            Ok(out)
        }
        dtype => Err(Error::Other(format!(
            "qwen38 image pixel_values must be f32 or bf16, got {dtype}"
        ))),
    }
}

/// Run the complete vision tower over one image's patch stream. Attention is
/// dense over all patches, which is what a single unsegmented image is.
///
/// `scratch` supplies stable per-shape buffers and `session` owns the prepared
/// GEMM/Attention executions. The first call for a given scratch must pass
/// `prepared = false` so the traversal may tune and create native resources;
/// later calls on the same scratch replay them without allocating. Sessions do
/// not nest, so this must not be called inside the language prefill session.
pub(crate) fn forward(
    weights: &VisionWeights,
    ctx: &CudaContext,
    scratch: &VisionScratch,
    session: &ops::ExecutionSession,
    prepared: bool,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    if weights.pos_embed.shape().dims() != [VISION_POSITIONS, VISION_HIDDEN] {
        return Err(Error::Other(
            "qwen38 image position table shape mismatch".into(),
        ));
    }
    let plan = prepare(pixel_values, grid_thw)?;
    let run = || {
        let mut pos_embeds = scratch.pos_embeds.clone();
        let mut pos_ids = scratch.pos_ids.clone();
        ops::vision_position(
            ctx,
            ops::VisionPositionArgs {
                table: &weights.pos_embed,
                embeddings: &mut pos_embeds,
                position_ids: &mut pos_ids,
                height: plan.height,
                width: plan.width,
                merge_size: VISION_MERGE,
            },
        )?;
        forward_device(weights, ctx, scratch, pixel_values, &plan)
    };
    let output = if prepared {
        ops::with_session(session, run)?
    } else {
        ops::prepare_with_session(session, run)?
    };
    ctx.synchronize().map_err(Error::Cuda)?;
    if let Ok(path) = std::env::var("APXINF_QWEN38_VISION_DUMP") {
        let values = transfers::to_cpu(&output)?.to_f32_vec()?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes)
            .map_err(|error| Error::Other(format!("write qwen38 vision dump {path}: {error}")))?;
        eprintln!(
            "qwen38 vision dump: {} rows x {} columns -> {path}",
            output.shape().dims()[0],
            output.shape().dims()[1]
        );
    }
    Ok(output)
}

/// The single-image contract, checked before any device work.
///
/// The tower runs Attention through the dense semantic, which is one
/// self-attention span. Several images, or a video's `t > 1`, need Attention
/// segmented per frame -- Transformers builds `cu_seqlens` with
/// `repeat_interleave(h * w, t)` -- so fusing them into one span would let
/// patches attend across an image or frame boundary. Reject instead.
fn single_image_grid(grid_thw: &[[u32; 3]], total_patches: usize) -> Result<(usize, usize)> {
    let &[t, h, w] = match grid_thw {
        [grid] => grid,
        grids => {
            return Err(Error::Other(format!(
                "qwen38 image input takes one image per request, got {} grids",
                grids.len()
            )))
        }
    };
    if t != 1 {
        return Err(Error::Other(format!(
            "qwen38 image input requires grid t = 1, got {t}; video needs per-frame \
             attention segmentation and three-axis positions this path does not build"
        )));
    }
    let (h, w) = (h as usize, w as usize);
    if h == 0 || w == 0 || h % VISION_MERGE != 0 || w % VISION_MERGE != 0 {
        return Err(Error::Other(format!(
            "qwen38 image grid [1, {h}, {w}] is invalid for merge {VISION_MERGE}"
        )));
    }
    let patches = h
        .checked_mul(w)
        .ok_or_else(|| Error::Other("qwen38 image grid is too large".into()))?;
    if patches != total_patches {
        return Err(Error::Other(format!(
            "qwen38 image grid covers {patches} patches but pixel_values has {total_patches}"
        )));
    }
    Ok((h, w))
}

fn prepare(pixel_values: &Tensor, grid_thw: &[[u32; 3]]) -> Result<PreparedVision> {
    let patch_width =
        VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
    let dims = pixel_values.shape().dims();
    if grid_thw.is_empty() || dims.len() != 2 || dims[1] != patch_width {
        return Err(Error::Other(format!(
            "qwen38 image pixel_values must be [patches, {patch_width}], got {dims:?}"
        )));
    }

    let total_patches = dims[0];
    let (height, width) = single_image_grid(grid_thw, total_patches)?;
    Ok(PreparedVision {
        height,
        width,
        total_patches,
    })
}

/// Device scratch for one patch count, allocated once and reused by every
/// block.
///
/// Reuse needs no in-place operator support: each block writes only buffers it
/// does not read. `x` alternates between two buffers because a residual add
/// cannot write its own input, and `normed` is rewritten only after the GEMM
/// that reads it has been enqueued. Addresses are stable for the lifetime of
/// the scratch, which is what lets the session's GEMM/Attention execution
/// cache hit on a repeated grid -- `ExecutionKey` includes binding pointers.
pub(crate) struct VisionScratch {
    patches: usize,
    pos_embeds: Tensor,
    pos_ids: Tensor,
    x: [Tensor; 2],
    normed: Tensor,
    qkv: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn: Tensor,
    projected: Tensor,
    mlp_hidden: Tensor,
    mlp_out: Tensor,
    merger_hidden: Tensor,
    merger_activated: Tensor,
    merger_out: Tensor,
}

impl VisionScratch {
    pub(crate) fn patches(&self) -> usize {
        self.patches
    }

    pub(crate) fn new(ctx: &CudaContext, patches: usize) -> Result<Self> {
        let factor = VISION_MERGE * VISION_MERGE;
        if patches == 0 || patches % factor != 0 {
            return Err(Error::Other(format!(
                "qwen38 vision scratch needs a non-zero patch count divisible by {factor}, got {patches}"
            )));
        }
        let merged = patches / factor;
        let rows = [patches, VISION_HIDDEN];
        let heads = [patches, VISION_HEADS, VISION_HEAD_DIM];
        let bf16 = DType::BF16;
        Ok(Self {
            patches,
            pos_embeds: device_tensor(ctx, &rows, bf16)?,
            pos_ids: device_tensor(ctx, &[patches, 2], DType::I32)?,
            x: [
                device_tensor(ctx, &rows, bf16)?,
                device_tensor(ctx, &rows, bf16)?,
            ],
            normed: device_tensor(ctx, &rows, bf16)?,
            qkv: device_tensor(ctx, &[patches, 3 * VISION_HIDDEN], bf16)?,
            query: device_tensor(ctx, &heads, bf16)?,
            key: device_tensor(ctx, &heads, bf16)?,
            value: device_tensor(ctx, &heads, bf16)?,
            // Dense Attention wants a leading batch axis; the rows are identical.
            attn: device_tensor(ctx, &[1, patches, VISION_HEADS, VISION_HEAD_DIM], bf16)?,
            projected: device_tensor(ctx, &rows, bf16)?,
            mlp_hidden: device_tensor(ctx, &[patches, VISION_INTERMEDIATE], bf16)?,
            mlp_out: device_tensor(ctx, &rows, bf16)?,
            merger_hidden: device_tensor(ctx, &[merged, VISION_HIDDEN * factor], bf16)?,
            merger_activated: device_tensor(ctx, &[merged, VISION_HIDDEN * factor], bf16)?,
            merger_out: device_tensor(ctx, &[merged, VISION_OUT_HIDDEN], bf16)?,
        })
    }
}

fn gemm_bias_into(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    if input_dims.len() != 2 || weight_dims.len() != 2 || input_dims[1] != weight_dims[0] {
        return Err(Error::Other("qwen38 vision GEMM shape mismatch".into()));
    }
    ops::gemm_bias(
        ctx,
        ops::GemmBiasArgs {
            gemm: ops::GemmArgs::new(input, weight, out),
            bias,
        },
    )
}

fn gemm_bias_gelu_into(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    if input_dims.len() != 2 || weight_dims.len() != 2 || input_dims[1] != weight_dims[0] {
        return Err(Error::Other(
            "qwen38 vision GELU GEMM shape mismatch".into(),
        ));
    }
    ops::gemm_bias_gelu(
        ctx,
        ops::GemmBiasGeluArgs {
            gemm: ops::GemmArgs::new(input, weight, out),
            bias,
        },
    )
}

fn layer_norm_into(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
    eps: f32,
) -> Result<()> {
    ops::norm::norm(ctx, ops::NormArgs::layer(input, weight, bias, out, eps))
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_layer_norm_into(
    ctx: &CudaContext,
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    norm_bias: &Tensor,
    hidden: &mut Tensor,
    normalized: &mut Tensor,
    eps: f32,
) -> Result<()> {
    ops::fused_add_norm(
        ctx,
        ops::FusedAddNormArgs {
            input,
            bias: None,
            residual,
            gate: None,
            hidden,
            normalized: Some(normalized),
            kind: ops::NormKind::Layer,
            affine: ops::Affine::Weight {
                weight,
                bias: Some(norm_bias),
                offset: false,
            },
            quantized: None,
            numerics: ops::NormNumerics::Flexible,
            bias_order: ops::BiasOrder::SingleRound,
            eps,
        },
    )
}

fn forward_device(
    weights: &VisionWeights,
    ctx: &CudaContext,
    scratch: &VisionScratch,
    pixel_values: &Tensor,
    prepared: &PreparedVision,
) -> Result<Tensor> {
    let eps = 1e-6f32;
    let patches = prepared.total_patches;
    let factor = VISION_MERGE * VISION_MERGE;
    if scratch.patches != patches {
        return Err(Error::Other(format!(
            "qwen38 vision scratch holds {} patches but this request has {patches}",
            scratch.patches
        )));
    }
    let Some(first) = weights.blocks.first() else {
        return Err(Error::Other("qwen38 vision tower has no blocks".into()));
    };

    // Cloning a Tensor copies its shape and an Arc handle, not device memory,
    // so these alias the scratch buffers at stable addresses. Cloning avoids
    // borrowing several fields of `scratch` mutably at once.
    let batched = [1, patches, VISION_HEADS, VISION_HEAD_DIM];
    let mut current = scratch.x[0].clone();
    let mut spare = scratch.x[1].clone();
    let mut normed = scratch.normed.clone();
    let mut qkv = scratch.qkv.clone();
    let mut query = scratch.query.clone();
    let mut key = scratch.key.clone();
    let mut value = scratch.value.clone();
    let mut attn = scratch.attn.clone();
    let mut projected = scratch.projected.clone();
    let mut mlp_hidden = scratch.mlp_hidden.clone();
    let mut mlp_out = scratch.mlp_out.clone();

    gemm_bias_into(
        ctx,
        pixel_values,
        &weights.patch_w,
        &weights.patch_b,
        &mut current,
    )?;
    ops::add_into(ctx, &scratch.pos_embeds, &current)?;
    dump_vision_stage("00_patch_position", &current)?;
    layer_norm_into(
        ctx,
        &current,
        &first.norm1_w,
        &first.norm1_b,
        &mut normed,
        eps,
    )?;

    for (block_index, block) in weights.blocks.iter().enumerate() {
        gemm_bias_into(ctx, &normed, &block.qkv_w, &block.qkv_b, &mut qkv)?;
        ops::split_qkv_axial_rope(
            ctx,
            ops::AxialRopeArgs {
                qkv: &qkv,
                bias: None,
                positions: &scratch.pos_ids,
                query: &mut query,
                key: &mut key,
                value: &mut value,
                theta: 10_000.0,
                numerics: ops::AxialRopeNumerics::Flexible,
            },
        )?;
        // Dense Attention, not `ops::segmented_attention`: a single image is
        // one unsegmented span, and only the dense semantic has FA2 registered
        // as a candidate. The segmented semantic carries one shape-general
        // fallback kernel, which costs 267 ms per block at 2,500 patches
        // against 0.94 ms here for a bitwise-comparable result.
        ops::attention(
            ctx,
            ops::AttentionArgs::new(
                &query.reshape(batched.to_vec())?,
                &key.reshape(batched.to_vec())?,
                &value.reshape(batched.to_vec())?,
                &mut attn,
            ),
        )?;
        gemm_bias_into(
            ctx,
            &attn.reshape(vec![patches, VISION_HIDDEN])?,
            &block.proj_w,
            &block.proj_b,
            &mut projected,
        )?;
        // Writes `spare`, reads `current`: the residual add cannot target its
        // own input. `normed` is free here because the QKV GEMM above consumed
        // it, and it becomes this block's norm2 output.
        bias_residual_layer_norm_into(
            ctx,
            &projected,
            &current,
            &block.norm2_w,
            &block.norm2_b,
            &mut spare,
            &mut normed,
            eps,
        )?;
        std::mem::swap(&mut current, &mut spare);

        gemm_bias_gelu_into(ctx, &normed, &block.fc1_w, &block.fc1_b, &mut mlp_hidden)?;
        gemm_bias_into(ctx, &mlp_hidden, &block.fc2_w, &block.fc2_b, &mut mlp_out)?;
        // The fused norm for the next block, or the merger's own norm after
        // the last one, so no separate LayerNorm pass is needed.
        let (next_w, next_b) = weights
            .blocks
            .get(block_index + 1)
            .map(|next| (&next.norm1_w, &next.norm1_b))
            .unwrap_or((&weights.merger_norm_w, &weights.merger_norm_b));
        bias_residual_layer_norm_into(
            ctx,
            &mlp_out,
            &current,
            next_w,
            next_b,
            &mut spare,
            &mut normed,
            eps,
        )?;
        std::mem::swap(&mut current, &mut spare);
        dump_vision_stage(&format!("block_{block_index:02}_output"), &current)?;
    }

    let merged = normed.reshape(vec![patches / factor, VISION_HIDDEN * factor])?;
    dump_vision_stage("merger_norm", &merged)?;
    let mut merger_hidden = scratch.merger_hidden.clone();
    let mut merger_activated = scratch.merger_activated.clone();
    let mut merger_out = scratch.merger_out.clone();
    gemm_bias_into(
        ctx,
        &merged,
        &weights.merger_fc1_w,
        &weights.merger_fc1_b,
        &mut merger_hidden,
    )?;
    let mut activation = ops::PointwiseArgs::new(
        ops::PointwiseSemantic::BiasActivation,
        &merger_hidden,
        &mut merger_activated,
    );
    activation.activation = ops::PointwiseActivation::GeluExact;
    ops::pointwise(ctx, activation)?;
    gemm_bias_into(
        ctx,
        &merger_activated,
        &weights.merger_fc2_w,
        &weights.merger_fc2_b,
        &mut merger_out,
    )?;
    dump_vision_stage("merger_output", &merger_out)?;
    Ok(merger_out)
}

/// Replace image-token embedding rows with visual features. The replacement
/// is assignment (`masked_scatter` in Transformers), not residual addition.
pub(crate) fn embed_and_replace(
    ctx: &CudaContext,
    embedding_table: &Tensor,
    token_ids: &[u32],
    visual: &Tensor,
) -> Result<Tensor> {
    let ids: Vec<i32> = token_ids.iter().map(|&value| value as i32).collect();
    let ids = upload_i32(ctx, vec![token_ids.len()], &ids)?;
    let hidden = embedding_table.shape().dims()[1];
    let base = device_tensor(ctx, &[token_ids.len(), hidden], DType::BF16)?;
    ops::embedding_gather(ctx, embedding_table, &ids, &base)?;
    let mut ordinal = 0u32;
    let row_map: Vec<u32> = token_ids
        .iter()
        .map(|&token| {
            if token == IMAGE_TOKEN_ID {
                let current = ordinal;
                ordinal += 1;
                current
            } else {
                u32::MAX
            }
        })
        .collect();
    if ordinal as usize != visual.shape().dims()[0] {
        return Err(Error::Other(format!(
            "qwen38 image token count {} does not match {} visual rows",
            ordinal,
            visual.shape().dims()[0]
        )));
    }
    let row_map = upload_u32(ctx, &row_map)?;
    let mut output = device_tensor(ctx, &[token_ids.len(), hidden], DType::BF16)?;
    let mut gather = ops::GatherArgs::new(ops::GatherSemantic::ReplaceRows, &base, &mut output);
    gather.replacement = Some(visual);
    gather.row_map = Some(&row_map);
    ops::gather(ctx, gather)?;
    Ok(output)
}

/// Qwen3.5 three-axis language positions for an image-only multimodal prompt.
pub(crate) fn rope_positions(token_ids: &[u32], grid_thw: &[[u32; 3]]) -> Result<Vec<[u32; 3]>> {
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut current = 0u32;
    let mut grid_index = 0usize;
    let mut index = 0usize;
    while index < token_ids.len() {
        if token_ids[index] != IMAGE_TOKEN_ID {
            positions.push([current; 3]);
            current += 1;
            index += 1;
            continue;
        }
        let &[t, h, w] = grid_thw.get(grid_index).ok_or_else(|| {
            Error::Other("qwen38 image prompt has more image groups than grids".into())
        })?;
        grid_index += 1;
        if h % VISION_MERGE as u32 != 0 || w % VISION_MERGE as u32 != 0 {
            return Err(Error::Other(
                "qwen38 image grid is not merge-aligned".into(),
            ));
        }
        let merged_h = h / VISION_MERGE as u32;
        let merged_w = w / VISION_MERGE as u32;
        let image_tokens = t as usize * merged_h as usize * merged_w as usize;
        if index + image_tokens > token_ids.len()
            || token_ids[index..index + image_tokens]
                .iter()
                .any(|&token| token != IMAGE_TOKEN_ID)
        {
            return Err(Error::Other(format!(
                "qwen38 image placeholder group at token {index} does not contain {image_tokens} tokens"
            )));
        }
        for temporal in 0..t {
            for row in 0..merged_h {
                for column in 0..merged_w {
                    positions.push([current + temporal, current + row, current + column]);
                }
            }
        }
        current += merged_h.max(merged_w);
        index += image_tokens;
    }
    if grid_index != grid_thw.len() {
        return Err(Error::Other(format!(
            "qwen38 image prompt used {grid_index} grids but received {}",
            grid_thw.len()
        )));
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_positions_match_qwen35_group_semantics() {
        let tokens = [
            7,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            8,
        ];
        let positions = rope_positions(&tokens, &[[1, 4, 4]]).unwrap();
        assert_eq!(
            positions,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3],
            ]
        );
    }

    #[test]
    fn image_positions_reject_placeholder_mismatch() {
        let error = rope_positions(&[IMAGE_TOKEN_ID], &[[1, 4, 4]])
            .err()
            .expect("one placeholder cannot represent four merged image tokens");
        assert!(error.to_string().contains("does not contain 4"));
    }

    #[test]
    fn single_image_grid_accepts_one_merge_aligned_frame() {
        assert_eq!(single_image_grid(&[[1, 4, 6]], 24).unwrap(), (4, 6));
    }

    #[test]
    fn single_image_grid_rejects_multiple_images() {
        let error = single_image_grid(&[[1, 4, 4], [1, 4, 4]], 32)
            .err()
            .expect("dense Attention would fuse two images into one span");
        assert!(error.to_string().contains("one image per request"));
    }

    #[test]
    fn single_image_grid_rejects_a_temporal_grid() {
        let error = single_image_grid(&[[2, 4, 4]], 32)
            .err()
            .expect("a video's t > 1 needs per-frame segmentation");
        assert!(error.to_string().contains("grid t = 1"));
    }

    #[test]
    fn single_image_grid_rejects_merge_misalignment_and_patch_mismatch() {
        assert!(single_image_grid(&[[1, 5, 4]], 20)
            .err()
            .expect("odd height breaks the merger reshape")
            .to_string()
            .contains("invalid for merge"));
        assert!(single_image_grid(&[[1, 4, 4]], 15)
            .err()
            .expect("grid must cover exactly the supplied patches")
            .to_string()
            .contains("but pixel_values has 15"));
    }
}

//! Native BF16 Qwen3.5 vision tower and image-embedding injection.
//!
//! Processor output enters as canonical `[patches, 1536]` F32/BF16 plus
//! `grid_thw`. All per-request tower computation remains on CUDA. The learned
//! 48x48 position table is interpolated on the host once per distinct grid and
//! retained in a small device cache, matching the maintained Qwen-Drive path.
//! Vision and language computation share the same cuda-new context and stream.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};

use super::config::*;
use super::vision_weights::VisionWeights;

#[derive(Default)]
pub(crate) struct VisionState {
    pos_table_host: OnceLock<Vec<f32>>,
    pos_embed_cache: Mutex<Vec<(Vec<[u32; 3]>, Tensor)>>,
}

struct PreparedVision {
    offsets: Vec<u32>,
    offsets_dev: CudaBuffer,
    pos_embeds: Tensor,
    pos_ids_dev: Tensor,
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

/// Run the complete vision tower over one or more concatenated image patch
/// streams. Attention is segmented at image boundaries.
pub(crate) fn forward(
    weights: &VisionWeights,
    state: &VisionState,
    ctx: &CudaContext,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    let prepared = prepare(weights, state, ctx, pixel_values, grid_thw)?;
    let output = forward_device(weights, ctx, pixel_values, &prepared)?;
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

fn prepare(
    weights: &VisionWeights,
    state: &VisionState,
    ctx: &CudaContext,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<PreparedVision> {
    let patch_width =
        VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
    let dims = pixel_values.shape().dims();
    if grid_thw.is_empty() || dims.len() != 2 || dims[1] != patch_width {
        return Err(Error::Other(format!(
            "qwen38 image pixel_values must be [patches, {patch_width}], got {dims:?}"
        )));
    }

    let mut offsets = vec![0u32];
    for &[t, h, w] in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        if t == 0 || h == 0 || w == 0 || h % VISION_MERGE != 0 || w % VISION_MERGE != 0 {
            return Err(Error::Other(format!(
                "qwen38 image grid [{t}, {h}, {w}] is invalid for merge {VISION_MERGE}"
            )));
        }
        let patches = t
            .checked_mul(h)
            .and_then(|value| value.checked_mul(w))
            .ok_or_else(|| Error::Other("qwen38 image grid is too large".into()))?;
        let next = offsets[offsets.len() - 1]
            .checked_add(
                u32::try_from(patches)
                    .map_err(|_| Error::Other("qwen38 image patch count exceeds u32".into()))?,
            )
            .ok_or_else(|| Error::Other("qwen38 image patch count overflow".into()))?;
        offsets.push(next);
    }
    let total_patches = dims[0];
    if offsets[offsets.len() - 1] as usize != total_patches {
        return Err(Error::Other(format!(
            "qwen38 image grids cover {} patches but pixel_values has {total_patches}",
            offsets[offsets.len() - 1]
        )));
    }

    let offsets_dev = upload_u32(ctx, &offsets)?;
    let pos_embeds = compute_pos_embeds(weights, state, ctx, grid_thw)?;
    let pos_ids = vision_position_ids(grid_thw);
    let pos_ids: Vec<i32> = pos_ids.into_iter().map(|value| value as i32).collect();
    let pos_ids_dev = upload_i32(ctx, vec![total_patches, 2], &pos_ids)?;
    Ok(PreparedVision {
        offsets,
        offsets_dev,
        pos_embeds,
        pos_ids_dev,
        total_patches,
    })
}

fn gemm_bias(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    if input_dims.len() != 2 || weight_dims.len() != 2 || input_dims[1] != weight_dims[0] {
        return Err(Error::Other("qwen38 vision GEMM shape mismatch".into()));
    }
    let mut output = device_tensor(ctx, &[input_dims[0], weight_dims[1]], DType::BF16)?;
    ops::gemm_bias(
        ctx,
        ops::GemmBiasArgs {
            gemm: ops::GemmArgs::new(input, weight, &mut output),
            bias,
        },
    )?;
    Ok(output)
}

fn gemm_bias_gelu(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    if input_dims.len() != 2 || weight_dims.len() != 2 || input_dims[1] != weight_dims[0] {
        return Err(Error::Other("qwen38 vision GELU GEMM shape mismatch".into()));
    }
    let mut output = device_tensor(ctx, &[input_dims[0], weight_dims[1]], DType::BF16)?;
    ops::gemm_bias_gelu(
        ctx,
        ops::GemmBiasGeluArgs {
            gemm: ops::GemmArgs::new(input, weight, &mut output),
            bias,
        },
    )?;
    Ok(output)
}

fn layer_norm(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = device_tensor(ctx, input.shape().dims(), DType::BF16)?;
    ops::layer_norm(
        ctx,
        ops::LayerNormArgs::new(input, weight, bias, &mut output, eps),
    )?;
    Ok(output)
}

fn bias_residual_layer_norm(
    ctx: &CudaContext,
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let dims = input.shape().dims();
    let mut hidden = device_tensor(ctx, dims, DType::BF16)?;
    let mut normalized = device_tensor(ctx, dims, DType::BF16)?;
    ops::bias_residual_layer_norm(
        ctx,
        ops::BiasResidualLayerNormArgs::new(
            input,
            None,
            residual,
            weight,
            norm_bias,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok((hidden, normalized))
}

fn forward_device(
    weights: &VisionWeights,
    ctx: &CudaContext,
    pixel_values: &Tensor,
    prepared: &PreparedVision,
) -> Result<Tensor> {
    let eps = 1e-6f32;
    let mut x = gemm_bias(ctx, pixel_values, &weights.patch_w, &weights.patch_b)?;
    ops::mlp::add_into(ctx, &prepared.pos_embeds, &x)?;
    dump_vision_stage("00_patch_position", &x)?;

    let mut normalized = None;
    for (block_index, block) in weights.blocks.iter().enumerate() {
        let normed = match normalized.take() {
            Some(value) => value,
            None => layer_norm(ctx, &x, &block.norm1_w, &block.norm1_b, eps)?,
        };
        let qkv = gemm_bias(ctx, &normed, &block.qkv_w, &block.qkv_b)?;
        let qkv_shape = [prepared.total_patches, VISION_HEADS, VISION_HEAD_DIM];
        let mut query = device_tensor(ctx, &qkv_shape, DType::BF16)?;
        let mut key = device_tensor(ctx, &qkv_shape, DType::BF16)?;
        let mut value = device_tensor(ctx, &qkv_shape, DType::BF16)?;
        ops::split_qkv_axial_rope(
            ctx,
            ops::AxialRopeArgs {
                qkv: &qkv,
                bias: None,
                positions: &prepared.pos_ids_dev,
                query: &mut query,
                key: &mut key,
                value: &mut value,
                theta: 10_000.0,
            },
        )?;
        let mut attn = device_tensor(ctx, &qkv_shape, DType::BF16)?;
        ops::segmented_attention(
            ctx,
            ops::SegmentedAttentionArgs::new(
                &query,
                &key,
                &value,
                &mut attn,
                &prepared.offsets_dev,
                &prepared.offsets,
            ),
        )?;
        let attn = attn.reshape(vec![prepared.total_patches, VISION_HIDDEN])?;
        let projected = gemm_bias(ctx, &attn, &block.proj_w, &block.proj_b)?;
        let (hidden, norm2) = bias_residual_layer_norm(
            ctx,
            &projected,
            &x,
            &block.norm2_w,
            &block.norm2_b,
            eps,
        )?;
        x = hidden;

        let hidden = gemm_bias_gelu(ctx, &norm2, &block.fc1_w, &block.fc1_b)?;
        let mlp = gemm_bias(ctx, &hidden, &block.fc2_w, &block.fc2_b)?;
        let (next_w, next_b) = weights
            .blocks
            .get(block_index + 1)
            .map(|next| (&next.norm1_w, &next.norm1_b))
            .unwrap_or((&weights.merger_norm_w, &weights.merger_norm_b));
        let (hidden, next_normalized) =
            bias_residual_layer_norm(ctx, &mlp, &x, next_w, next_b, eps)?;
        x = hidden;
        dump_vision_stage(&format!("block_{block_index:02}_output"), &x)?;
        normalized = Some(next_normalized);
    }

    let normalized = match normalized {
        Some(value) => value,
        None => layer_norm(ctx, &x, &weights.merger_norm_w, &weights.merger_norm_b, eps)?,
    };
    let rows = normalized.shape().dims()[0];
    let factor = VISION_MERGE * VISION_MERGE;
    if rows % factor != 0 {
        return Err(Error::Other(format!(
            "qwen38 image merger rows {rows} do not divide by {factor}"
        )));
    }
    let merged = normalized.reshape(vec![rows / factor, VISION_HIDDEN * factor])?;
    dump_vision_stage("merger_norm", &merged)?;
    let hidden = gemm_bias(ctx, &merged, &weights.merger_fc1_w, &weights.merger_fc1_b)?;
    let mut activated = device_tensor(ctx, hidden.shape().dims(), DType::BF16)?;
    let mut activation = ops::PointwiseArgs::new(
        ops::PointwiseSemantic::BiasActivation,
        &hidden,
        &mut activated,
    );
    activation.activation = ops::PointwiseActivation::GeluExact;
    ops::pointwise(ctx, activation)?;
    let output = gemm_bias(ctx, &activated, &weights.merger_fc2_w, &weights.merger_fc2_b)?;
    dump_vision_stage("merger_output", &output)?;
    Ok(output)
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
    let mut gather = ops::GatherArgs::new(
        ops::GatherSemantic::ReplaceRows,
        &base,
        &mut output,
    );
    gather.replacement = Some(visual);
    gather.row_map = Some(&row_map);
    ops::gather(ctx, gather)?;
    Ok(output)
}

fn compute_pos_embeds(
    weights: &VisionWeights,
    state: &VisionState,
    ctx: &CudaContext,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    if let Ok(cache) = state.pos_embed_cache.lock() {
        if let Some((_, tensor)) = cache.iter().find(|(key, _)| key.as_slice() == grid_thw) {
            return Ok(tensor.clone());
        }
    }
    if state.pos_table_host.get().is_none() {
        let table = transfers::to_cpu(&weights.pos_embed)?.to_f32_vec()?;
        let _ = state.pos_table_host.set(table);
    }
    let table = state
        .pos_table_host
        .get()
        .ok_or_else(|| Error::Other("qwen38 image position table unavailable".into()))?;
    let grid_side = (VISION_POSITIONS as f64).sqrt().round() as usize;
    if table.len() != grid_side * grid_side * VISION_HIDDEN {
        return Err(Error::Other(
            "qwen38 image position table shape mismatch".into(),
        ));
    }

    let total: usize = grid_thw
        .iter()
        .map(|grid| grid[0] as usize * grid[1] as usize * grid[2] as usize)
        .sum();
    let mut output = vec![0.0f32; total * VISION_HIDDEN];
    let mut destination_token = 0usize;
    for &[t, h, w] in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        for _ in 0..t {
            for merged_h in 0..h / VISION_MERGE {
                for merged_w in 0..w / VISION_MERGE {
                    for inner_h in 0..VISION_MERGE {
                        for inner_w in 0..VISION_MERGE {
                            let row = merged_h * VISION_MERGE + inner_h;
                            let column = merged_w * VISION_MERGE + inner_w;
                            let row_position =
                                row as f32 * (grid_side - 1) as f32 / (h - 1).max(1) as f32;
                            let column_position =
                                column as f32 * (grid_side - 1) as f32 / (w - 1).max(1) as f32;
                            let row0 = row_position.floor() as usize;
                            let row1 = (row0 + 1).min(grid_side - 1);
                            let column0 = column_position.floor() as usize;
                            let column1 = (column0 + 1).min(grid_side - 1);
                            let row_weight = row_position - row0 as f32;
                            let column_weight = column_position - column0 as f32;
                            let destination = destination_token * VISION_HIDDEN;
                            for channel in 0..VISION_HIDDEN {
                                let value00 =
                                    table[(row0 * grid_side + column0) * VISION_HIDDEN + channel];
                                let value01 =
                                    table[(row0 * grid_side + column1) * VISION_HIDDEN + channel];
                                let value10 =
                                    table[(row1 * grid_side + column0) * VISION_HIDDEN + channel];
                                let value11 =
                                    table[(row1 * grid_side + column1) * VISION_HIDDEN + channel];
                                output[destination + channel] =
                                    (1.0 - row_weight) * (1.0 - column_weight) * value00
                                        + (1.0 - row_weight) * column_weight * value01
                                        + row_weight * (1.0 - column_weight) * value10
                                        + row_weight * column_weight * value11;
                            }
                            destination_token += 1;
                        }
                    }
                }
            }
        }
    }
    let rounded: Vec<half::bf16> = output
        .iter()
        .map(|&value| half::bf16::from_f32(value))
        .collect();
    let host = Tensor::from_bf16(vec![total, VISION_HIDDEN], &rounded)?;
    let uploaded = transfers::to_cuda(&host, ctx.device_id())?;
    if let Ok(mut cache) = state.pos_embed_cache.lock() {
        const MAX_GRIDS: usize = 4;
        if cache.len() >= MAX_GRIDS {
            cache.remove(0);
        }
        cache.push((grid_thw.to_vec(), uploaded.clone()));
    }
    Ok(uploaded)
}

fn vision_position_ids(grid_thw: &[[u32; 3]]) -> Vec<u32> {
    let mut ids = Vec::new();
    for &[t, h, w] in grid_thw {
        for _ in 0..t {
            for merged_h in 0..h as usize / VISION_MERGE {
                for merged_w in 0..w as usize / VISION_MERGE {
                    for inner_h in 0..VISION_MERGE {
                        for inner_w in 0..VISION_MERGE {
                            ids.push((merged_h * VISION_MERGE + inner_h) as u32);
                            ids.push((merged_w * VISION_MERGE + inner_w) as u32);
                        }
                    }
                }
            }
        }
    }
    ids
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
}

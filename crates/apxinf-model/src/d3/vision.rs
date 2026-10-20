//! Qwen3.5 vision tower and multimodal position preparation for d3-mini.

use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};

use crate::d3::backend::{copy_i32, read_bf16, upload, zeros};
use crate::d3::model::D3Policies;
use crate::d3::weights::bf16::{D3DeviceWeights, VisionWeights};
use crate::d3::D3PackageConfig;

const CHANNELS: usize = 3;
const VISION_THETA: f32 = 10_000.0;

pub(crate) struct D3VisionScratch {
    patches: usize,
    merged_rows: usize,
    pixels_f32: Tensor,
    pixels_bf16: Tensor,
    hidden: Tensor,
    normalized: Tensor,
    projected: Tensor,
    qkv: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention: Tensor,
    fc1: Tensor,
    activation: Tensor,
    merger_fc1: Tensor,
    merger_activation: Tensor,
    visual_output: Tensor,
    visual_final: Tensor,
    position_embeddings: Tensor,
    vision_positions: Tensor,
    offsets: Tensor,
    host_offsets: Vec<u32>,
    row_sources: Tensor,
    mrope_positions: Tensor,
}

impl D3VisionScratch {
    pub(crate) fn new(
        ctx: &CudaContext,
        config: &D3PackageConfig,
        weights: &D3DeviceWeights,
        token_ids: &[u32],
        grid_thw: &[[u32; 3]],
        patches: usize,
    ) -> Result<Self> {
        let vision = &config.model.vision_config;
        let patch_width =
            CHANNELS * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
        let merged_rows = patches / vision.spatial_merge_size.pow(2);
        let layout = prepare_layout(config, token_ids, grid_thw, patches, merged_rows)?;
        let table = read_bf16(ctx, &weights.vision.pos_embed)?;
        let position_embeddings = upload(
            ctx,
            &Tensor::from_bf16(
                vec![patches, vision.hidden_size],
                &interpolate_positions(config, grid_thw, &table)?
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
            )?,
        )?;
        let vision_positions = zeros(ctx, vec![patches, 2], DType::I32)?;
        copy_i32(
            ctx,
            &vision_positions,
            &vision_position_ids(grid_thw, vision.spatial_merge_size)?
                .into_iter()
                .map(|value| value as i32)
                .collect::<Vec<_>>(),
        )?;
        let offsets = zeros(ctx, vec![layout.offsets.len()], DType::I32)?;
        copy_i32(
            ctx,
            &offsets,
            &layout
                .offsets
                .iter()
                .map(|value| *value as i32)
                .collect::<Vec<_>>(),
        )?;
        let row_sources = zeros(ctx, vec![token_ids.len()], DType::I32)?;
        copy_i32(ctx, &row_sources, &layout.row_sources)?;
        let mrope_positions = zeros(ctx, vec![token_ids.len(), 3], DType::I32)?;
        copy_i32(ctx, &mrope_positions, &layout.mrope_positions)?;

        let hidden = vision.hidden_size;
        let intermediate = vision.intermediate_size;
        let heads = vision.num_heads;
        let head_dim = hidden / heads;
        let merged = hidden * vision.spatial_merge_size.pow(2);
        Ok(Self {
            patches,
            merged_rows,
            pixels_f32: zeros(ctx, vec![patches, patch_width], DType::F32)?,
            pixels_bf16: zeros(ctx, vec![patches, patch_width], DType::BF16)?,
            hidden: zeros(ctx, vec![patches, hidden], DType::BF16)?,
            normalized: zeros(ctx, vec![patches, hidden], DType::BF16)?,
            projected: zeros(ctx, vec![patches, hidden], DType::BF16)?,
            qkv: zeros(ctx, vec![patches, 3 * hidden], DType::BF16)?,
            query: zeros(ctx, vec![patches, heads, head_dim], DType::BF16)?,
            key: zeros(ctx, vec![patches, heads, head_dim], DType::BF16)?,
            value: zeros(ctx, vec![patches, heads, head_dim], DType::BF16)?,
            attention: zeros(ctx, vec![patches, heads, head_dim], DType::BF16)?,
            fc1: zeros(ctx, vec![patches, intermediate], DType::BF16)?,
            activation: zeros(ctx, vec![patches, intermediate], DType::BF16)?,
            merger_fc1: zeros(ctx, vec![merged_rows, merged], DType::BF16)?,
            merger_activation: zeros(ctx, vec![merged_rows, merged], DType::BF16)?,
            visual_output: zeros(ctx, vec![merged_rows, vision.out_hidden_size], DType::BF16)?,
            visual_final: zeros(ctx, vec![merged_rows, vision.out_hidden_size], DType::BF16)?,
            position_embeddings,
            vision_positions,
            offsets,
            host_offsets: layout.offsets,
            row_sources,
            mrope_positions,
        })
    }

    pub(crate) fn write_pixels(&self, ctx: &CudaContext, pixels: &Tensor) -> Result<()> {
        if pixels.shape() != self.pixels_f32.shape() || pixels.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "d3 vision pixels must be F32 {:?}, got {} {:?}",
                self.pixels_f32.shape().dims(),
                pixels.dtype(),
                pixels.shape().dims()
            )));
        }
        match pixels.device() {
            Device::Cpu => transfers::copy_cpu_to_cuda(pixels, &self.pixels_f32),
            Device::Cuda(device) if device == ctx.device_id() => {
                transfers::copy_cuda_to_cuda(ctx, pixels, &self.pixels_f32)
            }
            device => Err(Error::Other(format!(
                "d3 vision pixels are on unsupported device {device}"
            ))),
        }
    }

    pub(crate) fn mrope_positions(&self) -> &Tensor {
        &self.mrope_positions
    }

    pub(crate) fn row_sources(&self) -> &Tensor {
        &self.row_sources
    }

    pub(crate) fn visual_output(&self) -> &Tensor {
        &self.visual_final
    }
}

pub(crate) fn forward_vision_device(
    ctx: &CudaContext,
    config: &D3PackageConfig,
    weights: &VisionWeights,
    policies: &D3Policies,
    scratch: &mut D3VisionScratch,
) -> Result<()> {
    let vision = &config.model.vision_config;
    let hidden = vision.hidden_size;
    let eps = 1e-6;

    let mut pixels_bf16 = scratch.pixels_bf16.clone();
    ops::quantization(
        ctx,
        ops::QuantizationArgs::new(
            ops::QuantizationSemantic::CastF32ToBf16,
            &scratch.pixels_f32,
            &mut pixels_bf16,
        ),
    )?;
    let mut projected = scratch.projected.clone();
    gemm(
        ctx,
        &pixels_bf16,
        &weights.patch_w,
        &mut projected,
        &policies.gemm,
    )?;
    bias_activation(
        ctx,
        &projected,
        &weights.patch_b,
        &mut scratch.hidden,
        ops::PointwiseActivation::None,
    )?;
    ops::add_into(ctx, &scratch.position_embeddings, &scratch.hidden)?;
    trace_vision(ctx, "input", &scratch.hidden)?;

    for (block_index, block) in weights.blocks.iter().enumerate() {
        ops::layer_norm(
            ctx,
            ops::LayerNormArgs::new(
                &scratch.hidden,
                &block.norm1_w,
                &block.norm1_b,
                &mut scratch.normalized,
                eps,
            ),
        )?;
        if block_index == 0 {
            trace_vision(ctx, "block-0-norm1", &scratch.normalized)?;
        }
        gemm(
            ctx,
            &scratch.normalized,
            &block.qkv_w,
            &mut scratch.qkv,
            &policies.gemm,
        )?;
        let mut query = scratch.query.clone();
        let mut key = scratch.key.clone();
        let mut value = scratch.value.clone();
        ops::split_qkv_axial_rope(
            ctx,
            ops::AxialRopeArgs {
                qkv: &scratch.qkv,
                bias: Some(&block.qkv_b),
                positions: &scratch.vision_positions,
                query: &mut query,
                key: &mut key,
                value: &mut value,
                theta: VISION_THETA,
            },
        )?;
        let offsets = CudaBuffer::from_tensor(&scratch.offsets).map_err(Error::Cuda)?;
        let mut attention_args = ops::SegmentedAttentionArgs::new(
            &scratch.query,
            &scratch.key,
            &scratch.value,
            &mut scratch.attention,
            &offsets,
            &scratch.host_offsets,
        );
        attention_args.policy = policies.attention.clone();
        ops::segmented_attention(ctx, attention_args)?;
        if block_index == 0 {
            trace_vision(ctx, "block-0-attention", &scratch.attention)?;
        }
        let attention = scratch.attention.reshape(vec![scratch.patches, hidden])?;
        gemm(
            ctx,
            &attention,
            &block.proj_w,
            &mut scratch.projected,
            &policies.gemm,
        )?;
        bias_activation(
            ctx,
            &scratch.projected,
            &block.proj_b,
            &mut scratch.normalized,
            ops::PointwiseActivation::None,
        )?;
        if block_index == 0 {
            trace_vision(ctx, "block-0-attn-branch", &scratch.normalized)?;
        }
        ops::add_into(ctx, &scratch.normalized, &scratch.hidden)?;
        if matches!(block_index, 0 | 1 | 5 | 11 | 17 | 23) {
            trace_vision(
                ctx,
                &format!("block-{block_index}-after-attn"),
                &scratch.hidden,
            )?;
        }

        ops::layer_norm(
            ctx,
            ops::LayerNormArgs::new(
                &scratch.hidden,
                &block.norm2_w,
                &block.norm2_b,
                &mut scratch.normalized,
                eps,
            ),
        )?;
        gemm(
            ctx,
            &scratch.normalized,
            &block.fc1_w,
            &mut scratch.fc1,
            &policies.gemm,
        )?;
        bias_activation(
            ctx,
            &scratch.fc1,
            &block.fc1_b,
            &mut scratch.activation,
            ops::PointwiseActivation::Gelu,
        )?;
        gemm(
            ctx,
            &scratch.activation,
            &block.fc2_w,
            &mut scratch.projected,
            &policies.gemm,
        )?;
        bias_activation(
            ctx,
            &scratch.projected,
            &block.fc2_b,
            &mut scratch.normalized,
            ops::PointwiseActivation::None,
        )?;
        if block_index == 0 {
            trace_vision(ctx, "block-0-mlp-branch", &scratch.normalized)?;
        }
        ops::add_into(ctx, &scratch.normalized, &scratch.hidden)?;
        if matches!(block_index, 0 | 1 | 5 | 11 | 17 | 23) {
            trace_vision(ctx, &format!("block-{block_index}"), &scratch.hidden)?;
        }
    }

    ops::layer_norm(
        ctx,
        ops::LayerNormArgs::new(
            &scratch.hidden,
            &weights.merger_norm_w,
            &weights.merger_norm_b,
            &mut scratch.normalized,
            eps,
        ),
    )?;
    trace_vision(ctx, "merger-norm", &scratch.normalized)?;
    let merged_width = hidden * vision.spatial_merge_size.pow(2);
    let merged = scratch
        .normalized
        .reshape(vec![scratch.merged_rows, merged_width])?;
    gemm(
        ctx,
        &merged,
        &weights.merger_fc1_w,
        &mut scratch.merger_fc1,
        &policies.gemm,
    )?;
    trace_vision(ctx, "merger-fc1", &scratch.merger_fc1)?;
    bias_activation(
        ctx,
        &scratch.merger_fc1,
        &weights.merger_fc1_b,
        &mut scratch.merger_activation,
        ops::PointwiseActivation::GeluExact,
    )?;
    trace_vision(ctx, "merger-gelu", &scratch.merger_activation)?;
    gemm(
        ctx,
        &scratch.merger_activation,
        &weights.merger_fc2_w,
        &mut scratch.visual_output,
        &policies.gemm,
    )?;
    bias_activation(
        ctx,
        &scratch.visual_output,
        &weights.merger_fc2_b,
        &mut scratch.visual_final,
        ops::PointwiseActivation::None,
    )?;
    if std::env::var_os("APXINF_D3_TRACE").is_some() {
        let values = read_bf16(ctx, &scratch.visual_final)?;
        let sum = values.iter().sum::<f32>();
        let abs_sum = values.iter().map(|value| value.abs()).sum::<f32>();
        eprintln!(
            "[d3 trace] vision rows={} width={} sum={sum:.8} abs_sum={abs_sum:.8} first={:?}",
            scratch.merged_rows,
            vision.out_hidden_size,
            &values[..values.len().min(8)]
        );
    }
    Ok(())
}

fn gemm(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    policy: &ops::GemmPolicy,
) -> Result<()> {
    let mut args = ops::GemmArgs::new(input, weight, output);
    args.policy = policy.clone();
    ops::gemm(ctx, args)
}

fn bias_activation(
    ctx: &CudaContext,
    input: &Tensor,
    bias: &Tensor,
    output: &mut Tensor,
    activation: ops::PointwiseActivation,
) -> Result<()> {
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, output);
    args.bias = Some(bias);
    args.activation = activation;
    ops::pointwise(ctx, args)
}

struct PreparedLayout {
    offsets: Vec<u32>,
    row_sources: Vec<i32>,
    mrope_positions: Vec<i32>,
}

fn prepare_layout(
    config: &D3PackageConfig,
    token_ids: &[u32],
    grid_thw: &[[u32; 3]],
    patches: usize,
    merged_rows: usize,
) -> Result<PreparedLayout> {
    let merge = config.model.vision_config.spatial_merge_size as u32;
    let mut offsets = vec![0u32];
    let mut covered = 0usize;
    for &[t, h, w] in grid_thw {
        if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
            return Err(Error::Other(format!(
                "d3 vision grid [{t}, {h}, {w}] is incompatible with merge {merge}"
            )));
        }
        let frame = (h as usize)
            .checked_mul(w as usize)
            .ok_or_else(|| Error::Other("d3 vision grid overflow".into()))?;
        for _ in 0..t {
            covered += frame;
            offsets.push(covered as u32);
        }
    }
    if covered != patches || patches % (merge as usize).pow(2) != 0 {
        return Err(Error::Other(format!(
            "d3 vision grids cover {covered} patches, expected {patches}"
        )));
    }

    let image = config.model.image_token_id;
    let video = config.model.video_token_id;
    let mut row_sources = Vec::with_capacity(token_ids.len());
    let mut source_row = 0i32;
    for &token in token_ids {
        if token == image || token == video {
            row_sources.push(source_row);
            source_row += 1;
        } else {
            row_sources.push(-1);
        }
    }
    if source_row as usize != merged_rows {
        return Err(Error::Other(format!(
            "d3 vision prompt has {source_row} placeholder tokens for {merged_rows} merged rows"
        )));
    }

    let positions = multimodal_positions(config, token_ids, grid_thw)?;
    Ok(PreparedLayout {
        offsets,
        row_sources,
        mrope_positions: positions
            .into_iter()
            .flat_map(|position| position.map(|value| value as i32))
            .collect(),
    })
}

fn multimodal_positions(
    config: &D3PackageConfig,
    token_ids: &[u32],
    grid_thw: &[[u32; 3]],
) -> Result<Vec<[u32; 3]>> {
    let merge = config.model.vision_config.spatial_merge_size as u32;
    let image = config.model.image_token_id;
    let video = config.model.video_token_id;
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut grid_index = 0usize;
    let mut video_frames_remaining = 0u32;
    let mut current = 0u32;
    let mut index = 0usize;
    while index < token_ids.len() {
        let token = token_ids[index];
        if token != image && token != video {
            positions.push([current; 3]);
            current += 1;
            index += 1;
            continue;
        }
        let start = index;
        while index < token_ids.len() && token_ids[index] == token {
            index += 1;
        }
        let run = index - start;
        let &[t, h, w] = grid_thw
            .get(grid_index)
            .ok_or_else(|| Error::Other("d3 vision has more placeholder runs than grids".into()))?;
        let merged_h = h / merge;
        let merged_w = w / merge;
        let per_frame = (merged_h * merged_w) as usize;
        // Qwen3.5's processor inserts timestamps/text between video frames.
        // Transformers therefore expands a [T,H,W] video grid into T separate
        // [1,H,W] mRoPE groups, while the vision tower still sees one T-frame
        // tensor. Images remain one group and normally have T=1.
        let group_t = if token == video { 1 } else { t };
        if run != (group_t as usize) * per_frame {
            return Err(Error::Other(format!(
                "d3 vision placeholder run {run} does not match grid [{t}, {h}, {w}]"
            )));
        }
        for ti in 0..group_t {
            for hi in 0..merged_h {
                for wi in 0..merged_w {
                    positions.push([current + ti, current + hi, current + wi]);
                }
            }
        }
        current += merged_h.max(merged_w);
        if token == video {
            if video_frames_remaining == 0 {
                video_frames_remaining = t;
            }
            video_frames_remaining -= 1;
            if video_frames_remaining == 0 {
                grid_index += 1;
            }
        } else {
            grid_index += 1;
        }
    }
    if grid_index != grid_thw.len()
        || video_frames_remaining != 0
        || positions.len() != token_ids.len()
    {
        return Err(Error::Other(
            "d3 vision grids and prompt placeholder layout disagree".into(),
        ));
    }
    Ok(positions)
}

fn vision_position_ids(grid_thw: &[[u32; 3]], merge: usize) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for &[t, h, w] in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        for _ in 0..t {
            for mh in 0..h / merge {
                for mw in 0..w / merge {
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

fn interpolate_positions(
    config: &D3PackageConfig,
    grid_thw: &[[u32; 3]],
    table: &[f32],
) -> Result<Vec<f32>> {
    let vision = &config.model.vision_config;
    let side = (vision.num_position_embeddings as f64).sqrt() as usize;
    let hidden = vision.hidden_size;
    let merge = vision.spatial_merge_size;
    let mut output = Vec::new();
    for &[t, h, w] in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        let mut interpolated = vec![0.0f32; h * w * hidden];
        for hi in 0..h {
            let hf = hi as f32 * (side - 1) as f32 / (h - 1).max(1) as f32;
            let h0 = hf.floor() as usize;
            let h1 = (h0 + 1).min(side - 1);
            let dh = hf - h0 as f32;
            for wi in 0..w {
                let wf = wi as f32 * (side - 1) as f32 / (w - 1).max(1) as f32;
                let w0 = wf.floor() as usize;
                let w1 = (w0 + 1).min(side - 1);
                let dw = wf - w0 as f32;
                for channel in 0..hidden {
                    let values = [
                        table[(h0 * side + w0) * hidden + channel],
                        table[(h0 * side + w1) * hidden + channel],
                        table[(h1 * side + w0) * hidden + channel],
                        table[(h1 * side + w1) * hidden + channel],
                    ];
                    let weights = [
                        (1.0 - dh) * (1.0 - dw),
                        (1.0 - dh) * dw,
                        dh * (1.0 - dw),
                        dh * dw,
                    ];
                    let weighted = values
                        .into_iter()
                        .zip(weights)
                        .map(|(value, weight)| bf16_round(value * bf16_round(weight)))
                        .collect::<Vec<_>>();
                    interpolated[(hi * w + wi) * hidden + channel] = weighted
                        .into_iter()
                        .reduce(|left, right| bf16_round(left + right))
                        .unwrap();
                }
            }
        }
        for _ in 0..t {
            for mh in 0..h / merge {
                for mw in 0..w / merge {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            let source = ((mh * merge + ih) * w + mw * merge + iw) * hidden;
                            output.extend_from_slice(&interpolated[source..source + hidden]);
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn bf16_round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn trace_vision(ctx: &CudaContext, label: &str, tensor: &Tensor) -> Result<()> {
    if std::env::var_os("APXINF_D3_TRACE").is_none() {
        return Ok(());
    }
    let values = read_bf16(ctx, tensor)?;
    let sum = values.iter().sum::<f32>();
    let abs_sum = values.iter().map(|value| value.abs()).sum::<f32>();
    eprintln!(
        "[d3 trace] vision {label} sum={sum:.8} abs_sum={abs_sum:.8} first={:?}",
        &values[..values.len().min(8)]
    );
    Ok(())
}

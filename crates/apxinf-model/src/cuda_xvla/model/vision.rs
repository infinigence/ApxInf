//! DaViT vision tower forward pass for X-VLA.
//!
//! Input is a token-major normalized image `[S*S, 3]`; output is
//! `cnn_feats` `[1 + S4*S4, projection_dim]` where row zero is the spatial
//! average and the remaining rows are the stage-4 tokens, matching
//! `Florence2ForConditionalGeneration._encode_image` for a single frame.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::cuda_xvla::backend::{kernels, Context, CublasTranspose, DeviceBuffer};
use crate::cuda_xvla::backend::kernels::xvla_ops;
use crate::cuda_xvla::weights::{
    DepthwiseWeights, LayerNormWeights, VisionBlockWeights, VisionStageWeights, VisionWeights,
};

type Probe<'a> = &'a mut dyn FnMut(&str, &Tensor) -> Result<()>;

/// Encode one image view; `pixel_tokens` is BF16 `[side*side, 3]`, token-major.
pub fn encode_image(
    ctx: &Context,
    weights: &VisionWeights,
    pixel_tokens: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    encode_image_probed(ctx, weights, pixel_tokens, eps, &mut |_, _| Ok(()))
}

pub(crate) fn encode_image_probed(
    ctx: &Context,
    weights: &VisionWeights,
    pixel_tokens: &Tensor,
    eps: f32,
    probe: Probe<'_>,
) -> Result<Tensor> {
    let dims = pixel_tokens.shape().dims();
    let side = weights.input_side;
    if dims != [side * side, 3] || pixel_tokens.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "X-VLA vision input must be BF16 [{}, 3], got {} {:?}",
            side * side,
            pixel_tokens.dtype(),
            dims
        )));
    }

    let mut x = pixel_tokens.clone();
    let mut current_side = side;
    for (index, stage) in weights.stages.iter().enumerate() {
        if stage.pre_norm {
            x = layer_norm(ctx, &stage.norm, &x, eps)?;
        }
        let columns = xvla_ops::im2col_bf16(
            ctx,
            &x,
            current_side,
            current_side,
            stage.kernel,
            stage.kernel,
            stage.stride,
            stage.stride,
            stage.padding,
            stage.padding,
        )?;
        let embedded =
            kernels::gemm::bf16_bias(ctx, &columns, &stage.conv.weight, &stage.conv.bias)?;
        x = if stage.pre_norm {
            embedded
        } else {
            layer_norm(ctx, &stage.norm, &embedded, eps)?
        };
        current_side = stage.plan.height;
        probe(&format!("conv{index}"), &x)?;

        for (pair, (spatial, channel)) in stage.blocks.iter().enumerate() {
            let label = format!("s{index}.{pair}.spatial_block");
            x = spatial_block(ctx, spatial, stage, &x, eps, &label, probe)?;
            let label = format!("s{index}.{pair}.channel_block");
            x = channel_block(ctx, channel, stage, &x, eps, &label, probe)?;
        }
    }
    probe("tower", &x)?;

    let tokens = x.shape().dims()[0];
    if tokens != weights.image_position_table.shape().dims()[0] {
        return Err(Error::Other(format!(
            "X-VLA vision produced {tokens} tokens, position table has {}",
            weights.image_position_table.shape().dims()[0]
        )));
    }
    let positioned = kernels::elementwise::add(ctx, &x, &weights.image_position_table)?;
    let pooled = kernels::gemm::matmul(ctx, &weights.spatial_mean_row, &positioned)?;
    let pooled = kernels::elementwise::scale(ctx, &pooled, 1.0 / tokens as f32)?;
    let features = kernels::elementwise::concat_rows_bf16(ctx, &pooled, &positioned)?;
    let projected = kernels::gemm::matmul(ctx, &features, &weights.image_projection)?;
    let features = kernels::norm::layer_bf16(
        ctx,
        &projected,
        &weights.image_proj_norm.weight,
        &weights.image_proj_norm.bias,
        eps,
    )?;
    probe("features", &features)?;
    Ok(features)
}

fn spatial_block(
    ctx: &Context,
    weights: &VisionBlockWeights,
    stage: &VisionStageWeights,
    x: &Tensor,
    eps: f32,
    label: &str,
    probe: Probe<'_>,
) -> Result<Tensor> {
    let plan = &stage.plan;
    let x = depthwise(ctx, &weights.conv1, x, plan.height, plan.width)?;
    probe(&format!("{label}.conv1"), &x)?;
    let normalized = layer_norm(ctx, &weights.norm_attn, &x, eps)?;
    let padded = kernels::elementwise::concat_rows_bf16(ctx, &normalized, &stage.zero_row)?;
    let windows = xvla_ops::gather_rows_bf16(ctx, &padded, &plan.partition)?;
    let qkv = kernels::gemm::bf16_bias(ctx, &windows, &weights.qkv.weight, &weights.qkv.bias)?;
    let split = kernels::attention::split_qkv_bias_bf16(
        ctx,
        &qkv,
        None,
        stage.heads,
        stage.channels / stage.heads,
    )?;
    let attended = window_attention(ctx, &split, plan.windows, plan.window, stage.heads)?;
    let tokens = xvla_ops::gather_rows_bf16(ctx, &attended, &plan.reverse)?;
    let projected = kernels::gemm::matmul(ctx, &tokens, &weights.proj.weight)?;
    let x = kernels::fused::bias_residual_bf16(ctx, &projected, Some(&weights.proj.bias), &x)?;
    probe(&format!("{label}.attn"), &x)?;

    let x = depthwise(ctx, &weights.conv2, &x, plan.height, plan.width)?;
    probe(&format!("{label}.conv2"), &x)?;
    let normalized = layer_norm(ctx, &weights.norm_ffn, &x, eps)?;
    let hidden =
        kernels::gemm::bf16_bias(ctx, &normalized, &weights.fc1.weight, &weights.fc1.bias)?;
    let hidden = xvla_ops::gelu_erf_bf16(ctx, &hidden)?;
    let projected = kernels::gemm::matmul(ctx, &hidden, &weights.fc2.weight)?;
    let x = kernels::fused::bias_residual_bf16(ctx, &projected, Some(&weights.fc2.bias), &x)?;
    probe(&format!("{label}.ffn"), &x)?;
    Ok(x)
}

fn channel_block(
    ctx: &Context,
    weights: &VisionBlockWeights,
    stage: &VisionStageWeights,
    x: &Tensor,
    eps: f32,
    label: &str,
    probe: Probe<'_>,
) -> Result<Tensor> {
    let tokens = stage.plan.height * stage.plan.width;
    let channels = stage.channels;
    let groups = stage.groups;
    let head_dim = channels / groups;

    let x = depthwise(ctx, &weights.conv1, x, stage.plan.height, stage.plan.width)?;
    probe(&format!("{label}.conv1"), &x)?;
    let normalized = layer_norm(ctx, &weights.norm_attn, &x, eps)?;
    let qkv = kernels::gemm::bf16_bias(ctx, &normalized, &weights.qkv.weight, &weights.qkv.bias)?;
    let split = kernels::attention::split_qkv_bias_bf16(ctx, &qkv, None, groups, head_dim)?;
    let q = kernels::elementwise::scale(ctx, &split.q, (tokens as f32).powf(-0.5))?;

    let q_buffer = DeviceBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = DeviceBuffer::from_tensor(&split.k).map_err(Error::Cuda)?;
    let v_buffer = DeviceBuffer::from_tensor(&split.v).map_err(Error::Cuda)?;

    let element = std::mem::size_of::<half::bf16>();
    let row_bytes = channels * element;
    let span = (tokens - 1) * row_bytes + head_dim * element;
    let group_bytes = head_dim * element;
    let output = kernels::scratch_buffer(ctx, tokens * row_bytes)?;

    for group in 0..groups {
        let offset = group * group_bytes;
        let q_view = q_buffer.view(offset, span).map_err(Error::Cuda)?;
        let k_view = k_buffer.view(offset, span).map_err(Error::Cuda)?;
        let v_view = v_buffer.view(offset, span).map_err(Error::Cuda)?;

        let scores = kernels::scratch_buffer(ctx, head_dim * group_bytes)?;
        kernels::gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::Transpose,
            CublasTranspose::None,
            head_dim,
            head_dim,
            tokens,
            1.0,
            &q_view,
            channels as i32,
            &k_view,
            channels as i32,
            0.0,
            &scores,
            head_dim as i32,
        )?;
        let scores = scores
            .as_tensor(Shape::new(vec![head_dim, head_dim]), DType::BF16)
            .map_err(Error::Cuda)?;
        let probabilities = kernels::attention::softmax(ctx, &scores)?;
        let probabilities = DeviceBuffer::from_tensor(&probabilities).map_err(Error::Cuda)?;

        let out_view = output.view(offset, span).map_err(Error::Cuda)?;
        kernels::gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            tokens,
            head_dim,
            head_dim,
            1.0,
            &v_view,
            channels as i32,
            &probabilities,
            head_dim as i32,
            0.0,
            &out_view,
            channels as i32,
        )?;
    }

    let attended = output
        .as_tensor(Shape::new(vec![tokens, channels]), DType::BF16)
        .map_err(Error::Cuda)?;
    let projected = kernels::gemm::matmul(ctx, &attended, &weights.proj.weight)?;
    let x = kernels::fused::bias_residual_bf16(ctx, &projected, Some(&weights.proj.bias), &x)?;
    probe(&format!("{label}.attn"), &x)?;

    let x = depthwise(ctx, &weights.conv2, &x, stage.plan.height, stage.plan.width)?;
    probe(&format!("{label}.conv2"), &x)?;
    let normalized = layer_norm(ctx, &weights.norm_ffn, &x, eps)?;
    let hidden =
        kernels::gemm::bf16_bias(ctx, &normalized, &weights.fc1.weight, &weights.fc1.bias)?;
    let hidden = xvla_ops::gelu_erf_bf16(ctx, &hidden)?;
    let projected = kernels::gemm::matmul(ctx, &hidden, &weights.fc2.weight)?;
    let x = kernels::fused::bias_residual_bf16(ctx, &projected, Some(&weights.fc2.bias), &x)?;
    probe(&format!("{label}.ffn"), &x)?;
    Ok(x)
}

/// Padded windows have no crossing attention, so each window is an independent
/// attention call; FA2 batches them when available, otherwise we loop.
fn window_attention(
    ctx: &Context,
    qkv: &kernels::attention::QkvTensors,
    windows: usize,
    window: usize,
    heads: usize,
) -> Result<Tensor> {
    let head_dim = qkv.q.shape().dims()[2];
    if windows >= 2 {
        if let Some(batched) = kernels::attention::try_noncausal_batched_equal(
            ctx,
            &qkv.q,
            &qkv.k,
            &qkv.v,
            windows,
            window * window,
            heads,
            head_dim,
        )? {
            return Ok(batched);
        }
    }
    if windows == 1 {
        return kernels::attention::noncausal(ctx, &qkv.q, &qkv.k, &qkv.v, heads, head_dim);
    }

    let q = DeviceBuffer::from_tensor(&qkv.q).map_err(Error::Cuda)?;
    let k = DeviceBuffer::from_tensor(&qkv.k).map_err(Error::Cuda)?;
    let v = DeviceBuffer::from_tensor(&qkv.v).map_err(Error::Cuda)?;
    let bytes = window * window * heads * head_dim * std::mem::size_of::<half::bf16>();
    let shape = Shape::new(vec![window * window, heads, head_dim]);
    let mut parts = Vec::with_capacity(windows);
    for index in 0..windows {
        let offset = index * bytes;
        let q = q
            .view(offset, bytes)
            .and_then(|buffer| buffer.as_tensor(shape.clone(), DType::BF16))
            .map_err(Error::Cuda)?;
        let k = k
            .view(offset, bytes)
            .and_then(|buffer| buffer.as_tensor(shape.clone(), DType::BF16))
            .map_err(Error::Cuda)?;
        let v = v
            .view(offset, bytes)
            .and_then(|buffer| buffer.as_tensor(shape.clone(), DType::BF16))
            .map_err(Error::Cuda)?;
        parts.push(kernels::attention::noncausal(
            ctx, &q, &k, &v, heads, head_dim,
        )?);
    }
    let mut merged = parts.remove(0);
    for part in &parts {
        merged = kernels::elementwise::concat_rows_bf16(ctx, &merged, part)?;
    }
    Ok(merged)
}

fn layer_norm(
    ctx: &Context,
    weights: &LayerNormWeights,
    x: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    kernels::norm::layer_bf16(ctx, x, &weights.weight, &weights.bias, eps)
}

fn depthwise(
    ctx: &Context,
    weights: &DepthwiseWeights,
    x: &Tensor,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let convolved =
        xvla_ops::dwconv3x3_bf16(ctx, x, &weights.weight, Some(&weights.bias), height, width)?;
    kernels::elementwise::add(ctx, x, &convolved)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::Path;

    use apxinf_core::Tensor;
    use half::bf16;

    use crate::cuda_xvla::backend::transfers;
    use crate::cuda_xvla::config::XvlaConfig;

    use super::{encode_image, encode_image_probed, VisionWeights};

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", path.display());
        });
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn setup(case: &Path) -> (XvlaConfig, crate::cuda_xvla::backend::Context, VisionWeights, Tensor) {
        let checkpoint = std::env::var("XVLA_CKPT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/refs/checkpoints/X-VLA-Libero",
                std::env::var("HOME").expect("HOME")
            )
        });
        let config = XvlaConfig::from_checkpoint(Path::new(&checkpoint)).unwrap();
        let ctx = crate::cuda_xvla::backend::Context::new(0).unwrap();
        let weights =
            VisionWeights::from_checkpoint(&config, Path::new(&checkpoint), &ctx).unwrap();

        let side = config.profile.image_size;
        let pixels = read_f32(&case.join("pixel_values.f32"));
        assert_eq!(pixels.len(), 3 * side * side);
        let mut token_major = vec![0.0f32; side * side * 3];
        for channel in 0..3 {
            for position in 0..side * side {
                token_major[position * 3 + channel] = pixels[channel * side * side + position];
            }
        }
        let rounded: Vec<bf16> = token_major
            .iter()
            .map(|value| bf16::from_f32(*value))
            .collect();
        let host = Tensor::from_bf16(vec![side * side, 3], &rounded).unwrap();
        let input = transfers::to_cuda(&host, 0).unwrap();
        (config, ctx, weights, input)
    }

    /// GPU parity harness against an oracle dump:
    /// `XVLA_VISION_CASE=<dir>` with `pixel_values.f32` (CHW 3×224×224) and
    /// `cnn_feats.f32` ([50, 1024], one view).
    #[test]
    #[ignore = "GPU parity harness: set XVLA_VISION_CASE"]
    fn encode_image_matches_oracle() {
        let case = std::env::var("XVLA_VISION_CASE").expect("XVLA_VISION_CASE is required");
        let case = Path::new(&case);
        // BF16 storage floor: the preflight 1e-3 gate targets fp32 ports; this
        // tower runs end-to-end BF16 against an fp32 oracle (max abs ≈ 0.05 on
        // activations whose scale is ≈ 4, cosine > 0.9999).
        let tolerance: f32 = std::env::var("XVLA_VISION_TOL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.06);

        let (config, ctx, weights, input) = setup(case);
        let expected = read_f32(&case.join("cnn_feats.f32"));

        let output = encode_image(&ctx, &weights, &input, config.layer_norm_eps).unwrap();
        let host = transfers::to_cpu(&output).unwrap();
        let actual: Vec<f32> = host
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect();
        assert_eq!(actual.len(), expected.len());

        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        for (actual, expected) in actual.iter().zip(&expected) {
            let difference = (actual - expected).abs();
            max_abs = max_abs.max(difference);
            sum_abs += difference as f64;
            dot += (*actual as f64) * (*expected as f64);
            actual_norm += (*actual as f64) * (*actual as f64);
            expected_norm += (*expected as f64) * (*expected as f64);
        }
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        println!(
            "X-VLA vision parity: max_abs={max_abs:.6} mean_abs={:.6} cosine={cosine:.8} tolerance={tolerance}",
            sum_abs / actual.len() as f64
        );
        assert!(
            max_abs <= tolerance,
            "X-VLA vision parity failed: max_abs={max_abs} > {tolerance}"
        );
        assert!(cosine >= 0.9995, "X-VLA vision cosine {cosine} below 0.9995");
    }

    /// Dump every intermediate BF16 stage as raw f32 for reference diffing.
    #[test]
    #[ignore = "GPU debug harness: set XVLA_VISION_CASE and XVLA_VISION_DUMP"]
    fn dump_intermediates() {
        let case = std::env::var("XVLA_VISION_CASE").expect("XVLA_VISION_CASE is required");
        let dump = std::env::var("XVLA_VISION_DUMP").expect("XVLA_VISION_DUMP is required");
        let dump = Path::new(&dump);
        std::fs::create_dir_all(dump).unwrap();

        let (config, ctx, weights, input) = setup(Path::new(&case));
        let mut index = Vec::new();
        let mut probe = |name: &str, tensor: &Tensor| -> apxinf_core::Result<()> {
            let host = transfers::to_cpu(tensor).unwrap();
            let values: Vec<f32> = host
                .as_bf16()
                .unwrap()
                .iter()
                .map(|value| value.to_f32())
                .collect();
            let mut file = std::fs::File::create(dump.join(format!("{name}.f32"))).unwrap();
            for value in &values {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            index.push(format!(
                "{name} {:?}",
                tensor.shape().dims()
            ));
            Ok(())
        };
        encode_image_probed(&ctx, &weights, &input, config.layer_norm_eps, &mut probe).unwrap();
        std::fs::write(dump.join("index.txt"), index.join("\n")).unwrap();
        println!("wrote {} dumps to {}", index.len(), dump.display());
    }
}

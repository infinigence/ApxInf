//! Soft-prompted action transformer + linear flow (`generate_actions`).
//!
//! The head consumes `vlm_features` `[merged_tokens, hidden]` and
//! `aux_visual_inputs` `[aux_tokens, hidden]`, encodes the noisy action chunk
//! together with proprioception and the sinusoidal timestep, and runs `depth`
//! pre-norm blocks over `[action; vlm; aux; soft prompt]`: positional
//! embeddings cover only the first three groups, soft prompts are appended
//! afterwards without positions, and only the first `num_actions` rows are
//! decoded. Inference is a linear flow loop from the sampled `x1` to the
//! final chunk.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::cuda_xvla::backend::kernels;
use crate::cuda_xvla::backend::{Context, DeviceBuffer};
use crate::cuda_xvla::config::XvlaConfig;
use crate::cuda_xvla::weights::{ActionBlockWeights, ActionHeadWeights, DomainLinearWeights};

/// Runs the linear flow loop and applies the ee6d gripper sigmoid.
pub fn generate_actions(
    ctx: &Context,
    weights: &ActionHeadWeights,
    config: &XvlaConfig,
    vlm_features: &Tensor,
    aux_visual_inputs: &Tensor,
    proprio: &Tensor,
    domain: usize,
    x1: &Tensor,
) -> Result<Tensor> {
    let action = denoise_actions_probed(
        ctx,
        weights,
        config,
        vlm_features,
        aux_visual_inputs,
        proprio,
        domain,
        x1,
        &mut |_, _| Ok(()),
    )?;
    kernels::xvla_ops::sigmoid_columns_bf16(ctx, &action, config.action_space.gripper_indices)
}

/// Linear flow loop, returning the raw (pre-postprocess) final chunk.
///
/// `probe` receives `("step.{k}", raw_action)` after every transformer call,
/// `k = 0` for `t = 1.0`.
pub fn denoise_actions_probed(
    ctx: &Context,
    weights: &ActionHeadWeights,
    config: &XvlaConfig,
    vlm_features: &Tensor,
    aux_visual_inputs: &Tensor,
    proprio: &Tensor,
    domain: usize,
    x1: &Tensor,
    probe: &mut dyn FnMut(&str, &Tensor) -> Result<()>,
) -> Result<Tensor> {
    let head = &config.head;
    let hidden = head.hidden_size;
    let num_actions = config.profile.action_horizon;
    let steps = config.profile.num_flow_steps;
    let dim_action = config.action_space.dim_action;

    let image_tokens = config.vision.feature_sides(config.profile.image_size)[3].pow(2) + 1;
    let merged_tokens = image_tokens + config.profile.tokens;
    let aux_tokens = config.profile.tokens * (config.profile.num_views - 1);
    let expected_positions = num_actions + merged_tokens + aux_tokens;

    if vlm_features.dtype() != DType::BF16
        || vlm_features.shape().dims() != [merged_tokens, hidden]
        || aux_visual_inputs.dtype() != DType::BF16
        || aux_visual_inputs.shape().dims() != [aux_tokens, hidden]
    {
        return Err(Error::Other(format!(
            "X-VLA action head expects BF16 visual streams [{merged_tokens}, {hidden}] and [{aux_tokens}, {hidden}], got {} {:?} and {} {:?}",
            vlm_features.dtype(),
            vlm_features.shape().dims(),
            aux_visual_inputs.dtype(),
            aux_visual_inputs.shape().dims()
        )));
    }
    if x1.dtype() != DType::BF16 || x1.shape().dims() != [num_actions, dim_action] {
        return Err(Error::Other(format!(
            "X-VLA action head expects a BF16 noise chunk [{num_actions}, {dim_action}], got {} {:?}",
            x1.dtype(),
            x1.shape().dims()
        )));
    }
    if proprio.dtype() != DType::BF16 || proprio.shape().dims() != [config.action_space.dim_proprio]
    {
        return Err(Error::Other(format!(
            "X-VLA action head expects BF16 proprioception [{}], got {} {:?}",
            config.action_space.dim_proprio,
            proprio.dtype(),
            proprio.shape().dims()
        )));
    }
    if weights.pos_emb.shape().dims() != [expected_positions, hidden] {
        return Err(Error::Other(format!(
            "X-VLA action head weights carry {:?} positions, expected [{expected_positions}, {hidden}]",
            weights.pos_emb.shape().dims()
        )));
    }
    if weights.time_tokens.shape().dims() != [steps, head.dim_time] {
        return Err(Error::Other(format!(
            "X-VLA action head weights carry {:?} timestep rows, expected [{steps}, {}]",
            weights.time_tokens.shape().dims(),
            head.dim_time
        )));
    }

    let mut action = zeros_bf16(ctx, num_actions, dim_action)?;
    for step in (1..=steps).rev() {
        let t = step as f32 / steps as f32;
        let scaled_noise = kernels::elementwise::scale(ctx, x1, t)?;
        let scaled_action = kernels::elementwise::scale(ctx, &action, 1.0 - t)?;
        let noisy = kernels::elementwise::add(ctx, &scaled_noise, &scaled_action)?;
        let time_row =
            rows_slice(&weights.time_tokens, step - 1, 1)?.reshape(vec![head.dim_time])?;
        let tokens = kernels::xvla_ops::pack_action_tokens_bf16(
            ctx,
            &noisy,
            proprio,
            &time_row,
            config.action_space.gripper_indices,
        )?;
        action = forward(
            ctx,
            weights,
            config,
            &tokens,
            vlm_features,
            aux_visual_inputs,
            domain,
        )?;
        probe(&format!("step.{}", steps - step), &action)?;
    }
    Ok(action)
}

/// One `SoftPromptedTransformer.forward` call for a packed action block.
fn forward(
    ctx: &Context,
    weights: &ActionHeadWeights,
    config: &XvlaConfig,
    action_tokens: &Tensor,
    vlm_features: &Tensor,
    aux_visual_inputs: &Tensor,
    domain: usize,
) -> Result<Tensor> {
    let head = &config.head;
    let hidden = head.hidden_size;
    let num_actions = config.profile.action_horizon;

    let mut sequence = domain_linear(ctx, &weights.action_encoder, action_tokens, domain)?;
    let vlm = kernels::gemm::bf16_bias(
        ctx,
        vlm_features,
        &weights.vlm_proj.weight,
        &weights.vlm_proj.bias,
    )?;
    let aux = kernels::gemm::bf16_bias(
        ctx,
        aux_visual_inputs,
        &weights.aux_visual_proj.weight,
        &weights.aux_visual_proj.bias,
    )?;
    sequence = kernels::elementwise::concat_rows_bf16(ctx, &sequence, &vlm)?;
    sequence = kernels::elementwise::concat_rows_bf16(ctx, &sequence, &aux)?;
    sequence = kernels::elementwise::add(ctx, &sequence, &weights.pos_emb)?;

    let soft_prompts = rows_slice(
        &weights.soft_prompt,
        domain * head.len_soft_prompts,
        head.len_soft_prompts,
    )?;
    sequence = kernels::elementwise::concat_rows_bf16(ctx, &sequence, &soft_prompts)?;

    let head_dim = hidden / head.num_heads;
    for block in &weights.blocks {
        sequence = action_block(
            ctx,
            block,
            &sequence,
            head.num_heads,
            head_dim,
            config.layer_norm_eps,
        )?;
    }

    let pooled = rows_slice(&sequence, 0, num_actions)?;
    let normalized = kernels::norm::layer_bf16(
        ctx,
        &pooled,
        &weights.norm.weight,
        &weights.norm.bias,
        config.layer_norm_eps,
    )?;
    domain_linear(ctx, &weights.action_decoder, &normalized, domain)
}

fn action_block(
    ctx: &Context,
    weights: &ActionBlockWeights,
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let normalized =
        kernels::norm::layer_bf16(ctx, x, &weights.norm1.weight, &weights.norm1.bias, eps)?;
    let qkv = kernels::gemm::bf16_bias(ctx, &normalized, &weights.qkv.weight, &weights.qkv.bias)?;
    let split = kernels::attention::split_qkv_bias_bf16(ctx, &qkv, None, heads, head_dim)?;
    let attended =
        kernels::attention::noncausal(ctx, &split.q, &split.k, &split.v, heads, head_dim)?;
    let projected = kernels::gemm::matmul(ctx, &attended, &weights.proj.weight)?;
    let residual =
        kernels::fused::bias_residual_bf16(ctx, &projected, Some(&weights.proj.bias), x)?;

    let normalized = kernels::norm::layer_bf16(
        ctx,
        &residual,
        &weights.norm2.weight,
        &weights.norm2.bias,
        eps,
    )?;
    let hidden = kernels::gemm::bf16_bias_gelu_tanh(
        ctx,
        &normalized,
        &weights.fc1.weight,
        &weights.fc1.bias,
    )?;
    let projected = kernels::gemm::matmul(ctx, &hidden, &weights.fc2.weight)?;
    kernels::fused::bias_residual_bf16(ctx, &projected, Some(&weights.fc2.bias), &residual)
}

fn domain_linear(
    ctx: &Context,
    weights: &DomainLinearWeights,
    x: &Tensor,
    domain: usize,
) -> Result<Tensor> {
    let weight = weights.weight_view(domain)?;
    let bias = weights.bias_view(domain)?;
    kernels::gemm::bf16_bias(ctx, x, &weight, &bias)
}

fn rows_slice(tensor: &Tensor, offset_rows: usize, rows: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 || rows == 0 || offset_rows + rows > dims[0] {
        return Err(Error::Other(format!(
            "X-VLA action head row slice {offset_rows}+{rows} is out of range for {dims:?}"
        )));
    }
    let row_bytes = dims[1] * DType::BF16.size_in_bytes();
    let buffer = DeviceBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let view = buffer
        .view(offset_rows * row_bytes, rows * row_bytes)
        .map_err(Error::Cuda)?;
    view.as_tensor(Shape::new(vec![rows, dims[1]]), DType::BF16)
        .map_err(Error::Cuda)
}

fn zeros_bf16(ctx: &Context, rows: usize, cols: usize) -> Result<Tensor> {
    let bytes = rows
        .checked_mul(cols)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| Error::Other("X-VLA action head: zero block size overflow".into()))?;
    let buffer = kernels::scratch_buffer_zeroed(ctx, bytes)?;
    buffer
        .as_tensor(Shape::new(vec![rows, cols]), DType::BF16)
        .map_err(Error::Cuda)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use apxinf_core::Tensor;
    use half::bf16;

    use crate::cuda_xvla::backend::transfers;
    use crate::cuda_xvla::config::XvlaConfig;
    use crate::cuda_xvla::model::text::encode_vlm;
    use crate::cuda_xvla::model::vision::encode_image;
    use crate::cuda_xvla::weights::{ActionHeadWeights, VlmWeights};

    use super::{denoise_actions_probed, generate_actions};

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", path.display());
        });
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_i64(path: &Path) -> Vec<i64> {
        let bytes = std::fs::read(path).unwrap_or_else(|error| {
            panic!("failed to read {}: {error}", path.display());
        });
        bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn upload(ctx: &crate::cuda_xvla::backend::Context, values: &[f32], dims: Vec<usize>) -> Tensor {
        let rounded: Vec<bf16> = values.iter().map(|value| bf16::from_f32(*value)).collect();
        let host = Tensor::from_bf16(dims, &rounded).unwrap();
        transfers::to_cuda(&host, ctx.device_id()).unwrap()
    }

    fn download(tensor: &Tensor) -> Vec<f32> {
        let host = transfers::to_cpu(tensor).unwrap();
        host.as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect()
    }

    /// Returns `(max_abs, mean_abs, cosine)` without asserting.
    fn stats(label: &str, actual: &[f32], expected: &[f32]) -> (f32, f64, f64) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        for (actual, expected) in actual.iter().zip(expected) {
            let difference = (actual - expected).abs();
            max_abs = max_abs.max(difference);
            sum_abs += difference as f64;
            dot += (*actual as f64) * (*expected as f64);
            actual_norm += (*actual as f64) * (*actual as f64);
            expected_norm += (*expected as f64) * (*expected as f64);
        }
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        let mean_abs = sum_abs / actual.len() as f64;
        println!(
            "X-VLA {label} parity: max_abs={max_abs:.6} mean_abs={mean_abs:.6} cosine={cosine:.8}"
        );
        (max_abs, mean_abs, cosine)
    }

    fn compare(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        let (max_abs, _, cosine) = stats(label, actual, expected);
        assert!(
            max_abs <= tolerance,
            "X-VLA {label} parity failed: max_abs={max_abs} > {tolerance}"
        );
        assert!(cosine >= 0.999, "X-VLA {label} cosine {cosine} below 0.999");
    }

    fn token_major(pixels: &[f32], side: usize) -> Vec<f32> {
        let mut tokens = vec![0.0f32; side * side * 3];
        for channel in 0..3 {
            for position in 0..side * side {
                tokens[position * 3 + channel] = pixels[channel * side * side + position];
            }
        }
        tokens
    }

    /// Full pipeline parity: pixels → DaViT → BART merge → linear flow.
    /// `XVLA_ACTION_CASE=<dir>` must hold the vision/text inputs as well.
    #[test]
    #[ignore = "GPU parity harness: set XVLA_ACTION_CASE"]
    fn pipeline_matches_oracle() {
        let case = std::env::var("XVLA_ACTION_CASE").expect("XVLA_ACTION_CASE is required");
        let case = Path::new(&case);
        let tolerance: f32 = std::env::var("XVLA_PIPELINE_TOL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.01);

        let checkpoint = std::env::var("XVLA_CKPT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/refs/checkpoints/X-VLA-Libero",
                std::env::var("HOME").expect("HOME")
            )
        });
        let config = XvlaConfig::from_checkpoint(Path::new(&checkpoint)).unwrap();
        let ctx = crate::cuda_xvla::backend::Context::new(0).unwrap();
        let vlm_weights =
            VlmWeights::from_checkpoint(&config, Path::new(&checkpoint), &ctx).unwrap();
        let action_weights =
            ActionHeadWeights::from_checkpoint(&config, Path::new(&checkpoint), &ctx).unwrap();

        let side = config.profile.image_size;
        let mask = std::fs::read(case.join("view_mask.u8")).unwrap();
        let mut features = Vec::new();
        let mut valid = 0usize;
        for present in mask {
            if present != 0 {
                let pixels = read_f32(&case.join(format!("pixel_{valid}.f32")));
                let tokens = upload(&ctx, &token_major(&pixels, side), vec![side * side, 3]);
                features.push(Some(
                    encode_image(&ctx, &vlm_weights.vision, &tokens, config.layer_norm_eps)
                        .unwrap(),
                ));
                valid += 1;
            } else {
                features.push(None);
            }
        }
        let refs: Vec<Option<&Tensor>> = features.iter().map(Option::as_ref).collect();
        let ids: Vec<u32> = read_i64(&case.join("input_ids.i64"))
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        let output = encode_vlm(&ctx, &vlm_weights, &refs, &ids, config.layer_norm_eps).unwrap();

        let proprio = upload(
            &ctx,
            &read_f32(&case.join("proprio.f32")),
            vec![config.action_space.dim_proprio],
        );
        let x1 = upload(
            &ctx,
            &read_f32(&case.join("x1.f32")),
            vec![
                config.profile.action_horizon,
                config.action_space.dim_action,
            ],
        );
        let domain = read_i64(&case.join("domain_id.i64"))[0] as usize;
        let post = generate_actions(
            &ctx,
            &action_weights,
            &config,
            &output.vlm_features,
            &output.aux_visual_inputs,
            &proprio,
            domain,
            &x1,
        )
        .unwrap();
        compare(
            "pipeline_final_action",
            &download(&post),
            &read_f32(&case.join("final_action.f32")),
            tolerance,
        );
    }

    /// GPU parity harness against the oracle action dump:
    /// `XVLA_ACTION_CASE=<dir>` with `vlm_features.f32`, `aux_visual_inputs.f32`,
    /// `proprio.f32`, `x1.f32`, `step_actions.f32`, `final_action.f32`,
    /// `domain_id.i64`.
    #[test]
    #[ignore = "GPU parity harness: set XVLA_ACTION_CASE"]
    fn denoise_actions_matches_oracle() {
        let case = std::env::var("XVLA_ACTION_CASE").expect("XVLA_ACTION_CASE is required");
        let case = Path::new(&case);
        // The action gate: final chunk max-abs vs the fp32 oracle.
        let tolerance: f32 = std::env::var("XVLA_ACTION_TOL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.01);

        let checkpoint = std::env::var("XVLA_CKPT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/refs/checkpoints/X-VLA-Libero",
                std::env::var("HOME").expect("HOME")
            )
        });
        let config = XvlaConfig::from_checkpoint(Path::new(&checkpoint)).unwrap();
        let ctx = crate::cuda_xvla::backend::Context::new(0).unwrap();
        let weights =
            ActionHeadWeights::from_checkpoint(&config, Path::new(&checkpoint), &ctx).unwrap();

        let hidden = config.head.hidden_size;
        let image_tokens = config.vision.feature_sides(config.profile.image_size)[3].pow(2) + 1;
        let merged_tokens = image_tokens + config.profile.tokens;
        let aux_tokens = config.profile.tokens * (config.profile.num_views - 1);
        let num_actions = config.profile.action_horizon;
        let dim_action = config.action_space.dim_action;

        let vlm = upload(
            &ctx,
            &read_f32(&case.join("vlm_features.f32")),
            vec![merged_tokens, hidden],
        );
        let aux = upload(
            &ctx,
            &read_f32(&case.join("aux_visual_inputs.f32")),
            vec![aux_tokens, hidden],
        );
        let proprio = upload(
            &ctx,
            &read_f32(&case.join("proprio.f32")),
            vec![config.action_space.dim_proprio],
        );
        let x1 = upload(
            &ctx,
            &read_f32(&case.join("x1.f32")),
            vec![num_actions, dim_action],
        );
        let domain = read_i64(&case.join("domain_id.i64"))[0] as usize;

        let steps = config.profile.num_flow_steps;
        let step_reference = read_f32(&case.join("step_actions.f32"));
        let step_len = num_actions * dim_action;

        let mut probe = |name: &str, action: &Tensor| -> apxinf_core::Result<()> {
            let index: usize = name
                .strip_prefix("step.")
                .and_then(|value| value.parse().ok())
                .unwrap();
            let values = download(action);
            stats(
                name,
                &values,
                &step_reference[index * step_len..(index + 1) * step_len],
            );
            Ok(())
        };
        let final_raw = denoise_actions_probed(
            &ctx, &weights, &config, &vlm, &aux, &proprio, domain, &x1, &mut probe,
        )
        .unwrap();
        compare(
            "final_raw",
            &download(&final_raw),
            &step_reference[(steps - 1) * step_len..],
            tolerance,
        );

        let post =
            generate_actions(&ctx, &weights, &config, &vlm, &aux, &proprio, domain, &x1).unwrap();
        compare(
            "final_action",
            &download(&post),
            &read_f32(&case.join("final_action.f32")),
            tolerance,
        );
    }
}

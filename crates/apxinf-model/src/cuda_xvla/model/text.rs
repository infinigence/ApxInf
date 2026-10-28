//! Florence-2 BART text encoder + view merge for X-VLA (`forward_vlm`).
//!
//! The merged decoder input is `[view0 features (50); token embeddings (50)]`
//! plus absolute positions (offset 2), embedded-LayerNorm, then 12 post-norm
//! BART encoder layers. Remaining views become `aux_visual_inputs`, with
//! invalid views filled by zero rows.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::cuda_xvla::backend::kernels;
use crate::cuda_xvla::backend::kernels::xvla_ops;
use crate::cuda_xvla::backend::Context;
use crate::cuda_xvla::weights::{TextLayerWeights, VlmWeights};

type Probe<'a> = &'a mut dyn FnMut(&str, &Tensor) -> Result<()>;

/// Encoder output consumed by the action head.
pub struct VlmOutput {
    /// Merged encoder states `[image_tokens + tokens, d_model]`.
    pub vlm_features: Tensor,
    /// Flattened auxiliary views `[(views - 1) * image_tokens, d_model]`.
    pub aux_visual_inputs: Tensor,
}

/// Build `vlm_features` and `aux_visual_inputs` exactly as `forward_vlm`.
pub fn encode_vlm(
    ctx: &Context,
    weights: &VlmWeights,
    view_features: &[Option<&Tensor>],
    input_ids: &[u32],
    eps: f32,
) -> Result<VlmOutput> {
    encode_vlm_probed(ctx, weights, view_features, input_ids, eps, &mut |_, _| {
        Ok(())
    })
}

pub(crate) fn encode_vlm_probed(
    ctx: &Context,
    weights: &VlmWeights,
    view_features: &[Option<&Tensor>],
    input_ids: &[u32],
    eps: f32,
    probe: Probe<'_>,
) -> Result<VlmOutput> {
    let table_rows = weights.text.embedding.shape().dims()[0];
    let plan = xvla_ops::prepare_row_gather(ctx, input_ids, table_rows)?;
    encode_vlm_with_plan_probed(
        ctx,
        weights,
        view_features,
        &plan,
        input_ids.len(),
        eps,
        probe,
    )
}

/// Encoder pass over a pre-uploaded instruction row table.
///
/// CUDA-graph plans refresh [`xvla_ops::RowGatherPlan`] outside capture and
/// replay the same device index buffer, so no host copy happens in a graph.
pub(crate) fn encode_vlm_with_plan(
    ctx: &Context,
    weights: &VlmWeights,
    view_features: &[Option<&Tensor>],
    plan: &xvla_ops::RowGatherPlan,
    token_count: usize,
    eps: f32,
) -> Result<VlmOutput> {
    encode_vlm_with_plan_probed(
        ctx,
        weights,
        view_features,
        plan,
        token_count,
        eps,
        &mut |_, _| Ok(()),
    )
}

pub(crate) fn encode_vlm_with_plan_probed(
    ctx: &Context,
    weights: &VlmWeights,
    view_features: &[Option<&Tensor>],
    plan: &xvla_ops::RowGatherPlan,
    token_count: usize,
    eps: f32,
    probe: Probe<'_>,
) -> Result<VlmOutput> {
    if token_count == 0 || plan.rows() != token_count {
        return Err(Error::Other(format!(
            "X-VLA text: instruction plan carries {} rows for {token_count} tokens",
            plan.rows()
        )));
    }
    let text = &weights.text;
    let d_model = text.embedding.shape().dims()[1];
    let image_tokens = text.positions.shape().dims()[0]
        .checked_sub(token_count)
        .ok_or_else(|| Error::Other("X-VLA text: position table shorter than tokens".into()))?;
    if image_tokens == 0 {
        return Err(Error::Other(
            "X-VLA text: empty token or image sequence".into(),
        ));
    }
    if view_features.is_empty() {
        return Err(Error::Other(
            "X-VLA text: at least one image view is required".into(),
        ));
    }
    let primary = view_features[0]
        .ok_or_else(|| Error::Other("X-VLA text: first image view must carry features".into()))?;
    for (index, view) in view_features.iter().enumerate() {
        if let Some(view) = view {
            if view.dtype() != DType::BF16 || view.shape().dims() != [image_tokens, d_model] {
                return Err(Error::Other(format!(
                    "X-VLA text: view {index} must be BF16 [{image_tokens}, {d_model}], got {} {:?}",
                    view.dtype(),
                    view.shape().dims()
                )));
            }
        }
    }

    let embedded = xvla_ops::gather_rows_bf16(ctx, &text.embedding, plan)?;
    let merged = kernels::elementwise::concat_rows_bf16(ctx, primary, &embedded)?;
    let positioned = kernels::elementwise::add(ctx, &merged, &text.positions)?;
    let mut x = kernels::norm::layer_bf16(
        ctx,
        &positioned,
        &text.embed_norm.weight,
        &text.embed_norm.bias,
        eps,
    )?;
    probe("embed", &x)?;

    for (index, layer) in text.layers.iter().enumerate() {
        x = text_layer(ctx, layer, &x, text.heads, text.head_dim, eps)?;
        probe(&format!("layer.{index}"), &x)?;
    }

    if view_features.len() < 2 {
        return Err(Error::Other(
            "X-VLA text: expected an auxiliary view stream".into(),
        ));
    }
    let zeros = zeros_bf16(ctx, image_tokens, d_model)?;
    let block = |view: &Option<&Tensor>| -> Result<Tensor> {
        Ok(match view {
            Some(view) => (*view).clone(),
            None => zeros.clone(),
        })
    };
    let mut aux = block(&view_features[1])?;
    for view in &view_features[2..] {
        aux = kernels::elementwise::concat_rows_bf16(ctx, &aux, &block(view)?)?;
    }
    probe("aux", &aux)?;

    Ok(VlmOutput {
        vlm_features: x,
        aux_visual_inputs: aux,
    })
}

fn text_layer(
    ctx: &Context,
    weights: &TextLayerWeights,
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let rows = x.shape().dims()[0];
    let attended = attention(ctx, weights, x, rows, heads, head_dim)?;
    let projected = kernels::gemm::matmul(ctx, &attended, &weights.out_proj.weight)?;
    let normalized = kernels::fused::bias_residual_layer_bf16(
        ctx,
        &projected,
        Some(&weights.out_proj.bias),
        x,
        &weights.attn_norm.weight,
        &weights.attn_norm.bias,
        eps,
    )?
    .normalized;

    let hidden =
        kernels::gemm::bf16_bias(ctx, &normalized, &weights.fc1.weight, &weights.fc1.bias)?;
    let hidden = kernels::xvla_ops::gelu_erf_bf16(ctx, &hidden)?;
    let projected = kernels::gemm::matmul(ctx, &hidden, &weights.fc2.weight)?;
    Ok(kernels::fused::bias_residual_layer_bf16(
        ctx,
        &projected,
        Some(&weights.fc2.bias),
        &normalized,
        &weights.ffn_norm.weight,
        &weights.ffn_norm.bias,
        eps,
    )?
    .normalized)
}

fn attention(
    ctx: &Context,
    weights: &TextLayerWeights,
    x: &Tensor,
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let q = kernels::gemm::bf16_bias(ctx, x, &weights.q_proj.weight, &weights.q_proj.bias)?;
    let k = kernels::gemm::bf16_bias(ctx, x, &weights.k_proj.weight, &weights.k_proj.bias)?;
    let v = kernels::gemm::bf16_bias(ctx, x, &weights.v_proj.weight, &weights.v_proj.bias)?;
    let split = |tensor: &Tensor| tensor.reshape(vec![rows, heads, head_dim]);
    kernels::attention::noncausal(ctx, &split(&q)?, &split(&k)?, &split(&v)?, heads, head_dim)
}

fn zeros_bf16(ctx: &Context, rows: usize, cols: usize) -> Result<Tensor> {
    let bytes = rows
        .checked_mul(cols)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| Error::Other("X-VLA text: zero block size overflow".into()))?;
    let buffer = kernels::scratch_buffer_zeroed(ctx, bytes)?;
    buffer
        .as_tensor(Shape::new(vec![rows, cols]), DType::BF16)
        .map_err(Error::Cuda)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::Path;

    use apxinf_core::Tensor;
    use half::bf16;

    use crate::cuda_xvla::backend::transfers;
    use crate::cuda_xvla::config::XvlaConfig;

    use super::{encode_vlm, encode_vlm_probed, VlmWeights};

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

    fn setup(
        case: &Path,
    ) -> (
        XvlaConfig,
        crate::cuda_xvla::backend::Context,
        VlmWeights,
        Vec<Option<Tensor>>,
        Vec<u32>,
    ) {
        let checkpoint = std::env::var("XVLA_CKPT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/refs/checkpoints/X-VLA-Libero",
                std::env::var("HOME").expect("HOME")
            )
        });
        let config = XvlaConfig::from_checkpoint(Path::new(&checkpoint)).unwrap();
        let ctx = crate::cuda_xvla::backend::Context::new(0).unwrap();
        let weights = VlmWeights::from_checkpoint(&config, Path::new(&checkpoint), &ctx).unwrap();

        let mask = std::fs::read(case.join("view_mask.u8")).unwrap();
        let d_model = config.text.d_model;
        let tokens = config.vision.feature_sides(config.profile.image_size)[3].pow(2) + 1;
        let mut features = Vec::new();
        let mut valid = 0usize;
        for present in mask {
            if present != 0 {
                let values = read_f32(&case.join(format!("cnn_feats_{valid}.f32")));
                assert_eq!(values.len(), tokens * d_model);
                features.push(Some(upload(&ctx, &values, vec![tokens, d_model])));
                valid += 1;
            } else {
                features.push(None);
            }
        }
        let ids = read_i64(&case.join("input_ids.i64"))
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        (config, ctx, weights, features, ids)
    }

    fn compare(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
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
        println!(
            "X-VLA {label} parity: max_abs={max_abs:.6} mean_abs={:.6} cosine={cosine:.8} tolerance={tolerance}",
            sum_abs / actual.len() as f64
        );
        assert!(
            max_abs <= tolerance,
            "X-VLA {label} parity failed: max_abs={max_abs} > {tolerance}"
        );
        assert!(cosine >= 0.999, "X-VLA {label} cosine {cosine} below 0.999");
    }

    /// GPU parity harness against an oracle `forward_vlm` dump:
    /// `XVLA_VLM_CASE=<dir>` with `input_ids.i64`, `view_mask.u8`,
    /// `cnn_feats_<n>.f32`, `vlm_features.f32`, `aux_visual_inputs.f32`.
    #[test]
    #[ignore = "GPU parity harness: set XVLA_VLM_CASE"]
    fn encode_vlm_matches_oracle() {
        let case = std::env::var("XVLA_VLM_CASE").expect("XVLA_VLM_CASE is required");
        let case = Path::new(&case);
        // BF16 storage floor over 12 post-norm layers: the fp32 oracle shows
        // mean abs ≈ 0.01 and max abs ≈ 0.27 on activations peaking at ≈ 15
        // (cosine > 0.99995). The action-level gate decides whether the full
        // pipeline needs an fp32 text fallback.
        let tolerance: f32 = std::env::var("XVLA_VLM_TOL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.3);
        let aux_tolerance: f32 = std::env::var("XVLA_VLM_AUX_TOL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.02);

        let (config, ctx, weights, features, ids) = setup(case);
        let refs: Vec<Option<&Tensor>> = features.iter().map(Option::as_ref).collect();
        let output = encode_vlm(&ctx, &weights, &refs, &ids, config.layer_norm_eps).unwrap();

        let actual = transfers::to_cpu(&output.vlm_features).unwrap();
        let actual: Vec<f32> = actual
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect();
        compare(
            "vlm_features",
            &actual,
            &read_f32(&case.join("vlm_features.f32")),
            tolerance,
        );

        let actual_aux = transfers::to_cpu(&output.aux_visual_inputs).unwrap();
        let actual_aux: Vec<f32> = actual_aux
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect();
        compare(
            "aux_visual_inputs",
            &actual_aux,
            &read_f32(&case.join("aux_visual_inputs.f32")),
            aux_tolerance,
        );
    }

    /// Dump encoder intermediates as raw f32 for reference diffing.
    #[test]
    #[ignore = "GPU debug harness: set XVLA_VLM_CASE and XVLA_VLM_DUMP"]
    fn dump_vlm_intermediates() {
        let case = std::env::var("XVLA_VLM_CASE").expect("XVLA_VLM_CASE is required");
        let dump = std::env::var("XVLA_VLM_DUMP").expect("XVLA_VLM_DUMP is required");
        let dump = Path::new(&dump);
        std::fs::create_dir_all(dump).unwrap();

        let (config, ctx, weights, features, ids) = setup(Path::new(&case));
        let refs: Vec<Option<&Tensor>> = features.iter().map(Option::as_ref).collect();
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
            index.push(format!("{name} {:?}", tensor.shape().dims()));
            Ok(())
        };
        encode_vlm_probed(
            &ctx,
            &weights,
            &refs,
            &ids,
            config.layer_norm_eps,
            &mut probe,
        )
        .unwrap();
        std::fs::write(dump.join("index.txt"), index.join("\n")).unwrap();
        println!("wrote {} dumps to {}", index.len(), dump.display());
    }
}

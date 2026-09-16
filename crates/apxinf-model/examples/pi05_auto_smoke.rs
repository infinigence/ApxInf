//! Public-entry smoke test for AutoModel -> LoadedModel::Vla -> prepare/run.

use std::path::{Path, PathBuf};

use apxinf_core::{standard_normal_f32, DType, Device, RngKey, Tensor};
use apxinf_model::{
    AutoModel, ExecutionMode, ExecutionPolicy, ImageLayout, LoadOptions, Observation, Pi05Config,
    PreparationStatus, VisionObservation, VlaRequest,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.len() < 2 || arguments.len() > 4 {
        return Err(format!(
            "usage: {} <checkpoint-or-directory> [token-count=21] [bf16|fp8_static|int8_dynamic]",
            arguments
                .first()
                .map(String::as_str)
                .unwrap_or("pi05_auto_smoke")
        )
        .into());
    }
    let checkpoint = PathBuf::from(&arguments[1]);
    let token_count = arguments
        .get(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(21usize);
    let root = if checkpoint.is_dir() {
        checkpoint.as_path()
    } else {
        checkpoint.parent().unwrap_or_else(|| Path::new("."))
    };
    let config_path = root.join("config.json");
    let config = if config_path.is_file() {
        Pi05Config::from_json_file(&config_path)?
    } else {
        Pi05Config::default()
    };

    let options = LoadOptions {
        model_name: Some("pi05".to_owned()),
        compute_variant: Some(
            arguments
                .get(3)
                .map(String::as_str)
                .unwrap_or("bf16")
                .parse::<apxinf_model::pi05::ComputeVariant>()?
                .as_str()
                .into(),
        ),
        ..LoadOptions::default()
    };
    let model = AutoModel::load_model(Device::Cuda(0), &checkpoint, &options)?;
    let patch_rows = config.num_views * config.patches_per_view();
    let patch_width = 3 * config.patch_size * config.patch_size;
    let observation = Observation {
        vision: VisionObservation::Patches(Tensor::zeros(
            vec![patch_rows, patch_width],
            DType::F32,
        )),
        token_ids: vec![0; token_count],
        state: None,
        action_mask: None,
    };
    let noise = Tensor::zeros(vec![config.action_horizon, config.action_dim], DType::F32);
    let request = VlaRequest::provided(&observation, &noise);
    let spec = observation.inference_spec();
    let eager = model.prepare_with_policy(&spec, ExecutionPolicy::Eager)?;
    assert_eq!(
        eager.status(),
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: None,
        }
    );
    let eager_values =
        apxinf_cuda::transfers::to_cpu(eager.run(&request)?.tensor())?.to_f32_vec()?;
    drop(eager);
    let prepared = model.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
    assert_eq!(
        prepared.status(),
        PreparationStatus::Ready {
            mode: ExecutionMode::Graph,
            fallback_reason: None,
        }
    );
    // Populate the implicit cache so eviction exercises real resource release,
    // while a separately owned explicit plan remains usable.
    drop(model.infer(&request)?);
    assert!(matches!(model.vla()?.execution_mode(), "graph" | "eager"));
    model.clear_prepared()?;
    assert_eq!(model.vla()?.execution_mode(), "unprepared");
    let mut invalid = observation.clone();
    invalid.token_ids.push(0);
    assert!(prepared
        .run(&VlaRequest::provided(&invalid, &noise))
        .is_err());
    let prepared_action = prepared.run(&request)?;
    let graph_values = apxinf_cuda::transfers::to_cpu(prepared_action.tensor())?.to_f32_vec()?;
    let eager_graph_max_abs = eager_values
        .iter()
        .zip(&graph_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if eager_values.len() != graph_values.len()
        || eager_values
            .iter()
            .chain(&graph_values)
            .any(|v| !v.is_finite())
        || eager_graph_max_abs > 0.01
    {
        return Err(format!("explicit eager/graph mismatch: {eager_graph_max_abs}").into());
    }
    drop(prepared);
    let inferred_action = model.infer(&request)?;
    let cached_action = model.infer(&request)?;

    // Exercise the new VLA latent policy through the public API. Reusing a key
    // must replay exactly; a CPU-generated latent using the same Philox stream
    // must remain numerically equivalent to direct device generation.
    let rng = RngKey::new(0x1234_5678_9abc_def0, 7, 3);
    let generated_request = VlaRequest::generated(&observation, rng);
    let generated_first = model.infer_host_f32(&generated_request)?;
    let generated_second = model.infer_host_f32(&generated_request)?;
    if generated_first != generated_second {
        return Err("seeded PI0.5 inference is not exactly reproducible".into());
    }
    let cpu_noise = Tensor::from_f32(
        vec![config.action_horizon, config.action_dim],
        &standard_normal_f32(config.action_horizon * config.action_dim, rng),
    )?;
    let cpu_noise_request = VlaRequest::provided(&observation, &cpu_noise);
    let provided_rng_action = model.infer_host_f32(&cpu_noise_request)?;
    let dot = generated_first
        .iter()
        .zip(&provided_rng_action)
        .map(|(left, right)| left * right)
        .sum::<f32>();
    let left_norm = generated_first
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    let right_norm = provided_rng_action
        .iter()
        .map(|value| value * value)
        .sum::<f32>();
    let cosine = dot / (left_norm * right_norm).sqrt();
    let max_abs = generated_first
        .iter()
        .zip(&provided_rng_action)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    if !cosine.is_finite() || cosine < 0.9999 {
        return Err(format!(
            "device-generated latent diverges from CPU Philox reference: cosine={cosine}, max_abs={max_abs}"
        )
        .into());
    }
    let different_request = VlaRequest::generated(
        &observation,
        RngKey {
            sequence: rng.sequence + 1,
            ..rng
        },
    );
    let different_action = model.infer_host_f32(&different_request)?;
    if generated_first == different_action {
        return Err("distinct PI0.5 RNG streams produced identical actions".into());
    }
    // Cover canonical raw RGB -> prepared Session -> device Action as well as
    // the preprocessed patch route above, without reloading model weights.
    model.clear_prepared()?;
    let rgb_observation = Observation {
        vision: VisionObservation::RgbU8 {
            bytes: vec![0; config.num_views * config.image_size * config.image_size * 3],
            layout: ImageLayout::Nhwc,
        },
        token_ids: observation.token_ids.clone(),
        state: None,
        action_mask: None,
    };
    let rgb_request = VlaRequest::provided(&rgb_observation, &noise);
    let rgb_spec = rgb_observation.inference_spec();
    let rgb_eager = model.prepare_with_policy(&rgb_spec, ExecutionPolicy::Eager)?;
    let rgb_eager_values =
        apxinf_cuda::transfers::to_cpu(rgb_eager.run(&rgb_request)?.tensor())?.to_f32_vec()?;
    drop(rgb_eager);
    let rgb_graph = model.prepare_with_policy(&rgb_spec, ExecutionPolicy::RequireGraph)?;
    let rgb_graph_values =
        apxinf_cuda::transfers::to_cpu(rgb_graph.run(&rgb_request)?.tensor())?.to_f32_vec()?;
    let rgb_max_abs = rgb_eager_values
        .iter()
        .zip(&rgb_graph_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if rgb_eager_values.len() != rgb_graph_values.len()
        || rgb_eager_values
            .iter()
            .chain(&rgb_graph_values)
            .any(|v| !v.is_finite())
        || rgb_max_abs > 0.01
    {
        return Err(format!("raw RGB eager/graph mismatch: {rgb_max_abs}").into());
    }
    drop(rgb_graph);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "explicit_policy_and_eviction_passed": true,
            "populated_cache_eviction_passed": true,
            "raw_rgb_eager_graph_max_abs": rgb_max_abs,
            "eager_graph_max_abs": eager_graph_max_abs,
            "device": cached_action.tensor().device().to_string(),
            "dtype": cached_action.tensor().dtype().to_string(),
            "prepared_shape": prepared_action.tensor().shape().dims(),
            "infer_shape": inferred_action.tensor().shape().dims(),
            "cached_infer_shape": cached_action.tensor().shape().dims(),
            "token_count": token_count,
            "seeded_reproducible": true,
            "generated_vs_cpu_rng_cosine": cosine,
            "generated_vs_cpu_rng_max_abs": max_abs,
        }))?
    );
    Ok(())
}

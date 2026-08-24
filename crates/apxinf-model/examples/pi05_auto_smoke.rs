//! Public-entry smoke test for AutoModel -> LoadedModel::Vla -> prepare/run.

use std::path::{Path, PathBuf};

use apxinf_core::{standard_normal_f32, DType, Device, RngKey, Tensor};
use apxinf_model::{
    AutoModel, LoadOptions, ModelPrecision, Observation, Pi05Config,
    VisionObservation, VlaRequest,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.len() < 2 || arguments.len() > 3 {
        return Err(format!(
            "usage: {} <checkpoint-or-directory> [token-count=21]",
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
        precision: ModelPrecision::Bf16,
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
    };
    let noise = Tensor::zeros(vec![config.action_horizon, config.action_dim], DType::F32);
    let request = VlaRequest::provided(&observation, &noise);
    let prepared = model.prepare(&observation.inference_spec())?;
    let prepared_action = prepared.run(&request)?;
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
    let left_norm = generated_first.iter().map(|value| value * value).sum::<f32>();
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
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
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

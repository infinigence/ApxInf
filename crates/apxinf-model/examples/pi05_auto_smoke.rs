//! Public-entry smoke test for AutoModel -> LoadedModel::Vla -> prepare/run.

use std::path::{Path, PathBuf};

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{
    AutoModel, LoadOptions, ModelPrecision, Observation, Pi05Config, VisionObservation,
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
        noise: Tensor::zeros(vec![config.action_horizon, config.action_dim], DType::F32),
    };
    let prepared = model.prepare(&observation.inference_spec())?;
    let prepared_action = prepared.run(&observation)?;
    drop(prepared);
    let inferred_action = model.infer(&observation)?;
    let cached_action = model.infer(&observation)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "device": cached_action.tensor().device().to_string(),
            "dtype": cached_action.tensor().dtype().to_string(),
            "prepared_shape": prepared_action.tensor().shape().dims(),
            "infer_shape": inferred_action.tensor().shape().dims(),
            "cached_infer_shape": cached_action.tensor().shape().dims(),
            "token_count": token_count,
        }))?
    );
    Ok(())
}

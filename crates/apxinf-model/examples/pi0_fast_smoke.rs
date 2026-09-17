//! Public-entry smoke test for π0-FAST: AutoModel -> LoadedModel::Vla -> token decode.
//!
//! Drives the token-VLA surface end to end on a real checkpoint: the discrete
//! action-token shape, one full decode, and — when a stop token is passed — the
//! early-stopped decode, asserting that stopping at the terminator returns the
//! identical prefix. That prefix property is what lets the deployment skip the
//! ~90% of `max_action_tokens` the FAST detokenizer discards anyway.
//!
//! usage: pi0_fast_smoke <checkpoint> [stop-token] [token-count=21]

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{
    AutoModel, LoadOptions, LoadedModel, ModelPrecision, Observation, VisionObservation, VlaRequest,
};

fn decode(
    model: &LoadedModel,
    request: &VlaRequest<'_>,
    stop_token: Option<u32>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(model
        .infer_action_tokens(request, stop_token)?
        .to_f32_vec()?
        .into_iter()
        .map(|value| value as u32)
        .collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if !(2..=4).contains(&arguments.len()) {
        return Err(format!(
            "usage: {} <checkpoint> [stop-token] [token-count=21]",
            arguments.first().map(String::as_str).unwrap_or("pi0_fast_smoke")
        )
        .into());
    }
    let checkpoint = PathBuf::from(&arguments[1]);
    let stop_token = arguments
        .get(2)
        .map(|value| value.parse::<u32>())
        .transpose()?;
    let token_count = arguments
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(21usize);

    let options = LoadOptions {
        model_name: Some("pi0_fast".to_owned()),
        precision: ModelPrecision::Bf16,
        ..LoadOptions::default()
    };
    let model = AutoModel::load_model(Device::Cuda(0), &checkpoint, &options)?;
    let contract = model.vla()?.contract();
    let token_shape = model
        .action_token_shape()?
        .ok_or("π0-FAST did not report a discrete action-token output shape")?;

    // Zeros are enough: the smoke test exercises the wiring and the shape, not
    // the numerics, and π0-FAST takes raw pixels as FP32 patches.
    let observation = Observation {
        vision: VisionObservation::Patches(Tensor::zeros(
            contract.patch_shape.to_vec(),
            DType::F32,
        )),
        token_ids: vec![0; token_count],
        state: None,
        action_mask: None,
    };
    let unused_latent = Tensor::zeros(vec![1, 1], DType::F32);
    let request = VlaRequest::provided(&observation, &unused_latent);

    let started = Instant::now();
    let full = decode(&model, &request, None)?;
    let full_ms = started.elapsed().as_secs_f64() * 1000.0;
    if full.len() != token_shape[1] {
        return Err(format!(
            "full decode returned {} tokens, expected {}",
            full.len(),
            token_shape[1]
        )
        .into());
    }

    let mut early_stop = None;
    if let Some(stop) = stop_token {
        let started = Instant::now();
        let stopped = decode(&model, &request, Some(stop))?;
        let stopped_ms = started.elapsed().as_secs_f64() * 1000.0;
        if stopped.last() != Some(&stop) {
            return Err(format!(
                "early stop returned {} tokens without ending on stop token {stop}",
                stopped.len()
            )
            .into());
        }
        if stopped != full[..stopped.len()] {
            return Err("early-stopped token stream is not a prefix of the full decode".into());
        }
        early_stop = Some(serde_json::json!({
            "stop_token": stop,
            "tokens": stopped.len(),
            "ms": stopped_ms,
            "prefix_of_full_decode": true,
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "num_views": contract.num_views,
            "image_size": contract.image_size,
            "patch_shape": contract.patch_shape,
            "max_token_len": contract.max_token_len,
            "action_token_shape": token_shape,
            "token_count": token_count,
            "full_decode": {
                "tokens": full.len(),
                "ms": full_ms,
                "head": &full[..full.len().min(8)],
            },
            "early_stop": early_stop,
        }))?
    );
    Ok(())
}

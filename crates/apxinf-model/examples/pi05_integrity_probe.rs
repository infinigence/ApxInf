//! Low-level numerical integrity probe for PI0.5.
//!
//! This example intentionally bypasses the unified `AutoModel`/`infer` frontend
//! to reach `Pi05CudaRuntime` internals and `apxinf_cuda` tuning/kernel
//! signatures directly, which the model abstraction does not (and should not)
//! expose. See `pi05_auto_smoke` for the abstraction-level entry point.

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Tensor};
use apxinf_cuda::{CudaBackend, CudaBuffer};
use apxinf_model::pi05::{
    upload_time_embeddings, vision_layer, vision_patch_embed, vision_qkv_packed_from_env,
    Pi05ActivationScales, Pi05Config, Pi05CudaRuntime, Pi05Weights, StaticFp8Calibration,
    StaticFp8Pi05Weights,
};

fn signature(values: &[f32]) -> serde_json::Value {
    let elements = values.len();
    let sample_count = elements.min(256);
    let mut sum = 0.0f64;
    let mut abs_checksum = 0.0f64;
    let mut square_sum = 0.0f64;
    let mut max_abs = 0.0f64;
    for &value in values {
        let value = f64::from(value);
        sum += value;
        abs_checksum += value.abs();
        square_sum += value * value;
        max_abs = max_abs.max(value.abs());
    }
    let sample = (0..sample_count)
        .map(|index| {
            let source = if sample_count == 1 {
                0
            } else {
                index * (elements - 1) / (sample_count - 1)
            };
            values[source]
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "elements": elements,
        "sum": sum,
        "abs_checksum": abs_checksum,
        "l2": square_sum.sqrt(),
        "max_abs": max_abs,
        "sample": sample,
    })
}

fn device_signature(
    backend: &CudaBackend,
    tensor: &Tensor,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let values = backend.to_cpu(tensor)?.to_f32_vec()?;
    Ok(signature(&values))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.len() != 5 {
        return Err(format!(
            "usage: {} <checkpoint-or-index> <calibration.json> <tactics.json> <token-count>",
            arguments
                .first()
                .map(String::as_str)
                .unwrap_or("pi05_integrity_probe")
        )
        .into());
    }
    let token_count = arguments[4].parse::<usize>()?;
    let config = Arc::new(Pi05Config::thor_two_view());
    let calibration = StaticFp8Calibration::from_json_file(Path::new(&arguments[2]))?;
    let scales = Arc::new(Pi05ActivationScales::from_calibration(
        &config,
        &calibration,
    )?);
    let backend = Arc::new(CudaBackend::new(0)?);
    let tuning = apxinf_cuda::tuning::TuningDb::from_json_file(Path::new(&arguments[3]))?;
    apxinf_cuda::kernels::gemm::install_tuning_db(backend.context(), &tuning)?;
    eprintln!("loading π0.5 checkpoint...");
    let host_weights = Pi05Weights::from_safetensors(&config, Path::new(&arguments[1]))?;
    eprintln!("quantizing and uploading static FP8 weights...");
    let device_weights = Arc::new(StaticFp8Pi05Weights::from_host(&host_weights, &*backend)?);
    drop(host_weights);

    let patch_tokens = config.num_views * config.patches_per_view();
    let patch_width = 3 * config.patch_size * config.patch_size;
    let patches = backend.to_device(&Tensor::zeros(vec![patch_tokens, patch_width], DType::F16))?;
    let noise = backend.to_device(&Tensor::zeros(
        vec![config.action_horizon, config.action_dim],
        DType::F16,
    ))?;
    let token_ids = CudaBuffer::alloc_zeros(token_count * 4, backend.device_id())
        .map_err(std::io::Error::other)?;
    let time_embeddings = upload_time_embeddings(&config, &*backend)?;
    let runtime = Pi05CudaRuntime::new(
        backend.clone(),
        config.clone(),
        device_weights.clone(),
        scales.clone(),
    )?;

    let mut signatures = serde_json::Map::new();
    eprintln!("probing patch embedding and each vision layer...");
    let mut vision_hidden = vision_patch_embed(
        backend.context(),
        &device_weights.patch_embedding,
        &device_weights.position_embedding,
        &patches,
        config.patches_per_view(),
        scales.vision_patch_input,
    )?;
    signatures.insert(
        "vision_patch_embed".into(),
        device_signature(&backend, &vision_hidden)?,
    );
    let packed_vision_qkv = vision_qkv_packed_from_env()?;
    for (index, (weights, layer_scales)) in device_weights
        .vision_layers
        .iter()
        .zip(&scales.vision_layers)
        .enumerate()
    {
        vision_hidden = vision_layer(
            backend.context(),
            weights,
            *layer_scales,
            &vision_hidden,
            config.patches_per_view(),
            config.vision_heads,
            config.vision_head_dim,
            packed_vision_qkv,
            config.layer_norm_eps,
        )?;
        signatures.insert(
            format!("vision_layer_{index}"),
            device_signature(&backend, &vision_hidden)?,
        );
    }
    eprintln!("probing vision projection...");
    let vision = runtime.encode_vision(&patches)?;
    signatures.insert(
        "vision_projected".into(),
        device_signature(&backend, &vision)?,
    );
    eprintln!("probing language prefix K/V...");
    let prefix_input = runtime.embed_prefix(&vision, &token_ids, token_count)?;
    let prefix = runtime.prefix_forward(&prefix_input)?;
    for layer in [0usize, config.language.depth - 1] {
        signatures.insert(
            format!("prefix_v_layer{layer}"),
            device_signature(&backend, &prefix.values[layer])?,
        );
    }

    eprintln!("probing ten denoising steps...");
    let mut state = noise;
    let dt = -1.0 / config.num_flow_steps as f32;
    for (step, embedding) in time_embeddings.iter().enumerate() {
        state = runtime.denoise_step(&state, embedding, &prefix, dt)?;
        signatures.insert(
            format!("denoise_step_{step}"),
            device_signature(&backend, &state)?,
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.pi05.stage-probe.v1",
            "token_count": token_count,
            "intermediate_signatures": signatures,
        }))?
    );
    Ok(())
}

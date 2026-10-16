//! GR00T model-core benchmark with deterministic in-memory inputs.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::{
    AutoModel, LoadOptions, ModelPrecision, Observation, VisionObservation, VlaMetadata, VlaRequest,
};
use half::bf16;
use serde_json::Value;

struct Arguments {
    checkpoint: PathBuf,
    views: usize,
    precision: ModelPrecision,
    device: usize,
    warmup: usize,
    iterations: usize,
    calibration: Option<PathBuf>,
    tactics: Option<PathBuf>,
    output: Option<PathBuf>,
    autotune: bool,
}

struct Inputs {
    pixel_values: Tensor,
    grid: Vec<[u32; 3]>,
    token_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    state: Tensor,
    noise: Tensor,
    embodiment_id: usize,
    name: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_arguments()?;
    let input = generated_inputs(&args)?;
    let options = LoadOptions {
        model_name: Some("gr00t".into()),
        precision: args.precision,
        calibration_path: args.calibration.clone(),
        tuning_path: args.tactics.clone(),
        autotune: args.autotune,
        ..LoadOptions::default()
    };
    let load_started = Instant::now();
    let model = AutoModel::load_model(Device::Cuda(args.device), &args.checkpoint, &options)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
    let observation = Observation {
        vision: VisionObservation::Patches(input.pixel_values),
        token_ids: input.token_ids,
        state: Some(input.state),
        action_mask: None,
    };
    let metadata = VlaMetadata {
        attention_mask: Some(&input.attention_mask),
        image_grid_thw: Some(&input.grid),
        embodiment_id: Some(input.embodiment_id),
        planning: None,
    };
    let request = VlaRequest::provided_with_metadata(&observation, &input.noise, metadata);

    for _ in 0..args.warmup {
        black_box(model.infer_host_f32(&request)?);
    }
    let execution = model.vla()?.execution_mode();
    if execution != "cuda-graph" {
        return Err(format!(
            "GR00T benchmark requires CUDA Graph after warmup, runtime reported {execution}"
        )
        .into());
    }

    let [horizon, action_dim] = model.vla()?.action_shape();
    let mut first_output: Option<Vec<f32>> = None;
    let mut samples = Vec::with_capacity(args.iterations);
    let mut last_output = Vec::new();
    for _ in 0..args.iterations {
        let started = Instant::now();
        last_output = model.infer_host_f32(&request)?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        if last_output.len() != horizon * action_dim || last_output.iter().any(|v| !v.is_finite()) {
            return Err("model returned an invalid action shape or non-finite actions".into());
        }
        if let Some(first) = &first_output {
            if first != &last_output {
                return Err("identical benchmark inputs produced different actions".into());
            }
        } else {
            first_output = Some(last_output.clone());
        }
        black_box(&last_output);
    }
    let summary = latency_summary(&samples)?;
    let report = serde_json::json!({
        "schema": "apxinf.gr00t-n1.7.benchmark.v2",
        "input_profile": input.name,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "precision": precision_name(args.precision),
        "execution": execution,
        "timing_boundary": "constructed host tensors through synchronized model core and action D2H",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_ms": load_ms,
        "latency_ms": summary,
        "input": {
            "pixel_values": observation_shape(&observation),
            "image_grid_thw": input.grid,
            "token_count": observation.token_ids.len(),
            "state": observation.state.as_ref().map(|value| value.shape().dims()),
            "embodiment_id": input.embodiment_id,
        },
        "output": {
            "shape": [horizon, action_dim],
            "sum": last_output.iter().copied().map(f64::from).sum::<f64>(),
            "head": last_output.iter().take(16).copied().collect::<Vec<_>>(),
            "values": last_output,
        },
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        std::fs::write(&path, format!("{rendered}\n"))?;
        println!("wrote {}", path.display());
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn observation_shape(observation: &Observation) -> &[usize] {
    match &observation.vision {
        VisionObservation::Patches(value) => value.shape().dims(),
        VisionObservation::RgbU8 { .. } => &[],
    }
}

fn parse_arguments() -> Result<Arguments, Box<dyn std::error::Error>> {
    let values = std::env::args().collect::<Vec<_>>();
    if !(4..=11).contains(&values.len()) {
        return Err(format!(
            "usage: {} <model-dir> <views:1|2> <bf16|fp8|int8> [device=0] [warmup=10] [iterations=50] [calibration|-] [tactics|-] [output|-] [--autotune]",
            values.first().map(String::as_str).unwrap_or("gr00t_bench")
        )
        .into());
    }
    let integer = |index: usize, default: usize, label: &str| {
        values
            .get(index)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {label} {value:?}: {error}"))
            })
            .transpose()
            .map(|value| value.unwrap_or(default))
    };
    let optional_path = |index: usize| {
        values
            .get(index)
            .filter(|value| value.as_str() != "-")
            .map(PathBuf::from)
    };
    let precision = match values[3].as_str() {
        "bf16" => ModelPrecision::Bf16,
        "fp8" => ModelPrecision::Fp8,
        "int8" => ModelPrecision::W8A8,
        value => return Err(format!("invalid precision {value:?}; expected bf16|fp8|int8").into()),
    };
    let iterations = integer(6, 50, "iteration count")?;
    if iterations == 0 {
        return Err("iteration count must be non-zero".into());
    }
    let autotune = match values.get(10).map(String::as_str) {
        None => false,
        Some("--autotune") => true,
        Some(value) => {
            return Err(format!("invalid trailing argument {value:?}; expected --autotune").into())
        }
    };
    Ok(Arguments {
        checkpoint: PathBuf::from(&values[1]),
        views: integer(2, 2, "view count")?,
        precision,
        device: integer(4, 0, "CUDA device")?,
        warmup: integer(5, 10, "warmup count")?,
        iterations,
        calibration: optional_path(7),
        tactics: optional_path(8),
        output: optional_path(9),
        autotune,
    })
}

/// Match the published 256-patch/view workload without external input files.
fn generated_inputs(args: &Arguments) -> Result<Inputs, Box<dyn std::error::Error>> {
    if !(1..=2).contains(&args.views) {
        return Err("views must be 1 or 2".into());
    }
    let config: Value =
        serde_json::from_slice(&std::fs::read(args.checkpoint.join("config.json"))?)?;
    let cosmos: Value = serde_json::from_slice(&std::fs::read(
        args.checkpoint.join("assets/cosmos/config.json"),
    )?)?;
    let dimension = |key: &str, default| config[key].as_u64().unwrap_or(default) as usize;
    let state_dim = dimension("max_state_dim", 132);
    let action_dim = dimension("max_action_dim", 132);
    let horizon = dimension("action_horizon", 40);
    let image_token = cosmos["image_token_id"].as_u64().unwrap_or(151655) as u32;
    let start = cosmos["vision_start_token_id"].as_u64().unwrap_or(151652) as u32;
    let end = cosmos["vision_end_token_id"].as_u64().unwrap_or(151653) as u32;
    let vision = &cosmos["vision_config"];
    if vision["patch_size"].as_u64().unwrap_or(16) != 16
        || vision["temporal_patch_size"].as_u64().unwrap_or(2) != 2
        || vision["spatial_merge_size"].as_u64().unwrap_or(2) != 2
    {
        return Err("benchmark profile requires patch=16, temporal=2, spatial merge=2".into());
    }
    let mut token_ids = vec![42; 24];
    for _ in 0..args.views {
        token_ids.push(start);
        token_ids.extend(std::iter::repeat_n(image_token, 64));
        token_ids.push(end);
    }
    // Bounded deterministic nonzero pixels avoid constant-zero arithmetic.
    let pixels: Vec<bf16> = (0..args.views * 256 * 1536)
        .map(|i| bf16::from_f32(((i * 17 % 251) as f32 - 125.0) / 125.0))
        .collect();
    Ok(Inputs {
        pixel_values: Tensor::from_bf16(vec![args.views * 256, 1536], &pixels)?,
        grid: vec![[1, 16, 16]; args.views],
        attention_mask: vec![1; token_ids.len()],
        token_ids,
        state: Tensor::from_bf16(vec![1, 1, state_dim], &vec![bf16::ZERO; state_dim])?,
        noise: Tensor::from_bf16(
            vec![1, horizon, action_dim],
            &vec![bf16::ZERO; horizon * action_dim],
        )?,
        embodiment_id: 2,
        name: format!("synthetic-libero-{}-view-v1", args.views),
    })
}

fn latency_summary(samples: &[f64]) -> Result<Value, Box<dyn std::error::Error>> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.is_empty() {
        return Err("latency sample list is empty".into());
    }
    Ok(serde_json::json!({
        "samples": samples,
        "p50": percentile(&sorted, 0.50),
        "p95": percentile(&sorted, 0.95),
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
        "min": sorted[0],
        "max": sorted[sorted.len() - 1],
    }))
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let rank = ((sorted.len() as f64) * quantile).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn precision_name(precision: ModelPrecision) -> &'static str {
    match precision {
        ModelPrecision::Auto => "auto",
        ModelPrecision::Bf16 => "bf16",
        ModelPrecision::Fp8 => "fp8",
        ModelPrecision::W8A8 => "int8",
    }
}

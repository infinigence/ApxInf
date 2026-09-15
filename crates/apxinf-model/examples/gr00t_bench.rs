//! Fixed-input GR00T N1.7 model-core benchmark.
//!
//! The input directory is produced by NVIDIA's official processor. Timing
//! begins with those preprocessed host tensors and ends after action D2H.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::{
    AutoModel, LoadOptions, ModelPrecision, Observation, VisionObservation, VlaMetadata, VlaRequest,
};
use half::bf16;
use serde_json::Value;

const FIXTURE_SCHEMA: &str = "apxinf.gr00t-n1.7.preprocessed-fixture.v1";

struct Arguments {
    checkpoint: PathBuf,
    backbone: PathBuf,
    fixture: PathBuf,
    precision: ModelPrecision,
    device: usize,
    warmup: usize,
    iterations: usize,
    calibration: Option<PathBuf>,
    tactics: Option<PathBuf>,
    output: Option<PathBuf>,
    autotune: bool,
}

struct Fixture {
    pixel_values: Tensor,
    grid: Vec<[u32; 3]>,
    token_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    state: Tensor,
    noise: Tensor,
    embodiment_id: usize,
    name: String,
}

struct TensorEntry {
    path: PathBuf,
    shape: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_arguments()?;
    let fixture = load_fixture(&args.fixture)?;
    let mut options = LoadOptions {
        model_name: Some("gr00t".into()),
        precision: args.precision,
        calibration_path: args.calibration.clone(),
        tuning_path: args.tactics.clone(),
        autotune: args.autotune,
        ..LoadOptions::default()
    };
    options
        .assets
        .insert("backbone".into(), args.backbone.clone());

    let load_started = Instant::now();
    let model = AutoModel::load_model(Device::Cuda(args.device), &args.checkpoint, &options)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
    let observation = Observation {
        vision: VisionObservation::Patches(fixture.pixel_values),
        token_ids: fixture.token_ids,
        state: Some(fixture.state),
        action_mask: None,
    };
    let metadata = VlaMetadata {
        attention_mask: Some(&fixture.attention_mask),
        image_grid_thw: Some(&fixture.grid),
        embodiment_id: Some(fixture.embodiment_id),
    };
    let request = VlaRequest::provided_with_metadata(&observation, &fixture.noise, metadata);

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

    let mut samples = Vec::with_capacity(args.iterations);
    let mut last_output = Vec::new();
    for _ in 0..args.iterations {
        let started = Instant::now();
        last_output = model.infer_host_f32(&request)?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        black_box(&last_output);
    }
    let summary = latency_summary(&samples)?;
    let [horizon, action_dim] = model.vla()?.action_shape();
    let report = serde_json::json!({
        "schema": "apxinf.gr00t-n1.7.benchmark.v2",
        "fixture": fixture.name,
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "device": args.device,
        "precision": precision_name(args.precision),
        "execution": execution,
        "timing_boundary": "official-processor tensors through synchronized model core and action D2H",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_ms": load_ms,
        "latency_ms": summary,
        "input": {
            "pixel_values": observation_shape(&observation),
            "image_grid_thw": fixture.grid,
            "token_count": observation.token_ids.len(),
            "state": observation.state.as_ref().map(|value| value.shape().dims()),
            "embodiment_id": fixture.embodiment_id,
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
    if !(5..=12).contains(&values.len()) {
        return Err(format!(
            "usage: {} <checkpoint> <backbone> <fixture> <bf16|fp8|int8> [device=0] [warmup=10] [iterations=50] [calibration|-] [tactics|-] [output|-] [--autotune]",
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
    let precision = match values[4].as_str() {
        "bf16" => ModelPrecision::Bf16,
        "fp8" => ModelPrecision::Fp8,
        "int8" => ModelPrecision::W8A8,
        value => return Err(format!("invalid precision {value:?}; expected bf16|fp8|int8").into()),
    };
    let iterations = integer(7, 50, "iteration count")?;
    if iterations == 0 {
        return Err("iteration count must be non-zero".into());
    }
    let autotune = match values.get(11).map(String::as_str) {
        None => false,
        Some("--autotune") => true,
        Some(value) => {
            return Err(format!("invalid trailing argument {value:?}; expected --autotune").into())
        }
    };
    Ok(Arguments {
        checkpoint: PathBuf::from(&values[1]),
        backbone: PathBuf::from(&values[2]),
        fixture: PathBuf::from(&values[3]),
        precision,
        device: integer(5, 0, "CUDA device")?,
        warmup: integer(6, 10, "warmup count")?,
        iterations,
        calibration: optional_path(8),
        tactics: optional_path(9),
        output: optional_path(10),
        autotune,
    })
}

fn load_fixture(root: &Path) -> Result<Fixture, Box<dyn std::error::Error>> {
    let manifest: Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let schema = string(&manifest, "schema")?;
    if schema != FIXTURE_SCHEMA {
        return Err(format!("unsupported fixture schema {schema:?}").into());
    }
    let pixels = entry(root, &manifest, "pixel_values", "bfloat16")?;
    let grid = entry(root, &manifest, "image_grid_thw", "uint32")?;
    let tokens = entry(root, &manifest, "token_ids", "uint32")?;
    let mask = entry(root, &manifest, "attention_mask", "uint8")?;
    let state = entry(root, &manifest, "state", "bfloat16")?;
    let noise = entry(root, &manifest, "noise", "bfloat16")?;
    if grid.shape.len() != 2 || grid.shape[1] != 3 {
        return Err(format!("image_grid_thw must be [images, 3], got {:?}", grid.shape).into());
    }
    let grid = read_u32(&grid)?
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    Ok(Fixture {
        pixel_values: read_bf16(&pixels)?,
        grid,
        token_ids: read_u32(&tokens)?,
        attention_mask: read_u8(&mask)?,
        state: read_bf16(&state)?,
        noise: read_bf16(&noise)?,
        embodiment_id: usize::try_from(
            manifest
                .get("embodiment_id")
                .and_then(Value::as_u64)
                .ok_or("fixture embodiment_id must be an integer")?,
        )?,
        name: string(&manifest, "fixture")?.to_owned(),
    })
}

fn entry(
    root: &Path,
    manifest: &Value,
    name: &str,
    dtype: &str,
) -> Result<TensorEntry, Box<dyn std::error::Error>> {
    let value = manifest
        .get("tensors")
        .and_then(|tensors| tensors.get(name))
        .ok_or_else(|| format!("fixture is missing tensors.{name}"))?;
    if string(value, "dtype")? != dtype {
        return Err(format!("fixture tensor {name} must have dtype {dtype}").into());
    }
    let shape = value
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("fixture tensor {name} has no shape"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| format!("invalid {name} shape"))
        })
        .map(|value| {
            value.and_then(|value| usize::try_from(value).map_err(|error| error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(TensorEntry {
        path: root.join(string(value, "file")?),
        shape,
    })
}

fn read_bf16(entry: &TensorEntry) -> Result<Tensor, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&entry.path)?;
    let expected = elements(&entry.shape)?
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or("fixture tensor byte size overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {:?}",
            entry.path.display(),
            bytes.len(),
            entry.shape
        )
        .into());
    }
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    Ok(Tensor::from_bf16(entry.shape.clone(), &values)?)
}

fn read_u32(entry: &TensorEntry) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&entry.path)?;
    let expected = elements(&entry.shape)?
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or("fixture tensor byte size overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {:?}",
            entry.path.display(),
            bytes.len(),
            entry.shape
        )
        .into());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok(values)
}

fn read_u8(entry: &TensorEntry) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let values = std::fs::read(&entry.path)?;
    let expected = elements(&entry.shape)?;
    if values.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {:?}",
            entry.path.display(),
            values.len(),
            entry.shape
        )
        .into());
    }
    Ok(values)
}

fn elements(shape: &[usize]) -> Result<usize, Box<dyn std::error::Error>> {
    shape
        .iter()
        .try_fold(1usize, |count, value| count.checked_mul(*value))
        .ok_or_else(|| "fixture tensor size overflow".into())
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

fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("fixture field {field:?} must be a string").into())
}

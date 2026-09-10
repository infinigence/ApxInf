//! GR00T N1.7 benchmark over an official processor fixture.
//!
//! Unlike `gr00t_bench`, this entry consumes the exact non-zero pixel, token
//! and state values dumped by the NVIDIA processor.  The fixture conversion
//! is performed by `scripts/prepare_gr00t_n1d7_fixture.py`.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use apxinf_core::{Backend, Device, Tensor};
use apxinf_model::gr00t::{Gr00tConfig, Gr00tLoadOptions, Gr00tObservation, Gr00tVlaRuntime};
use apxinf_model::ModelPrecision;
use half::bf16;
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE_SCHEMA: &str = "apxinf.gr00t-n1.7.preprocessed-fixture.v1";

#[derive(Debug)]
struct Arguments {
    checkpoint: PathBuf,
    backbone: PathBuf,
    fixture: PathBuf,
    device_id: usize,
    warmup: usize,
    iterations: usize,
    execution: ExecutionMode,
    output_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
enum ExecutionMode {
    Graph,
    Eager,
}

#[derive(Debug)]
struct TensorEntry {
    path: PathBuf,
    shape: Vec<usize>,
    dtype: String,
}

struct TacticProvenance {
    path: PathBuf,
    sha256: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = parse_arguments()?;
    let precision = precision_from_env()?;
    let tactics = tactic_provenance_from_env()?;
    let manifest_path = arguments.fixture.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    expect_string(&manifest, "schema", FIXTURE_SCHEMA)?;
    validate_fp8_calibration_fixture(precision, &manifest_path, &manifest_bytes)?;
    let fixture_tensor_sha256 = fixture_tensor_sha256(&arguments.fixture, &manifest)?;
    let fixture_name = string(&manifest, "fixture")?.to_owned();
    let mut observation = load_observation(&arguments.fixture, &manifest)?;
    let config = Gr00tConfig::from_json_file(&arguments.checkpoint.join("config.json"))?;
    observation.validate(&config)?;

    let _input_backend = match arguments.execution {
        ExecutionMode::Graph => None,
        ExecutionMode::Eager => {
            let backend = apxinf_cuda::CudaBackend::new(arguments.device_id)?;
            observation.pixel_values = backend.to_device(&observation.pixel_values)?;
            observation.state = backend.to_device(&observation.state)?;
            observation.noise = backend.to_device(&observation.noise)?;
            Some(backend)
        }
    };
    let calibration_path = std::env::var_os("APXINF_GR00T_FP8_CALIBRATION").map(PathBuf::from);
    let options = Gr00tLoadOptions {
        config: Some(config),
        precision,
        backbone_path: Some(arguments.backbone.clone()),
        fp8_calibration_path: calibration_path.clone(),
        tuning_path: tactics.as_ref().map(|tactics| tactics.path.clone()),
    };

    let load_start = Instant::now();
    let mut runtime = Gr00tVlaRuntime::from_dir(
        &arguments.checkpoint,
        options,
        Device::Cuda(arguments.device_id),
    )?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    for _ in 0..arguments.warmup {
        black_box(runtime.infer(&observation)?);
    }
    if matches!(arguments.execution, ExecutionMode::Graph) && !runtime.has_captured_graph() {
        return Err(
            "GR00T benchmark requested CUDA Graph execution, but capture fell back to eager".into(),
        );
    }

    let mut model_core_samples = Vec::with_capacity(arguments.iterations);
    let mut output_sums = Vec::with_capacity(arguments.iterations);
    let mut last_output = Vec::new();
    for _ in 0..arguments.iterations {
        let start = Instant::now();
        let output = runtime.infer(&observation)?;
        model_core_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
        last_output = output.to_f32_vec()?;
        output_sums.push(last_output.iter().copied().map(f64::from).sum::<f64>());
        black_box(output);
    }

    let model_core = latency_summary(&model_core_samples)?;
    let model_core_median = model_core["median"]
        .as_f64()
        .ok_or("model-core latency summary is missing a numeric median")?;
    let processor_median = processor_median_ms(&manifest);
    let noise_metadata = manifest.get("noise").cloned().unwrap_or(Value::Null);
    let executable = std::env::current_exe()?;
    let checkpoint_config = arguments.checkpoint.join("config.json");
    let backbone_config = arguments.backbone.join("config.json");
    let e2e_component_median_sum = processor_median.map(|processor| {
        serde_json::json!({
            "method": "sum-of-component-medians",
            "processor_median_ms": processor,
            "model_core_median_ms": model_core_median,
            "estimated_median_ms": processor + model_core_median,
            "note": "not a single contiguous in-process timing",
        })
    });
    let report = serde_json::json!({
        "schema": "apxinf.gr00t-n1.7.fixture-benchmark.v1",
        "fixture": fixture_name,
        "fixture_manifest": manifest_path,
        "fixture_source": manifest.get("source"),
        "device": arguments.device_id,
        "precision": match precision {
            ModelPrecision::Bf16 => "bf16",
            ModelPrecision::Fp8 => "fp8-e4m3-bf16-output",
            ModelPrecision::W8A8 => "int8",
            ModelPrecision::Auto => "auto",
        },
        "execution": match arguments.execution {
            ExecutionMode::Graph => "auto-graph",
            ExecutionMode::Eager => "forced-eager-device-input",
        },
        "tactics": tactics.as_ref().map(|tactics| serde_json::json!({
            "path": tactics.path.display().to_string(),
            "sha256": tactics.sha256,
            "runtime_records": runtime.tuning_record_count(),
        })),
        "provenance": {
            "benchmark_binary": {
                "path": executable,
                "sha256": sha256_file(&executable)?,
            },
            "apxinf_source": {
                "path": std::env::var("APXINF_BENCH_SOURCE_DIR").ok(),
                "revision": std::env::var("APXINF_BENCH_SOURCE_REVISION").ok(),
                "dirty": std::env::var("APXINF_BENCH_SOURCE_DIRTY").ok(),
            },
            "checkpoint_config": {
                "path": checkpoint_config,
                "sha256": sha256_file(&checkpoint_config)?,
            },
            "backbone_config": {
                "path": backbone_config,
                "sha256": sha256_file(&backbone_config)?,
            },
            "fixture_manifest_sha256": format!("{:x}", Sha256::digest(&manifest_bytes)),
            "fixture_tensor_sha256": fixture_tensor_sha256,
            "calibration": calibration_path.as_ref().map(|path| -> Result<Value, Box<dyn std::error::Error>> {
                Ok(serde_json::json!({
                    "path": path,
                    "sha256": sha256_file(path)?,
                }))
            }).transpose()?,
            "device_model": std::fs::read("/proc/device-tree/model").ok().map(|bytes| {
                String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_owned()
            }),
        },
        "input": {
            "pixel_values": observation.pixel_values.shape().dims(),
            "image_grid_thw": observation.image_grid_thw,
            "token_count": observation.token_ids.len(),
            "state": observation.state.shape().dims(),
            "embodiment_id": observation.embodiment_id,
            "noise": noise_metadata,
        },
        "action_shape": [1, runtime.config().action_horizon, runtime.config().max_action_dim],
        "warmup": arguments.warmup,
        "iterations": arguments.iterations,
        "load_seconds": load_seconds,
        "model_core_latency_ms": model_core,
        "processor_report": manifest.get("processor_report"),
        "e2e_component_median_sum_ms": e2e_component_median_sum,
        "timing_boundary": {
            "model_core": "host preprocessed tensors through synchronized CUDA execution and final action D2H",
            "e2e": "sum of independently measured processor and Model-Core medians; not one enclosing wall-clock timer",
        },
        "output_sums": output_sums,
        "output_shape": [1, runtime.config().action_horizon, runtime.config().max_action_dim],
        "output": last_output,
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = arguments.output_path {
        std::fs::write(&path, format!("{rendered}\n"))?;
        println!("wrote {}", path.display());
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}

fn fixture_tensor_sha256(
    fixture_root: &Path,
    manifest: &Value,
) -> Result<Value, Box<dyn std::error::Error>> {
    let tensors = manifest
        .get("tensors")
        .and_then(Value::as_object)
        .ok_or("fixture manifest is missing object field tensors")?;
    let mut hashes = serde_json::Map::new();
    for (name, entry) in tensors {
        let path = fixture_root.join(string(entry, "file")?);
        hashes.insert(name.clone(), Value::String(sha256_file(&path)?));
    }
    Ok(Value::Object(hashes))
}

fn validate_fp8_calibration_fixture(
    precision: ModelPrecision,
    manifest_path: &Path,
    manifest_bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    if precision != ModelPrecision::Fp8 {
        return Ok(());
    }
    let Some(calibration_path) = std::env::var_os("APXINF_GR00T_FP8_CALIBRATION") else {
        return Ok(());
    };
    let calibration_path = PathBuf::from(calibration_path);
    let calibration: Value = serde_json::from_slice(&std::fs::read(&calibration_path)?)?;
    let expected = calibration
        .get("fixture_manifest_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            format!(
                "FP8 calibration {} is missing string field fixture_manifest_sha256",
                calibration_path.display()
            )
        })?;
    let actual = format!("{:x}", Sha256::digest(manifest_bytes));
    if expected != actual {
        return Err(format!(
            "FP8 calibration fixture mismatch: calibration {} expects manifest SHA-256 {}, but fixture {} has {}; regenerate or select calibration for this exact fixture",
            calibration_path.display(),
            expected,
            manifest_path.display(),
            actual
        )
        .into());
    }
    Ok(())
}

fn precision_from_env() -> Result<ModelPrecision, Box<dyn std::error::Error>> {
    match std::env::var("APXINF_GR00T_PRECISION")
        .unwrap_or_else(|_| "bf16".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "bf16" => Ok(ModelPrecision::Bf16),
        "fp8" | "fp8-e4m3" => Ok(ModelPrecision::Fp8),
        "int8" => Ok(ModelPrecision::W8A8),
        value => Err(format!(
            "unsupported APXINF_GR00T_PRECISION={value:?}; expected bf16, fp8, or int8"
        )
        .into()),
    }
}

fn tactic_provenance_from_env() -> Result<Option<TacticProvenance>, Box<dyn std::error::Error>> {
    let Some(path) = std::env::var_os("APXINF_GR00T_BF16_TACTICS") else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let sha256 = sha256_file(&path)?;
    Ok(Some(TacticProvenance { path, sha256 }))
}

fn parse_arguments() -> Result<Arguments, Box<dyn std::error::Error>> {
    let values = std::env::args().collect::<Vec<_>>();
    if !(4..=9).contains(&values.len()) {
        return Err(format!(
            "usage: {} <gr00t-checkpoint-dir> <cosmos-backbone-dir-or-config> <fixture-dir> [cuda-device] [warmup] [iterations] [graph|eager] [output-json]",
            values.first().map(String::as_str).unwrap_or("gr00t_fixture_bench")
        )
        .into());
    }
    let parse = |index: usize, default: usize, name: &str| {
        values
            .get(index)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {name} {value:?}: {error}"))
            })
            .transpose()
            .map(|value| value.unwrap_or(default))
    };
    let arguments = Arguments {
        checkpoint: PathBuf::from(&values[1]),
        backbone: PathBuf::from(&values[2]),
        fixture: PathBuf::from(&values[3]),
        device_id: parse(4, 0, "CUDA device")?,
        warmup: parse(5, 5, "warmup count")?,
        iterations: parse(6, 20, "iteration count")?,
        execution: match values.get(7).map(String::as_str).unwrap_or("graph") {
            "graph" => ExecutionMode::Graph,
            "eager" => ExecutionMode::Eager,
            value => return Err(format!("invalid execution mode {value:?}").into()),
        },
        output_path: values.get(8).map(PathBuf::from),
    };
    if arguments.iterations == 0 {
        return Err("iteration count must be non-zero".into());
    }
    Ok(arguments)
}

fn load_observation(
    fixture_root: &Path,
    manifest: &Value,
) -> Result<Gr00tObservation, Box<dyn std::error::Error>> {
    let pixels = tensor_entry(fixture_root, manifest, "pixel_values", "bfloat16")?;
    let grids = tensor_entry(fixture_root, manifest, "image_grid_thw", "uint32")?;
    let tokens = tensor_entry(fixture_root, manifest, "token_ids", "uint32")?;
    let mask = tensor_entry(fixture_root, manifest, "attention_mask", "uint8")?;
    let state = tensor_entry(fixture_root, manifest, "state", "bfloat16")?;
    let noise = tensor_entry(fixture_root, manifest, "noise", "bfloat16")?;

    let pixel_values = read_bf16_tensor(&pixels)?;
    let state = read_bf16_tensor(&state)?;
    let noise = read_bf16_tensor(&noise)?;
    let token_ids = read_u32(&tokens)?;
    let attention_mask = read_u8(&mask)?;
    let grid_values = read_u32(&grids)?;
    if grids.shape.len() != 2 || grids.shape[1] != 3 {
        return Err(format!(
            "image_grid_thw fixture shape must be [images,3], got {:?}",
            grids.shape
        )
        .into());
    }
    let image_grid_thw = grid_values
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect::<Vec<_>>();
    let embodiment_id = usize::try_from(
        manifest
            .get("embodiment_id")
            .and_then(Value::as_u64)
            .ok_or("fixture manifest is missing integer embodiment_id")?,
    )?;
    Ok(Gr00tObservation {
        pixel_values,
        image_grid_thw,
        token_ids,
        attention_mask,
        state,
        embodiment_id,
        noise,
    })
}

fn tensor_entry(
    root: &Path,
    manifest: &Value,
    name: &str,
    expected_dtype: &str,
) -> Result<TensorEntry, Box<dyn std::error::Error>> {
    let value = manifest
        .get("tensors")
        .and_then(|value| value.get(name))
        .ok_or_else(|| format!("fixture manifest is missing tensors.{name}"))?;
    let dtype = string(value, "dtype")?.to_owned();
    if dtype != expected_dtype {
        return Err(
            format!("fixture {name} dtype is {dtype:?}, expected {expected_dtype:?}").into(),
        );
    }
    let shape = value
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("fixture tensors.{name}.shape must be an array"))?
        .iter()
        .map(|dimension| {
            dimension
                .as_u64()
                .ok_or_else(|| format!("fixture tensors.{name}.shape is not integral"))
                .and_then(|dimension| {
                    usize::try_from(dimension)
                        .map_err(|_| format!("fixture tensors.{name}.shape exceeds usize"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err(
            format!("fixture tensors.{name}.shape must be non-empty, got {shape:?}").into(),
        );
    }
    Ok(TensorEntry {
        path: root.join(string(value, "file")?),
        shape,
        dtype,
    })
}

fn read_bf16_tensor(entry: &TensorEntry) -> Result<Tensor, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&entry.path)?;
    let expected = element_count(&entry.shape)?
        .checked_mul(2)
        .ok_or("BF16 fixture byte size overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected} for {:?} {}",
            entry.path.display(),
            bytes.len(),
            entry.shape,
            entry.dtype
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
    let expected = element_count(&entry.shape)?
        .checked_mul(4)
        .ok_or("u32 fixture byte size overflow")?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            entry.path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_u8(entry: &TensorEntry) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(&entry.path)?;
    let expected = element_count(&entry.shape)?;
    if bytes.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            entry.path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(bytes)
}

fn element_count(shape: &[usize]) -> Result<usize, Box<dyn std::error::Error>> {
    shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| "fixture element count overflow".into())
}

fn latency_summary(samples: &[f64]) -> Result<Value, Box<dyn std::error::Error>> {
    if samples.is_empty() {
        return Err("latency sample list is empty".into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(serde_json::json!({
        // Preserve measurement order so a caller can combine same-iteration
        // processor and Model-Core samples before computing E2E percentiles.
        "samples": samples,
        "min": sorted[0],
        "median": median(&sorted),
        "p90": percentile(&sorted, 0.90),
        "p95": percentile(&sorted, 0.95),
        "max": sorted[sorted.len() - 1],
        "mean": sorted.iter().sum::<f64>() / sorted.len() as f64,
    }))
}

fn processor_median_ms(manifest: &Value) -> Option<f64> {
    let report = manifest.get("processor_report")?;
    report
        .pointer("/latency_ms/median")
        .or_else(|| report.pointer("/data_processing/median"))
        .and_then(Value::as_f64)
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let rank = ((sorted.len() as f64) * quantile).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[index]
}

fn median(sorted: &[f64]) -> f64 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("fixture manifest field {name:?} must be a string").into())
}

fn expect_string(
    value: &Value,
    name: &str,
    expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual = string(value, name)?;
    if actual != expected {
        return Err(format!("fixture {name} is {actual:?}, expected {expected:?}").into());
    }
    Ok(())
}

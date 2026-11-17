//! Fixed-input GR00T N1.7 MUSA model-core benchmark.
//!
//! The input directory is produced by NVIDIA's official processor. Timing
//! begins with those preprocessed host tensors and ends after action D2H.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::{
    AutoModel, ExecutionPolicy, LoadOptions, ModelPrecision, Observation, VisionObservation,
    VlaMetadata, VlaRequest,
};
use half::bf16;
use serde_json::Value;

const FIXTURE_SCHEMA: &str = "apxinf.gr00t-n1.7.preprocessed-fixture.v1";
const DEFAULT_EXECUTION_POLICY: ExecutionPolicy = ExecutionPolicy::RequireGraph;

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
    execution_policy: ExecutionPolicy,
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

struct RequestInput {
    observation: Observation,
    attention_mask: Vec<u8>,
    grid: Vec<[u32; 3]>,
    noise: Tensor,
    embodiment_id: usize,
    name: String,
}

impl RequestInput {
    fn from_fixture(fixture: Fixture) -> Self {
        let Fixture {
            pixel_values,
            grid,
            token_ids,
            attention_mask,
            state,
            noise,
            embodiment_id,
            name,
        } = fixture;
        Self {
            observation: Observation {
                vision: VisionObservation::Patches(pixel_values),
                token_ids,
                state: Some(state),
                action_mask: None,
            },
            attention_mask,
            grid,
            noise,
            embodiment_id,
            name,
        }
    }

    fn request(&self) -> VlaRequest<'_> {
        let metadata = VlaMetadata {
            attention_mask: Some(&self.attention_mask),
            image_grid_thw: Some(&self.grid),
            embodiment_id: Some(self.embodiment_id),
            planning: None,
        };
        VlaRequest::provided_with_metadata(&self.observation, &self.noise, metadata)
    }
}

struct TensorEntry {
    path: PathBuf,
    shape: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_arguments()?;
    let input = RequestInput::from_fixture(load_fixture(&args.fixture)?);
    let mut options = LoadOptions {
        model_name: Some("gr00t".into()),
        precision: args.precision,
        calibration_path: args.calibration.clone(),
        tuning_path: args.tactics.clone(),
        autotune: args.autotune,
        execution_policy: Some(args.execution_policy),
        ..LoadOptions::default()
    };
    options
        .assets
        .insert("backbone".into(), args.backbone.clone());

    let load_started = Instant::now();
    let model = AutoModel::load_model(Device::Musa(args.device), &args.checkpoint, &options)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;

    for _ in 0..args.warmup {
        let request = input.request();
        black_box(model.infer_host_f32(&request)?);
    }
    let execution = model.vla()?.execution_mode();
    let execution_matches_policy = match args.execution_policy {
        ExecutionPolicy::Eager => execution == "no_graph",
        ExecutionPolicy::PreferGraph => matches!(execution, "musa-graph" | "no_graph"),
        ExecutionPolicy::RequireGraph => execution == "musa-graph",
    };
    if !execution_matches_policy {
        return Err(format!(
            "GR00T MUSA benchmark requested {}, runtime reported {execution}",
            execution_policy_name(args.execution_policy),
        )
        .into());
    }

    let mut samples = Vec::with_capacity(args.iterations);
    let mut first_output: Option<Vec<f32>> = None;
    let mut deterministic = true;
    let mut repeat_max_abs = 0.0_f64;
    let mut sample_details = Vec::with_capacity(args.iterations);
    for iteration in 0..args.iterations {
        let request = input.request();
        let started = Instant::now();
        let output = model.infer_host_f32(&request)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        samples.push(elapsed_ms);
        match &first_output {
            Some(first) => {
                if first != &output {
                    deterministic = false;
                }
                for (left, right) in first.iter().zip(&output) {
                    repeat_max_abs = repeat_max_abs
                        .max((f64::from(*left) - f64::from(*right)).abs());
                }
            }
            None => first_output = Some(output.clone()),
        }
        sample_details.push(serde_json::json!({
            "iteration": iteration,
            "fixture": input.name,
            "token_count": input.observation.token_ids.len(),
            "latency_ms": elapsed_ms,
        }));
        black_box(&output);
    }
    let summary = latency_summary(&samples)?;
    let reported_output = first_output.unwrap_or_default();
    let [horizon, action_dim] = model.vla()?.action_shape();
    let report = serde_json::json!({
        "schema": "apxinf.gr00t-n1.7.benchmark.v2",
        "mode": "fixed",
        "fixture": &input.name,
        "checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "device": args.device,
        "precision": precision_name(args.precision),
        "requested_execution": execution_policy_name(args.execution_policy),
        "execution": execution,
        "timing_boundary": "official-processor tensors through synchronized model core and action D2H",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_ms": load_ms,
        "deterministic": deterministic,
        "repeat_normalized_max_abs": repeat_max_abs,
        "noise": {
            "shape": input.noise.shape().dims(),
            "values": input.noise.to_f32_vec()?,
        },
        "latency_ms": summary,
        "samples_detail": sample_details,
        "input": {
            "pixel_values": observation_shape(&input.observation),
            "image_grid_thw": &input.grid,
            "token_count": input.observation.token_ids.len(),
            "state": input.observation.state.as_ref().map(|value| value.shape().dims()),
            "embodiment_id": input.embodiment_id,
        },
        "output": {
            "shape": [horizon, action_dim],
            "sum": reported_output.iter().copied().map(f64::from).sum::<f64>(),
            "head": reported_output.iter().take(16).copied().collect::<Vec<_>>(),
            "values": reported_output,
        },
        "model": "gr00t",
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        std::fs::write(&path, format!("{rendered}\n"))?;
        println!("wrote {}", path.display());
    } else {
        println!("{rendered}");
    }
    if !deterministic {
        return Err(format!(
            "GR00T benchmark outputs are not deterministic; max abs drift {repeat_max_abs}"
        )
        .into());
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
    parse_argument_values(&values)
}

fn parse_argument_values(values: &[String]) -> Result<Arguments, Box<dyn std::error::Error>> {
    if values.len() < 5 {
        return Err(format!(
            "usage: {} <checkpoint> <backbone> <fixture> <bf16|fp8|int8> [device=0] [warmup=10] [iterations=50] [calibration|-] [tactics|-] [output|-] [--autotune] [--execution <eager|prefer-graph|require-graph>]",
            values.first().map(String::as_str).unwrap_or("gr00t_bench_musa")
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
    let warmup = integer(6, 10, "warmup count")?;
    if warmup == 0 {
        return Err("warmup count must be non-zero".into());
    }
    let iterations = integer(7, 50, "iteration count")?;
    if iterations == 0 {
        return Err("iteration count must be non-zero".into());
    }
    let mut autotune = false;
    let mut execution_policy = DEFAULT_EXECUTION_POLICY;
    let mut execution_was_set = false;
    let mut trailing = values.iter().skip(11);
    while let Some(value) = trailing.next() {
        match value.as_str() {
            "--autotune" => {
                if autotune {
                    return Err("--autotune was specified more than once".into());
                }
                autotune = true;
            }
            "--execution" => {
                if execution_was_set {
                    return Err("--execution was specified more than once".into());
                }
                let value = trailing.next().ok_or("--execution requires a value")?;
                execution_policy = parse_execution_policy(value)?;
                execution_was_set = true;
            }
            _ => {
                return Err(format!(
                    "invalid trailing argument {value:?}; expected --autotune or --execution"
                )
                .into());
            }
        }
    }
    Ok(Arguments {
        checkpoint: PathBuf::from(&values[1]),
        backbone: PathBuf::from(&values[2]),
        fixture: PathBuf::from(&values[3]),
        precision,
        device: integer(5, 0, "MUSA device")?,
        warmup,
        iterations,
        calibration: optional_path(8),
        tactics: optional_path(9),
        output: optional_path(10),
        autotune,
        execution_policy,
    })
}

fn parse_execution_policy(value: &str) -> Result<ExecutionPolicy, Box<dyn std::error::Error>> {
    match value {
        "eager" => Ok(ExecutionPolicy::Eager),
        "prefer-graph" => Ok(ExecutionPolicy::PreferGraph),
        "require-graph" => Ok(ExecutionPolicy::RequireGraph),
        _ => Err(format!(
            "invalid execution policy {value:?}; expected eager|prefer-graph|require-graph"
        )
        .into()),
    }
}

fn execution_policy_name(policy: ExecutionPolicy) -> &'static str {
    match policy {
        ExecutionPolicy::Eager => "eager",
        ExecutionPolicy::PreferGraph => "prefer-graph",
        ExecutionPolicy::RequireGraph => "require-graph",
    }
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
    let root = root.canonicalize().map_err(|error| {
        format!(
            "fixture directory {} is not readable: {error}",
            root.display()
        )
    })?;
    let path = root.join(string(value, "file")?);
    let path = path.canonicalize().map_err(|error| {
        format!(
            "fixture tensor {name} path {} is not readable: {error}",
            path.display()
        )
    })?;
    if !path.starts_with(&root) {
        return Err(format!("fixture tensor {name} escapes fixture directory").into());
    }
    Ok(TensorEntry { path, shape })
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
    if let Some(value) = values.iter().find(|value| !value.is_finite()) {
        return Err(format!(
            "{} contains non-finite BF16 value {value}",
            entry.path.display()
        )
        .into());
    }
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

#[cfg(test)]
mod gpu_lifecycle_tests {
    use super::*;
    use apxinf_model::vla::{ExecutionMode, PreparationStatus, PreparedInference};
    use apxinf_model::LoadedModel;

    fn setup() -> Result<(LoadedModel, RequestInput, RequestInput), Box<dyn std::error::Error>> {
        let checkpoint = PathBuf::from(std::env::var("APXINF_TEST_CHECKPOINT")?);
        let backbone = PathBuf::from(std::env::var("APXINF_TEST_BACKBONE")?);
        let fixture = PathBuf::from(std::env::var("APXINF_TEST_FIXTURE")?);
        let input = RequestInput::from_fixture(load_fixture(&fixture)?);
        let mut changed = RequestInput::from_fixture(load_fixture(&fixture)?);
        changed.embodiment_id = std::env::var("APXINF_TEST_OTHER_EMBODIMENT")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        assert_ne!(input.embodiment_id, changed.embodiment_id);
        let device = std::env::var("APXINF_TEST_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let mut options = LoadOptions {
            model_name: Some("gr00t".into()),
            precision: ModelPrecision::Bf16,
            execution_policy: Some(ExecutionPolicy::RequireGraph),
            ..LoadOptions::default()
        };
        options.assets.insert("backbone".into(), backbone);
        println!(
            "Loading BF16 on MUSA device {device}; A embodiment={}, B embodiment={} (forces selected_embodiment replacement)",
            input.embodiment_id, changed.embodiment_id
        );
        let model = AutoModel::load_model(Device::Musa(device), &checkpoint, &options)?;
        Ok((model, input, changed))
    }

    fn assert_graph(plan: &dyn PreparedInference) {
        assert_eq!(
            plan.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                fallback_reason: None,
            }
        );
    }

    fn assert_retired(plan: &dyn PreparedInference, input: &RequestInput) {
        assert_eq!(plan.status(), PreparationStatus::Invalidated);
        let error = match plan.run(&input.request()) {
            Ok(_) => panic!("retired Graph must not execute"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("invalidated"), "unexpected error: {error}");
        println!("PASS old Graph rejected: {error}");
    }

    fn assert_output(label: &str, actual: &[f32], reference: &[f32]) {
        assert!(!actual.is_empty());
        assert_eq!(actual.len(), reference.len());
        assert!(actual
            .iter()
            .chain(reference)
            .all(|value| value.is_finite()));
        assert_eq!(
            actual, reference,
            "{label}: output changed after cache replacement"
        );
        println!(
            "PASS {label}: {} finite values, exactly equal f32 values",
            actual.len()
        );
    }

    #[test]
    #[ignore = "requires MUSA GPU, GR00T checkpoint and preprocessed fixture"]
    fn graph_calibration_and_runtime_managed_gpu() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(std::env::var("APXINF_GR00T_PREPARED_ARENA")?, "1");
        let (model, input, changed) = setup()?;
        let runtime = model.vla()?;
        let request = input.request();
        let changed_request = changed.request();
        let legacy = runtime.prepare(&input.observation.inference_spec())?;
        assert_eq!(legacy.status(), PreparationStatus::RuntimeManaged);
        let eager_reference = legacy.run(&changed_request)?.tensor().to_f32_vec()?;

        println!("[1/5] Explicit RequireGraph(A): capture and replay twice");
        let graph = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert_graph(&*graph);
        let reference = graph.run(&request)?.tensor().to_f32_vec()?;
        assert_output(
            "repeated Graph(A)",
            &graph.run(&request)?.tensor().to_f32_vec()?,
            &reference,
        );

        println!("[2/5] calibration_amax(B): replace embodiment cache, retire Graph(A)");
        let records = runtime.calibration_amax(&changed_request)?;
        let expected = runtime.calibration_plan()?;
        assert!(!records.is_empty());
        assert_eq!(records.len(), expected.len());
        assert!(expected.iter().all(|name| records.contains_key(name)));
        assert!(records
            .values()
            .all(|value| value.is_finite() && *value >= 0.0));
        println!(
            "PASS calibration coverage: {} finite amax sites",
            records.len()
        );
        assert_retired(&*graph, &input);
        assert_output(
            "Graph(A) rebuilt after calibration(B)",
            &runtime.infer_host_f32(&request)?,
            &reference,
        );
        assert_eq!(runtime.execution_mode(), "musa-graph");

        println!("[3/5] Implicit Graph(A) -> calibration(B) -> implicit infer(A) rebuild");
        runtime.calibration_amax(&changed_request)?;
        assert_ne!(runtime.execution_mode(), "musa-graph");
        assert_output(
            "invalidated implicit cache rebuilt",
            &runtime.infer_host_f32(&request)?,
            &reference,
        );
        assert_eq!(runtime.execution_mode(), "musa-graph");

        println!("[4/5] Explicit Graph(A) -> old shape-only prepare handle runs eager(B)");
        let graph = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert_graph(&*graph);
        graph.run(&request)?;
        assert_output(
            "runtime-managed eager(B)",
            &legacy.run(&changed_request)?.tensor().to_f32_vec()?,
            &eager_reference,
        );
        assert_retired(&*graph, &input);
        assert_output(
            "Graph(A) rebuilt after runtime-managed(B)",
            &runtime.infer_host_f32(&request)?,
            &reference,
        );
        assert_eq!(runtime.execution_mode(), "musa-graph");

        println!("[5/5] Explicit Graph(A) -> prepare_for(B, Eager) -> Graph(A) rebuilt");
        let graph = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert_graph(&*graph);
        graph.run(&request)?;
        let eager = runtime.prepare_for(&changed_request, ExecutionPolicy::Eager)?;
        assert!(matches!(
            eager.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                ..
            }
        ));
        assert_retired(&*graph, &input);
        let output = eager.run(&changed_request)?.tensor().to_f32_vec()?;
        assert!(!output.is_empty() && output.iter().all(|value| value.is_finite()));
        assert_output(
            "Graph(A) rebuilt after prepared eager(B)",
            &runtime.infer_host_f32(&request)?,
            &reference,
        );
        assert_eq!(runtime.execution_mode(), "musa-graph");
        assert_eq!(eager.status(), PreparationStatus::Invalidated);
        println!("PASS all calibration/runtime-managed/prepared-eager GPU lifecycle checks");
        Ok(())
    }

    #[test]
    #[ignore = "requires MUSA GPU, GR00T checkpoint and preprocessed fixture"]
    fn graph_token_cache_gpu() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(std::env::var("APXINF_GR00T_PREPARED_ARENA")?, "1");
        let (model, input, mut changed) = setup()?;
        changed.embodiment_id = input.embodiment_id;
        let text_positions = input
            .observation
            .token_ids
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (1000..100000).contains(token).then_some(index))
            .collect::<Vec<_>>();
        let first = *text_positions
            .first()
            .ok_or("fixture needs ordinary text tokens")?;
        let last = *text_positions
            .last()
            .ok_or("fixture needs ordinary text tokens")?;
        assert_ne!(
            input.observation.token_ids[first],
            input.observation.token_ids[last]
        );
        changed.observation.token_ids.swap(first, last);
        println!(
            "Token-cache scenario: same embodiment={}, length={}, swap ordinary tokens at {first}/{last}; image tokens unchanged",
            input.embodiment_id, input.observation.token_ids.len()
        );
        let runtime = model.vla()?;
        let request = input.request();
        let changed_request = changed.request();
        let legacy = runtime.prepare(&input.observation.inference_spec())?;
        let eager_reference = legacy.run(&request)?.tensor().to_f32_vec()?;
        let calibration_reference = runtime.calibration_amax(&request)?;

        println!("[1/3] Capture Graph(A), replay A then B using the SAME plan");
        let graph = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert_graph(&*graph);
        let graph_reference = graph.run(&request)?.tensor().to_f32_vec()?;
        let changed_output = graph.run(&changed_request)?.tensor().to_f32_vec()?;
        let changed_max_abs = changed_output
            .iter()
            .zip(&graph_reference)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(changed_output.iter().all(|value| value.is_finite()));
        assert!(
            changed_max_abs > 1e-5,
            "changed tokens must affect model output"
        );
        println!("PASS B changes output: max_abs(A,B)={changed_max_abs}");

        println!("[2/3] Eager(A) after Graph(B) must match clean eager(A), not read B tokens");
        let actual = legacy.run(&request)?.tensor().to_f32_vec()?;
        assert_eq!(actual.len(), eager_reference.len());
        assert!(actual.iter().all(|value| value.is_finite()));
        let max_abs = actual
            .iter()
            .zip(&eager_reference)
            .map(|(value, expected)| (value - expected).abs())
            .fold(0.0_f32, f32::max);
        println!("Eager(A) vs clean eager(A): max_abs={max_abs}");
        assert_eq!(max_abs, 0.0, "stale token cache changed eager(A) output");
        assert_retired(&*graph, &input);

        println!("[3/3] Capture A, replay B, then calibration(A) must match clean calibration(A)");
        let graph = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        graph.run(&changed_request)?;
        let calibration = runtime.calibration_amax(&request)?;
        assert!(!calibration.is_empty());
        assert_eq!(calibration.len(), calibration_reference.len());
        let mismatches = calibration
            .iter()
            .filter(|(name, value)| {
                !value.is_finite() || calibration_reference.get(*name) != Some(*value)
            })
            .count();
        assert_eq!(mismatches, 0, "calibration(A) used stale token data");
        println!(
            "PASS {} calibration sites exactly match clean A",
            calibration.len()
        );
        assert_retired(&*graph, &input);
        assert_output(
            "Graph(A) rebuilt after token changes",
            &runtime.infer_host_f32(&request)?,
            &graph_reference,
        );
        assert_eq!(runtime.execution_mode(), "musa-graph");
        println!("PASS token-cache GPU regression");
        Ok(())
    }

    #[test]
    #[ignore = "requires a separate process with APXINF_GR00T_PREPARED_ARENA=0"]
    fn graph_direct_eager_gpu() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(std::env::var("APXINF_GR00T_PREPARED_ARENA")?, "0");
        let (model, input, changed) = setup()?;
        let runtime = model.vla()?;
        let request = input.request();
        let changed_request = changed.request();
        let eager_reference = runtime.infer_host_f32(&changed_request)?;

        println!("[1/3] Explicit RequireGraph(A) even though implicit arena is disabled");
        let graph = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert_graph(&*graph);
        let reference = graph.run(&request)?.tensor().to_f32_vec()?;
        println!("[2/3] Direct infer(B) takes eager path and must retire Graph(A)");
        assert_output(
            "direct eager(B)",
            &runtime.infer_host_f32(&changed_request)?,
            &eager_reference,
        );
        assert_eq!(runtime.execution_mode(), "no_graph");
        assert_retired(&*graph, &input);

        println!("[3/3] Explicit RequireGraph(A) rebuilt and replayed");
        let rebuilt = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert_graph(&*rebuilt);
        assert_output(
            "Graph(A) rebuilt after direct eager(B)",
            &rebuilt.run(&request)?.tensor().to_f32_vec()?,
            &reference,
        );
        println!("PASS direct-eager GPU lifecycle checks");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        execution_policy_name, parse_execution_policy, ExecutionPolicy, DEFAULT_EXECUTION_POLICY,
    };

    #[test]
    fn benchmark_rejects_zero_warmup() {
        let mut values: Vec<String> = ["bench", "checkpoint", "backbone", "fixture", "bf16", "0", "0"]
            .into_iter()
            .map(String::from)
            .collect();
        let error = super::parse_argument_values(&values).err().unwrap();
        assert_eq!(error.to_string(), "warmup count must be non-zero");
        values[6] = "1".into();
        assert_eq!(super::parse_argument_values(&values).unwrap().warmup, 1);
        values.truncate(5);
        assert_eq!(super::parse_argument_values(&values).unwrap().warmup, 10);
    }

    #[test]
    fn execution_policy_defaults_to_strict_graph_and_parses_all_modes() {
        assert_eq!(DEFAULT_EXECUTION_POLICY, ExecutionPolicy::RequireGraph);
        for (name, expected) in [
            ("eager", ExecutionPolicy::Eager),
            ("prefer-graph", ExecutionPolicy::PreferGraph),
            ("require-graph", ExecutionPolicy::RequireGraph),
        ] {
            assert_eq!(parse_execution_policy(name).unwrap(), expected);
            assert_eq!(execution_policy_name(expected), name);
        }
        assert!(parse_execution_policy("graph").is_err());
    }
}

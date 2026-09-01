//! Unified low-level latency benchmark for PI0.5 (BF16 / FP8 / INT8-W8A8).
//!
//! This single example replaces the former `pi05_{bf16,thor,int8}_bench.rs`. It
//! bypasses the unified `AutoModel`/`infer` frontend and drives the dtype-native
//! `Pi05{Bf16,,Int8}CudaRuntime` directly, because it needs `apxinf_cuda`
//! profiler hooks, tuning-DB install and raw device inputs that the model
//! abstraction does not (and should not) expose. For an abstraction-level entry
//! point see `pi05_auto_smoke`.
//!
//! The benchmark runs **checkpoint-free** with deterministic random weights
//! (`<source>` == `random`), because graph-replay latency depends only on tensor
//! shape and dtype, not on trained values. Pass a checkpoint path/index instead
//! to measure a real model and validate against a captured reference.
//!
//! ```text
//! pi05_bench <checkpoint-or-index|random> --dtype {bf16,fp8,int8}
//!     [--calibration <json|uniform:SCALE>] [--tactics <json>] [--autotune]
//!     [--views N] [--image-size N] [--action-horizon N] [--action-dim N]
//!     [--num-flow-steps N] [--max-token-len N]            (random-only overrides)
//!     [--token-count T] [--iterations N] [--seed N]
//!     [--image-input patches|nhwc|nchw] [--reference <json>] [--min-cosine C]
//!     [--images-u8 <raw>] [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>]
//! ```
//!
//! nsys: set `APXINF_PI05_PROFILE_REPLAY=1` to wrap exactly one steady-state
//! graph replay in an NVTX range under the CUDA profiler API, e.g.
//!   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
//! `APXINF_PI05_EAGER_ONLY=1` stops after the eager integrity pass, and
//! `APXINF_PI05_IMAGE_INPUT` mirrors `--image-input` for scripted runs.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{Backend, DType, Tensor};
use apxinf_cuda::{CudaBackend, CudaBuffer};
use apxinf_model::pi05::{
    upload_time_embeddings, upload_time_embeddings_bf16, upload_time_embeddings_int8,
    Pi05ActivationScales, Pi05Bf16CapturedGraph, Pi05Bf16CudaRuntime, Pi05CapturedGraph,
    Pi05Config, Pi05CudaRuntime, Pi05ImageLayout, Pi05Int8CapturedGraph, Pi05Int8CudaRuntime,
    Pi05Weights, StaticBf16Pi05Weights, StaticFp8Calibration, StaticFp8Pi05Weights,
    StaticInt8Pi05Weights,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dtype {
    Bf16,
    Fp8,
    Int8,
}

impl Dtype {
    fn parse(spec: &str) -> Result<Self, String> {
        match spec {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            "int8" | "w8a8" => Ok(Self::Int8),
            other => Err(format!("--dtype must be bf16, fp8, or int8; got {other}")),
        }
    }

    /// Device dtype of the patch/noise inputs feeding the graph.
    fn io_dtype(self) -> DType {
        match self {
            Self::Fp8 => DType::F16,
            Self::Bf16 | Self::Int8 => DType::BF16,
        }
    }

    fn precision_label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
            Self::Int8 => "int8_w8a8",
        }
    }

    fn patches_label(self) -> &'static str {
        match self {
            Self::Fp8 => "patches_f16",
            Self::Bf16 | Self::Int8 => "patches_bf16",
        }
    }

    /// Eager-vs-graph and reference-vs-graph acceptance gates. FP8 tightens the
    /// eager/graph max-abs; INT8 additionally bounds max-abs against a reference.
    fn thresholds(self) -> Thresholds {
        match self {
            Self::Bf16 => Thresholds {
                eager_graph_max_abs: 1e-2,
                reference_min_cosine: 0.999,
                reference_max_relative_l2: 0.05,
                reference_max_abs: None,
            },
            Self::Fp8 => Thresholds {
                eager_graph_max_abs: 1e-3,
                reference_min_cosine: 0.997,
                reference_max_relative_l2: 0.10,
                reference_max_abs: None,
            },
            Self::Int8 => Thresholds {
                eager_graph_max_abs: 1e-2,
                reference_min_cosine: 0.995,
                reference_max_relative_l2: 0.10,
                reference_max_abs: Some(0.125),
            },
        }
    }
}

/// Per-dtype integrity gates (see `Dtype::thresholds`). `eager_graph_min_cosine`
/// is a shared `0.999_999`; the reference cosine floor may be overridden.
#[derive(Clone, Copy, Debug)]
struct Thresholds {
    eager_graph_max_abs: f64,
    reference_min_cosine: f64,
    reference_max_relative_l2: f64,
    reference_max_abs: Option<f64>,
}

const EAGER_GRAPH_MIN_COSINE: f64 = 0.999_999;

/// The dtype-native runtime. The three concrete runtimes expose an identical
/// `infer`/`capture_infer`/`capture_infer_rgb_u8` surface, so the only per-dtype
/// fork in the whole benchmark is this construction + delegation.
enum Bench {
    Bf16(Pi05Bf16CudaRuntime),
    Fp8(Pi05CudaRuntime),
    Int8(Pi05Int8CudaRuntime),
}

impl Bench {
    fn infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Bf16(rt) => rt.infer(patches, token_ids, token_count, noise, time_embeddings)?,
            Self::Fp8(rt) => rt.infer(patches, token_ids, token_count, noise, time_embeddings)?,
            Self::Int8(rt) => rt.infer(patches, token_ids, token_count, noise, time_embeddings)?,
        })
    }

    fn capture_infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Graph, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Bf16(rt) => Graph::Bf16(rt.capture_infer(
                patches,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?),
            Self::Fp8(rt) => Graph::Fp8(rt.capture_infer(
                patches,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?),
            Self::Int8(rt) => Graph::Int8(rt.capture_infer(
                patches,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?),
        })
    }

    fn capture_infer_rgb_u8(
        &self,
        layout: Pi05ImageLayout,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Graph, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Bf16(rt) => Graph::Bf16(rt.capture_infer_rgb_u8(
                layout,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?),
            Self::Fp8(rt) => Graph::Fp8(rt.capture_infer_rgb_u8(
                layout,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?),
            Self::Int8(rt) => Graph::Int8(rt.capture_infer_rgb_u8(
                layout,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?),
        })
    }
}

/// The captured CUDA graph. Like `Bench`, the three concrete graph types share
/// an identical method set, delegated here so the harness stays dtype-agnostic.
enum Graph {
    Bf16(Pi05Bf16CapturedGraph),
    Fp8(Pi05CapturedGraph),
    Int8(Pi05Int8CapturedGraph),
}

impl Graph {
    fn replay(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Bf16(g) => g.replay()?,
            Self::Fp8(g) => g.replay()?,
            Self::Int8(g) => g.replay()?,
        }
        Ok(())
    }

    fn replay_and_synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Bf16(g) => g.replay_and_synchronize()?,
            Self::Fp8(g) => g.replay_and_synchronize()?,
            Self::Int8(g) => g.replay_and_synchronize()?,
        }
        Ok(())
    }

    fn output(&self) -> &Tensor {
        match self {
            Self::Bf16(g) => g.output(),
            Self::Fp8(g) => g.output(),
            Self::Int8(g) => g.output(),
        }
    }

    fn update_raw_image_inputs(
        &self,
        images: &[u8],
        token_ids: &[u32],
        noise: &Tensor,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Self::Bf16(g) => g.update_raw_image_inputs(images, token_ids, noise)?,
            Self::Fp8(g) => g.update_raw_image_inputs(images, token_ids, noise)?,
            Self::Int8(g) => g.update_raw_image_inputs(images, token_ids, noise)?,
        }
        Ok(())
    }

    fn workspace_bytes(&self) -> usize {
        match self {
            Self::Bf16(g) => g.workspace_bytes(),
            Self::Fp8(g) => g.workspace_bytes(),
            Self::Int8(g) => g.workspace_bytes(),
        }
    }

    fn workspace_used_bytes(&self) -> usize {
        match self {
            Self::Bf16(g) => g.workspace_used_bytes(),
            Self::Fp8(g) => g.workspace_used_bytes(),
            Self::Int8(g) => g.workspace_used_bytes(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ImageInput {
    Patches,
    Rgb(Pi05ImageLayout),
}

impl ImageInput {
    /// `--image-input` (via `spec`) takes precedence over the
    /// `APXINF_PI05_IMAGE_INPUT` env var, which defaults to `patches`.
    fn resolve(spec: Option<&str>) -> Result<Self, String> {
        let owned;
        let value = match spec {
            Some(value) => value,
            None => {
                owned =
                    std::env::var("APXINF_PI05_IMAGE_INPUT").unwrap_or_else(|_| "patches".into());
                owned.as_str()
            }
        };
        match value {
            "patches" => Ok(Self::Patches),
            "nhwc" => Ok(Self::Rgb(Pi05ImageLayout::Nhwc)),
            "nchw" => Ok(Self::Rgb(Pi05ImageLayout::Nchw)),
            other => Err(format!(
                "image input must be patches, nhwc, or nchw; got {other}"
            )),
        }
    }

    fn label(self, dtype: Dtype) -> &'static str {
        match self {
            Self::Patches => dtype.patches_label(),
            Self::Rgb(Pi05ImageLayout::Nhwc) => "rgb_u8_nhwc",
            Self::Rgb(Pi05ImageLayout::Nchw) => "rgb_u8_nchw",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ErrorMetrics {
    cosine: f64,
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
    relative_l2: f64,
    bitwise_equal: bool,
}

impl ErrorMetrics {
    fn measure(actual: &[f32], expected: &[f32]) -> Result<Self, Box<dyn std::error::Error>> {
        if actual.len() != expected.len() || actual.is_empty() {
            return Err(format!(
                "comparison length mismatch: actual {}, expected {}",
                actual.len(),
                expected.len()
            )
            .into());
        }
        if actual
            .iter()
            .chain(expected)
            .any(|value| !value.is_finite())
        {
            return Err("integrity comparison contains a non-finite value".into());
        }

        let mut dot = 0.0f64;
        let mut actual_squared = 0.0f64;
        let mut expected_squared = 0.0f64;
        let mut error_squared = 0.0f64;
        let mut error_sum = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut bitwise_equal = true;
        for (&actual, &expected) in actual.iter().zip(expected) {
            let actual_f64 = f64::from(actual);
            let expected_f64 = f64::from(expected);
            let error = (actual_f64 - expected_f64).abs();
            dot += actual_f64 * expected_f64;
            actual_squared += actual_f64 * actual_f64;
            expected_squared += expected_f64 * expected_f64;
            error_squared += error * error;
            error_sum += error;
            max_abs = max_abs.max(error);
            bitwise_equal &= actual.to_bits() == expected.to_bits();
        }
        let cosine = if actual_squared == 0.0 || expected_squared == 0.0 {
            f64::from(actual_squared == expected_squared)
        } else {
            dot / (actual_squared.sqrt() * expected_squared.sqrt())
        };
        let rmse = (error_squared / actual.len() as f64).sqrt();
        let relative_l2 = if expected_squared == 0.0 {
            if error_squared == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (error_squared / expected_squared).sqrt()
        };
        Ok(Self {
            cosine,
            max_abs,
            mean_abs: error_sum / actual.len() as f64,
            rmse,
            relative_l2,
            bitwise_equal,
        })
    }

    fn as_json(self) -> serde_json::Value {
        serde_json::json!({
            "cosine": self.cosine,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rmse": self.rmse,
            "relative_l2": self.relative_l2,
            "bitwise_equal": self.bitwise_equal,
        })
    }
}

fn raw_image_fixture(
    config: &Pi05Config,
    layout: Pi05ImageLayout,
    io_dtype: DType,
    fixture_path: Option<&str>,
) -> Result<(Vec<u8>, Tensor), Box<dyn std::error::Error>> {
    let image_bytes = config.num_views * config.image_size * config.image_size * 3;
    let nhwc = if let Some(path) = fixture_path {
        if !matches!(layout, Pi05ImageLayout::Nhwc) {
            return Err("--images-u8 currently requires --image-input nhwc".into());
        }
        let bytes = std::fs::read(path)?;
        if bytes.len() != image_bytes {
            return Err(format!(
                "--images-u8 length mismatch: expected {image_bytes}, got {}",
                bytes.len()
            )
            .into());
        }
        bytes
    } else {
        (0..image_bytes)
            .map(|index| ((index * 73 + index / 11 + 19) & 0xff) as u8)
            .collect::<Vec<_>>()
    };
    let wire_images = match layout {
        Pi05ImageLayout::Nhwc => nhwc.clone(),
        Pi05ImageLayout::Nchw => {
            let mut nchw = vec![0u8; image_bytes];
            for view in 0..config.num_views {
                for y in 0..config.image_size {
                    for x in 0..config.image_size {
                        for channel in 0..3 {
                            let source = ((view * config.image_size + y) * config.image_size + x)
                                * 3
                                + channel;
                            let destination = ((view * 3 + channel) * config.image_size + y)
                                * config.image_size
                                + x;
                            nchw[destination] = nhwc[source];
                        }
                    }
                }
            }
            nchw
        }
    };

    let patches_per_side = config.image_size / config.patch_size;
    let patch_rows = config.num_views * config.patches_per_view();
    let patch_width = 3 * config.patch_size * config.patch_size;
    let mut normalized = vec![0.0f32; patch_rows * patch_width];
    for view in 0..config.num_views {
        for patch_y in 0..patches_per_side {
            for patch_x in 0..patches_per_side {
                let row = view * config.patches_per_view() + patch_y * patches_per_side + patch_x;
                for channel in 0..3 {
                    for dy in 0..config.patch_size {
                        for dx in 0..config.patch_size {
                            let y = patch_y * config.patch_size + dy;
                            let x = patch_x * config.patch_size + dx;
                            let source = ((view * config.image_size + y) * config.image_size + x)
                                * 3
                                + channel;
                            let column = channel * config.patch_size * config.patch_size
                                + dy * config.patch_size
                                + dx;
                            normalized[row * patch_width + column] =
                                (f32::from(nhwc[source]) / 255.0) * 2.0 - 1.0;
                        }
                    }
                }
            }
        }
    }
    let patches = match io_dtype {
        DType::F16 => {
            let values = normalized
                .iter()
                .map(|&value| half::f16::from_f32(value))
                .collect::<Vec<_>>();
            Tensor::from_f16(vec![patch_rows, patch_width], &values)?
        }
        _ => {
            let values = normalized
                .iter()
                .map(|&value| half::bf16::from_f32(value))
                .collect::<Vec<_>>();
            Tensor::from_bf16(vec![patch_rows, patch_width], &values)?
        }
    };
    Ok((wire_images, patches))
}

fn token_fixture(
    path: Option<&str>,
    token_count: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let Some(path) = path else {
        return Ok(vec![0; token_count]);
    };
    let bytes = std::fs::read(path)?;
    if bytes.len() != token_count * 4 {
        return Err(format!(
            "--token-ids-u32le length mismatch: expected {}, got {}",
            token_count * 4,
            bytes.len()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn noise_fixture(
    path: Option<&str>,
    config: &Pi05Config,
    io_dtype: DType,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let elements = config.action_horizon * config.action_dim;
    let Some(path) = path else {
        return Ok(Tensor::zeros(
            vec![config.action_horizon, config.action_dim],
            io_dtype,
        ));
    };
    let bytes = std::fs::read(path)?;
    if bytes.len() != elements * 2 {
        return Err(format!(
            "--noise-bf16-u16le length mismatch: expected {}, got {}",
            elements * 2,
            bytes.len()
        )
        .into());
    }
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
        .collect::<Vec<_>>();
    Ok(match io_dtype {
        DType::F16 => Tensor::from_f16(
            vec![config.action_horizon, config.action_dim],
            &values
                .iter()
                .map(|&value| half::f16::from_f32(value))
                .collect::<Vec<_>>(),
        )?,
        _ => Tensor::from_bf16(
            vec![config.action_horizon, config.action_dim],
            &values
                .iter()
                .map(|&value| half::bf16::from_f32(value))
                .collect::<Vec<_>>(),
        )?,
    })
}

fn latency_json(mut milliseconds: Vec<f64>) -> serde_json::Value {
    milliseconds.sort_by(f64::total_cmp);
    let sample_count = milliseconds.len() as f64;
    let mean = milliseconds.iter().sum::<f64>() / sample_count;
    let variance = milliseconds
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / sample_count;
    let percentile = |fraction: f64| {
        let index = ((milliseconds.len() - 1) as f64 * fraction).round() as usize;
        milliseconds[index]
    };
    serde_json::json!({
        "min": milliseconds[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": milliseconds[milliseconds.len() - 1],
        "mean": mean,
        "standard_deviation": variance.sqrt()
    })
}

/// Reference actions parser. FP8 uses a strict schema-validated fixture (matching
/// the former `pi05_thor_bench`); BF16/INT8 accept a bare `{ "raw_actions": [..] }`.
fn reference_actions(
    path: &Path,
    dtype: Dtype,
    config: &Pi05Config,
    token_count: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let document: serde_json::Value = serde_json::from_str(&raw)?;
    if dtype == Dtype::Fp8 {
        let expected_integer = |name: &str, expected: usize| -> Result<(), String> {
            let actual = document
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("reference `{name}` is missing or not an integer"))?;
            if actual as usize != expected {
                return Err(format!(
                    "reference `{name}` mismatch: expected {expected}, got {actual}"
                ));
            }
            Ok(())
        };
        if document.get("schema").and_then(serde_json::Value::as_str)
            != Some("apxinf.pi05.integrity.v1")
        {
            return Err("unsupported π0.5 integrity reference schema".into());
        }
        expected_integer("num_views", config.num_views)?;
        expected_integer("token_count", token_count)?;
        expected_integer("action_horizon", config.action_horizon)?;
        expected_integer("action_dim", config.action_dim)?;
        expected_integer("flow_steps", config.num_flow_steps)?;
        for name in ["normalized_images", "token_ids", "diffusion_noise"] {
            let value = document
                .get("fixture")
                .and_then(|fixture| fixture.get(name))
                .and_then(serde_json::Value::as_str);
            if value != Some("zeros") {
                return Err(format!("reference fixture `{name}` must be `zeros`").into());
            }
        }
    }
    let actions = document
        .get("raw_actions")
        .and_then(serde_json::Value::as_array)
        .ok_or("reference `raw_actions` is missing or not an array")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map(|value| value as f32)
                .ok_or("reference action is not numeric")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = config.action_horizon * config.action_dim;
    if actions.len() != expected {
        return Err(format!(
            "reference action length mismatch: expected {expected}, got {}",
            actions.len()
        )
        .into());
    }
    Ok(actions)
}

/// Parsed command line. `overrides` are architecture fields that only make sense
/// in random mode (a checkpoint's weights are fixed to its own config).
#[derive(Debug)]
struct Args {
    source: String,
    dtype: Dtype,
    calibration: Option<String>,
    tactics: Option<String>,
    autotune: bool,
    views: Option<usize>,
    image_size: Option<usize>,
    action_horizon: Option<usize>,
    action_dim: Option<usize>,
    num_flow_steps: Option<usize>,
    max_token_len: Option<usize>,
    token_count: usize,
    iterations: usize,
    seed: u64,
    reference: Option<String>,
    min_cosine: Option<f64>,
    image_input: Option<String>,
    images_u8: Option<String>,
    token_ids_u32le: Option<String>,
    noise_bf16_u16le: Option<String>,
}

impl Args {
    fn parse(raw: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        // Consume the value that follows a flag, advancing the cursor past it.
        fn expect_value(raw: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
            *index += 1;
            raw.get(*index)
                .cloned()
                .ok_or_else(|| format!("{flag} requires a value"))
        }

        let mut source: Option<String> = None;
        let mut dtype: Option<Dtype> = None;
        let mut calibration = None;
        let mut tactics = None;
        let mut autotune = false;
        let mut views = None;
        let mut image_size = None;
        let mut action_horizon = None;
        let mut action_dim = None;
        let mut num_flow_steps = None;
        let mut max_token_len = None;
        let mut token_count = 200usize;
        let mut iterations = 30usize;
        let mut seed = 0u64;
        let mut reference = None;
        let mut min_cosine = None;
        let mut image_input = None;
        let mut images_u8 = None;
        let mut token_ids_u32le = None;
        let mut noise_bf16_u16le = None;

        let mut index = 1;
        while index < raw.len() {
            let argument = raw[index].as_str();
            match argument {
                "--dtype" => {
                    dtype = Some(Dtype::parse(&expect_value(raw, &mut index, "--dtype")?)?)
                }
                "--calibration" => {
                    calibration = Some(expect_value(raw, &mut index, "--calibration")?)
                }
                "--tactics" => tactics = Some(expect_value(raw, &mut index, "--tactics")?),
                "--autotune" => autotune = true,
                "--views" => views = Some(expect_value(raw, &mut index, "--views")?.parse()?),
                "--image-size" => {
                    image_size = Some(expect_value(raw, &mut index, "--image-size")?.parse()?)
                }
                "--action-horizon" => {
                    action_horizon =
                        Some(expect_value(raw, &mut index, "--action-horizon")?.parse()?)
                }
                "--action-dim" => {
                    action_dim = Some(expect_value(raw, &mut index, "--action-dim")?.parse()?)
                }
                "--num-flow-steps" => {
                    num_flow_steps =
                        Some(expect_value(raw, &mut index, "--num-flow-steps")?.parse()?)
                }
                "--max-token-len" => {
                    max_token_len = Some(expect_value(raw, &mut index, "--max-token-len")?.parse()?)
                }
                "--token-count" => {
                    token_count = expect_value(raw, &mut index, "--token-count")?.parse()?
                }
                "--iterations" => {
                    iterations = expect_value(raw, &mut index, "--iterations")?.parse()?
                }
                "--seed" => seed = expect_value(raw, &mut index, "--seed")?.parse()?,
                "--reference" => reference = Some(expect_value(raw, &mut index, "--reference")?),
                "--min-cosine" => {
                    min_cosine = Some(expect_value(raw, &mut index, "--min-cosine")?.parse()?)
                }
                "--image-input" => {
                    image_input = Some(expect_value(raw, &mut index, "--image-input")?)
                }
                "--images-u8" => images_u8 = Some(expect_value(raw, &mut index, "--images-u8")?),
                "--token-ids-u32le" => {
                    token_ids_u32le = Some(expect_value(raw, &mut index, "--token-ids-u32le")?)
                }
                "--noise-bf16-u16le" => {
                    noise_bf16_u16le = Some(expect_value(raw, &mut index, "--noise-bf16-u16le")?)
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown flag `{other}`").into());
                }
                positional => {
                    if source.replace(positional.to_string()).is_some() {
                        return Err("unexpected second positional argument".into());
                    }
                }
            }
            index += 1;
        }

        let source = source.ok_or("missing <checkpoint-or-index|random> positional argument")?;
        let dtype = dtype.ok_or("missing required --dtype {bf16,fp8,int8}")?;
        validate_explicit_tactics_path(tactics.as_deref(), autotune)?;
        if iterations == 0 {
            return Err("--iterations must be non-zero".into());
        }
        if let Some(cosine) = min_cosine {
            if !(0.0..=1.0).contains(&cosine) {
                return Err("--min-cosine must be in 0..=1".into());
            }
        }
        Ok(Self {
            source,
            dtype,
            calibration,
            tactics,
            autotune,
            views,
            image_size,
            action_horizon,
            action_dim,
            num_flow_steps,
            max_token_len,
            token_count,
            iterations,
            seed,
            reference,
            min_cosine,
            image_input,
            images_u8,
            token_ids_u32le,
            noise_bf16_u16le,
        })
    }
}

fn validate_explicit_tactics_path(tactics: Option<&str>, autotune: bool) -> Result<(), String> {
    let Some(path) = tactics else {
        return Ok(());
    };
    if autotune || Path::new(path).is_file() {
        return Ok(());
    }
    Err(format!(
        "explicit --tactics path `{path}` does not exist or is not a file; pass --autotune to create a new database"
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::env::args().collect::<Vec<_>>();
    let args = Args::parse(&raw).map_err(|error| {
        format!(
            "{error}\nusage: {} <checkpoint-or-index|random> --dtype {{bf16,fp8,int8}} \
             [--calibration <json|uniform:SCALE>] [--tactics <json>] [--autotune] [--views N] \
             [--image-size N] [--action-horizon N] [--action-dim N] [--num-flow-steps N] \
             [--max-token-len N] [--token-count T] [--iterations N] [--seed N] \
             [--image-input patches|nhwc|nchw] [--reference <json>] [--min-cosine C] \
             [--images-u8 <raw>] [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>]",
            raw.first().map(String::as_str).unwrap_or("pi05_bench")
        )
    })?;

    let random = args.source == "random";
    let dtype = args.dtype;
    let thresholds = dtype.thresholds();
    let token_count = args.token_count;
    let iterations = args.iterations;
    let image_input = ImageInput::resolve(args.image_input.as_deref())?;
    if args.images_u8.is_some() && !matches!(image_input, ImageInput::Rgb(_)) {
        return Err("--images-u8 requires --image-input nhwc".into());
    }

    // Architecture overrides only apply to synthetic weights; a real checkpoint's
    // tensors are fixed to the config it was exported with.
    let has_overrides = args.views.is_some()
        || args.image_size.is_some()
        || args.action_horizon.is_some()
        || args.action_dim.is_some()
        || args.num_flow_steps.is_some()
        || args.max_token_len.is_some();
    if !random && has_overrides {
        return Err(
            "architecture overrides (--views/--image-size/--action-horizon/--action-dim/\
             --num-flow-steps/--max-token-len) are only valid with the `random` source"
                .into(),
        );
    }
    if random && args.reference.is_some() {
        return Err(
            "--reference is not valid with the `random` source (no trained weights to match); \
             eager-vs-graph integrity still runs as a self-test"
                .into(),
        );
    }
    if matches!(image_input, ImageInput::Rgb(_)) && args.reference.is_some() {
        return Err(
            "a raw-image fixture cannot be validated against the zero-input reference".into(),
        );
    }
    // All GEMM precisions share the hardware tactic database; calibration is
    // still specific to FP8 activations.
    if dtype != Dtype::Fp8 && args.calibration.is_some() {
        return Err("--calibration only applies to --dtype fp8".into());
    }

    let config = if random {
        // Random-benchmark defaults mirror `apxinf_py.Model.random` (2-view / H10)
        // so the Rust and Python entry points measure the same shape by default,
        // rather than `Pi05Config::default()` (3-view / H50). Untouched architecture
        // fields (patch/vision/language/action widths) come from `default()`.
        let base = Pi05Config::default();
        Pi05Config {
            num_views: args.views.unwrap_or(2),
            image_size: args.image_size.unwrap_or(224),
            action_horizon: args.action_horizon.unwrap_or(10),
            action_dim: args.action_dim.unwrap_or(32),
            num_flow_steps: args.num_flow_steps.unwrap_or(10),
            max_token_len: args.max_token_len.unwrap_or(200),
            ..base
        }
    } else {
        // A real checkpoint is benchmarked at its native config (H50 for
        // pi05_libero_base), matching the Python L2/L3 path and the LIBERO eval —
        // not a reduced thor_two_view() workload. Resolve `config.json` from the
        // source root the same way the loader does (see `load_config` in
        // apxinf-py/src/lib.rs), falling back to `default()` when it is absent.
        let source = Path::new(&args.source);
        let root = if source.is_dir() {
            source
        } else {
            source.parent().unwrap_or_else(|| Path::new("."))
        };
        let config_path = root.join("config.json");
        if config_path.is_file() {
            Pi05Config::from_json_file(&config_path)?
        } else {
            Pi05Config::default()
        }
    };
    config.validate()?;
    let config = Arc::new(config);

    let backend = Arc::new(CudaBackend::new(0)?);

    // BF16/FP8 tactics are optional: kernels retain their fallback route when no
    // tuning DB is installed. Supplying the production DB is required for a
    // benchmark that claims production routing.
    let tuning_paths = args
        .tactics
        .as_deref()
        .map(apxinf_cuda::tuning::TuningPaths::from_tactics)
        .unwrap_or_else(|| {
            apxinf_cuda::tuning::TuningPaths::for_cuda("configs/tuning", backend.context().caps())
        });
    let tuning = if tuning_paths.tactics.is_file() {
        Some(apxinf_cuda::tuning::TuningDb::from_json_file(
            &tuning_paths.tactics,
        )?)
    } else {
        None
    };
    let tuning_mode = if args.autotune {
        apxinf_cuda::tuning::TuningMode::AutoTune
    } else {
        apxinf_cuda::tuning::TuningMode::Inference
    };
    apxinf_cuda::kernels::gemm::configure_tuning(
        backend.context(),
        tuning_mode,
        tuning.as_ref().map(std::slice::from_ref).unwrap_or(&[]),
        Some(tuning_paths),
    )?;

    if random {
        eprintln!(
            "building deterministic random π0.5 weights (seed {})...",
            args.seed
        );
    } else {
        eprintln!("loading π0.5 checkpoint...");
    }
    let host_weights = if random {
        Pi05Weights::synthetic(&config, args.seed)?
    } else {
        Pi05Weights::from_safetensors(&config, Path::new(&args.source))?
    };

    let bench = match dtype {
        Dtype::Bf16 => {
            eprintln!("converting and uploading native BF16 weights...");
            let device_weights = Arc::new(StaticBf16Pi05Weights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            Bench::Bf16(Pi05Bf16CudaRuntime::new(
                backend.clone(),
                config.clone(),
                device_weights,
            )?)
        }
        Dtype::Fp8 => {
            let scales =
                match args.calibration.as_deref() {
                    Some(spec) if spec.starts_with("uniform:") => {
                        eprintln!(
                            "warning: uniform activation scales are for smoke/latency tests only"
                        );
                        Arc::new(Pi05ActivationScales::uniform(
                            &config,
                            spec["uniform:".len()..].parse()?,
                        )?)
                    }
                    Some(path) => {
                        let calibration = StaticFp8Calibration::from_json_file(Path::new(path))?;
                        Arc::new(Pi05ActivationScales::from_calibration(
                            &config,
                            &calibration,
                        )?)
                    }
                    None if random => {
                        eprintln!("warning: no --calibration; using uniform activation scale 1.0");
                        Arc::new(Pi05ActivationScales::uniform(&config, 1.0)?)
                    }
                    None => return Err(
                        "--dtype fp8 requires --calibration <json|uniform:SCALE> for a checkpoint"
                            .into(),
                    ),
                };
            eprintln!("quantizing and uploading static FP8 weights...");
            let device_weights = Arc::new(StaticFp8Pi05Weights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            Bench::Fp8(Pi05CudaRuntime::new(
                backend.clone(),
                config.clone(),
                device_weights,
                scales,
            )?)
        }
        Dtype::Int8 => {
            eprintln!("quantizing and uploading per-channel INT8 weights...");
            let device_weights =
                Arc::new(StaticInt8Pi05Weights::from_host(&host_weights, &backend)?);
            Bench::Int8(Pi05Int8CudaRuntime::new(
                backend.clone(),
                config.clone(),
                device_weights,
            )?)
        }
    };
    drop(host_weights);

    let io_dtype = dtype.io_dtype();
    let patch_rows = config.num_views * config.patches_per_view();
    let patch_width = 3 * config.patch_size * config.patch_size;
    let (raw_images, patches_host) = match image_input {
        ImageInput::Patches => (None, Tensor::zeros(vec![patch_rows, patch_width], io_dtype)),
        ImageInput::Rgb(layout) => {
            let (images, patches) =
                raw_image_fixture(&config, layout, io_dtype, args.images_u8.as_deref())?;
            (Some(images), patches)
        }
    };
    let patches = backend.to_device(&patches_host)?;
    let noise_host = noise_fixture(args.noise_bf16_u16le.as_deref(), &config, io_dtype)?;
    let noise = backend.to_device(&noise_host)?;
    let host_token_ids = token_fixture(args.token_ids_u32le.as_deref(), token_count)?;
    let token_bytes = host_token_ids
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<_>>();
    let token_ids = CudaBuffer::alloc_zeros(token_bytes.len(), backend.device_id())
        .map_err(std::io::Error::other)?;
    token_ids
        .copy_from_host(&token_bytes)
        .map_err(std::io::Error::other)?;
    let time_embeddings = match dtype {
        Dtype::Bf16 => upload_time_embeddings_bf16(&config, &*backend)?,
        Dtype::Fp8 => upload_time_embeddings(&config, &*backend)?,
        Dtype::Int8 => upload_time_embeddings_int8(&config, &*backend)?,
    };

    eprintln!(
        "running eager {} integrity pass...",
        dtype.precision_label()
    );
    let eager_output = bench.infer(&patches, &token_ids, token_count, &noise, &time_embeddings)?;
    let eager = backend.to_cpu(&eager_output)?.to_f32_vec()?;
    drop(eager_output);

    if std::env::var_os("APXINF_PI05_EAGER_ONLY").is_some() {
        let checksum = eager.iter().map(|value| value.abs() as f64).sum::<f64>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "precision": dtype.precision_label(),
                "mode": "eager_only",
                "token_count": token_count,
                "output_abs_checksum": checksum,
            }))?
        );
        return Ok(());
    }

    eprintln!(
        "capturing {} graph with {} input...",
        dtype.precision_label(),
        image_input.label(dtype)
    );
    let graph = match image_input {
        ImageInput::Patches => {
            bench.capture_infer(&patches, &token_ids, token_count, &noise, &time_embeddings)?
        }
        ImageInput::Rgb(layout) => {
            bench.capture_infer_rgb_u8(layout, &token_ids, token_count, &noise, &time_embeddings)?
        }
    };
    if let Some(images) = raw_images.as_ref() {
        graph.update_raw_image_inputs(images, &host_token_ids, &noise_host)?;
    }
    graph.replay_and_synchronize()?;
    let captured = backend.to_cpu(graph.output())?.to_f32_vec()?;
    let eager_graph = ErrorMetrics::measure(&captured, &eager)?;
    let eager_graph_passed = eager_graph.cosine >= EAGER_GRAPH_MIN_COSINE
        && eager_graph.max_abs <= thresholds.eager_graph_max_abs;

    let reference_min_cosine = args.min_cosine.unwrap_or(thresholds.reference_min_cosine);
    let reference_metrics = args
        .reference
        .as_ref()
        .map(|path| reference_actions(Path::new(path), dtype, &config, token_count))
        .transpose()?
        .map(|expected| ErrorMetrics::measure(&captured, &expected))
        .transpose()?;
    let reference_passed = reference_metrics.is_none_or(|metrics| {
        metrics.cosine >= reference_min_cosine
            && metrics.relative_l2 <= thresholds.reference_max_relative_l2
            && thresholds
                .reference_max_abs
                .is_none_or(|limit| metrics.max_abs <= limit)
    });

    for _ in 0..10 {
        graph.replay()?;
    }
    backend.synchronize()?;

    // Nsight Systems can use the CUDA profiler API as a robust one-shot capture
    // boundary and retain the NVTX name as a human-readable label:
    //
    //   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
    //
    // Keeping the range opt-in prevents profiling instrumentation from changing
    // the normal benchmark loop while allowing a trace to contain exactly one
    // steady-state graph replay (and none of model loading, graph construction,
    // integrity validation, or warm-up).
    let profile_replay = std::env::var_os("APXINF_PI05_PROFILE_REPLAY").is_some();
    if profile_replay {
        eprintln!("profiling one steady-state CUDA graph replay...");
        apxinf_cuda::profiler::start().map_err(std::io::Error::other)?;
        let replay_result = {
            let _range = apxinf_cuda::nvtx::range("pi05.graph_replay");
            graph.replay_and_synchronize()
        };
        let stop_result = apxinf_cuda::profiler::stop().map_err(std::io::Error::other);
        replay_result?;
        stop_result?;
    }

    let mut milliseconds = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        graph.replay_and_synchronize()?;
        milliseconds.push(start.elapsed().as_secs_f64() * 1e3);
    }
    let graph_latency = latency_json(milliseconds);
    let update_plus_graph_latency = if let Some(images) = raw_images.as_ref() {
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            graph.update_raw_image_inputs(images, &host_token_ids, &noise_host)?;
            graph.replay_and_synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1e3);
        }
        Some(latency_json(samples))
    } else {
        None
    };
    let output = backend.to_cpu(graph.output())?.to_f32_vec()?;
    let checksum = output.iter().map(|value| value.abs() as f64).sum::<f64>();
    let profile = format!(
        "pi05_{}view_h{}_steps{}",
        config.num_views, config.action_horizon, config.num_flow_steps
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "profile": profile,
            "precision": dtype.precision_label(),
            "weights": if random { "synthetic" } else { "checkpoint" },
            "image_input": {
                "kind": image_input.label(dtype),
                "graph_includes_cuda_preprocess": matches!(image_input, ImageInput::Rgb(_)),
                "graph_latency_includes_h2d": false,
                "input_update_plus_graph_latency_ms": update_plus_graph_latency,
            },
            "token_count": token_count,
            "fixture_inputs": {
                "images_u8": args.images_u8,
                "token_ids_u32le": args.token_ids_u32le,
                "noise_bf16_u16le": args.noise_bf16_u16le,
            },
            "iterations": iterations,
            "profile_replay": profile_replay,
            "latency_ms": graph_latency,
            "workspace": {
                "capacity_bytes": graph.workspace_bytes(),
                "used_bytes": graph.workspace_used_bytes()
            },
            "output_abs_checksum": checksum,
            "integrity": {
                "passed": eager_graph_passed && reference_passed,
                "eager_vs_graph": eager_graph.as_json(),
                "reference": reference_metrics.map(ErrorMetrics::as_json),
                "thresholds": {
                    "eager_graph_min_cosine": EAGER_GRAPH_MIN_COSINE,
                    "eager_graph_max_abs": thresholds.eager_graph_max_abs,
                    "reference_min_cosine": reference_min_cosine,
                    "reference_max_relative_l2": thresholds.reference_max_relative_l2,
                    "reference_max_abs": thresholds.reference_max_abs,
                }
            }
        }))?
    );
    if !eager_graph_passed {
        return Err(format!(
            "eager/graph integrity failed: cosine {:.9}, max_abs {:.6}",
            eager_graph.cosine, eager_graph.max_abs
        )
        .into());
    }
    if !reference_passed {
        let metrics = reference_metrics.expect("reference failure requires metrics");
        return Err(format!(
            "reference integrity failed: cosine {:.9}, relative_l2 {:.6}, max_abs {:.6}",
            metrics.cosine, metrics.relative_l2, metrics.max_abs
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(extra: &[&str]) -> Vec<String> {
        ["pi05_bench", "random", "--dtype", "fp8"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn missing_explicit_tactics_is_rejected_in_inference_mode() {
        let path = std::env::temp_dir().join(format!(
            "apxinf-missing-tactics-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let error = Args::parse(&arguments(&["--tactics", path.to_str().unwrap()])).unwrap_err();
        assert!(error.to_string().contains("does not exist"));
        assert!(error.to_string().contains("--autotune"));
    }

    #[test]
    fn missing_explicit_tactics_is_allowed_for_autotune_creation() {
        let path = std::env::temp_dir().join(format!(
            "apxinf-new-tactics-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let parsed = Args::parse(&arguments(&[
            "--tactics",
            path.to_str().unwrap(),
            "--autotune",
        ]));
        assert!(parsed.is_ok());
    }
}

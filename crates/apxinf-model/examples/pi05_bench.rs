//! Unified low-level latency benchmark for PI0.5 (BF16 / FP8 / INT8-W8A8).
//!
//! This single example replaces the former `pi05_{bf16,thor,int8}_bench.rs`. It
//! bypasses the unified `AutoModel`/`infer` frontend and drives the variant-specific
//! `Pi05{Bf16,,Int8Dynamic}CudaRuntime` directly, because it needs raw device inputs that the model
//! abstraction does not (and should not) expose. For an abstraction-level entry
//! point see `pi05_auto_smoke`.
//!
//! The benchmark runs **checkpoint-free** with deterministic random weights
//! (`<source>` == `random`), because graph-replay latency depends only on tensor
//! shape and model_variant, not on trained values. Pass a checkpoint path/index instead
//! to measure a real model and validate against a captured reference.
//!
//! ```text
//! pi05_bench <checkpoint-or-index|random> --model-variant {bf16,fp8_static,int8_dynamic}
//!     [--calibration <json|uniform:SCALE>]
//!     [--views N] [--image-size N] [--action-horizon N] [--action-dim N]
//!     [--num-flow-steps N] [--max-token-len N]  (views/H also work with checkpoints)
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

use apxinf_model::pi05::{build_bf16_model, build_fp8_static_model, build_int8_dynamic_model};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda_new::{transfers, CudaBuffer, CudaContext};
use apxinf_model::pi05::{
    upload_time_embeddings_bf16, upload_time_embeddings_fp8_static,
    upload_time_embeddings_int8_dynamic, Bf16Model, Bf16Weights, CapturedGraph,
    Fp8StaticActivationScales, Fp8StaticCalibration, Fp8StaticModel, Fp8StaticWeights,
    Int8DynamicModel, Int8DynamicWeights, Pi05Config, Pi05ImageLayout, Pi05Weights,
    ModelVariantChoice,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchVariant {
    Bf16,
    Fp8Static,
    Int8Dynamic,
}

impl BenchVariant {
    fn parse(spec: &str) -> Result<Self, String> {
        match spec {
            "bf16" => Ok(Self::Bf16),
            "fp8_static" => Ok(Self::Fp8Static),
            "int8_dynamic" => Ok(Self::Int8Dynamic),
            other => Err(format!(
                "--model-variant must be bf16, fp8_static, or int8_dynamic; got {other}"
            )),
        }
    }

    /// Device dtype of the patch/noise inputs feeding the graph.
    fn io_dtype(self) -> DType {
        match self {
            Self::Fp8Static => DType::F16,
            Self::Bf16 | Self::Int8Dynamic => DType::BF16,
        }
    }

    fn variant_label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8Static => "fp8_static",
            Self::Int8Dynamic => "int8_w8a8",
        }
    }

    fn patches_label(self) -> &'static str {
        match self {
            Self::Fp8Static => "patches_f16",
            Self::Bf16 | Self::Int8Dynamic => "patches_bf16",
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
            Self::Fp8Static => Thresholds {
                eager_graph_max_abs: 1e-3,
                reference_min_cosine: 0.997,
                reference_max_relative_l2: 0.10,
                reference_max_abs: None,
            },
            Self::Int8Dynamic => Thresholds {
                eager_graph_max_abs: 1e-2,
                reference_min_cosine: 0.995,
                reference_max_relative_l2: 0.10,
                reference_max_abs: Some(0.125),
            },
        }
    }
}

/// Per-variant integrity gates (see `BenchVariant::thresholds`). `eager_graph_min_cosine`
/// is a shared `0.999_999`; the reference cosine floor may be overridden.
#[derive(Clone, Copy, Debug)]
struct Thresholds {
    eager_graph_max_abs: f64,
    reference_min_cosine: f64,
    reference_max_relative_l2: f64,
    reference_max_abs: Option<f64>,
}

const EAGER_GRAPH_MIN_COSINE: f64 = 0.999_999;

fn to_device(context: &CudaContext, tensor: &Tensor) -> apxinf_core::Result<Tensor> {
    transfers::to_cuda(tensor, context.device_id())
}

fn to_cpu(tensor: &Tensor) -> apxinf_core::Result<Tensor> {
    transfers::to_cpu(tensor)
}

/// Benchmark-only dispatch over statically typed Networks. Capture uses the
/// same prepare implementation and graph owner for every compute variant.
enum Bench {
    Bf16(Bf16Model),
    Fp8Static(Fp8StaticModel),
    Int8Dynamic(Int8DynamicModel),
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
            Self::Fp8Static(rt) => {
                rt.infer(patches, token_ids, token_count, noise, time_embeddings)?
            }
            Self::Int8Dynamic(rt) => {
                rt.infer(patches, token_ids, token_count, noise, time_embeddings)?
            }
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
            Self::Bf16(rt) => apxinf_model::pi05::capture_patches(
                &rt,
                patches,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?,
            Self::Fp8Static(rt) => apxinf_model::pi05::capture_patches(
                &rt,
                patches,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?,
            Self::Int8Dynamic(rt) => apxinf_model::pi05::capture_patches(
                &rt,
                patches,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?,
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
            Self::Bf16(rt) => apxinf_model::pi05::capture_rgb(
                &rt,
                layout,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?,
            Self::Fp8Static(rt) => apxinf_model::pi05::capture_rgb(
                &rt,
                layout,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?,
            Self::Int8Dynamic(rt) => apxinf_model::pi05::capture_rgb(
                &rt,
                layout,
                token_ids,
                token_count,
                noise,
                time_embeddings,
            )?,
        })
    }
}

type Graph = CapturedGraph;

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

    fn label(self, model_variant: BenchVariant) -> &'static str {
        match self {
            Self::Patches => model_variant.patches_label(),
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

fn latency_json(samples_ms: Vec<f64>) -> serde_json::Value {
    let mut ordered = samples_ms.clone();
    ordered.sort_by(f64::total_cmp);
    let sample_count = ordered.len() as f64;
    let mean = ordered.iter().sum::<f64>() / sample_count;
    let variance = ordered
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / sample_count;
    let percentile = |fraction: f64| {
        let index = ((ordered.len() - 1) as f64 * fraction).round() as usize;
        ordered[index]
    };
    serde_json::json!({
        "samples_ms": samples_ms,
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[ordered.len() - 1],
        "mean": mean,
        "standard_deviation": variance.sqrt()
    })
}

/// Reference action parser. Artifact identity is verified by the regression
/// manifest, so every precision consumes the same pinned `raw_actions` payload.
fn reference_actions(
    path: &Path,
    config: &Pi05Config,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let document: serde_json::Value = serde_json::from_str(&raw)?;
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
    model_variant: BenchVariant,
    calibration: Option<String>,
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
        let mut model_variant: Option<BenchVariant> = None;
        let mut calibration = None;
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
                "--model-variant" => {
                    model_variant = Some(BenchVariant::parse(&expect_value(
                        raw,
                        &mut index,
                        "--model-variant",
                    )?)?)
                }
                "--calibration" => {
                    calibration = Some(expect_value(raw, &mut index, "--calibration")?)
                }
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
        let model_variant = model_variant
            .ok_or("missing required --model-variant {bf16,fp8_static,int8_dynamic}")?;
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
            model_variant,
            calibration,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::env::args().collect::<Vec<_>>();
    let args = Args::parse(&raw).map_err(|error| {
        format!(
            "{error}\nusage: {} <checkpoint-or-index|random> --model-variant {{bf16,fp8_static,int8_dynamic}} \
             [--calibration <json|uniform:SCALE>] [--views N] \
             [--image-size N] [--action-horizon N] [--action-dim N] [--num-flow-steps N] \
             [--max-token-len N] [--token-count T] [--iterations N] [--seed N] \
             [--image-input patches|nhwc|nchw] [--reference <json>] [--min-cosine C] \
             [--images-u8 <raw>] [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>]",
            raw.first().map(String::as_str).unwrap_or("pi05_bench")
        )
    })?;

    let random = args.source == "random";
    let model_variant = args.model_variant;
    let thresholds = model_variant.thresholds();
    let token_count = args.token_count;
    let iterations = args.iterations;
    let image_input = ImageInput::resolve(args.image_input.as_deref())?;
    if args.images_u8.is_some() && !matches!(image_input, ImageInput::Rgb(_)) {
        return Err("--images-u8 requires --image-input nhwc".into());
    }

    // View count and action horizon do not change checkpoint tensor shapes, so
    // the fixed H10 regression matrix may override them for real weights. The
    // remaining architecture fields still describe learned tensor dimensions.
    let has_checkpoint_shape_overrides = args.image_size.is_some()
        || args.action_dim.is_some()
        || args.num_flow_steps.is_some()
        || args.max_token_len.is_some();
    if !random && has_checkpoint_shape_overrides {
        return Err(
            "checkpoint mode permits --views and --action-horizon, but \
             --image-size/--action-dim/--num-flow-steps/--max-token-len require `random`"
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
    // All GEMM precisions share the hardware tactic database; calibration is
    // still specific to FP8 activations.
    if model_variant != BenchVariant::Fp8Static && args.calibration.is_some() {
        return Err("--calibration only applies to --model-variant fp8".into());
    }

    let config = if random {
        // Random-benchmark defaults mirror `apxinf_py.ModelRunner.random` (2-view / H10)
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
        let mut config = if config_path.is_file() {
            Pi05Config::from_json_file(&config_path)?
        } else {
            Pi05Config::default()
        };
        if let Some(num_views) = args.views {
            config.num_views = num_views;
        }
        if let Some(action_horizon) = args.action_horizon {
            config.action_horizon = action_horizon;
        }
        config
    };
    config.validate()?;
    let config = Arc::new(config);

    let context = Arc::new(CudaContext::new(0).map_err(std::io::Error::other)?);
    if model_variant == BenchVariant::Fp8Static {
        ModelVariantChoice::Fp8Static.ensure_supported(context.caps().sm)?;
    }
    // cuda-new selects and prepares exact L3 executions before graph capture.
    // That one-time work is outside both timed steady-state boundaries below.

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

    let bench = match model_variant {
        BenchVariant::Bf16 => {
            eprintln!("converting and uploading native BF16 weights...");
            let device_weights = Arc::new(Bf16Weights::from_host(
                &host_weights,
                &context,
            )?);
            Bench::Bf16(build_bf16_model(
                context.clone(),
                config.clone(),
                device_weights,
            )?)
        }
        BenchVariant::Fp8Static => {
            let scales = match args.calibration.as_deref() {
                Some(spec) if spec.starts_with("uniform:") => {
                    eprintln!(
                        "warning: uniform activation scales are for smoke/latency tests only"
                    );
                    Arc::new(Fp8StaticActivationScales::uniform(
                        &config,
                        spec["uniform:".len()..].parse()?,
                    )?)
                }
                Some(path) => {
                    if random {
                        return Err(
                            "random FP8 benchmarks require uniform:SCALE, not a checkpoint profile"
                                .into(),
                        );
                    }
                    let checkpoint =
                        apxinf_model::pi05::checkpoint_identity(Path::new(&args.source))?;
                    let calibration = Fp8StaticCalibration::from_json_file(
                        Path::new(path),
                        &config,
                        &checkpoint,
                    )?;
                    Arc::new(Fp8StaticActivationScales::from_calibration(
                        &config,
                        &calibration,
                    )?)
                }
                None if random => {
                    eprintln!("warning: no --calibration; using uniform activation scale 1.0");
                    Arc::new(Fp8StaticActivationScales::uniform(&config, 1.0)?)
                }
                None => {
                    return Err(
                        "--model-variant fp8_static requires --calibration <json|uniform:SCALE> for a checkpoint"
                            .into(),
                    )
                }
            };
            eprintln!("quantizing and uploading static FP8 weights...");
            let device_weights = Arc::new(Fp8StaticWeights::from_host(
                &host_weights,
                &context,
            )?);
            Bench::Fp8Static(build_fp8_static_model(
                context.clone(),
                config.clone(),
                device_weights,
                scales,
            )?)
        }
        BenchVariant::Int8Dynamic => {
            eprintln!("quantizing and uploading per-channel INT8 weights...");
            let device_weights = Arc::new(Int8DynamicWeights::from_host(&host_weights, &context)?);
            Bench::Int8Dynamic(build_int8_dynamic_model(
                context.clone(),
                config.clone(),
                device_weights,
            )?)
        }
    };
    drop(host_weights);

    let io_dtype = model_variant.io_dtype();
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
    let patches = to_device(&context, &patches_host)?;
    let noise_host = noise_fixture(args.noise_bf16_u16le.as_deref(), &config, io_dtype)?;
    let noise = to_device(&context, &noise_host)?;
    let host_token_ids = token_fixture(args.token_ids_u32le.as_deref(), token_count)?;
    let token_bytes = host_token_ids
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<_>>();
    let token_ids = CudaBuffer::alloc_zeros(token_bytes.len(), context.device_id())
        .map_err(std::io::Error::other)?;
    token_ids
        .copy_from_host(&token_bytes)
        .map_err(std::io::Error::other)?;
    let time_embeddings = match model_variant {
        BenchVariant::Bf16 => upload_time_embeddings_bf16(&config, &context)?,
        BenchVariant::Fp8Static => upload_time_embeddings_fp8_static(&config, &context)?,
        BenchVariant::Int8Dynamic => upload_time_embeddings_int8_dynamic(&config, &context)?,
    };

    eprintln!(
        "running eager {} integrity pass...",
        model_variant.variant_label()
    );
    let eager_output = bench.infer(&patches, &token_ids, token_count, &noise, &time_embeddings)?;
    let eager = to_cpu(&eager_output)?.to_f32_vec()?;
    drop(eager_output);

    if std::env::var_os("APXINF_PI05_EAGER_ONLY").is_some() {
        let checksum = eager.iter().map(|value| value.abs() as f64).sum::<f64>();
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "passed": true,
                "model_variant": model_variant.variant_label(),
                "mode": "eager_only",
                "token_count": token_count,
                "output_abs_checksum": checksum,
            }))?
        );
        return Ok(());
    }

    eprintln!(
        "capturing {} graph with {} input...",
        model_variant.variant_label(),
        image_input.label(model_variant)
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
    let captured = to_cpu(graph.output())?.to_f32_vec()?;
    let eager_graph = ErrorMetrics::measure(&captured, &eager)?;
    let eager_graph_passed = eager_graph.cosine >= EAGER_GRAPH_MIN_COSINE
        && eager_graph.max_abs <= thresholds.eager_graph_max_abs;

    let reference_min_cosine = args.min_cosine.unwrap_or(thresholds.reference_min_cosine);
    let reference_metrics = args
        .reference
        .as_ref()
        .map(|path| reference_actions(Path::new(path), &config))
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
    context.synchronize().map_err(std::io::Error::other)?;
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
    let output = to_cpu(graph.output())?.to_f32_vec()?;
    let checksum = output.iter().map(|value| value.abs() as f64).sum::<f64>();
    let profile = format!(
        "pi05_{}view_h{}_steps{}",
        config.num_views, config.action_horizon, config.num_flow_steps
    );
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "passed": eager_graph_passed && reference_passed,
            "profile": profile,
            "model_variant": model_variant.variant_label(),
            "weights": if random { "synthetic" } else { "checkpoint" },
            "image_input": {
                "kind": image_input.label(model_variant),
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

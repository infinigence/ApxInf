//! π0.5 on HIP against fixture inputs: accuracy against a reference, eager
//! latency, and optionally every intermediate stage for comparison with CUDA.
//!
//! Inference goes through the same path AutoModel and Python use —
//! `hip_pi05::load_runner`, a prepared plan, `run` — so what is measured and
//! gated here is what a policy gets. The gate is `pi05_bench`'s BF16 one:
//! cosine >= 0.999 and relative L2 <= 0.05 against `--reference`.
//!
//! ```text
//! pi05_hip_bench <checkpoint> [--views N] [--action-horizon N] [--token-count T]
//!     [--iterations N] [--image-input nhwc|patches]
//!     [--images-u8 <raw>] [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>]
//!     [--reference <json>] [--dump-stages <dir>]
//! ```
//!
//! `--dump-stages` additionally runs the model's stage methods, synchronizing
//! after each to time it, and writes the files `pi05_stage_dump` writes on CUDA
//! (compare them with `scripts/compare_pi05_stages.py`).

#[path = "support/pi05_fixtures.rs"]
mod fixtures;

use std::path::{Path, PathBuf};
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_hip::native::{self, HipArena, HipDeviceBytes};
use apxinf_model::hip_pi05::{load_runner, Config, HipPi05Model};
use apxinf_model::vla::{ImageLayout, Observation, VisionObservation, VlaRequest, VlaRuntime};
use apxinf_model::LoadOptions;
use fixtures::{
    latency_json, metrics, passes_bf16_gate, reference_actions, write_npy, AnyResult, Fixture,
    Geometry,
};

fn bf16_tensor(shape: Vec<usize>, values: &[f32]) -> AnyResult<Tensor> {
    let values: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
    Ok(Tensor::from_bf16(shape, &values)?)
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let usage = "usage: pi05_hip_bench <checkpoint> [--views N] [--action-horizon N] [--token-count T] \
                 [--iterations N] [--image-input nhwc|patches] [--images-u8 <raw>] \
                 [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>] [--reference <json>] [--dump-stages <dir>]";
    let checkpoint = args.get(1).filter(|a| !a.starts_with("--")).ok_or(usage)?;
    let token_count: usize = flag("--token-count").map_or(Ok(10), |v| v.parse())?;
    let iterations: usize = flag("--iterations").map_or(Ok(30), |v| v.parse())?;
    let rgb = match flag("--image-input").as_deref() {
        None | Some("nhwc") => true,
        Some("patches") => false,
        Some(other) => {
            return Err(format!("--image-input must be nhwc or patches, got {other}").into())
        }
    };

    let source = Path::new(checkpoint);
    let root = if source.is_dir() {
        source
    } else {
        source.parent().unwrap_or(Path::new("."))
    };
    // Resolve the checkpoint config with the view/horizon overrides, as the
    // port reads it, and hand it to the loader.
    let mut config = match root.join("config.json") {
        path if path.is_file() => Config::from_json_file(&path)?,
        _ => Config::default(),
    };
    if let Some(views) = flag("--views") {
        config.num_views = views.parse()?;
    }
    if let Some(horizon) = flag("--action-horizon") {
        config.action_horizon = horizon.parse()?;
    }
    config.validate()?;

    let ctx = std::sync::Arc::new(apxinf_hip::HipContext::new(0)?);
    eprintln!(
        "loading π0.5 checkpoint onto {} ({})...",
        ctx.caps().name,
        ctx.caps().arch
    );
    let started = Instant::now();
    let runner = load_runner(
        source,
        std::sync::Arc::clone(&ctx),
        &LoadOptions::default(),
        Some(config),
    )?;
    let load_seconds = started.elapsed().as_secs_f64();
    let config: &Config = runner.model().config();
    let geometry = Geometry {
        views: config.num_views,
        image_size: config.image_size,
        patch_size: config.patch_size,
        action_horizon: config.action_horizon,
        action_dim: config.action_dim,
    };

    let fixture = Fixture::load(
        geometry,
        token_count,
        flag("--images-u8").as_deref(),
        flag("--token-ids-u32le").as_deref(),
        flag("--noise-bf16-u16le").as_deref(),
    )?;
    let noise_shape = vec![config.action_horizon, config.action_dim];
    let patch_shape = vec![config.patch_rows(), config.patch_width()];
    let noise = bf16_tensor(noise_shape.clone(), &fixture.noise)?;
    let observation = Observation {
        vision: if rgb {
            VisionObservation::RgbU8 {
                bytes: fixture.images.clone(),
                layout: ImageLayout::Nhwc,
            }
        } else {
            VisionObservation::Patches(Tensor::from_bf16(
                patch_shape.clone(),
                &fixture.patches(geometry),
            )?)
        },
        token_ids: fixture.token_ids.clone(),
        state: None,
        action_mask: None,
    };
    let request = VlaRequest::provided(&observation, &noise);

    let prepared = runner.prepare(&observation.inference_spec())?;
    let started = Instant::now();
    let first = ctx.to_cpu(prepared.run(&request)?.tensor())?.to_f32_vec()?;
    let first_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut samples = Vec::with_capacity(iterations);
    let mut actions = first.clone();
    for _ in 0..iterations {
        let started = Instant::now();
        let action = prepared.run(&request)?;
        ctx.synchronize()?;
        samples.push(started.elapsed().as_secs_f64() * 1e3);
        actions = ctx.to_cpu(action.tensor())?.to_f32_vec()?;
    }
    let repeat = metrics(&actions, &first)?;

    let reference = flag("--reference")
        .map(|path| reference_actions(&path, geometry))
        .transpose()?;
    let reference = reference
        .as_ref()
        .map(|expected| metrics(&actions, expected))
        .transpose()?;
    let passed =
        reference.as_ref().is_none_or(passes_bf16_gate) && actions.iter().all(|v| v.is_finite());

    let stages = match flag("--dump-stages") {
        Some(dir) => Some(dump_stages(
            runner.model(),
            &fixture,
            token_count,
            &PathBuf::from(dir),
        )?),
        None => None,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "passed": passed,
            "device": format!("{} ({})", ctx.caps().name, ctx.caps().arch),
            "model_variant": "bf16",
            "execution": "eager",
            "views": config.num_views,
            "action_horizon": config.action_horizon,
            "token_count": token_count,
            "image_input": if rgb { "rgb_u8_nhwc" } else { "patches" },
            "load_seconds": load_seconds,
            "first_run_ms": first_ms,
            "latency_ms": latency_json(&samples),
            "repeat_vs_first": repeat,
            "reference": reference,
            "stages": stages,
            "output_abs_checksum": actions.iter().map(|v| f64::from(v.abs())).sum::<f64>(),
        }))?
    );
    if !passed {
        std::process::exit(1);
    }
    Ok(())
}

/// Run the stages one at a time, synchronizing after each to time it, and
/// write each stage's output.
fn dump_stages(
    model: &HipPi05Model,
    fixture: &Fixture,
    token_count: usize,
    out: &Path,
) -> AnyResult<serde_json::Value> {
    let ctx = model.context();
    let config = model.config();
    let arena = HipArena::new(ctx, HipPi05Model::arena_bytes(config, token_count))?;
    let images = HipDeviceBytes::new(ctx, fixture.images.len())?;
    images.upload(&fixture.images)?;
    let token_ids = HipDeviceBytes::new(ctx, token_count * 4)?;
    token_ids.upload_u32(&fixture.token_ids)?;
    let noise_shape = vec![config.action_horizon, config.action_dim];
    let noise = ctx.to_device(&bf16_tensor(noise_shape.clone(), &fixture.noise)?)?;
    let host = |t: &Tensor| -> AnyResult<Vec<f32>> { Ok(ctx.to_cpu(t)?.to_f32_vec()?) };

    let mut timings = serde_json::Map::new();
    let mut timed = |name: &str, started: Instant| -> AnyResult<()> {
        ctx.synchronize()?;
        timings.insert(name.into(), (started.elapsed().as_secs_f64() * 1e3).into());
        Ok(())
    };
    ctx.synchronize()?;
    let started = Instant::now();
    let patches = arena.tensor(vec![config.patch_rows(), config.patch_width()], DType::BF16)?;
    native::rgb_to_patches(
        ctx,
        &images,
        native::ImageLayout::Nhwc,
        config.num_views,
        config.image_size,
        config.patch_size,
        &patches,
    )?;
    timed("rgb_to_patches_ms", started)?;
    let started = Instant::now();
    let vision = model.encode_vision(&arena, &patches)?;
    timed("vision_ms", started)?;
    let started = Instant::now();
    let prefix = model.embed_prefix(&arena, &vision, &token_ids, token_count)?;
    timed("embed_prefix_ms", started)?;
    let started = Instant::now();
    let cache = model.prefix_forward(&arena, &prefix)?;
    timed("prefix_forward_ms", started)?;
    let started = Instant::now();
    let mut state = noise;
    let mut first_step = None;
    for step in 0..config.num_flow_steps {
        state = model.denoise_step(&arena, &state, step, &cache)?;
        if step == 0 {
            first_step = Some(state.clone());
        }
    }
    timed("denoise_ms", started)?;

    write_npy(out, "patches", patches.shape().dims(), &host(&patches)?)?;
    write_npy(out, "vision", vision.shape().dims(), &host(&vision)?)?;
    write_npy(out, "prefix", prefix.shape().dims(), &host(&prefix)?)?;
    let kv_width = config.language.num_kv_heads * config.language.head_dim;
    for (layer, (key, value)) in cache.keys.iter().zip(&cache.values).enumerate() {
        let rows = cache.tokens * kv_width;
        write_npy(
            out,
            &format!("k_{layer:02}"),
            &[cache.tokens, kv_width],
            &host(key)?[..rows],
        )?;
        write_npy(
            out,
            &format!("v_{layer:02}"),
            &[cache.tokens, kv_width],
            &host(value)?[..rows],
        )?;
    }
    write_npy(
        out,
        "state_step00",
        &noise_shape,
        &host(first_step.as_ref().unwrap())?,
    )?;
    write_npy(out, "actions", &noise_shape, &host(&state)?)?;
    eprintln!("wrote HIP BF16 stages to {}", out.display());
    timings.insert(
        "arena_peak_mib".into(),
        (arena.peak() as f64 / (1 << 20) as f64).into(),
    );
    timings.insert(
        "arena_capacity_mib".into(),
        (arena.capacity() as f64 / (1 << 20) as f64).into(),
    );
    Ok(serde_json::Value::Object(timings))
}

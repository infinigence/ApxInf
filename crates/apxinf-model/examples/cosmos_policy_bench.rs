//! Reproducible Cosmos Policy BF16 benchmark through the public VLA contract.
//!
//! cosmos_policy_bench MODEL_DIR FIXTURE_DIR OUTPUT_JSON [--recipes DIR]
//!     [--autotune] [--warmup N] [--iterations N] [--profile]
//!
//! FIXTURE_DIR contains joint-fixture.safetensors, public-rgb.bin and
//! public-task-id.txt captured from the declared policy input boundary.
//! Timings include request upload, Graph replay, independent Action publication
//! and synchronization; Python transforms and host result conversion are outside.
//! --profile brackets the timed region with cudaProfilerStart/Stop for nsys.
use apxinf_core::{Device, Error, Result, Tensor};
use apxinf_cuda_new::CudaBuffer;
use apxinf_model::{
    AutoModel, ExecutionPolicy, ImageLayout, LoadOptions, Observation, VisionObservation,
    VlaRequest,
};
use std::{path::PathBuf, time::Instant};

fn host(tensor: &Tensor) -> Result<Vec<f32>> {
    let mut bytes = vec![0; tensor.size_in_bytes()];
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_to_host(&mut bytes)
        .map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
        .collect())
}
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * q;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: cosmos_policy_bench MODEL_DIR FIXTURE_DIR OUTPUT_JSON [--recipes DIR] [--autotune] [--warmup N] [--iterations N] [--profile]";
    let model_dir = PathBuf::from(args.next().ok_or(usage)?);
    let fixtures = PathBuf::from(args.next().ok_or(usage)?);
    let output = PathBuf::from(args.next().ok_or(usage)?);
    let (mut warmup, mut iterations) = (5usize, 20usize);
    let (mut autotune, mut profile, mut recipes) = (false, false, None);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--autotune" => autotune = true,
            "--profile" => profile = true,
            "--recipes" => recipes = Some(PathBuf::from(args.next().ok_or(usage)?)),
            "--warmup" => warmup = args.next().ok_or(usage)?.parse()?,
            "--iterations" => iterations = args.next().ok_or(usage)?.parse()?,
            _ => return Err(format!("unknown argument {arg}; {usage}").into()),
        }
    }
    if iterations == 0 || warmup == 0 {
        return Err("warmup and iterations must be positive".into());
    }
    let (fixture, _) =
        apxinf_loader::safetensors::load_native_path(&fixtures.join("joint-fixture.safetensors"))?;
    let observation = Observation {
        vision: VisionObservation::RgbU8 {
            bytes: std::fs::read(fixtures.join("public-rgb.bin"))?,
            layout: ImageLayout::Nhwc,
        },
        token_ids: vec![
            std::fs::read_to_string(fixtures.join("public-task-id.txt"))?
                .trim()
                .parse()?,
        ],
        state: Some(fixture["state"].clone()),
        action_mask: None,
    };
    let request = VlaRequest::provided(&observation, &fixture["noise"]);
    let loaded = AutoModel::load_model(
        Device::Cuda(0),
        &model_dir,
        &LoadOptions {
            model_name: Some("cosmos-policy".into()),
            model_variant: Some("bf16".into()),
            autotune,
            tuning_path: recipes,
            ..Default::default()
        },
    )?;
    let runtime = loaded.vla()?;
    let plan = runtime.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
    let first = host(plan.run(&request)?.tensor())?;
    let expected = fixture["expected_actions"].as_f32()?;
    if first.len() != 112 || expected.len() != first.len() {
        return Err("fixture or runtime action geometry differs from [16,7]".into());
    }
    let max_abs = first
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rmse = (first
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / first.len() as f32)
        .sqrt();
    if first.iter().any(|x| !x.is_finite()) || max_abs > 0.03 || rmse > 0.01 {
        return Err(format!("action reference mismatch: max_abs={max_abs}, rmse={rmse}").into());
    }
    for row in 0..16 {
        if (first[row * 7 + 6] > 0.0) != (expected[row * 7 + 6] > 0.0) {
            return Err(format!("gripper mismatch at row {row}").into());
        }
    }
    for _ in 0..warmup {
        plan.run(&request)?;
    }
    let mut milliseconds = Vec::with_capacity(iterations);
    if profile {
        apxinf_cuda_new::profiler::start()?;
    }
    let mut last = None;
    for _ in 0..iterations {
        let start = Instant::now();
        let action = plan.run(&request)?;
        milliseconds.push(start.elapsed().as_secs_f64() * 1000.0);
        last = Some(action);
    }
    if profile {
        apxinf_cuda_new::profiler::stop()?;
    }
    let final_action = host(last.as_ref().unwrap().tensor())?;
    if final_action != first {
        return Err("fixed-input graph replay changed actions".into());
    }
    let mut sorted = milliseconds.clone();
    sorted.sort_by(f64::total_cmp);
    let result = serde_json::json!({
        "model_type":"cosmos-policy", "model_variant":"bf16",
        "scope":"request binding, model Graph, independent Action publication and synchronization; excludes Python preprocessing and D2H conversion",
        "preparation_status":format!("{:?}",plan.status()),
        "warmup":warmup, "iterations":iterations,
        "milliseconds":milliseconds, "median_ms":percentile(&sorted,0.5),
        "p95_ms":percentile(&sorted,0.95),
        "normalized_action_max_abs":max_abs,"normalized_action_rmse":rmse,
        "fixed_replay_bitwise":true, "actions":first,
    });
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

//! Fixed-shape d3-mini CUDA Graph latency benchmark.
//!
//! Usage:
//! `cargo run -p apxinf-model --release --features cuda --example d3_bench -- MODEL_DIR 1,2,3 2 30 [--autotune] [--tuning-dir DIR]`

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{AutoModel, DecisionInput, LoadOptions};

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("missing d3-mini model directory")?;
    let token_ids = args
        .next()
        .ok_or("missing comma-separated token IDs")?
        .split(',')
        .map(str::parse::<u32>)
        .collect::<Result<Vec<_>, _>>()?;
    let candidate_count = args
        .next()
        .ok_or("missing candidate count")?
        .parse::<usize>()?;
    let iterations = args.next().as_deref().unwrap_or("30").parse::<usize>()?;
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }

    let mut options = LoadOptions::default();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--autotune" => options.autotune = true,
            "--tuning-dir" => {
                options.tuning_path = Some(
                    args.next()
                        .map(PathBuf::from)
                        .ok_or("--tuning-dir requires a directory")?,
                );
            }
            _ => return Err(format!("unknown argument {argument:?}").into()),
        }
    }

    let load_started = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), model_dir, &options)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let input = DecisionInput::text(&token_ids, candidate_count);

    let capture_started = Instant::now();
    let first = model.decide(input)?;
    let capture_ms = capture_started.elapsed().as_secs_f64() * 1_000.0;

    let mut milliseconds = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let output = model.decide(input)?;
        milliseconds.push(started.elapsed().as_secs_f64() * 1_000.0);
        if output.selected_index != first.selected_index {
            return Err("d3 replay changed the selected answer".into());
        }
    }
    milliseconds.sort_by(f64::total_cmp);
    println!(
        "tokens={} candidates={} autotune={} tuning_dir={} load_s={load_seconds:.3} prepare_capture_ms={capture_ms:.3} replay_p50_ms={:.3} replay_p95_ms={:.3} replay_min_ms={:.3}",
        token_ids.len(),
        candidate_count,
        options.autotune,
        options
            .tuning_path
            .as_deref()
            .map_or("<default>", |path| path.to_str().unwrap_or("<non-utf8>")),
        percentile(&milliseconds, 0.50),
        percentile(&milliseconds, 0.95),
        milliseconds[0],
    );
    let selected = first.selected()?;
    println!(
        "selected={} token_id={} probability={:.8}",
        selected.code, selected.token_id, selected.probability
    );
    Ok(())
}

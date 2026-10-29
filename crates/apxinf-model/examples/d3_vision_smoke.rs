//! Official-processor d3-mini image smoke test.
//!
//! Usage: `d3_vision_smoke MODEL_DIR REFERENCE_JSON PIXELS_F32 [ITERATIONS] [--autotune] [--tuning-dir DIR]`

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::{AutoModel, DecisionInput, ImageInput, LoadOptions};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    token_ids: Vec<u32>,
    pixel_shape: Vec<usize>,
    grid_thw: Vec<[u32; 3]>,
    probabilities: Vec<f32>,
    selected_index: usize,
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model directory")?);
    let reference_path = PathBuf::from(args.next().ok_or("missing reference JSON")?);
    let pixels_path = PathBuf::from(args.next().ok_or("missing pixel F32 file")?);
    let mut remaining = args.collect::<Vec<_>>().into_iter().peekable();
    let iterations = remaining
        .next_if(|argument| !argument.starts_with("--"))
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if iterations == Some(0) {
        return Err("iterations must be positive".into());
    }
    let mut options = LoadOptions::default();
    while let Some(argument) = remaining.next() {
        match argument.as_str() {
            "--autotune" => options.autotune = true,
            "--tuning-dir" => {
                options.tuning_path = Some(
                    remaining
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--tuning-dir requires a directory")?,
                );
            }
            _ => return Err(format!("unknown argument {argument:?}").into()),
        }
    }
    let reference: Reference = serde_json::from_slice(&std::fs::read(reference_path)?)?;
    let bytes = std::fs::read(pixels_path)?;
    let pixels = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    let patches = reference.pixel_shape[0];
    let pixel_values = Tensor::from_f32(reference.pixel_shape, &pixels)?;
    let mut model = AutoModel::load_model(Device::Cuda(0), model_dir, &options)?;
    let input = DecisionInput::with_vision(
        &reference.token_ids,
        ImageInput::new(&pixel_values, &reference.grid_thw),
        reference.probabilities.len(),
    );
    let prepare_started = Instant::now();
    let output = model.decide(input)?;
    let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1_000.0;
    let actual = output
        .scores
        .iter()
        .map(|score| score.probability)
        .collect::<Vec<_>>();
    let max_abs = actual
        .iter()
        .zip(&reference.probabilities)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    println!(
        "selected={} expected={} max_abs={max_abs:.8}",
        output.selected_index, reference.selected_index
    );
    println!("actual={actual:?}");
    println!("reference={:?}", reference.probabilities);
    if output.selected_index != reference.selected_index {
        return Err("d3 vision selected answer differs from official reference".into());
    }
    // BF16 CUDA providers and the official Torch reference use different GEMM
    // accumulation/rounding paths. Selection is the primary contract; 2e-2 is
    // a tight probability guard for non-saturated image/video examples.
    const MAX_ABS_TOLERANCE: f32 = 2e-2;
    if max_abs > MAX_ABS_TOLERANCE {
        return Err(format!(
            "d3 vision probability max_abs {max_abs:.8} exceeds {MAX_ABS_TOLERANCE:.3}"
        )
        .into());
    }
    if let Some(iterations) = iterations {
        let mut milliseconds = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let started = Instant::now();
            let replay = model.decide(input)?;
            milliseconds.push(started.elapsed().as_secs_f64() * 1_000.0);
            if replay.selected_index != reference.selected_index {
                return Err("d3 vision replay changed the selected answer".into());
            }
        }
        milliseconds.sort_by(f64::total_cmp);
        println!(
            "tokens={} patches={} prepare_capture_ms={prepare_ms:.3} replay_p50_ms={:.3} replay_p95_ms={:.3} replay_min_ms={:.3}",
            reference.token_ids.len(),
            patches,
            percentile(&milliseconds, 0.50),
            percentile(&milliseconds, 0.95),
            milliseconds[0],
        );
    }
    Ok(())
}

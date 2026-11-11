//! Fixed-length d3 Metal latency.
//!
//! `cargo run --release -p apxinf-model --features mlx --example d3_metal_latency -- MODEL_DIR --lengths 64,256,1024 [--count 4] [--rounds 12] [--warm 2]`
//!
//! Every listed length is requested `warm` times, then each round times one
//! request per length and prints `SAMPLE <length> <round> <ms>`. A length's
//! first request traces and warms its compiled profile, so the warm-up keeps
//! that cost out of the samples. The token IDs are fixed, so two builds see
//! the same inputs; the final line sums the returned probabilities.

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{AutoModel, DecisionInput, LoadOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("missing model directory")?;
    let mut lengths: Vec<usize> = vec![64, 256, 1024];
    let mut count = 4usize;
    let mut rounds = 12usize;
    let mut warm = 2usize;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--lengths" => {
                lengths = args
                    .next()
                    .ok_or("--lengths requires a comma-separated list")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--count" => count = args.next().ok_or("--count requires a value")?.parse()?,
            "--rounds" => rounds = args.next().ok_or("--rounds requires a value")?.parse()?,
            "--warm" => warm = args.next().ok_or("--warm requires a value")?.parse()?,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if lengths.contains(&0) || count == 0 || rounds == 0 {
        return Err("lengths, --count and --rounds must be positive".into());
    }

    let loaded = Instant::now();
    let mut model = AutoModel::load_model(Device::Metal(0), model_dir, &LoadOptions::default())?;
    eprintln!("load_ms {:.1}", loaded.elapsed().as_secs_f64() * 1_000.0);

    let tokens = |length: usize| -> Vec<u32> {
        (0..length)
            .map(|index| 1_000 + (index as u32 * 7_919) % 50_000)
            .collect()
    };
    for &length in &lengths {
        for _ in 0..warm {
            model.decide(DecisionInput::text(&tokens(length), count))?;
        }
    }
    let mut checksum = 0.0f32;
    for round in 0..rounds {
        for &length in &lengths {
            let started = Instant::now();
            let output = model.decide(DecisionInput::text(&tokens(length), count))?;
            let ms = started.elapsed().as_secs_f64() * 1_000.0;
            checksum += output
                .scores
                .iter()
                .map(|score| score.probability)
                .sum::<f32>();
            println!("SAMPLE {length} {round} {ms:.3}");
        }
    }
    eprintln!("probability_sum {checksum:.3}");
    Ok(())
}

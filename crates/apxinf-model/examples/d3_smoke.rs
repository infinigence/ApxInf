//! Public-entry d3-mini smoke test.
//!
//! Usage:
//! `cargo run -p apxinf-model --features cuda --example d3_smoke -- MODEL_DIR 1,2,3 2 [4,5,6] [--autotune] [--tuning-dir DIR]`

use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_model::{AutoModel, DecisionInput, LoadOptions};

fn parse_ids(value: &str) -> Result<Vec<u32>, std::num::ParseIntError> {
    value.split(',').map(str::parse::<u32>).collect()
}

fn print_output(
    label: &str,
    output: apxinf_model::DecisionOutput,
) -> Result<(), Box<dyn std::error::Error>> {
    let selected = output.selected()?;
    println!(
        "{label} selected={} token_id={} probability={:.8}",
        selected.code, selected.token_id, selected.probability
    );
    let mut top = output.scores;
    top.sort_by(|left, right| right.probability.total_cmp(&left.probability));
    for score in top.into_iter().take(5) {
        println!(
            "  {} token_id={} probability={:.8}",
            score.code, score.token_id, score.probability
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("missing d3-mini model directory")?;
    let token_ids = parse_ids(&args.next().ok_or("missing comma-separated token IDs")?)?;
    let candidate_count = args
        .next()
        .ok_or("missing candidate count")?
        .parse::<usize>()?;
    let mut remaining = args.collect::<Vec<_>>().into_iter().peekable();
    let changed = remaining
        .next_if(|argument| !argument.starts_with("--"))
        .map(|value| parse_ids(&value))
        .transpose()?;
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
    let mut model = AutoModel::load_model(Device::Cuda(0), model_dir, &options)?;
    print_output(
        "primary",
        model.decide(DecisionInput::text(&token_ids, candidate_count))?,
    )?;
    if let Some(changed) = changed {
        if changed.len() != token_ids.len() {
            return Err("changed token IDs must reuse the primary fixed-length profile".into());
        }
        print_output(
            "changed",
            model.decide(DecisionInput::text(&changed, candidate_count))?,
        )?;
    }
    Ok(())
}

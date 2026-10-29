//! Evaluate d3-mini over pre-tokenized decision records.
//!
//! Usage:
//! `d3_eval MODEL_DIR RECORDS_JSONL [PREDICTIONS_JSONL] [--autotune] [--tuning-dir DIR]`

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{AutoModel, DecisionInput, LoadOptions};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Record {
    id: String,
    dataset: String,
    task_type: String,
    token_ids: Vec<u32>,
    candidate_count: usize,
    expected_index: usize,
    oos_index: Option<usize>,
}

#[derive(Serialize)]
struct Prediction<'a> {
    id: &'a str,
    dataset: &'a str,
    task_type: &'a str,
    tokens: usize,
    expected_index: usize,
    selected_index: usize,
    probabilities: Vec<f32>,
    correct: bool,
    latency_ms: f64,
}

#[derive(Default)]
struct Counts {
    correct: usize,
    total: usize,
}

fn selected_index(record: &Record, probabilities: &[f32], argmax: usize) -> usize {
    if record.task_type == "noul" && probabilities.len() == 2 {
        return usize::from(probabilities[1] >= 0.5);
    }
    if let Some(oos) = record.oos_index {
        if probabilities.iter().copied().fold(0.0f32, f32::max) < 0.35 {
            return oos;
        }
    }
    argmax
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model directory")?);
    let records_path = PathBuf::from(args.next().ok_or("missing records JSONL")?);
    let mut remaining = args.collect::<Vec<_>>().into_iter().peekable();
    let predictions_path = remaining
        .next_if(|argument| !argument.starts_with("--"))
        .map(PathBuf::from);
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

    let reader = BufReader::new(File::open(records_path)?);
    let mut records = reader
        .lines()
        .map(|line| Ok(serde_json::from_str::<Record>(&line?)?))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    records.sort_by_key(|record| record.token_ids.len());

    let load_started = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), model_dir, &options)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let mut writer = predictions_path
        .map(File::create)
        .transpose()?
        .map(BufWriter::new);
    let mut datasets = BTreeMap::<String, Counts>::new();
    let run_started = Instant::now();

    for (index, record) in records.iter().enumerate() {
        let started = Instant::now();
        let output = model.decide(DecisionInput::text(
            &record.token_ids,
            record.candidate_count,
        ))?;
        let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let probabilities = output
            .scores
            .iter()
            .map(|score| score.probability)
            .collect::<Vec<_>>();
        let selected = selected_index(record, &probabilities, output.selected_index);
        let correct = selected == record.expected_index;
        let counts = datasets.entry(record.dataset.clone()).or_default();
        counts.total += 1;
        counts.correct += usize::from(correct);
        if let Some(writer) = &mut writer {
            serde_json::to_writer(
                &mut *writer,
                &Prediction {
                    id: &record.id,
                    dataset: &record.dataset,
                    task_type: &record.task_type,
                    tokens: record.token_ids.len(),
                    expected_index: record.expected_index,
                    selected_index: selected,
                    probabilities,
                    correct,
                    latency_ms,
                },
            )?;
            writer.write_all(b"\n")?;
        }
        if (index + 1) % 25 == 0 || index + 1 == records.len() {
            eprintln!("progress={}/{}", index + 1, records.len());
        }
    }
    if let Some(writer) = &mut writer {
        writer.flush()?;
    }

    let total = datasets.values().map(|counts| counts.total).sum::<usize>();
    let correct = datasets
        .values()
        .map(|counts| counts.correct)
        .sum::<usize>();
    let macro_accuracy = datasets
        .values()
        .map(|counts| counts.correct as f64 / counts.total as f64)
        .sum::<f64>()
        / datasets.len() as f64;
    for (dataset, counts) in &datasets {
        println!(
            "dataset={dataset} correct={} total={} accuracy={:.4}",
            counts.correct,
            counts.total,
            counts.correct as f64 / counts.total as f64,
        );
    }
    println!(
        "overall_correct={correct} overall_total={total} micro_accuracy={:.4} macro_accuracy={macro_accuracy:.4} load_s={load_seconds:.3} run_s={:.3}",
        correct as f64 / total as f64,
        run_started.elapsed().as_secs_f64(),
    );
    Ok(())
}

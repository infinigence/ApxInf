//! Run pre-tokenized Decision Index requests and emit its native result format.
//!
//! Usage:
//! `d3_decision_index MODEL_DIR RECORDS_JSONL RESULTS_JSONL [--autotune] [--tuning-dir DIR] [--profile-limit N] [--profile-token-budget N]`

use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{
    AutoModel, DecisionInput, LoadOptions, D3_DEFAULT_TEXT_PROFILE_LIMIT,
    D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};

#[derive(Deserialize)]
struct Question {
    key: String,
    #[serde(rename = "type")]
    kind: String,
    candidate_keys: Vec<String>,
    token_ids: Vec<u32>,
}

#[derive(Deserialize)]
struct Record {
    run_id: String,
    questions: Vec<Question>,
    #[serde(flatten)]
    evaluation: Map<String, Value>,
}

fn positive_usize(value: String, option: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{option} must be positive").into());
    }
    Ok(parsed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model directory")?);
    let records_path = PathBuf::from(args.next().ok_or("missing records JSONL")?);
    let results_path = PathBuf::from(args.next().ok_or("missing results JSONL")?);
    let mut options = LoadOptions::default();
    options.d3_text_profile_limit = Some(D3_DEFAULT_TEXT_PROFILE_LIMIT);
    options.d3_text_profile_token_budget = Some(D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--autotune" => options.autotune = true,
            "--tuning-dir" => {
                options.tuning_path = Some(PathBuf::from(
                    args.next().ok_or("--tuning-dir requires a directory")?,
                ));
            }
            "--profile-limit" => {
                options.d3_text_profile_limit = Some(positive_usize(
                    args.next().ok_or("--profile-limit requires a value")?,
                    "--profile-limit",
                )?);
            }
            "--profile-token-budget" => {
                options.d3_text_profile_token_budget = Some(positive_usize(
                    args.next()
                        .ok_or("--profile-token-budget requires a value")?,
                    "--profile-token-budget",
                )?);
            }
            _ => return Err(format!("unknown argument {argument:?}").into()),
        }
    }

    let mut completed = HashSet::new();
    if results_path.exists() {
        let contents = std::fs::read_to_string(&results_path)?;
        let complete_len = if contents.ends_with('\n') {
            contents.len()
        } else {
            contents.rfind('\n').map_or(0, |index| index + 1)
        };
        if complete_len != contents.len() {
            OpenOptions::new()
                .write(true)
                .open(&results_path)?
                .set_len(complete_len as u64)?;
        }
        for line in contents[..complete_len].lines() {
            let result: Value = serde_json::from_str(line)?;
            if result.get("status").and_then(Value::as_str) == Some("ok") {
                if let Some(run_id) = result.get("run_id").and_then(Value::as_str) {
                    completed.insert(run_id.to_owned());
                }
            }
        }
    }

    let reader = BufReader::new(File::open(records_path)?);
    let mut records = reader
        .lines()
        .map(|line| Ok(serde_json::from_str::<Record>(&line?)?))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let total_records = records.len();
    records.retain(|record| !completed.contains(&record.run_id));
    records.sort_by_key(|record| {
        record
            .questions
            .iter()
            .map(|question| question.token_ids.len())
            .max()
            .unwrap_or_default()
    });

    let load_started = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), model_dir, &options)?;
    eprintln!("load_s={:.3}", load_started.elapsed().as_secs_f64());
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(results_path)?,
    );
    let run_started = Instant::now();
    eprintln!(
        "resume_completed={} remaining={} total={total_records} profile_limit={} profile_token_budget={}",
        completed.len(),
        records.len(),
        options.d3_text_profile_limit.unwrap(),
        options.d3_text_profile_token_budget.unwrap(),
    );

    for (index, record) in records.iter().enumerate() {
        let request_started = Instant::now();
        let mut answers = Map::new();
        let mut input_tokens = 0usize;
        for question in &record.questions {
            let candidate_count = question.candidate_keys.len();
            let output = model.decide(DecisionInput::text(&question.token_ids, candidate_count))?;
            let total = output
                .scores
                .iter()
                .map(|score| score.probability as f64)
                .sum::<f64>();
            if !total.is_finite() || total <= 0.0 {
                return Err(format!(
                    "{}:{} has invalid probabilities",
                    record.run_id, question.key
                )
                .into());
            }
            let probabilities = output
                .scores
                .iter()
                .map(|score| score.probability as f64 / total)
                .collect::<Vec<_>>();
            let answer = match question.kind.as_str() {
                "choice" => {
                    let shares = question
                        .candidate_keys
                        .iter()
                        .cloned()
                        .zip(probabilities.iter().copied())
                        .collect::<BTreeMap<_, _>>();
                    json!({
                        "type": "choice",
                        "choice": question.candidate_keys[output.selected_index],
                        "probabilities": shares,
                    })
                }
                "noul" if probabilities.len() == 2 => json!({
                    "type": "noul",
                    "noul": probabilities[1],
                }),
                other => {
                    return Err(format!(
                        "{}:{} has unsupported question type {other:?}",
                        record.run_id, question.key
                    )
                    .into())
                }
            };
            answers.insert(question.key.clone(), answer);
            input_tokens += question.token_ids.len();
        }
        let elapsed_ms = request_started.elapsed().as_secs_f64() * 1_000.0;
        let mut result = record.evaluation.clone();
        result.insert("run_id".into(), json!(record.run_id));
        result.insert("engine".into(), json!("apxinf-d3-cuda-new"));
        result.insert("status".into(), json!("ok"));
        result.insert(
            "response".into(),
            json!({
                "model": "vllm-sr/d3-mini",
                "answers": answers,
                "usage": {"input_tokens": input_tokens},
            }),
        );
        result.insert("total_wall_ms".into(), json!(elapsed_ms));
        result.insert("model_request_wall_ms".into(), json!(elapsed_ms));
        serde_json::to_writer(&mut writer, &result)?;
        writer.write_all(b"\n")?;
        if (index + 1) % 25 == 0 || index + 1 == records.len() {
            writer.flush()?;
            eprintln!(
                "progress={}/{} elapsed_s={:.1}",
                index + 1,
                records.len(),
                run_started.elapsed().as_secs_f64()
            );
        }
    }
    writer.flush()?;
    eprintln!("run_s={:.3}", run_started.elapsed().as_secs_f64());
    Ok(())
}

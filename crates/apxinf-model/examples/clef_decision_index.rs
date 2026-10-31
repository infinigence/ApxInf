//! Run Decision Index rows through native Clef on Metal and emit the kit's
//! result format.
//!
//! Usage:
//! `clef_decision_index MODEL_DIR ROWS_JSONL RESULTS_JSONL`
//!
//! `ROWS_JSONL` holds suite rows (`state`, `questions`, `_evaluation`), for
//! example from `python -m decision_index suite sample` or an export of
//! `Suite(...).rows(apply_exclusions=True)`. Clef encodes each request itself,
//! so no pre-tokenized records are needed. A request over the 4096-token
//! limit is recorded as `unsupported` (the kit scores it as wrong); an
//! unsupported question type is recorded the same way. Rerunning the same
//! command resumes after the last complete `ok` or `unsupported` row.

use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{AutoModel, DecisionSchema, LoadOptions};
use serde_json::{json, Map, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(args.next().ok_or("missing model directory")?);
    let rows_path = PathBuf::from(args.next().ok_or("missing rows JSONL")?);
    let results_path = PathBuf::from(args.next().ok_or("missing results JSONL")?);
    if let Some(extra) = args.next() {
        return Err(format!("unknown argument {extra:?}").into());
    }

    let mut completed = HashSet::new();
    if results_path.exists() {
        let contents = std::fs::read_to_string(&results_path)?;
        let complete_len = contents.rfind('\n').map_or(0, |index| index + 1);
        if complete_len != contents.len() {
            OpenOptions::new()
                .write(true)
                .open(&results_path)?
                .set_len(complete_len as u64)?;
        }
        for line in contents[..complete_len].lines() {
            let result: Value = serde_json::from_str(line)?;
            if matches!(
                result.get("status").and_then(Value::as_str),
                Some("ok" | "unsupported")
            ) {
                if let Some(run_id) = result.get("run_id").and_then(Value::as_str) {
                    completed.insert(run_id.to_owned());
                }
            }
        }
    }

    let reader = BufReader::new(File::open(&rows_path)?);
    let mut rows = Vec::new();
    for line in reader.lines() {
        let row: Map<String, Value> = serde_json::from_str(&line?)?;
        let run_id = row
            .get("_evaluation")
            .and_then(|e| e.get("run_id"))
            .and_then(Value::as_str)
            .ok_or("row lacks _evaluation.run_id")?
            .to_owned();
        if !completed.contains(&run_id) {
            rows.push(row);
        }
    }
    let total = rows.len() + completed.len();

    apxinf_mlx::reset_peak_memory()?;
    let load_started = Instant::now();
    let options = LoadOptions {
        model_name: Some("clef".into()),
        ..LoadOptions::default()
    };
    let mut model = AutoModel::load_model(Device::Metal(0), model_dir, &options)?;
    eprintln!("load_s={:.3}", load_started.elapsed().as_secs_f64());
    eprintln!(
        "resume_completed={} remaining={} total={total}",
        completed.len(),
        rows.len()
    );

    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&results_path)?,
    );
    let run_started = Instant::now();
    for (index, row) in rows.iter().enumerate() {
        let evaluation = row["_evaluation"].as_object().ok_or("_evaluation")?;
        let questions = row["questions"].as_object().ok_or("questions")?;
        let mut result = evaluation.clone();
        result.insert("engine".into(), json!("apxinf-clef-metal"));

        let request_started = Instant::now();
        let outcome = answer(&mut model, &row["state"], questions);
        let elapsed_ms = request_started.elapsed().as_secs_f64() * 1_000.0;
        match outcome {
            Ok(answers) => {
                result.insert("status".into(), json!("ok"));
                result.insert(
                    "response".into(),
                    json!({"model": "mlx-community/clef-flash-4bit", "answers": answers}),
                );
            }
            Err(Outcome::Unsupported(error)) => {
                result.insert("status".into(), json!("unsupported"));
                result.insert("error".into(), json!(error));
            }
            Err(Outcome::Failed(error)) => return Err(error.into()),
        }
        result.insert("total_wall_ms".into(), json!(elapsed_ms));
        result.insert("model_request_wall_ms".into(), json!(elapsed_ms));
        serde_json::to_writer(&mut writer, &result)?;
        writer.write_all(b"\n")?;
        if (index + 1) % 25 == 0 || index + 1 == rows.len() {
            writer.flush()?;
            let memory = apxinf_mlx::memory_stats()?;
            eprintln!(
                "progress={}/{} elapsed_s={:.1} peak_gib={:.2}",
                index + 1,
                rows.len(),
                run_started.elapsed().as_secs_f64(),
                memory.peak_bytes as f64 / (1u64 << 30) as f64
            );
        }
    }
    writer.flush()?;
    let memory = apxinf_mlx::memory_stats()?;
    eprintln!(
        "run_s={:.3} peak_bytes={} active_bytes={}",
        run_started.elapsed().as_secs_f64(),
        memory.peak_bytes,
        memory.active_bytes
    );
    Ok(())
}

enum Outcome {
    Unsupported(String),
    Failed(String),
}

/// One schema forward for the whole request, converted to the kit's answer
/// shape: `noul` reports P(true); `choice` reports every option's probability.
fn answer(
    model: &mut apxinf_model::LoadedModel,
    state: &Value,
    questions: &Map<String, Value>,
) -> Result<Map<String, Value>, Outcome> {
    for (key, question) in questions {
        let kind = question.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "noul" && kind != "choice" {
            return Err(Outcome::Unsupported(format!(
                "{key}: unsupported question type {kind:?}"
            )));
        }
    }
    let request = json!({"state": state, "questions": questions}).to_string();
    let schema = DecisionSchema::from_json(&request).map_err(|e| Outcome::Failed(e.to_string()))?;
    let output = model.decide_schema(&schema).map_err(|error| {
        let message = error.to_string();
        if message.contains("maximum is") {
            Outcome::Unsupported(message)
        } else {
            Outcome::Failed(message)
        }
    })?;
    let mut answers = Map::new();
    for scored in output.answers {
        let probabilities: BTreeMap<_, _> = scored
            .scores
            .iter()
            .map(|score| (score.code.clone(), score.probability as f64))
            .collect();
        let kind = questions[&scored.question_id]["type"]
            .as_str()
            .unwrap_or("");
        let entry = if kind == "noul" {
            json!({"type": "noul", "noul": probabilities["true"]})
        } else {
            let selected = scored
                .selected()
                .map_err(|e| Outcome::Failed(e.to_string()))?;
            json!({
                "type": "choice",
                "choice": selected.code,
                "probabilities": probabilities,
            })
        };
        answers.insert(scored.question_id, entry);
    }
    Ok(answers)
}

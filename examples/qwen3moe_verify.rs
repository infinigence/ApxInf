//! Compare apxinf's Qwen3-MoE AWQ INT4 output against the torch-CPU reference.
//!
//! The reference lives in `experiment/qwen3moe/reference/qwen3moe_awq_reference.py`
//! and writes `<case>.json` (prompt, token ids, greedy continuation) plus
//! `<case>.npz` (prefill logits, per-step logits, per-layer signatures). This
//! example replays the same token ids through the CUDA runtime and dumps the
//! logits it produced, so the comparison itself runs in NumPy, next to the
//! reference:
//!
//! ```text
//! cargo run --release --features cuda --example qwen3moe_verify -- \
//!     --model /opt/data/models/Qwen3-30B-A3B-Instruct-2507-AWQ \
//!     --case  /opt/data/dev/ref/raw0.json \
//!     --out   /opt/data/dev/ref/raw0.apxinf
//! python experiment/qwen3moe/reference/compare_reference.py \
//!     --reference /opt/data/dev/ref/raw0.npz --apxinf /opt/data/dev/ref/raw0.apxinf.json
//! ```
//!
//! `--out` produces `<out>.bin`, a flat little-endian `f32` array of
//! `[rows, vocab_size]`, and `<out>.json` describing it. Row `s` holds the
//! logits that *selected* greedy token `s`, which is exactly the reference's
//! `step_logits[s]` (and row 0 also equals its `prefill_logits`). One extra
//! row is written past the last reference step; the comparison ignores it.
//!
//! Loading goes through `AutoModel` rather than `Qwen3Moe::load` so the
//! registry wiring is on the tested path too. Set `APXINF_QWEN3MOE_GRAPHS=0`
//! to bisect against eager decode launches.

use std::io::Write;
use std::path::PathBuf;

use apxinf_core::{Device, Tensor};
use apxinf_model::auto::{AutoModel, LoadOptions};
use apxinf_model::LlmTrait;

fn main() {
    if let Err(error) = run() {
        eprintln!("qwen3moe_verify: {error}");
        std::process::exit(1);
    }
}

struct Args {
    model: PathBuf,
    case: PathBuf,
    out: PathBuf,
    steps: Option<usize>,
    free_running: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut case = None;
    let mut out = None;
    let mut steps = None;
    let mut free_running = false;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().ok_or(format!("`{flag}` needs a value"));
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value()?)),
            "--case" => case = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--steps" => {
                steps = Some(value()?.parse::<usize>().map_err(|e| format!("--steps: {e}"))?)
            }
            "--free-running" => free_running = true,
            other => return Err(format!("unknown flag `{other}`")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        case: case.ok_or("--case is required")?,
        out: out.ok_or("--out is required")?,
        steps,
        free_running,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let case: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&args.case)
            .map_err(|e| format!("read {}: {e}", args.case.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", args.case.display()))?;
    let token_ids = u32_array(&case["token_ids"]).ok_or("case json has no `token_ids` array")?;
    if token_ids.is_empty() {
        return Err("case json has an empty `token_ids`".into());
    }
    let reference_greedy = u32_array(&case["greedy_tokens"]).unwrap_or_default();
    let steps = args.steps.unwrap_or_else(|| reference_greedy.len().max(1));

    let mut loaded = AutoModel::load_model(Device::Cuda(0), &args.model, &LoadOptions::default())
        .map_err(|e| format!("load {}: {e}", args.model.display()))?;
    let model = loaded.text_mut().map_err(|e| e.to_string())?;
    let vocab = model.vocab_size();

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(steps + 1);
    let mut greedy: Vec<u32> = Vec::with_capacity(steps);

    // Teacher forcing: feed the *reference's* token at each step rather than
    // our own argmax. Once the two continuations diverge they are decoding
    // different sentences, and a per-step logit diff stops meaning anything.
    // Forcing keeps the context identical for every step, so the error trend
    // across steps is a real signal about the decode path. `greedy` still
    // records what apxinf would have picked, so the free-running continuation
    // is visible from the same run.
    let forced = !args.free_running && !reference_greedy.is_empty();

    let mut row = logits_row(model, &token_ids, 0, vocab, "prefill")?;
    let mut next = argmax(&row);
    row.shrink_to_fit();
    rows.push(row);

    for step in 0..steps {
        greedy.push(next);
        let fed = if forced {
            reference_greedy.get(step).copied().unwrap_or(next)
        } else {
            next
        };
        let position = (token_ids.len() + step) as u32;
        let row = logits_row(model, &[fed], position, vocab, "decode")?;
        next = argmax(&row);
        rows.push(row);
    }

    let bin = suffixed(&args.out, ".bin");
    let mut file = std::io::BufWriter::new(
        std::fs::File::create(&bin).map_err(|e| format!("create {}: {e}", bin.display()))?,
    );
    for row in &rows {
        let mut bytes = Vec::with_capacity(row.len() * 4);
        for value in row {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes)
            .map_err(|e| format!("write {}: {e}", bin.display()))?;
    }
    file.flush().map_err(|e| format!("flush: {e}"))?;

    let json = suffixed(&args.out, ".json");
    let meta = serde_json::json!({
        "logits": bin.file_name().and_then(|n| n.to_str()),
        "rows": rows.len(),
        "vocab_size": vocab,
        "token_ids": token_ids,
        "greedy_tokens": greedy,
        "reference_greedy_tokens": reference_greedy,
        "teacher_forced": forced,
        "cuda_graphs": std::env::var("APXINF_QWEN3MOE_GRAPHS").unwrap_or_else(|_| "1".into()),
    });
    std::fs::write(&json, serde_json::to_string_pretty(&meta).unwrap())
        .map_err(|e| format!("write {}: {e}", json.display()))?;

    println!(
        "context: {}",
        if forced {
            "teacher-forced on the reference tokens"
        } else {
            "free-running (apxinf feeds its own argmax)"
        }
    );
    println!("apxinf    argmax: {greedy:?}");
    if !reference_greedy.is_empty() {
        println!("reference greedy: {reference_greedy:?}");
        let agree = greedy
            .iter()
            .zip(&reference_greedy)
            .take_while(|(a, b)| a == b)
            .count();
        let common = greedy.len().min(reference_greedy.len());
        let disagreements = greedy
            .iter()
            .zip(&reference_greedy)
            .filter(|(a, b)| a != b)
            .count();
        if agree == common {
            println!("agree on all {common} compared steps");
        } else if forced {
            println!("first divergence at step {agree} of {common}; {disagreements} disagree");
        } else {
            println!(
                "first divergence at step {agree} of {common} \
                 (later steps decode different contexts and are not comparable)"
            );
        }
    }
    println!("wrote {} and {}", bin.display(), json.display());
    Ok(())
}

/// Append a suffix rather than replacing an extension: `--out raw0.apxinf`
/// should yield `raw0.apxinf.bin`, not overwrite the reference's `raw0.json`.
fn suffixed(base: &std::path::Path, suffix: &str) -> PathBuf {
    let mut name = base.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn u32_array(value: &serde_json::Value) -> Option<Vec<u32>> {    Some(
        value
            .as_array()?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as u32)
            .collect(),
    )
}

/// Run one forward and bring its logits back as f32 on the host.
fn logits_row(
    model: &mut dyn LlmTrait,
    token_ids: &[u32],
    start_pos: u32,
    vocab: usize,
    what: &str,
) -> Result<Vec<f32>, String> {
    let device: Tensor = model
        .forward(token_ids, start_pos)
        .map_err(|e| format!("{what}: {e}"))?;
    let host = model
        .backend()
        .to_cpu(&device)
        .map_err(|e| format!("{what} logits to host: {e}"))?;
    let mut row = host
        .to_f32_vec()
        .map_err(|e| format!("{what} logits to f32: {e}"))?;
    if row.len() < vocab {
        return Err(format!(
            "{what} returned {} logits, expected at least {vocab}",
            row.len()
        ));
    }
    // Keep the last position: prefill returns [1, vocab] today, but a runtime
    // that returned every position would put the one we want at the end.
    row.drain(..row.len() - vocab);
    Ok(row)
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best as u32
}

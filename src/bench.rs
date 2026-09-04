//! `apxinf bench` — prefill/decode latency sweep over input and output lengths.
//!
//! Reports the four numbers a deployment is sized on: time to first token,
//! prefill throughput, time per output token, and decode throughput. The sweep
//! is over synthetic token ids rather than a real prompt, because latency
//! depends on sequence length and cache state, not on what the tokens mean —
//! and a synthetic stream keeps the benchmark reproducible without a tokenizer.
//!
//! ```text
//! apxinf bench -m <model dir> -d cuda --isl 128,1024,4096 --osl 128
//! ```

use std::path::Path;
use std::time::Instant;

use apxinf_core::{Device, Result};
use apxinf_model::{AutoModel, LoadOptions};

/// One (ISL, OSL) measurement, averaged over `iters`.
struct Row {
    isl: usize,
    osl: usize,
    ttft_ms: f64,
    prefill_tps: f64,
    tpot_ms: f64,
    decode_tps: f64,
}

pub struct BenchArgs<'a> {
    pub model_dir: &'a Path,
    pub device: Device,
    pub dtype: Option<apxinf_core::DType>,
    pub isl: Vec<usize>,
    pub osl: Vec<usize>,
    pub warmup: usize,
    pub iters: usize,
    pub json: Option<&'a Path>,
}

/// Parse `--isl 128,1024,4096` into lengths.
pub fn parse_lengths(spec: &str, flag: &str) -> std::result::Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value: usize = part
            .parse()
            .map_err(|e| format!("{flag}: `{part}` is not a length: {e}"))?;
        if value == 0 {
            return Err(format!("{flag}: lengths must be positive"));
        }
        out.push(value);
    }
    if out.is_empty() {
        return Err(format!("{flag}: no lengths given"));
    }
    Ok(out)
}

/// Deterministic pseudo-random token ids. A constant id would let a cache or a
/// router memoize across positions and flatter the result; this keeps the
/// expert selection realistically scattered.
fn synthetic_tokens(count: usize, vocab: usize) -> Vec<u32> {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Stay clear of the low ids, which are special tokens in most
            // vocabularies and can trip early-stop logic.
            ((state >> 33) as usize % (vocab - 1000) + 1000) as u32
        })
        .collect()
}

pub fn run(args: BenchArgs<'_>) -> Result<()> {
    let options = LoadOptions {
        text_weight_dtype: args.dtype,
        ..Default::default()
    };
    let load_started = Instant::now();
    let mut loaded = AutoModel::load_model(args.device, args.model_dir, &options)?;
    let load_secs = load_started.elapsed().as_secs_f64();
    let model = loaded.text_mut()?;
    let vocab = model.vocab_size();
    println!(
        "model {} loaded in {load_secs:.1}s on {}",
        args.model_dir.display(),
        args.device
    );

    let max_isl = args.isl.iter().copied().max().unwrap_or(1);
    let max_osl = args.osl.iter().copied().max().unwrap_or(1);
    let tokens = synthetic_tokens(max_isl + max_osl, vocab);

    let mut rows = Vec::new();
    for &isl in &args.isl {
        for &osl in &args.osl {
            let mut ttft = Vec::with_capacity(args.iters);
            let mut decode = Vec::with_capacity(args.iters);
            for iteration in 0..(args.warmup + args.iters) {
                model.reset();
                let prompt = &tokens[..isl];

                model.backend().synchronize()?;
                let started = Instant::now();
                let _ = model.forward(prompt, 0)?;
                model.backend().synchronize()?;
                let prefill_secs = started.elapsed().as_secs_f64();

                let started = Instant::now();
                for step in 0..osl {
                    let token = tokens[isl + step];
                    let _ = model.forward(&[token], (isl + step) as u32)?;
                }
                model.backend().synchronize()?;
                let decode_secs = started.elapsed().as_secs_f64();

                if iteration >= args.warmup {
                    ttft.push(prefill_secs);
                    decode.push(decode_secs);
                }
            }
            // The median resists the one slow iteration that a background
            // process or a clock ramp will occasionally produce.
            let prefill_secs = median(&mut ttft);
            let decode_secs = median(&mut decode);
            let row = Row {
                isl,
                osl,
                ttft_ms: prefill_secs * 1e3,
                prefill_tps: isl as f64 / prefill_secs,
                tpot_ms: decode_secs * 1e3 / osl as f64,
                decode_tps: osl as f64 / decode_secs,
            };
            println!(
                "  ISL {:>6}  OSL {:>5}  TTFT {:>9.1} ms  prefill {:>9.1} tok/s  \
                 TPOT {:>7.2} ms  decode {:>7.2} tok/s",
                row.isl, row.osl, row.ttft_ms, row.prefill_tps, row.tpot_ms, row.decode_tps
            );
            rows.push(row);
        }
    }

    println!();
    println!(
        "{:>8} {:>7} {:>12} {:>14} {:>11} {:>13}",
        "ISL", "OSL", "TTFT (ms)", "prefill tok/s", "TPOT (ms)", "decode tok/s"
    );
    for row in &rows {
        println!(
            "{:>8} {:>7} {:>12.1} {:>14.1} {:>11.2} {:>13.2}",
            row.isl, row.osl, row.ttft_ms, row.prefill_tps, row.tpot_ms, row.decode_tps
        );
    }

    if let Some(path) = args.json {
        let payload = serde_json::json!({
            "model": args.model_dir.display().to_string(),
            "device": args.device.to_string(),
            "load_seconds": load_secs,
            "warmup": args.warmup,
            "iters": args.iters,
            "rows": rows.iter().map(|r| serde_json::json!({
                "isl": r.isl,
                "osl": r.osl,
                "ttft_ms": r.ttft_ms,
                "prefill_tokens_per_s": r.prefill_tps,
                "tpot_ms": r.tpot_ms,
                "decode_tokens_per_s": r.decode_tps,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(path, serde_json::to_string_pretty(&payload).unwrap()).map_err(|e| {
            apxinf_core::Error::Other(format!("write {}: {e}", path.display()))
        })?;
        println!();
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

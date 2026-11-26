//! Qwen3.8-27B-NVFP4 end-to-end latency benchmark through the public entry.
//!
//! Loads through `AutoModel` (registry detection from the checkpoint's
//! config.json), generates greedily through `LoadedModel::generate_streaming`,
//! and reports JSON per repeat plus a summary. The prompt is deterministic
//! and the run asserts that every repeat generates identical tokens.
//!
//! ```text
//! cargo run -p apxinf-model --features cuda --release --example qwen38_bench -- \
//!     <checkpoint-dir> [--prompt-len N] [--max-new N] [--repeats N]
//!     [--prepare] [--tuning-path DIR]
//! ```
//!
//! Run `--prepare` in a separate process to tune missing recipes. Measurement
//! disables autotuning; its first generation captures the decode graphs and is
//! reported with `"warmup": true`, excluded from the summary statistics.

use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::llm_trait::LlmInput;
use apxinf_model::{AutoModel, LoadOptions};

const VOCAB: usize = 248_320;

/// The fixed LCG prompt of the kernel harness (`APXINF_QWEN38_PROMPT_LEN`
/// path): reproducible, spread across the vocabulary, clear of the special
/// tokens near the top of the id range.
fn deterministic_prompt(n: usize) -> Vec<u32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 16) as usize % (VOCAB - 1000)) as u32
        })
        .collect()
}

fn arg_value(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect(name))
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = args
        .get(1)
        .filter(|arg| !arg.starts_with("--"))
        .expect("usage: qwen38_bench <checkpoint-dir> [--prompt-len N] [--max-new N] [--repeats N] [--prepare] [--tuning-path DIR]");
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--prepare" => index += 1,
            "--prompt-len" | "--max-new" | "--repeats" | "--tuning-path" => {
                if args
                    .get(index + 1)
                    .is_none_or(|value| value.starts_with("--"))
                {
                    return Err(format!("{} requires a value", args[index]).into());
                }
                index += 2;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    let prompt_len = arg_value(&args, "--prompt-len", 2048);
    let max_new = arg_value(&args, "--max-new", 128);
    let repeats = arg_value(&args, "--repeats", 3);
    let prepare = args.iter().any(|arg| arg == "--prepare");
    let tuning_path = args
        .iter()
        .position(|arg| arg == "--tuning-path")
        .map(|index| std::path::PathBuf::from(&args[index + 1]));
    if prompt_len == 0 || max_new < 2 || repeats == 0 {
        return Err("require --prompt-len > 0, --max-new >= 2 and --repeats > 0".into());
    }
    if let Some(path) = tuning_path.as_ref() {
        if prepare {
            std::fs::create_dir_all(path)?;
        } else if !path.is_dir() {
            return Err(format!(
                "recipe directory {} is missing; run --prepare first",
                path.display()
            )
            .into());
        }
    }
    let options = LoadOptions {
        autotune: prepare,
        tuning_path,
        ..LoadOptions::default()
    };

    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), checkpoint, &options)?;
    let load_secs = load_start.elapsed().as_secs_f64();

    let prompt = deterministic_prompt(prompt_len);
    let mode = if prepare { "prepare" } else { "benchmark" };
    println!(
        "{{\"schema\": \"apxinf.qwen38.benchmark.v1\", \"checkpoint\": {checkpoint:?}, \
         \"prompt_len\": {prompt_len}, \"max_new\": {max_new}, \"repeats\": {repeats}, \
         \"mode\": {mode:?}, \"autotune\": {prepare}, \"load_s\": {load_secs:.1}}}"
    );

    let mut ttfts = Vec::new();
    let mut tpots = Vec::new();
    let mut outputs: Vec<Vec<u32>> = Vec::new();
    // Preparation tunes and verifies a second generation without reporting
    // benchmark timings. Measurement warms only graphs and prepared executions.
    let generations = if prepare { 2 } else { repeats + 1 };
    for repeat in 0..generations {
        let (tokens, profile) = model.generate_streaming(
            LlmInput::text(&prompt),
            max_new,
            |_token| {},
            None, // fixed-length benchmark: EOS must not stop decode early
        )?;
        assert_eq!(tokens.len(), max_new, "fixed decode budget");
        if prepare {
            outputs.push(tokens);
            continue;
        }

        let ttft = profile.ttft_ms().expect("profile records first token");
        let tpot = profile.tpot_ms().expect("profile records decode");
        let warmup = repeat == 0;
        println!(
            "{{\"repeat\": {repeat}, \"warmup\": {warmup}, \"ttft_ms\": {ttft:.2}, \
             \"decode_ms_per_token\": {tpot:.3}, \"decode_tokens_per_second\": {:.2}}}",
            1e3 / tpot
        );
        if !warmup {
            ttfts.push(ttft);
            tpots.push(tpot);
        }
        outputs.push(tokens);
    }

    assert!(
        outputs.windows(2).all(|pair| pair[0] == pair[1]),
        "generation is not deterministic across repeats"
    );
    if prepare {
        println!("{{\"prepared\": true, \"deterministic_generations\": 2}}");
        return Ok(());
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    println!(
        "{{\"summary\": {{\"ttft_ms_mean\": {:.2}, \"decode_ms_per_token_mean\": {:.3}, \
         \"decode_tokens_per_second_mean\": {:.2}}}}}",
        mean(&ttfts),
        mean(&tpots),
        1e3 / mean(&tpots)
    );
    Ok(())
}

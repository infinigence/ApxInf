//! MiniCPM5-2B long-context greedy benchmark on Metal through the public entry.
//!
//! Reads the frozen EngineTailor long-context corpus (`corpus.json`, one prompt token-id list per
//! cell), generates a fixed greedy budget per cell with EOS disabled, and prints one JSON line per
//! cell and repeat: TTFT, decode ms/token and the SHA-256 of the comma-separated token ids. The
//! first visit is a warm-up and is reported with `"warmup": true`.
//!
//! ```text
//! cargo run -p apxinf-model --features mlx --release --example minicpm5_longctx_bench -- \
//!     <checkpoint-dir> <corpus.json> [--cells a,b] [--max-new N] [--repeats N] [--variant V]
//! ```

use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::llm_trait::LlmInput;
use apxinf_model::{AutoModel, LoadOptions};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Corpus {
    cells: Vec<Cell>,
}

#[derive(Deserialize)]
struct Cell {
    id: String,
    prompt_token_ids: Vec<u32>,
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn token_sha(tokens: &[u32]) -> String {
    let text = tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: minicpm5_longctx_bench <checkpoint-dir> <corpus.json> \
                 [--cells a,b] [--max-new N] [--repeats N] [--variant V]";
    let checkpoint = args.get(1).filter(|a| !a.starts_with("--")).expect(usage);
    let corpus_path = args.get(2).filter(|a| !a.starts_with("--")).expect(usage);
    let max_new: usize = arg(&args, "--max-new").map_or(128, |v| v.parse().expect("--max-new"));
    let repeats: usize = arg(&args, "--repeats").map_or(3, |v| v.parse().expect("--repeats"));
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(corpus_path)?)?;
    let cells: Vec<&Cell> = match arg(&args, "--cells") {
        Some(list) => list
            .split(',')
            .map(|id| {
                corpus
                    .cells
                    .iter()
                    .find(|c| c.id == id)
                    .expect("unknown --cells id")
            })
            .collect(),
        None => corpus.cells.iter().collect(),
    };

    let options = LoadOptions {
        model_name: Some("metal_minicpm5".into()),
        model_variant: arg(&args, "--variant").map(str::to_owned),
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Metal(0), checkpoint, &options)?;
    println!(
        "{{\"schema\": \"apxinf.minicpm5.longctx.v1\", \"checkpoint\": {checkpoint:?}, \
         \"max_new\": {max_new}, \"repeats\": {repeats}, \"load_s\": {:.1}}}",
        load_start.elapsed().as_secs_f64()
    );

    let warm = &cells[0].prompt_token_ids;
    model.generate_streaming(
        LlmInput::text(&warm[..warm.len().min(256)]),
        8,
        |_| {},
        None,
    )?;
    for cell in &cells {
        let mut first: Option<Vec<u32>> = None;
        for repeat in 0..=repeats {
            let (tokens, profile) = model.generate_streaming(
                LlmInput::text(&cell.prompt_token_ids),
                max_new,
                |_| {},
                None,
            )?;
            assert_eq!(tokens.len(), max_new, "fixed decode budget");
            if let Some(reference) = &first {
                assert_eq!(
                    &tokens, reference,
                    "{}: generation is not deterministic",
                    cell.id
                );
            } else {
                first = Some(tokens.clone());
            }
            println!(
                "{{\"cell\": {:?}, \"prompt\": {}, \"repeat\": {repeat}, \"warmup\": {}, \
                 \"ttft_ms\": {:.2}, \"tpot_ms\": {:.3}, \"tokens_sha256\": \"{}\"}}",
                cell.id,
                cell.prompt_token_ids.len(),
                repeat == 0,
                profile.ttft_ms().expect("profile records first token"),
                profile.tpot_ms().expect("profile records decode"),
                token_sha(&tokens)
            );
        }
    }
    Ok(())
}

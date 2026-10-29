//! Source-bound serial generation diagnostics. Run under the host's Metal lease.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native {
    use apxinf_core::{DType, Device};
    use apxinf_model::{AutoModel, GenerationOptions, GenerationRequest, LlmInput, LoadOptions};
    use apxinf_tokenizer::Tokenizer;
    use serde::Deserialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::io::Read;
    use std::{path::PathBuf, time::Instant};

    #[derive(Deserialize)]
    struct Corpus {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        id: String,
        messages: Vec<Value>,
        max_tokens: usize,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        if args.len() != 4 {
            return Err(
                "usage: metal_text_probe CHECKPOINT CORPUS REPEATS release-all|retain-code".into(),
            );
        }
        let model_path = PathBuf::from(&args[0]).canonicalize()?;
        let corpus: Corpus = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        let repeats = args[2].parse::<usize>()?;
        let retain = match args[3].as_str() {
            "release-all" => false,
            "retain-code" => true,
            _ => return Err("unknown cleanup policy".into()),
        };
        if repeats == 0 || corpus.cases.is_empty() {
            return Err("nonempty corpus and positive repeats required".into());
        }
        std::env::set_var("APXINF_MINICPM5_PREFILL_CHUNK", "256");
        apxinf_mlx::set_memory_limit(10 * 1024 * 1024 * 1024)?;
        apxinf_mlx::set_cache_limit(2 * 1024 * 1024 * 1024)?;
        let mut model = AutoModel::load_model(
            Device::Metal(0),
            &model_path,
            &LoadOptions {
                model_name: Some("metal_minicpm5".into()),
                model_variant: Some("bf16-compiled".into()),
                text_weight_dtype: Some(DType::BF16),
                ..Default::default()
            },
        )?;
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;
        let eos = model
            .generation_defaults()?
            .eos_token_ids
            .clone()
            .ok_or("checkpoint EOS is missing")?;
        model.text_mut()?.backend().synchronize()?;
        apxinf_mlx::clear_cache()?;
        let source = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()?;
        let dirty = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .output()?;
        if !source.status.success() || !dirty.status.success() {
            return Err("run from the measured source checkout".into());
        }
        let executable = std::env::current_exe()?;
        let mut file = std::fs::File::open(&executable)?;
        let mut checksum = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            checksum.update(&buffer[..count]);
        }
        println!(
            "{}",
            json!({"kind":"metadata", "checkout_commit":String::from_utf8(source.stdout)?.trim(),
        "checkout_dirty":!dirty.stdout.is_empty(), "executable_sha256":format!("{:x}", checksum.finalize()), "model_path":model_path, "cleanup_policy":args[3],
        "mlx_library":apxinf_mlx::loaded_library_path()?, "allocator_cache_limit_bytes":2147483648u64,
        "memory_limit_bytes":10737418240u64,"prefill_chunk_tokens":256})
        );
        let options = serde_json::Map::from_iter([("enable_thinking".into(), Value::Bool(false))]);
        for repeat in 0..repeats {
            for case in &corpus.cases {
                if case.id.is_empty() || case.max_tokens == 0 {
                    return Err("cases need an ID and positive output allowance".into());
                }
                let prompt = tokenizer.apply_rich_chat_template(&case.messages, &options)?;
                let ids = tokenizer.encode(&prompt)?;
                if ids
                    .len()
                    .checked_add(case.max_tokens)
                    .is_none_or(|n| n > 16384)
                {
                    return Err(format!(
                        "{} exceeds the comparable native service context",
                        case.id
                    )
                    .into());
                }
                let mut generation = GenerationOptions::greedy(case.max_tokens, None);
                generation.eos_token_ids = Some(eos.clone());
                let started = Instant::now();
                let result = model.text_mut()?.generate_streaming_controlled_dyn(
                    GenerationRequest {
                        input: LlmInput::text(&ids),
                        options: &generation,
                    },
                    &mut |_| Ok(true),
                )?;
                let generation_ms = started.elapsed().as_secs_f64() * 1000.;
                let output = result.token_ids();
                let visible = output
                    .iter()
                    .copied()
                    .take_while(|id| !eos.contains(id))
                    .collect::<Vec<_>>();
                let content = tokenizer.decode_with_options(&visible, false)?;
                let cleanup_started = Instant::now();
                if retain {
                    model
                        .text_mut()?
                        .release_request_state_retaining_preparation()?;
                } else {
                    model.text_mut()?.release_request_state()?;
                }
                apxinf_mlx::clear_cache()?;
                let mut memory = apxinf_mlx::memory_stats()?;
                if retain && memory.active_bytes + memory.cache_bytes > 9 * 1024 * 1024 * 1024 {
                    model.text_mut()?.release_request_state()?;
                    apxinf_mlx::clear_cache()?;
                    memory = apxinf_mlx::memory_stats()?;
                    if memory.active_bytes + memory.cache_bytes > 9 * 1024 * 1024 * 1024 {
                        return Err(
                            "native residency exceeds the comparable service allowance".into()
                        );
                    }
                }
                println!(
                    "{}",
                    json!({"kind":"result","id":case.id,"repeat":repeat,"input_tokens":ids.len(),
                "completion_tokens":output.len(),"token_ids":output,"content":content,
                "finish_reason":if result.tokens.last().is_some_and(|t| eos.contains(&t.token_id)) {"stop"} else {"length"},
                "generation_ms":generation_ms,"preparation_ms":result.profile.preparation_time().as_secs_f64()*1000.,
                "ttft_ms":result.profile.ttft_ms(),"tpot_ms":result.profile.tpot_ms(),
                "cleanup_ms":cleanup_started.elapsed().as_secs_f64()*1000.,
                "memory":{"active_bytes":memory.active_bytes,"cache_bytes":memory.cache_bytes,"peak_bytes":memory.peak_bytes}})
                );
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        native::run()
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Err("native Metal diagnostics require Apple Silicon macOS".into())
    }
}

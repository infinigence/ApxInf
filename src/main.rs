//! ApxInf LLM inference engine CLI.

use std::io::Write;
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use apxinf_core::{DType, Device, Tensor};
#[cfg(feature = "cuda")]
use apxinf_core::{NextTokenLogits, SamplingBackend, TokenSamplingInit, TokenSamplingSpec};
#[cfg(feature = "cuda")]
use apxinf_cuda::CudaBackend;
#[cfg(feature = "cuda")]
use apxinf_loader::safetensors;
#[cfg(feature = "cuda")]
use apxinf_model::qwen35::{load_embedding_row, HybridUnit, HybridUnitMode, Qwen35LmHead};
use apxinf_model::{
    AutoModel, GenerationConfigSource, GenerationOptions, ImageInput, LlmInput, LoadOptions,
    Qwen35CheckpointReport, Qwen35Config, SamplingMode,
};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use clap::{Parser, Subcommand};

#[cfg(feature = "cuda")]
mod qwen35_server;

#[derive(Parser)]
#[command(name = "apxinf")]
#[command(about = "LLM inference engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt
    Generate {
        /// Path to HuggingFace model directory (contains model.safetensors and tokenizer.json)
        #[arg(short, long)]
        model: PathBuf,

        /// Input prompt (treated as user message in chat mode, or raw text if no chat template)
        #[arg(short, long)]
        prompt: String,

        /// Path to an image file (for Qwen3-VL multimodal). When set, the
        /// image is preprocessed by a Python helper and fed alongside the
        /// prompt. Only for qwen3_vl models.
        #[arg(long)]
        image: Option<PathBuf>,

        /// Maximum new tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Explicitly enable random categorical sampling.
        #[arg(long, conflicts_with = "greedy")]
        sample: bool,

        /// Explicitly use greedy token selection.
        #[arg(long, conflicts_with = "sample")]
        greedy: bool,

        /// Sampling temperature. Zero selects greedy generation.
        #[arg(long)]
        temperature: Option<f32>,

        /// Retain only the highest-k logits; zero or negative disables top-k.
        #[arg(long)]
        top_k: Option<i64>,

        /// Nucleus probability mass.
        #[arg(long)]
        top_p: Option<f32>,

        /// Repetition penalty; 1 disables it.
        #[arg(long)]
        repetition_penalty: Option<f32>,

        /// Frequency penalty applied per token occurrence.
        #[arg(long)]
        frequency_penalty: Option<f32>,

        /// Presence penalty applied once to previously seen tokens.
        #[arg(long)]
        presence_penalty: Option<f32>,

        /// Counter-based sampling seed.
        #[arg(long)]
        seed: Option<u64>,

        /// Generation defaults: auto, apxinf, or a JSON file/directory path.
        #[arg(long, default_value = "auto")]
        generation_config: String,

        /// JSON object applied over model defaults and under request flags.
        #[arg(long)]
        override_generation_config: Option<String>,

        /// Disable EOS-based early stopping (generate until max_tokens)
        #[arg(long)]
        no_eos_stop: bool,

        /// System prompt for chat mode
        #[arg(long)]
        system: Option<String>,

        /// Device to run inference on (cpu or cuda)
        #[arg(short, long, default_value = "cpu")]
        device: String,

        /// Weight dtype ("fp32" or "bf16"). On CUDA, "bf16" halves weight-
        /// bandwidth and enables the bf16 fast path. Ignored on CPU.
        #[arg(long, default_value = "fp32")]
        dtype: String,
    },

    /// Validate Qwen3.8 identity, config, shards, and quantization layout.
    Inspect {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(long)]
        json: bool,
    },

    /// Start the resident OpenAI-compatible Qwen3.8 INT4 text service.
    Serve {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8001)]
        port: u16,
        #[arg(long, default_value_t = 32768)]
        max_model_len: usize,
        /// Allocate the M64 Marlin prefill workspace. Set
        /// apxinf_prefill_mode=m8 per request to select the M8 fallback.
        #[arg(long)]
        enable_marlin_m64: bool,
        /// Store full-attention KV as per-row E4M3. This raises the supported
        /// single-request context limit from 32K to 128K on a 24 GB GPU.
        #[arg(long)]
        enable_e4m3_kv: bool,
        /// Load the native Qwen3.8 visual encoder and enable one-image chat
        /// requests. Requires a Python executable with the pinned HF
        /// processor dependencies in APXINF_PROCESSOR_PYTHON.
        #[arg(long)]
        enable_multimodal: bool,
    },

    /// Run a quick test of the engine
    Test,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            model,
            prompt,
            image,
            max_tokens,
            sample,
            greedy,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            generation_config,
            override_generation_config,
            no_eos_stop,
            system,
            device,
            dtype,
        } => {
            let device = parse_device(&device);
            if read_model_type(&model).as_deref() == Some("qwen3_5") {
                if image.is_some() {
                    eprintln!(
                        "Qwen3.8 image execution is not implemented yet; the unified LlmInput image boundary remains unsupported for this native runtime"
                    );
                    std::process::exit(1);
                } else {
                    #[cfg(feature = "cuda")]
                    run_generate_qwen35(
                        &model,
                        &prompt,
                        max_tokens,
                        !no_eos_stop,
                        system.as_deref(),
                        device,
                        sample,
                        greedy,
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty,
                        frequency_penalty,
                        presence_penalty,
                        seed,
                        &generation_config,
                        override_generation_config.as_deref(),
                    );
                    #[cfg(not(feature = "cuda"))]
                    {
                        eprintln!("Qwen3.8 native generation requires an ApxInf CUDA build");
                        std::process::exit(1);
                    }
                }
            } else if let Err(error) = run_generate(
                &model,
                &prompt,
                image.as_ref(),
                max_tokens,
                !no_eos_stop,
                system.as_deref(),
                device,
                &dtype,
                sample,
                greedy,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                seed,
                &generation_config,
                override_generation_config.as_deref(),
            ) {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        Commands::Inspect { model, json } => {
            if let Err(error) = run_inspect(&model, json) {
                eprintln!("Model inspection failed: {error}");
                std::process::exit(2);
            }
        }
        Commands::Serve {
            model,
            host,
            port,
            max_model_len,
            enable_marlin_m64,
            enable_e4m3_kv,
            enable_multimodal,
        } => {
            #[cfg(feature = "cuda")]
            if let Err(error) = qwen35_server::serve(
                &model,
                &host,
                port,
                max_model_len,
                enable_marlin_m64,
                enable_multimodal,
                enable_e4m3_kv,
            ) {
                eprintln!("Server failed: {error}");
                std::process::exit(2);
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (
                    model,
                    host,
                    port,
                    max_model_len,
                    enable_marlin_m64,
                    enable_e4m3_kv,
                    enable_multimodal,
                );
                eprintln!("The native Qwen3.8 server requires an ApxInf CUDA build");
                std::process::exit(2);
            }
        }
        Commands::Test => {
            run_test();
        }
    }
}

fn parse_device(s: &str) -> Device {
    match s.to_lowercase().as_str() {
        "cuda" | "gpu" => Device::Cuda(0),
        "cpu" => Device::Cpu,
        _ => {
            eprintln!("Unknown device '{s}', defaulting to CPU. Use 'cpu' or 'cuda'.");
            Device::Cpu
        }
    }
}

fn run_generate(
    model_dir: &PathBuf,
    prompt: &str,
    image_path: Option<&PathBuf>,
    max_tokens: Option<usize>,
    eos_stop: bool,
    system_prompt: Option<&str>,
    device: Device,
    dtype: &str,
    sample: bool,
    greedy: bool,
    temperature: Option<f32>,
    top_k: Option<i64>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    generation_config: &str,
    override_generation_config: Option<&str>,
) -> Result<(), String> {
    println!("apxinf — LLM/VLM inference engine");
    println!();

    let model_name = AutoModel::detect_model_name(model_dir)
        .map_err(|error| format!("Failed to detect model type: {error}"))?;
    if image_path.is_some() && !matches!(model_name.as_str(), "qwen3_vl" | "qwen3vl") {
        return Err(format!("Model `{model_name}` does not support image input"));
    }

    let tokenizer_path = model_dir.join("tokenizer.json");
    println!("Loading tokenizer from {:?}...", tokenizer_path);
    let tok = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("Failed to load tokenizer: {error}"))?;
    println!("Vocab size: {}", tok.vocab_size());

    let eos_token_id = tok.eos_token_id();
    if let Some(eos) = eos_token_id {
        println!("EOS token ID: {eos}");
    }

    // Model-specific processors turn raw media into tensors, while generation
    // itself always receives the model-neutral LlmInput request.
    let (tokens, prepared_image) = if let Some(image_path) = image_path {
        println!("Preprocessing image via the Hugging Face processor...");
        let (data, shape, grid, tokens) =
            preprocess_image(model_dir, image_path, prompt, system_prompt)
                .map_err(|error| format!("Preprocessing failed: {error}"))?;
        println!(
            "pixel_values: {:?}, grid_thw: {:?}, prompt tokens: {}",
            shape,
            grid,
            tokens.len()
        );
        let pixels = Tensor::from_bf16(shape, &data)
            .map_err(|error| format!("Invalid processor output: {error}"))?;
        (tokens, Some((pixels, vec![grid])))
    } else {
        let tokens = encode_prompt(&tok, prompt, system_prompt)
            .map_err(|error| format!("Failed to encode prompt: {error}"))?;
        (tokens, None)
    };

    let text_weight_dtype = match dtype.to_ascii_lowercase().as_str() {
        "fp32" | "f32" => Some(DType::F32),
        "bf16" => Some(DType::BF16),
        other => {
            return Err(format!(
                "Unsupported text weight dtype `{other}`; use fp32 or bf16"
            ))
        }
    };
    let generation_overrides = override_generation_config
        .map(GenerationOptions::from_json_str)
        .transpose()
        .map_err(|error| format!("Invalid --override-generation-config: {error}"))?
        .unwrap_or_default();
    let options = LoadOptions {
        model_name: Some(model_name.clone()),
        text_weight_dtype,
        generation_config: GenerationConfigSource::from_cli_value(generation_config),
        generation_overrides,
        ..LoadOptions::default()
    };

    println!(
        "Loading {model_name} from {:?}... (dtype: {dtype})",
        model_dir
    );
    let mut model = AutoModel::load_model(device, model_dir, &options)
        .map_err(|error| format!("Failed to load model: {error}"))?;
    if prepared_image.is_some() {
        match model.text_capabilities() {
            Ok(capabilities) if capabilities.image => {}
            Ok(_) => return Err(format!("Model `{model_name}` does not support image input")),
            Err(error) => return Err(format!("Cannot generate with this model: {error}")),
        }
    }
    println!("Model ready.");

    let input = match prepared_image.as_ref() {
        Some((pixels, grids)) => LlmInput::with_image(&tokens, ImageInput::new(pixels, grids)),
        None => LlmInput::text(&tokens),
    };

    let configured_eos = model
        .generation_defaults()
        .map_err(|error| format!("Cannot read generation defaults: {error}"))?
        .eos_token_ids
        .is_some();
    let effective_max_tokens = max_tokens
        .or(model
            .generation_defaults()
            .ok()
            .and_then(|defaults| defaults.max_new_tokens))
        .unwrap_or(GenerationOptions::DEFAULT_MAX_NEW_TOKENS);

    println!();
    println!("Generating up to {effective_max_tokens} tokens...");
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut all_tokens = tokens.clone();

    let generation_options = GenerationOptions {
        max_new_tokens: max_tokens,
        eos_token_ids: if !eos_stop {
            Some(Vec::new())
        } else if configured_eos {
            None
        } else {
            eos_token_id.map(|id| vec![id])
        },
        sampling_mode: if sample {
            Some(SamplingMode::Random)
        } else if greedy {
            Some(SamplingMode::Greedy)
        } else {
            None
        },
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        seed,
        return_logprob: Some(false),
    };
    let output = model
        .generate_streaming_with_options(input, &generation_options, |token| {
            let token_id = token.token_id;
            all_tokens.push(token_id);
            if let Ok(text) = tok.decode(&all_tokens) {
                let previous = tok
                    .decode(&all_tokens[..all_tokens.len() - 1])
                    .unwrap_or_default();
                let delta = text.strip_prefix(&previous).unwrap_or(&text);
                print!("{delta}");
                out.flush().ok();
            }
        })
        .map_err(|error| format!("Generation failed: {error}"))?;

    println!();
    println!();
    println!("{}", output.profile.summary());
    Ok(())
}

fn encode_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    system_prompt: Option<&str>,
) -> Result<Vec<u32>, String> {
    if tokenizer.has_chat_template() {
        let mut messages = Vec::new();
        if let Some(system) = system_prompt {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(prompt));
        tokenizer
            .encode_chat(&messages)
            .map_err(|error| error.to_string())
    } else {
        tokenizer.encode(prompt).map_err(|error| error.to_string())
    }
}
#[cfg(feature = "cuda")]
fn run_generate_qwen35(
    model_dir: &PathBuf,
    prompt: &str,
    max_tokens: Option<usize>,
    eos_stop: bool,
    system_prompt: Option<&str>,
    device: Device,
    sample: bool,
    greedy: bool,
    temperature: Option<f32>,
    top_k: Option<i64>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    generation_config: &str,
    override_generation_config: Option<&str>,
) {
    if device != Device::Cuda(0) {
        eprintln!("Qwen3.5/Qwen3.8 native text generation currently requires --device cuda");
        return;
    }
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            eprintln!("Failed to load tokenizer: {error}");
            return;
        }
    };
    let prompt_tokens = if tokenizer.has_chat_template() {
        let mut messages = Vec::new();
        if let Some(system) = system_prompt {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(prompt));
        match tokenizer.encode_chat(&messages) {
            Ok(tokens) => tokens,
            Err(error) => {
                eprintln!("Failed to apply chat template: {error}");
                return;
            }
        }
    } else {
        match tokenizer.encode(prompt) {
            Ok(tokens) => tokens,
            Err(error) => {
                eprintln!("Failed to encode prompt: {error}");
                return;
            }
        }
    };
    if prompt_tokens.is_empty() {
        eprintln!("Tokenizer produced an empty prompt");
        return;
    }
    let deployment_overrides = match override_generation_config
        .map(GenerationOptions::from_json_str)
        .transpose()
    {
        Ok(options) => options.unwrap_or_default(),
        Err(error) => {
            eprintln!("Invalid --override-generation-config: {error}");
            return;
        }
    };
    let model_defaults =
        match GenerationConfigSource::from_cli_value(generation_config).load(model_dir) {
            Ok(options) => options,
            Err(error) => {
                eprintln!("Cannot load generation defaults: {error}");
                return;
            }
        };
    let request_options = GenerationOptions {
        max_new_tokens: max_tokens,
        eos_token_ids: if eos_stop {
            tokenizer.eos_token_id().map(|token| vec![token])
        } else {
            Some(Vec::new())
        },
        sampling_mode: if sample {
            Some(SamplingMode::Random)
        } else if greedy {
            Some(SamplingMode::Greedy)
        } else {
            None
        },
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        seed,
        return_logprob: Some(false),
    };
    let generation = match GenerationOptions::apxinf_defaults()
        .overlay(&model_defaults)
        .overlay(&deployment_overrides)
        .overlay(&request_options)
        .resolve()
    {
        Ok(options) => options,
        Err(error) => {
            eprintln!("Invalid generation options: {error}");
            return;
        }
    };
    let max_tokens = generation.max_new_tokens;
    let required = match prompt_tokens.len().checked_add(max_tokens) {
        Some(required) if required <= 32768 => required,
        _ => {
            eprintln!(
                "Current native text path supports prompt+output <=32768 tokens, got {}+{}",
                prompt_tokens.len(),
                max_tokens
            );
            return;
        }
    };
    let max_seq_len = required.next_power_of_two().min(32768);
    let manifest = match safetensors::inspect_path(model_dir) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("Failed to inspect checkpoint: {error}");
            return;
        }
    };
    let backend = match CudaBackend::new(0) {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("Failed to initialize CUDA: {error}");
            return;
        }
    };
    let context = backend.context();

    println!("apxinf — Qwen3.8 native INT4 text generation");
    println!(
        "Loading 64 decoder layers and W8 LM head (KV capacity {})...",
        max_seq_len
    );
    let load_start = Instant::now();
    let decoder = match HybridUnit::load_all(&manifest, context, max_seq_len) {
        Ok(decoder) => decoder,
        Err(error) => {
            eprintln!("Failed to load Qwen3.8 decoder: {error}");
            return;
        }
    };
    let lm_head = match Qwen35LmHead::load(&manifest, context) {
        Ok(head) => head,
        Err(error) => {
            eprintln!("Failed to load Qwen3.8 LM head: {error}");
            return;
        }
    };
    let mut sampler = match backend.create_token_sampler(TokenSamplingSpec {
        vocab_size: 248_320,
        max_sequence_len: max_seq_len,
    }) {
        Ok(sampler) => sampler,
        Err(error) => {
            eprintln!("Failed to create token sampler: {error}");
            return;
        }
    };
    if let Err(error) = sampler.begin(TokenSamplingInit {
        prompt_token_ids: &prompt_tokens,
        params: &generation.sampling,
        rng: generation.rng,
    }) {
        eprintln!("Failed to initialize token sampler: {error}");
        return;
    }
    let cache_shape = vec![4, max_seq_len, 256];
    let key_cache = Tensor::zeros(cache_shape.clone(), DType::BF16);
    let value_cache = Tensor::zeros(cache_shape, DType::BF16);
    let first_embedding = match load_embedding_row(&manifest, prompt_tokens[0]) {
        Ok(embedding) => embedding,
        Err(error) => {
            eprintln!("Failed to load embedding: {error}");
            return;
        }
    };
    if let Err(error) = decoder.reset(context, &first_embedding, &key_cache, &value_cache) {
        eprintln!("Failed to initialize decoder state: {error}");
        return;
    }
    println!("Model ready in {:.3}s", load_start.elapsed().as_secs_f64());
    println!("Prompt tokens: {}", prompt_tokens.len());

    let prefill_start = Instant::now();
    let tiled_tokens = prompt_tokens.len() / 8 * 8;
    for position in (0..tiled_tokens).step_by(8) {
        let embedding =
            match load_embedding_tile8(&manifest, &prompt_tokens[position..position + 8]) {
                Ok(embedding) => embedding,
                Err(error) => {
                    eprintln!("Embedding tile at {position}: {error}");
                    return;
                }
            };
        if let Err(error) = decoder.set_prefill8_input(context, &embedding) {
            eprintln!("Set prompt tile at {position}: {error}");
            return;
        }
        if let Err(error) = decoder.forward_prefill8(context, position, false) {
            eprintln!("Prompt tile forward at {position}: {error}");
            return;
        }
    }
    for (offset, &token) in prompt_tokens[tiled_tokens..].iter().enumerate() {
        let position = tiled_tokens + offset;
        if position > 0 || tiled_tokens > 0 {
            let embedding = match load_embedding_row(&manifest, token) {
                Ok(embedding) => embedding,
                Err(error) => {
                    eprintln!("Embedding row {token}: {error}");
                    return;
                }
            };
            if let Err(error) = decoder.set_token_input(context, &embedding) {
                eprintln!("Set prompt token {position}: {error}");
                return;
            }
        }
        let bucket = (position + 1).next_power_of_two().min(max_seq_len);
        if let Err(error) = decoder.forward(
            context,
            HybridUnitMode::ModelOptimized,
            bucket,
            position as u32,
            false,
        ) {
            eprintln!("Prompt forward at token {position}: {error}");
            return;
        }
    }
    if tiled_tokens == prompt_tokens.len() {
        if let Err(error) = decoder.commit_prefill8_last(context) {
            eprintln!("Commit final prompt tile: {error}");
            return;
        }
    }
    if let Err(error) = context.synchronize() {
        eprintln!("Prompt synchronization failed: {error}");
        return;
    }
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();

    let mut all_tokens = prompt_tokens.clone();
    let mut generated = Vec::with_capacity(max_tokens);
    let mut step_times = Vec::with_capacity(max_tokens);
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    print!("\n");
    for step in 0..max_tokens {
        let step_start = Instant::now();
        if step > 0 {
            let previous = generated[step - 1];
            let embedding = match load_embedding_row(&manifest, previous) {
                Ok(embedding) => embedding,
                Err(error) => {
                    eprintln!("Embedding row {previous}: {error}");
                    return;
                }
            };
            if let Err(error) = decoder.set_token_input(context, &embedding) {
                eprintln!("Set decode token {step}: {error}");
                return;
            }
            let position = prompt_tokens.len() + step - 1;
            let bucket = (position + 1).next_power_of_two().min(max_seq_len);
            if let Err(error) = decoder.forward(
                context,
                HybridUnitMode::ModelOptimized,
                bucket,
                position as u32,
                false,
            ) {
                eprintln!("Decode forward at step {step}: {error}");
                return;
            }
        }
        if let Err(error) = lm_head.forward(context, decoder.normalized_output()) {
            eprintln!("LM head at step {step}: {error}");
            return;
        }
        let token = match NextTokenLogits::last(lm_head.logits(), 248_320)
            .and_then(|logits| sampler.sample(logits))
        {
            Ok(sample) => sample.token_id,
            Err(error) => {
                eprintln!("Sampling at step {step}: {error}");
                return;
            }
        };
        generated.push(token);
        all_tokens.push(token);
        step_times.push(step_start.elapsed().as_secs_f64());
        if let Ok(text) = tokenizer.decode(&all_tokens) {
            let previous_text = tokenizer
                .decode(&all_tokens[..all_tokens.len() - 1])
                .unwrap_or_default();
            print!("{}", text.strip_prefix(&previous_text).unwrap_or(&text));
            out.flush().ok();
        }
        if generation.eos_token_ids.contains(&token) {
            break;
        }
    }
    println!("\n");
    let decode_seconds = step_times.iter().sum::<f64>();
    println!(
        "ApxInf Qwen3.8: prefill={} tokens in {:.3}s ({:.1} tok/s, M8 tiles + M1 tail); decode={} tokens in {:.3}s ({:.2} tok/s, {:.2} ms/token)",
        prompt_tokens.len(),
        prefill_seconds,
        prompt_tokens.len() as f64 / prefill_seconds,
        generated.len(),
        decode_seconds,
        generated.len() as f64 / decode_seconds,
        decode_seconds * 1000.0 / generated.len().max(1) as f64,
    );
}

#[cfg(feature = "cuda")]
fn load_embedding_tile8(
    manifest: &safetensors::CheckpointManifest,
    tokens: &[u32],
) -> Result<Tensor, String> {
    const HIDDEN: usize = 5120;
    if tokens.len() != 8 {
        return Err(format!(
            "Qwen3.8 prefill tile requires 8 tokens, got {}",
            tokens.len()
        ));
    }
    let mut values = Vec::with_capacity(tokens.len() * HIDDEN);
    for &token in tokens {
        let row = load_embedding_row(manifest, token).map_err(|error| error.to_string())?;
        values.extend_from_slice(row.as_bf16().map_err(|error| error.to_string())?);
    }
    Tensor::from_bf16(vec![tokens.len(), HIDDEN], &values).map_err(|error| error.to_string())
}

/// Read `model_type` from `config.json` if present. Empty string on any
/// error (falls back to the Llama path).
fn read_model_type(model_dir: &PathBuf) -> Option<String> {
    let cfg_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&cfg_path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    v.get("model_type")?.as_str().map(|s| s.to_string())
}

fn run_inspect(model_dir: &PathBuf, json: bool) -> Result<(), String> {
    let model_type = read_model_type(model_dir)
        .ok_or_else(|| format!("missing model_type in {}/config.json", model_dir.display()))?;
    if model_type != "qwen3_5" {
        return Err(format!(
            "inspection contract currently supports model_type `qwen3_5`, got `{model_type}`"
        ));
    }
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))
        .map_err(|error| error.to_string())?;
    let report =
        Qwen35CheckpointReport::inspect(model_dir, &config).map_err(|error| error.to_string())?;
    let dtype_counts = report
        .dtype_counts
        .iter()
        .map(|(dtype, count)| (dtype.to_string(), serde_json::json!(count)))
        .collect::<serde_json::Map<_, _>>();

    if json {
        let output = serde_json::json!({
            "status": "validated",
            "native_execution_ready": cfg!(feature = "cuda"),
            "native_capabilities": {
                "text_generate": cfg!(feature = "cuda"),
                "stateful_decode": cfg!(feature = "cuda"),
                "multimodal": cfg!(feature = "cuda"),
                "m_gt_1_prefill": cfg!(feature = "cuda"),
                "serial_prefill": cfg!(feature = "cuda"),
                "openai_compatible_service": cfg!(feature = "cuda"),
                "unified_llm_input": false,
            },
            "model_type": config.model_type,
            "architecture": config.architecture,
            "text": {
                "hidden_size": config.text.hidden_size,
                "intermediate_size": config.text.intermediate_size,
                "layers": config.text.n_layers,
                "linear_attention_layers": report.linear_attention_layers,
                "full_attention_layers": report.full_attention_layers,
                "attention_heads": config.text.n_heads,
                "kv_heads": config.text.n_kv_heads,
                "head_dim": config.text.head_dim,
                "max_position_embeddings": config.text.max_position_embeddings,
            },
            "vision": {
                "depth": config.vision.depth,
                "hidden_size": config.vision.hidden_size,
                "output_hidden_size": config.vision.out_hidden_size,
            },
            "quantization": {
                "method": config.quantization.method,
                "format": config.quantization.format,
                "bits": config.quantization.num_bits,
                "group_size": config.quantization.group_size,
                "symmetric": config.quantization.symmetric,
                "quantized_linears": report.quantized_linears,
                "ignored_modules": report.ignored_modules,
            },
            "checkpoint": {
                "shards": report.shard_count,
                "tensors": report.tensor_count,
                "tensor_bytes": report.tensor_bytes,
                "dtype_counts": dtype_counts,
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).map_err(|error| error.to_string())?
        );
    } else {
        println!("ApxInf Qwen3.5/Qwen3.8 checkpoint contract: VALID");
        println!("model: {} ({})", config.model_type, config.architecture);
        println!(
            "text: hidden={}, intermediate={}, layers={} ({} GDN + {} full attention), heads={}/{}, head_dim={}, max_context={}",
            config.text.hidden_size,
            config.text.intermediate_size,
            config.text.n_layers,
            report.linear_attention_layers,
            report.full_attention_layers,
            config.text.n_heads,
            config.text.n_kv_heads,
            config.text.head_dim,
            config.text.max_position_embeddings,
        );
        println!(
            "vision: depth={}, hidden={}, output_hidden={}",
            config.vision.depth, config.vision.hidden_size, config.vision.out_hidden_size
        );
        println!(
            "quantization: {} {}, W{} group={} asymmetric, {} packed linears, {} ignored modules",
            config.quantization.method,
            config.quantization.format,
            config.quantization.num_bits,
            config.quantization.group_size,
            report.quantized_linears,
            report.ignored_modules,
        );
        println!(
            "checkpoint: {} shards, {} tensors, {} bytes, dtypes={dtype_counts:?}",
            report.shard_count, report.tensor_count, report.tensor_bytes
        );
        println!(
            "native_text_execution_ready: {} (M8 prefill and service ready; unified LlmInput image path remains unsupported)",
            cfg!(feature = "cuda")
        );
    }
    Ok(())
}

/// Preprocess an image with the model's Hugging Face processor. Raw image
/// decoding and chat templating stay outside the model runtime; the resulting
/// borrowed tensor is attached to LlmInput for unified generation.
fn preprocess_image(
    model_dir: &PathBuf,
    image_path: &PathBuf,
    prompt: &str,
    system_prompt: Option<&str>,
) -> Result<(Vec<half::bf16>, Vec<usize>, [u32; 3], Vec<u32>), String> {
    use std::process::Command;

    let suffix = std::process::id();
    let pixel_path = std::env::temp_dir().join(format!("apxinf-cli-{suffix}-pixels.npy"));
    let metadata_path = std::env::temp_dir().join(format!("apxinf-cli-{suffix}-metadata.json"));
    let script = r#"
import json
import sys
import numpy as np
from transformers import AutoProcessor
from PIL import Image

model_dir, image_path, prompt, system, pixel_path, metadata_path = sys.argv[1:]
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
image = Image.open(image_path).convert("RGB")
messages = []
if system:
    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system}],
    })
messages.append({
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ],
})
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
pixels = inputs["pixel_values"].cpu().numpy().astype(np.float32)
grid = inputs["image_grid_thw"][0].cpu().numpy().tolist()
tokens = inputs["input_ids"][0].cpu().numpy().astype(np.int64).tolist()
np.save(pixel_path, pixels)
with open(metadata_path, "w") as output:
    json.dump({"grid": grid, "tokens": tokens}, output)
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(model_dir)
        .arg(image_path)
        .arg(prompt)
        .arg(system_prompt.unwrap_or(""))
        .arg(&pixel_path)
        .arg(&metadata_path)
        .output()
        .map_err(|error| format!("python3: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "python preprocessing failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let metadata_raw = std::fs::read_to_string(&metadata_path)
        .map_err(|error| format!("read {}: {error}", metadata_path.display()))?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_raw)
        .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?;
    let grid_values = metadata
        .get("grid")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "processor metadata has no grid array".to_string())?;
    if grid_values.len() != 3 {
        return Err(format!(
            "processor grid must have three values, got {}",
            grid_values.len()
        ));
    }
    let grid = [
        grid_values[0]
            .as_u64()
            .ok_or_else(|| "processor grid T is not an integer".to_string())? as u32,
        grid_values[1]
            .as_u64()
            .ok_or_else(|| "processor grid H is not an integer".to_string())? as u32,
        grid_values[2]
            .as_u64()
            .ok_or_else(|| "processor grid W is not an integer".to_string())? as u32,
    ];
    let tokens = metadata
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "processor metadata has no tokens array".to_string())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|token| token as u32)
                .ok_or_else(|| "processor returned a non-integer token".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (pixel_shape, pixel_data) = read_npy_f32_to_bf16(&pixel_path)?;

    let _ = std::fs::remove_file(&pixel_path);
    let _ = std::fs::remove_file(&metadata_path);
    Ok((pixel_data, pixel_shape, grid, tokens))
}

/// Read a NumPy v1 f32 array and convert it to bf16.
fn read_npy_f32_to_bf16(path: &std::path::Path) -> Result<(Vec<usize>, Vec<half::bf16>), String> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if buffer.len() < 10 || &buffer[..6] != b"\x93NUMPY" {
        return Err(format!("{} is not a NumPy array", path.display()));
    }
    if buffer[6] != 1 {
        return Err(format!(
            "{} uses unsupported NumPy format version {}",
            path.display(),
            buffer[6]
        ));
    }
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let data_start = 10usize
        .checked_add(header_len)
        .ok_or_else(|| "NumPy header length overflow".to_string())?;
    if data_start > buffer.len() {
        return Err("NumPy header exceeds file length".to_string());
    }
    let header = std::str::from_utf8(&buffer[10..data_start])
        .map_err(|error| format!("invalid NumPy header: {error}"))?;
    if !header.contains("<f4") {
        return Err("processor pixel array is not little-endian f32".to_string());
    }
    let shape = parse_npy_shape(header)?;
    let raw = &buffer[data_start..];
    let expected_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
    if raw.len() != expected_bytes {
        return Err(format!(
            "NumPy payload has {} bytes, expected {expected_bytes}",
            raw.len()
        ));
    }
    let data = raw
        .chunks_exact(4)
        .map(|bytes| half::bf16::from_f32(f32::from_le_bytes(bytes.try_into().unwrap())))
        .collect();
    Ok((shape, data))
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>, String> {
    let shape_offset = header
        .find("shape")
        .ok_or_else(|| "NumPy header has no shape".to_string())?;
    let open_offset = header[shape_offset..]
        .find('(')
        .ok_or_else(|| "NumPy shape has no opening parenthesis".to_string())?;
    let shape_start = shape_offset + open_offset + 1;
    let close_offset = header[shape_start..]
        .find(')')
        .ok_or_else(|| "NumPy shape has no closing parenthesis".to_string())?;
    let shape_text = &header[shape_start..shape_start + close_offset];
    let shape = shape_text
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid NumPy shape: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() {
        return Err("NumPy array has an empty shape".to_string());
    }
    Ok(shape)
}
fn run_test() {
    println!("apxinf — LLM inference engine (test mode)");
    println!();

    // ── CPU matmul smoke test ───────────────────────────────────────
    use apxinf_core::Tensor;

    let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c_cpu = a.matmul_cpu(&b).unwrap();
    println!("[CPU] A: {a}");
    println!("[CPU] B: {b}");
    println!("[CPU] C = A @ B: {c_cpu}");
    println!("[CPU] C data: {:?}", c_cpu.as_f32().unwrap());
    println!();

    #[cfg(feature = "cuda")]
    cuda_test();
}

#[cfg(feature = "cuda")]
fn cuda_test() {
    use apxinf_core::Tensor;
    use apxinf_cuda::{
        kernels::{activation, attention, elementwise, gemm, norm, rope},
        transfers, CudaContext,
    };

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("[CUDA] Not available: {e}");
            return;
        }
    };
    println!("[CUDA] Device: {}", ctx.device_id());

    // Matmul test
    let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    let a_gpu = transfers::to_cuda(&a, 0).unwrap();
    let b_gpu = transfers::to_cuda(&b, 0).unwrap();

    let c_gpu = gemm::matmul(&ctx, &a_gpu, &b_gpu).unwrap();
    let c_cpu = transfers::to_cpu(&c_gpu).unwrap();
    let data = c_cpu.as_f32().unwrap();
    println!("[CUDA] matmul: {:?}", data);

    // SiLU test
    let x = Tensor::from_f32(vec![4], &[1.0, -1.0, 0.0, 2.0]).unwrap();
    let x_gpu = transfers::to_cuda(&x, 0).unwrap();
    let silu_gpu = activation::silu(&ctx, &x_gpu).unwrap();
    let silu_cpu = transfers::to_cpu(&silu_gpu).unwrap();
    let silu_data = silu_cpu.as_f32().unwrap();
    let _silu_expected: Vec<f32> = [1.0f32, -1.0, 0.0, 2.0]
        .iter()
        .map(|x| x / (1.0 + (-x).exp()))
        .collect();
    println!("[CUDA] silu: {:?}", silu_data);

    // Add test
    let a2 = Tensor::from_f32(vec![4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b2 = Tensor::from_f32(vec![4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let a2_gpu = transfers::to_cuda(&a2, 0).unwrap();
    let b2_gpu = transfers::to_cuda(&b2, 0).unwrap();
    let add_gpu = elementwise::add(&ctx, &a2_gpu, &b2_gpu).unwrap();
    let add_cpu = transfers::to_cpu(&add_gpu).unwrap();
    println!("[CUDA] add: {:?}", add_cpu.as_f32().unwrap());

    // Mul test
    let mul_gpu = elementwise::mul(&ctx, &a2_gpu, &b2_gpu).unwrap();
    let mul_cpu = transfers::to_cpu(&mul_gpu).unwrap();
    println!("[CUDA] mul: {:?}", mul_cpu.as_f32().unwrap());

    // RMSNorm test
    let input = Tensor::from_f32(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let weight = Tensor::from_f32(vec![4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
    let input_gpu = transfers::to_cuda(&input, 0).unwrap();
    let weight_gpu = transfers::to_cuda(&weight, 0).unwrap();
    let norm_gpu = norm::rms(&ctx, &input_gpu, &weight_gpu, 1e-5).unwrap();
    let norm_cpu = transfers::to_cpu(&norm_gpu).unwrap();
    println!("[CUDA] rms_norm: {:?}", norm_cpu.as_f32().unwrap());

    // Softmax test
    let sm_input = Tensor::from_f32(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let sm_gpu = transfers::to_cuda(&sm_input, 0).unwrap();
    let softmax_gpu = attention::softmax(&ctx, &sm_gpu).unwrap();
    let softmax_cpu = transfers::to_cpu(&softmax_gpu).unwrap();
    println!("[CUDA] softmax: {:?}", softmax_cpu.as_f32().unwrap());

    // RoPE test
    let rope_input =
        Tensor::from_f32(vec![2, 4], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]).unwrap();
    let rope_gpu = transfers::to_cuda(&rope_input, 0).unwrap();
    let rope_out = rope::apply(&ctx, &rope_gpu, 2, 4, 10000.0, 0).unwrap();
    let rope_cpu = transfers::to_cpu(&rope_out).unwrap();
    println!("[CUDA] rope: {:?}", rope_cpu.as_f32().unwrap());

    // Causal mask test
    let mask_input = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask_gpu = transfers::to_cuda(&mask_input, 0).unwrap();
    let mask_out = attention::causal_mask(&ctx, &mask_gpu, 0).unwrap();
    let mask_cpu = transfers::to_cpu(&mask_out).unwrap();
    println!("[CUDA] causal_mask: {:?}", mask_cpu.as_f32().unwrap());

    println!("[CUDA] All kernel tests completed.");
}

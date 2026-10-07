//! ApxInf LLM inference engine CLI.

use std::io::Write;
use std::path::PathBuf;

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{
    AutoModel, GenerationConfigSource, GenerationOptions, ImageInput, LlmInput, LoadOptions,
    SamplingMode,
};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use clap::{Parser, Subcommand};

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

        /// Explicit registry family (e.g. minicpm5 when checkpoint model_type is llama)
        #[arg(long)]
        model_name: Option<String>,

        /// Family-local implementation variant; unsupported choices are rejected by its loader
        #[arg(long)]
        model_variant: Option<String>,

        /// Auxiliary checkpoint asset as NAME=PATH (for example draft=/path/to/DSpark)
        #[arg(long, value_parser = parse_asset)]
        asset: Vec<(String, PathBuf)>,

        /// JSON chat-template options, rendered with exact checkpoint whitespace
        #[arg(long)]
        chat_options: Option<String>,

        /// Preserve semantic control tokens such as tool-call XML tags.
        /// Also enabled automatically when chat options supply tools.
        #[arg(long)]
        keep_special_tokens: bool,

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

        /// Device to run inference on (cpu, cuda[:N], or metal[:N]; mlx is an alias)
        #[arg(short, long, default_value = "cpu")]
        device: String,

        /// Weight dtype ("fp32" or "bf16"). On CUDA, "bf16" halves weight-
        /// bandwidth and enables the bf16 fast path. Ignored on CPU.
        #[arg(long, default_value = "fp32")]
        dtype: String,
    },

    /// Run a quick test of the engine
    Test,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            model,
            model_name,
            model_variant,
            asset,
            chat_options,
            keep_special_tokens,
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
            let device = match parse_device(&device) {
                Ok(device) => device,
                Err(error) => {
                    eprintln!("{error}");
                    std::process::exit(1);
                }
            };
            // Report a failed generation through the exit status; a CLI that
            // printed an error and still exited 0 reads as success to any caller.
            if let Err(error) = run_generate(
                &model,
                model_name.as_deref(),
                model_variant.as_deref(),
                &asset,
                chat_options.as_deref(),
                keep_special_tokens,
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
        Commands::Test => {
            run_test();
        }
    }
}

fn parse_asset(value: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = value
        .split_once('=')
        .ok_or("expected auxiliary asset NAME=PATH")?;
    if name.is_empty() || path.is_empty() || name.chars().any(char::is_whitespace) {
        return Err("expected nonempty asset name and path".into());
    }
    Ok((name.to_owned(), PathBuf::from(path)))
}

fn parse_device(spec: &str) -> Result<Device, String> {
    let normalized = spec.to_ascii_lowercase();
    let (kind, index) = match normalized.split_once(':') {
        Some((kind, index)) => {
            let index = index.parse::<usize>().map_err(|_| {
                format!("Invalid device index in '{spec}'; use cpu, cuda:N, or metal:N")
            })?;
            (kind, index)
        }
        None => (normalized.as_str(), 0),
    };
    match kind {
        "cpu" if index == 0 => Ok(Device::Cpu),
        "cuda" | "gpu" => Ok(Device::Cuda(index)),
        "metal" | "mlx" => Ok(Device::Metal(index)),
        _ => Err(format!(
            "Unknown device '{spec}'; use cpu, cuda:N, or metal:N (mlx:N is an alias)"
        )),
    }
}

#[cfg(test)]
mod device_tests {
    use super::{parse_device, Cli, Commands};
    use apxinf_core::Device;
    use clap::Parser;

    #[test]
    fn parses_explicit_accelerator_devices() {
        assert_eq!(parse_device("cpu").unwrap(), Device::Cpu);
        assert_eq!(parse_device("gpu").unwrap(), Device::Cuda(0));
        assert_eq!(parse_device("cuda:2").unwrap(), Device::Cuda(2));
        assert_eq!(parse_device("metal").unwrap(), Device::Metal(0));
        assert_eq!(parse_device("MLX:1").unwrap(), Device::Metal(1));
    }

    #[test]
    fn malformed_devices_do_not_fall_back_to_cpu() {
        for spec in [
            "tpu",
            "",
            "metal:",
            "mlx:-1",
            "cuda:x",
            "cpu:1",
            "metal:0:1",
        ] {
            assert!(parse_device(spec).is_err(), "accepted {spec}");
        }
    }

    #[test]
    fn accepts_explicit_family_and_variant() {
        let cli = Cli::try_parse_from([
            "apxinf",
            "generate",
            "--model",
            "/checkpoint",
            "--prompt",
            "hello",
            "--device",
            "metal",
            "--model-name",
            "minicpm5",
            "--model-variant",
            "bf16-compiled",
        ])
        .unwrap();
        let Commands::Generate {
            model_name,
            model_variant,
            ..
        } = cli.command
        else {
            panic!("expected generate command");
        };
        assert_eq!(model_name.as_deref(), Some("minicpm5"));
        assert_eq!(model_variant.as_deref(), Some("bf16-compiled"));
    }
}

fn run_generate(
    model_dir: &PathBuf,
    model_name_override: Option<&str>,
    model_variant: Option<&str>,
    assets: &[(String, PathBuf)],
    chat_options: Option<&str>,
    keep_special_tokens: bool,
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

    let mut asset_map = std::collections::BTreeMap::new();
    for (name, path) in assets {
        if asset_map.insert(name.clone(), path.clone()).is_some() {
            return Err(format!("duplicate auxiliary asset: {name}"));
        }
    }
    if image_path.is_some() && chat_options.is_some() {
        return Err("--chat-options currently supports text prompts only".into());
    }
    let template_options = chat_options
        .map(|raw| {
            serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(raw)
                .map_err(|e| format!("invalid --chat-options object: {e}"))
        })
        .transpose()?;
    let skip_special_tokens = !(keep_special_tokens
        || template_options
            .as_ref()
            .is_some_and(|options| options.contains_key("tools")));
    let model_name = match model_name_override {
        Some(name) => name.to_owned(),
        None => AutoModel::detect_model_name(model_dir)
            .map_err(|error| format!("Failed to detect model type: {error}"))?,
    };
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
        let tokens = encode_prompt(&tok, prompt, system_prompt, template_options.as_ref())
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
        model_variant: model_variant.map(str::to_owned),
        assets: asset_map,
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
            if let Ok(text) = tok.decode_with_options(&all_tokens, skip_special_tokens) {
                let previous = tok
                    .decode_with_options(&all_tokens[..all_tokens.len() - 1], skip_special_tokens)
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
    template_options: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<Vec<u32>, String> {
    if tokenizer.has_chat_template() {
        let mut messages = Vec::new();
        if let Some(system) = system_prompt {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(prompt));
        match template_options {
            Some(options) => tokenizer
                .apply_chat_template_with_options(&messages, options)
                .and_then(|text| tokenizer.encode(&text))
                .map_err(|error| error.to_string()),
            None => tokenizer
                .encode_chat(&messages)
                .map_err(|error| error.to_string()),
        }
    } else {
        tokenizer.encode(prompt).map_err(|error| error.to_string())
    }
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

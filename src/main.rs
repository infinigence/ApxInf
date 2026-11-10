//! ApxInf LLM inference engine CLI.

use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{AutoModel, ImageInput, LlmInput, LoadOptions};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use clap::{Parser, Subcommand};

#[path = "server/mod.rs"]
mod server;

#[derive(Debug, Clone)]
struct ServeArgs {
    model: PathBuf,
    revision: String,
    gpu_uuid: String,
    bind: std::net::SocketAddr,
    max_model_len: usize,
    queue_capacity: usize,
}

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
        #[arg(long, default_value = "50")]
        max_tokens: usize,

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

    /// Run a quick test of the engine
    Test,

    /// Start the strict Qwen3.5 evaluator server.
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        revision: String,
        #[arg(long)]
        gpu_uuid: String,
        #[arg(long, default_value = "127.0.0.1:8000")]
        bind: std::net::SocketAddr,
        #[arg(long)]
        max_model_len: usize,
        #[arg(long, default_value_t = 1)]
        queue_capacity: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            model,
            prompt,
            image,
            max_tokens,
            no_eos_stop,
            system,
            device,
            dtype,
        } => {
            let device = parse_device(&device);
            // Report a failed generation through the exit status; a CLI that
            // printed an error and still exited 0 reads as success to any caller.
            if let Err(error) = run_generate(
                &model,
                &prompt,
                image.as_ref(),
                max_tokens,
                !no_eos_stop,
                system.as_deref(),
                device,
                &dtype,
            ) {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        Commands::Test => {
            run_test();
        }
        Commands::Serve {
            model,
            revision,
            gpu_uuid,
            bind,
            max_model_len,
            queue_capacity,
        } => {
            let args = ServeArgs {
                model,
                revision,
                gpu_uuid,
                bind,
                max_model_len,
                queue_capacity,
            };
            if let Err(error) = run_serve(args) {
                eprintln!("serve failed: {error}");
                std::process::exit(2);
            }
        }
    }
}

fn validate_serve_args(args: &ServeArgs) -> Result<(), String> {
    if !args.model.is_dir() {
        return Err(format!(
            "--model is not a directory: {}",
            args.model.display()
        ));
    }
    if args.revision != server::service::MODEL_REVISION {
        return Err(format!(
            "--revision must be {}, got {}",
            server::service::MODEL_REVISION,
            args.revision
        ));
    }
    if args.gpu_uuid.trim().is_empty() {
        return Err("--gpu-uuid must be non-empty".into());
    }
    if args.max_model_len == 0 {
        return Err("--max-model-len must be positive".into());
    }
    if args.max_model_len < 3 {
        return Err("--max-model-len must be at least 3 for startup warmup".into());
    }
    if args.queue_capacity == 0 {
        return Err("--queue-capacity must be positive".into());
    }
    Ok(())
}

fn run_serve(args: ServeArgs) -> Result<(), String> {
    validate_serve_args(&args)?;
    let inventory = apxinf_model::Qwen35CheckpointInventory::from_approved_checkpoint_dir(
        &args.model,
        args.revision.clone(),
    )
    .map_err(|error| format!("checkpoint validation failed: {error}"))?;
    if inventory.config.max_position_embeddings < args.max_model_len {
        return Err(format!(
            "--max-model-len {} exceeds checkpoint max position {}",
            args.max_model_len, inventory.config.max_position_embeddings
        ));
    }
    let observed_uuid = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
        .output()
        .map_err(|error| format!("nvidia-smi unavailable: {error}"))?;
    if !observed_uuid.status.success() {
        return Err(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&observed_uuid.stderr)
        ));
    }
    let mut physical_index = None;
    for line in String::from_utf8_lossy(&observed_uuid.stdout).lines() {
        let Some((index, uuid)) = line.split_once(',') else {
            continue;
        };
        if uuid.trim() == args.gpu_uuid {
            physical_index = Some(index.trim().parse::<usize>().map_err(|error| {
                format!("nvidia-smi returned invalid GPU index `{index}`: {error}")
            })?);
            break;
        }
    }
    let physical_index = physical_index
        .ok_or_else(|| format!("requested GPU UUID {} is not visible", args.gpu_uuid))?;
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").ok();
    let explicit_device = std::env::var("APXINF_CUDA_DEVICE").ok();
    if explicit_device.is_some() {
        return Err(
            "APXINF_CUDA_DEVICE is not accepted by strict serve; bind only through the requested GPU UUID"
                .into(),
        );
    }
    let device_id = match explicit_device.as_deref() {
        Some(value) => value
            .parse::<usize>()
            .map_err(|error| format!("APXINF_CUDA_DEVICE must be a CUDA ordinal: {error}"))?,
        None if visible.is_some() => 0,
        None => physical_index,
    };
    if let Some(visible) = visible.as_deref() {
        let entries = visible
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .collect::<Vec<_>>();
        if entries.len() != 1 || entries[0] != args.gpu_uuid {
            return Err(format!(
                "CUDA_VISIBLE_DEVICES must be exactly the requested GPU UUID {}, got {visible}",
                args.gpu_uuid,
            ));
        }
    }
    validate_cuda_device_selection(
        &args.gpu_uuid,
        physical_index,
        device_id,
        visible.as_deref(),
        explicit_device.is_some(),
    )?;
    #[cfg(not(any(feature = "cuda", feature = "cuda-no-nvtx")))]
    {
        let _ = inventory;
        return Err(
            "serve requires the `cuda` or `cuda-no-nvtx` feature; refusing CPU fallback".into(),
        );
    }
    #[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
    {
        use apxinf_model::qwen35::{request_resident_bytes, request_state_bytes, Qwen35CudaModel};
        use apxinf_model::RuntimeCapabilities;
        use server::qwen35_batch::{max_concurrency, Qwen35BatchProtocolRuntime};
        use server::qwen35_runtime::{Qwen35CudaStepExecutor, Qwen35ProtocolRuntime};
        use server::service::ProtocolService;

        let model = Qwen35CudaModel::from_inventory_attested(
            &inventory,
            device_id,
            &args.gpu_uuid,
            args.max_model_len,
        )
        .map_err(|error| format!("CUDA Qwen3.5 initialization failed: {error}"))?;
        let memory = model
            .memory_info()
            .map_err(|error| format!("CUDA memory query failed: {error}"))?;
        let per_request_bytes = request_state_bytes(&inventory.config, args.max_model_len)?;
        let concurrency = max_concurrency();

        macro_rules! finish_serve {
            ($runtime:expr) => {{
                let mut service = ProtocolService::new_unready($runtime, false);
                if apxinf_model::qwen35::multimodal_enabled() {
                    let chat = server::chat::ChatPreprocessor::from_model_dir(
                        &args.model,
                        args.max_model_len,
                    )
                    .map_err(|error| format!("multimodal chat initialization failed: {error}"))?;
                    service = service.with_chat(chat);
                }
                let service = Arc::new(service);
                service
                    .warmup()
                    .map_err(|error| format!("CUDA warmup failed: {error}"))?;
                let listener = TcpListener::bind(args.bind)
                    .map_err(|error| format!("bind {} failed: {error}", args.bind))?;
                server::http::serve(listener, service)
                    .map_err(|error| format!("HTTP server failed: {error}"))
            }};
        }

        if concurrency > 1 {
            // Concurrent batched runtime (multi-request bonus lane, gated by
            // APXINF_Q35_MAX_CONCURRENCY >= 2). Admission reserves each
            // request's actual KV+GDN bytes; the shared one-at-a-time prefill
            // workspace and one prefix-cache template slot are subtracted
            // from the budget up front so an over-committed mix still fails
            // closed at admission rather than at cudaMalloc.
            let workspace_reserve =
                request_state_bytes(&inventory.config, args.max_model_len)?.saturating_sub(
                    request_resident_bytes(&inventory.config, args.max_model_len)?,
                );
            let template_reserve = request_resident_bytes(&inventory.config, 4096)?;
            let device_budget_bytes = memory
                .free_bytes
                .saturating_sub(workspace_reserve)
                .saturating_sub(template_reserve);
            let single_request = request_resident_bytes(&inventory.config, args.max_model_len)?;
            if single_request > device_budget_bytes {
                // A request that budgets the full context cannot be admitted
                // under the concurrent accounting; it will receive the
                // contract's structured 503 capacity rejection at admission.
                // The evaluation workloads top out well below this.
                eprintln!(
                    "warning: full-length ({}-token) requests exceed the concurrent budget \
                     ({single_request} > {device_budget_bytes} bytes) and will be rejected \
                     with 503 capacity",
                    args.max_model_len
                );
            }
            let capabilities = RuntimeCapabilities {
                vocab_size: inventory.config.vocab_size,
                max_model_len: args.max_model_len,
                parallel_requests: concurrency,
                device_budget_bytes,
                per_request_bytes,
            };
            let runtime = Qwen35BatchProtocolRuntime::new(capabilities, args.queue_capacity, model)
                .map_err(|error| format!("batched runtime initialization failed: {error}"))?;
            finish_serve!(runtime)
        } else {
            if per_request_bytes > memory.free_bytes {
                return Err(format!(
                    "request state requires {per_request_bytes} bytes but CUDA{} has {} bytes free",
                    model.device_id(),
                    memory.free_bytes
                ));
            }
            let capabilities = RuntimeCapabilities {
                vocab_size: inventory.config.vocab_size,
                max_model_len: args.max_model_len,
                parallel_requests: 1,
                device_budget_bytes: memory.free_bytes,
                per_request_bytes,
            };
            let executor = Arc::new(Qwen35CudaStepExecutor::new(model));
            let runtime = Qwen35ProtocolRuntime::new(capabilities, args.queue_capacity, executor)
                .map_err(|error| format!("runtime initialization failed: {error}"))?;
            finish_serve!(runtime)
        }
    }
}

fn validate_cuda_device_selection(
    requested_uuid: &str,
    physical_index: usize,
    device_id: usize,
    visible_devices: Option<&str>,
    explicit_device: bool,
) -> Result<(), String> {
    if let Some(visible) = visible_devices {
        let entries = visible
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .collect::<Vec<_>>();
        if entries.len() != 1 {
            return Err(format!(
                "CUDA_VISIBLE_DEVICES must contain exactly one GPU UUID for strict binding, got {visible}"
            ));
        }
        let first = entries[0];
        if first != requested_uuid {
            return Err(format!(
                "CUDA_VISIBLE_DEVICES first entry `{first}` does not match requested GPU UUID {requested_uuid}"
            ));
        }
        if device_id != 0 {
            return Err(format!(
                "APXINF_CUDA_DEVICE {device_id} does not select the requested first visible GPU UUID {requested_uuid}"
            ));
        }
    } else if explicit_device && device_id != physical_index {
        return Err(format!(
            "APXINF_CUDA_DEVICE {device_id} does not select requested GPU UUID {requested_uuid} (physical index {physical_index})"
        ));
    }
    Ok(())
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
    max_tokens: usize,
    eos_stop: bool,
    system_prompt: Option<&str>,
    device: Device,
    dtype: &str,
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

    let eos_token_id = if eos_stop { tok.eos_token_id() } else { None };
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
    let options = LoadOptions {
        model_name: Some(model_name.clone()),
        text_weight_dtype,
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

    println!();
    println!("Generating {max_tokens} tokens...");
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut all_tokens = tokens.clone();

    let (_, profile) = model
        .generate_streaming(
            input,
            max_tokens,
            |token_id| {
                all_tokens.push(token_id);
                if let Ok(text) = tok.decode(&all_tokens) {
                    let previous = tok
                        .decode(&all_tokens[..all_tokens.len() - 1])
                        .unwrap_or_default();
                    let delta = text.strip_prefix(&previous).unwrap_or(&text);
                    print!("{delta}");
                    out.flush().ok();
                }
            },
            eos_token_id,
        )
        .map_err(|error| format!("Generation failed: {error}"))?;

    println!();
    println!();
    println!("{}", profile.summary());
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

#[cfg(test)]
mod serve_tests {
    use super::{validate_cuda_device_selection, validate_serve_args, ServeArgs};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::path::PathBuf;

    fn args(model: PathBuf) -> ServeArgs {
        ServeArgs {
            model,
            revision: "63768c10df38c0395e12ef49edac1bd539eaeeea".into(),
            gpu_uuid: "GPU-d074a13d-dbb6-fceb-4caf-a45be9be9281".into(),
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8000),
            max_model_len: 32,
            queue_capacity: 1,
        }
    }

    #[test]
    fn strict_serve_rejects_missing_model_and_wrong_revision() {
        let missing = args(PathBuf::from("/definitely/missing/apxinf-qwen35"));
        assert!(validate_serve_args(&missing)
            .unwrap_err()
            .contains("--model"));
        let directory =
            std::env::temp_dir().join(format!("apxinf-serve-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut wrong = args(directory.clone());
        wrong.revision = "wrong".into();
        assert!(validate_serve_args(&wrong)
            .unwrap_err()
            .contains("--revision"));
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn strict_serve_rejects_empty_gpu_and_zero_limits() {
        let directory =
            std::env::temp_dir().join(format!("apxinf-serve-limits-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let mut value = args(directory.clone());
        value.gpu_uuid.clear();
        assert!(validate_serve_args(&value)
            .unwrap_err()
            .contains("--gpu-uuid"));
        value.gpu_uuid = "GPU-test".into();
        value.max_model_len = 0;
        assert!(validate_serve_args(&value)
            .unwrap_err()
            .contains("--max-model-len"));
        value.max_model_len = 32;
        value.queue_capacity = 0;
        assert!(validate_serve_args(&value)
            .unwrap_err()
            .contains("--queue-capacity"));
        std::fs::remove_dir(&directory).unwrap();
    }

    #[test]
    fn cuda_selection_rejects_ordinal_mismatch_when_uuid_is_explicit() {
        let error = validate_cuda_device_selection("GPU-requested", 1, 0, None, true).unwrap_err();
        assert!(error.contains("does not select requested GPU UUID"));
        assert!(validate_cuda_device_selection("GPU-requested", 1, 1, None, true).is_ok());
    }

    #[test]
    fn cuda_selection_requires_requested_uuid_as_first_visible_device() {
        assert!(validate_cuda_device_selection(
            "GPU-requested",
            1,
            0,
            Some("GPU-requested"),
            false,
        )
        .is_ok());
        let error = validate_cuda_device_selection(
            "GPU-requested",
            1,
            0,
            Some("GPU-requested,GPU-other"),
            true,
        )
        .unwrap_err();
        assert!(error.contains("exactly one"));
        let error = validate_cuda_device_selection(
            "GPU-requested",
            1,
            0,
            Some("GPU-other,GPU-requested"),
            false,
        )
        .unwrap_err();
        assert!(error.contains("exactly one GPU UUID"));
    }

    #[cfg(feature = "cuda-no-nvtx")]
    #[test]
    fn cuda_no_nvtx_compiles_cuda_test_path() {
        let _: fn() = super::cuda_test;
    }
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

    #[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
    cuda_test();
}

#[cfg(any(feature = "cuda", feature = "cuda-no-nvtx"))]
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

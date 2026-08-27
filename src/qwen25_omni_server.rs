//! Serialized OpenAI-compatible service for the pinned native Omni runtime.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use apxinf_core::{Device, Tensor};
use apxinf_model::llm_trait::validate_generation_limits;
use apxinf_model::{
    AudioInput, AutoModel, ImageInput, LlmInput, LoadOptions, LoadedModel, ModelPrecision,
    Qwen25OmniConfig,
};
use apxinf_tokenizer::Tokenizer;
use serde_json::{json, Map, Value};

const MODEL_ID: &str = "Qwen/Qwen2.5-Omni-3B";
const MODEL_REVISION: &str = "f75b40e3da2003cdd6e1829b1f420ca70797c34e";
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn persistent_processor_enabled() -> Result<bool, String> {
    static ENABLED: OnceLock<Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_OMNI_PERSISTENT_PROCESSOR") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_OMNI_PERSISTENT_PROCESSOR must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_OMNI_PERSISTENT_PROCESSOR must be UTF-8".into())
            }
        })
        .clone()
}

pub fn serve(model_dir: &Path, host: &str, port: u16, max_model_len: usize) -> Result<(), String> {
    let config = Qwen25OmniConfig::from_model_dir(model_dir).map_err(|error| error.to_string())?;
    if max_model_len < 2 || max_model_len > config.text.max_position_embeddings {
        return Err(format!(
            "qwen2.5-omni max_model_len must be in 2..={}, got {max_model_len}",
            config.text.max_position_embeddings
        ));
    }
    let options = LoadOptions {
        precision: ModelPrecision::Bf16,
        text_weight_dtype: Some(apxinf_core::DType::BF16),
        ..LoadOptions::default()
    };
    let model = AutoModel::load_model(Device::Cuda(0), model_dir, &options)
        .map_err(|error| format!("load native qwen2.5-omni: {error}"))?;
    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json"))
        .map_err(|error| format!("load tokenizer: {error}"))?;
    let processor = persistent_processor_enabled()?
        .then(|| ProcessorWorker::new(model_dir))
        .transpose()?;
    let mut runtime = Runtime {
        model,
        tokenizer,
        model_dir: model_dir.to_path_buf(),
        processor,
        config,
        max_model_len,
    };
    let listener =
        TcpListener::bind((host, port)).map_err(|error| format!("bind {host}:{port}: {error}"))?;
    eprintln!("ApxInf native Qwen2.5-Omni ready on http://{host}:{port}");
    for connection in listener.incoming() {
        match connection {
            Ok(mut stream) => {
                if let Err(error) = handle_connection(&mut runtime, &mut stream) {
                    let _ = send_json(
                        &mut stream,
                        503,
                        &json!({"error":{"message":error,"type":"runtime_error"}}),
                    );
                }
            }
            Err(error) => eprintln!("qwen2.5-omni accept error: {error}"),
        }
    }
    Ok(())
}

struct Runtime {
    model: LoadedModel,
    tokenizer: Tokenizer,
    model_dir: PathBuf,
    processor: Option<ProcessorWorker>,
    config: Qwen25OmniConfig,
    max_model_len: usize,
}

struct Prepared {
    tokens: Vec<u32>,
    image: Option<PreparedImage>,
    audio: Option<PreparedAudio>,
}

struct PreparedImage {
    pixels: Tensor,
    grids: Vec<[u32; 3]>,
}

struct PreparedAudio {
    features: Tensor,
    mask: Tensor,
    lengths: Vec<u32>,
    counts: Vec<u32>,
}

struct Generation {
    tokens: Vec<u32>,
    text: String,
    prompt_tokens: usize,
    ttft_seconds: f64,
    tpot_seconds: f64,
}

impl Runtime {
    fn processor_mode(&mut self) -> Result<&'static str, String> {
        let Some(processor) = self.processor.as_mut() else {
            return Ok("per_request");
        };
        match processor
            .child
            .try_wait()
            .map_err(|error| format!("inspect Omni processor worker: {error}"))?
        {
            None => Ok("persistent"),
            Some(status) => Err(format!("Omni processor worker exited: {status}")),
        }
    }

    fn preprocess_chat(&mut self, body: &Value) -> Result<Prepared, String> {
        if let Some(processor) = self.processor.as_mut() {
            processor.preprocess(body)
        } else {
            ProcessorWorker::new(&self.model_dir)?.preprocess(body)
        }
    }

    fn generate(
        &mut self,
        prepared: &Prepared,
        max_tokens: usize,
        ignore_eos: bool,
    ) -> Result<Generation, String> {
        if prepared.tokens.is_empty() {
            return Err("processor produced an empty prompt".into());
        }
        validate_generation_limits(
            prepared.tokens.len(),
            max_tokens,
            Some(128),
            Some(self.max_model_len),
        )
        .map_err(|error| error.to_string())?;
        let image = prepared
            .image
            .as_ref()
            .map(|image| ImageInput::new(&image.pixels, &image.grids));
        let audio = prepared.audio.as_ref().map(|audio| {
            AudioInput::new(&audio.features, &audio.mask, &audio.lengths, &audio.counts)
        });
        let input = LlmInput::with_media(&prepared.tokens, image, audio);
        self.model.reset().map_err(|error| error.to_string())?;
        let eos = (!ignore_eos).then_some(self.config.eos_token_id);
        let result = self
            .model
            .generate_streaming(input, max_tokens, |_| {}, eos);
        let _ = self.model.reset();
        let (tokens, profile) = result.map_err(|error| error.to_string())?;
        let text = self
            .tokenizer
            .decode(&tokens)
            .map_err(|error| format!("decode generated tokens: {error}"))?;
        Ok(Generation {
            tokens,
            text,
            prompt_tokens: prepared.tokens.len(),
            ttft_seconds: profile.ttft_ms().unwrap_or_default() / 1000.0,
            tpot_seconds: profile.tpot_ms().unwrap_or_default() / 1000.0,
        })
    }
}

fn handle_connection(runtime: &mut Runtime, stream: &mut TcpStream) -> Result<(), String> {
    let request = match read_request(stream) {
        Ok(request) => request,
        Err(error) if error.contains("exceeds") => {
            return send_json(
                stream,
                413,
                &json!({"error":{"message":error,"type":"payload_too_large"}}),
            )
        }
        Err(error) => return Err(error),
    };
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            let processor_mode = match runtime.processor_mode() {
                Ok(mode) => mode,
                Err(error) => {
                    return send_json(
                        stream,
                        503,
                        &json!({"status":"error","error":error,"fallback_active":false}),
                    )
                }
            };
            send_json(
                stream,
                200,
                &json!({
                    "status":"ok",
                    "model":MODEL_ID,
                    "model_revision":MODEL_REVISION,
                    "precision":"bf16",
                    "parallel_requests":1,
                    "max_model_len":runtime.max_model_len,
                    "max_prompt_tokens":runtime.max_model_len.saturating_sub(1),
                    "max_output_tokens":128,
                    "context_contract":"prompt_tokens + requested_completion_tokens <= max_model_len",
                    "input_modalities":["text","image","audio"],
                    "output_modalities":["text"],
                    "talker_disabled":true,
                    "speech_output":false,
                    "video":false,
                    "processor_mode":processor_mode,
                    "fallback_active":false
                }),
            )
        }
        ("GET", "/v1/models") => send_json(
            stream,
            200,
            &json!({"object":"list","data":[{"id":MODEL_ID,"object":"model","owned_by":"apxinf"}]}),
        ),
        ("POST", "/v1/chat/completions") => handle_chat(runtime, stream, &request.body),
        ("POST", "/v1/evaluations/generate") => handle_evaluation(runtime, stream, &request.body),
        _ => send_json(
            stream,
            404,
            &json!({"error":{"message":"not found","type":"not_found"}}),
        ),
    }
}

fn handle_chat(runtime: &mut Runtime, stream: &mut TcpStream, raw: &[u8]) -> Result<(), String> {
    let body: Value = match serde_json::from_slice(raw) {
        Ok(value) => value,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":format!("invalid JSON: {error}"),"type":"invalid_request"}}),
            )
        }
    };
    let request = match validate_generation_request(&body, true) {
        Ok(request) => request,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":error,"type":"invalid_request"}}),
            )
        }
    };
    if let Err(error) = validate_chat_content(&body) {
        return send_json(
            stream,
            400,
            &json!({"error":{"message":error,"type":"invalid_request"}}),
        );
    }
    let prepared = match runtime.preprocess_chat(&body) {
        Ok(prepared) => prepared,
        Err(error) => {
            return send_json(
                stream,
                422,
                &json!({"error":{"message":error,"type":"unprocessable_media"}}),
            )
        }
    };
    let generation = match runtime.generate(&prepared, request.max_tokens, false) {
        Ok(generation) => generation,
        Err(error) => return send_generation_error(stream, error),
    };
    let id = request_id("chatcmpl");
    if request.stream {
        return send_stream(stream, &id, &generation, runtime);
    }
    let created = unix_seconds();
    send_json(
        stream,
        200,
        &json!({
            "id":id,"object":"chat.completion","created":created,"model":MODEL_ID,
            "choices":[{"index":0,"message":{"role":"assistant","content":generation.text},"finish_reason":"stop"}],
            "usage":{
                "prompt_tokens":generation.prompt_tokens,
                "completion_tokens":generation.tokens.len(),
                "total_tokens":generation.prompt_tokens+generation.tokens.len()
            },
            "apxinf":{
                "tokens":generation.tokens,
                "ttft_seconds":generation.ttft_seconds,
                "tpot_seconds":generation.tpot_seconds,
                "fallback_active":false,
                "output_modalities":["text"]
            }
        }),
    )
}

fn handle_evaluation(
    runtime: &mut Runtime,
    stream: &mut TcpStream,
    raw: &[u8],
) -> Result<(), String> {
    let body: Value = match serde_json::from_slice(raw) {
        Ok(value) => value,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":format!("invalid JSON: {error}"),"type":"invalid_request"}}),
            )
        }
    };
    let request = match validate_generation_request(&body, false) {
        Ok(request) => request,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":error,"type":"invalid_request"}}),
            )
        }
    };
    let tokens = match parse_input_ids(&body) {
        Ok(tokens) => tokens,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":error,"type":"invalid_request"}}),
            )
        }
    };
    let prepared = Prepared {
        tokens,
        image: None,
        audio: None,
    };
    let generation = match runtime.generate(
        &prepared,
        request.max_tokens,
        body.get("ignore_eos")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    ) {
        Ok(generation) => generation,
        Err(error) => return send_generation_error(stream, error),
    };
    send_json(
        stream,
        200,
        &json!({
            "model":MODEL_ID,
            "model_revision":MODEL_REVISION,
            "input_ids":prepared.tokens,
            "output_ids":generation.tokens,
            "text":generation.text,
            "ttft_seconds":generation.ttft_seconds,
            "tpot_seconds":generation.tpot_seconds,
            "fallback_active":false
        }),
    )
}

fn send_generation_error(stream: &mut TcpStream, error: String) -> Result<(), String> {
    if error.contains("exceeds") {
        send_json(
            stream,
            400,
            &json!({"error":{"message":error,"type":"invalid_request"}}),
        )
    } else {
        Err(error)
    }
}

#[derive(Debug)]
struct GenerationRequest {
    max_tokens: usize,
    stream: bool,
}

fn validate_generation_request(body: &Value, chat: bool) -> Result<GenerationRequest, String> {
    let object = body
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    let allowed_chat = [
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "n",
        "stream",
    ];
    let allowed_evaluation = [
        "input_ids",
        "max_new_tokens",
        "temperature",
        "ignore_eos",
        "stream",
    ];
    let allowed: &[&str] = if chat {
        &allowed_chat
    } else {
        &allowed_evaluation
    };
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(format!("unsupported generation field `{field}`"));
    }
    if chat && !object.get("messages").is_some_and(Value::is_array) {
        return Err("messages must be an array".into());
    }
    if chat && object.contains_key("max_tokens") && object.contains_key("max_completion_tokens") {
        return Err("specify only one of max_tokens or max_completion_tokens".into());
    }
    if let Some(model) = object.get("model") {
        let model = model
            .as_str()
            .ok_or_else(|| "model must be a string".to_string())?;
        if model != MODEL_ID {
            return Err(format!("model must be `{MODEL_ID}`"));
        }
    }
    if let Some(stream) = object.get("stream") {
        if !stream.is_boolean() {
            return Err("stream must be boolean".into());
        }
    }
    if !chat {
        if !matches!(object.get("ignore_eos"), Some(Value::Bool(_))) {
            return Err("ignore_eos must be boolean".into());
        }
        if object.get("stream").and_then(Value::as_bool) != Some(false) {
            return Err("evaluation v1 requires stream=false".into());
        }
    }
    neutral_number(object, "temperature", 0.0)?;
    neutral_number(object, "top_p", 1.0)?;
    neutral_integer(object, "top_k", 0)?;
    for field in [
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    ] {
        let neutral = if field == "repetition_penalty" {
            1.0
        } else {
            0.0
        };
        neutral_number(object, field, neutral)?;
    }
    neutral_integer(object, "n", 1)?;
    let key = if chat {
        if object.contains_key("max_tokens") {
            "max_tokens"
        } else {
            "max_completion_tokens"
        }
    } else {
        "max_new_tokens"
    };
    let max_tokens = match object.get(key) {
        Some(value) => value
            .as_u64()
            .ok_or_else(|| format!("{key} must be a positive integer"))?,
        None if chat => 16,
        None => return Err(format!("{key} must be a positive integer")),
    };
    if max_tokens == 0 || max_tokens > 128 {
        return Err(format!("{key} must be in 1..=128"));
    }
    Ok(GenerationRequest {
        max_tokens: max_tokens as usize,
        stream: object
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn validate_chat_content(body: &Value) -> Result<(), String> {
    let messages = body["messages"]
        .as_array()
        .ok_or_else(|| "messages must be an array".to_string())?;
    if messages.is_empty() {
        return Err("messages must be nonempty".into());
    }
    let mut images = 0;
    let mut audios = 0;
    for (message_index, message) in messages.iter().enumerate() {
        let role = message["role"]
            .as_str()
            .ok_or_else(|| format!("messages[{message_index}].role must be a string"))?;
        if !matches!(role, "system" | "user" | "assistant") {
            return Err(format!("unsupported message role `{role}`"));
        }
        let content = &message["content"];
        if content.is_string() {
            continue;
        }
        let parts = content
            .as_array()
            .ok_or_else(|| format!("messages[{message_index}].content must be string or array"))?;
        for (part_index, part) in parts.iter().enumerate() {
            let kind = part["type"].as_str().ok_or_else(|| {
                format!("messages[{message_index}].content[{part_index}].type must be a string")
            })?;
            match kind {
                "text" | "input_text" => {
                    if !part["text"].is_string() {
                        return Err("text content part is missing text".into());
                    }
                }
                "image_url" => {
                    images += 1;
                    if role != "user" || images > 1 {
                        return Err("at most one image in a user message is supported".into());
                    }
                    let url = part["image_url"]["url"]
                        .as_str()
                        .ok_or_else(|| "image_url content part is missing url".to_string())?;
                    if !url.starts_with("data:image/png;base64,") {
                        return Err(
                            "image input must be a PNG data URL; remote media URLs are forbidden"
                                .into(),
                        );
                    }
                }
                "input_audio" | "audio" => {
                    audios += 1;
                    if role != "user" || audios > 1 {
                        return Err("at most one audio clip in a user message is supported".into());
                    }
                    let audio = part.get("input_audio").unwrap_or(part);
                    if audio["format"].as_str() != Some("wav") || !audio["data"].is_string() {
                        return Err("audio input requires base64 data and format=wav".into());
                    }
                }
                "video" | "video_url" => {
                    return Err("video is outside the deployed capability".into())
                }
                other => return Err(format!("unsupported content part `{other}`")),
            }
        }
    }
    if images > 0 && audios > 0 {
        return Err("simultaneous image and audio input is outside the deployed capability".into());
    }
    Ok(())
}

fn neutral_number(object: &Map<String, Value>, field: &str, neutral: f64) -> Result<(), String> {
    if let Some(value) = object.get(field) {
        if value.as_f64() != Some(neutral) {
            return Err(format!("{field} supports only {neutral}"));
        }
    }
    Ok(())
}

fn neutral_integer(object: &Map<String, Value>, field: &str, neutral: u64) -> Result<(), String> {
    if let Some(value) = object.get(field) {
        if value.as_u64() != Some(neutral) {
            return Err(format!("{field} supports only {neutral}"));
        }
    }
    Ok(())
}

fn parse_input_ids(body: &Value) -> Result<Vec<u32>, String> {
    let values = body
        .get("input_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "input_ids must be an array".to_string())?;
    if values.is_empty() {
        return Err("input_ids must be nonempty".into());
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("input_ids[{index}] must be a u32"))
        })
        .collect()
}

const PROCESSOR_WORKER_SCRIPT: &str = r#"
import base64
import io
import json
import sys
import numpy as np
import soundfile as sf
from math import gcd
from PIL import Image
from scipy.signal import resample_poly
from transformers import AutoProcessor

model_dir = sys.argv[1]
processor = AutoProcessor.from_pretrained(
    model_dir, local_files_only=True, use_fast=False)

def emit(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def process(request):
    body = request["body"]
    pixel_path = request["pixel_path"]
    feature_path = request["feature_path"]
    mask_path = request["mask_path"]
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty array")
    images = 0
    audios = 0
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError("unsupported message role")
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [{"type":"text","text":content}]
            continue
        if not isinstance(content, list):
            raise ValueError("message content must be a string or array")
        rewritten = []
        for part in content:
            kind = part.get("type")
            if kind in ("text", "input_text"):
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError("text part is missing text")
                rewritten.append({"type":"text","text":text})
            elif kind == "image_url":
                if role != "user" or images >= 1:
                    raise ValueError("exactly one user image is supported")
                url = part.get("image_url", {}).get("url")
                prefix = "data:image/png;base64,"
                if not isinstance(url, str) or not url.startswith(prefix):
                    raise ValueError("image must be a data:image/png;base64 URL; remote URLs are forbidden")
                image = Image.open(io.BytesIO(base64.b64decode(
                    url[len(prefix):], validate=True))).convert("RGB")
                rewritten.append({"type":"image","image":image})
                images += 1
            elif kind in ("input_audio", "audio"):
                if role != "user" or audios >= 1:
                    raise ValueError("exactly one user audio clip is supported")
                item = part.get("input_audio", part)
                if item.get("format") != "wav" or not isinstance(item.get("data"), str):
                    raise ValueError("audio must contain base64 WAV data and format=wav")
                audio, rate = sf.read(io.BytesIO(base64.b64decode(
                    item["data"], validate=True)), dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if audio.size == 0 or not np.isfinite(audio).all():
                    raise ValueError("audio must be finite and nonempty")
                if rate != 16000:
                    divisor = gcd(int(rate), 16000)
                    audio = resample_poly(
                        audio, 16000 // divisor, int(rate) // divisor).astype(np.float32)
                rewritten.append({"type":"audio","audio":audio})
                audios += 1
            elif kind in ("video", "video_url"):
                raise ValueError("video is outside the deployed capability")
            else:
                raise ValueError("unsupported content part: %s" % kind)
        message["content"] = rewritten
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", sampling_rate=16000)
    tokens = inputs["input_ids"][0].cpu().numpy().astype(np.int64).tolist()
    metadata = {"tokens":tokens,"has_image":images == 1,"has_audio":audios == 1}
    if images:
        pixels = inputs["pixel_values"].cpu().numpy().astype(np.float32)
        grids = inputs["image_grid_thw"].cpu().numpy().astype(np.int64).tolist()
        np.save(pixel_path, pixels)
        metadata["grids"] = grids
    if audios:
        features = inputs["input_features"][0].cpu().numpy().astype(np.float32)
        if features.shape[0] == 128:
            features = features.T
        mask = inputs.get("feature_attention_mask")
        if mask is None:
            mask = np.ones((features.shape[0],), dtype=np.float32)
        else:
            mask = mask[0].cpu().numpy().astype(np.float32).reshape(-1)[:features.shape[0]]
        valid = int(mask.sum())
        if valid <= 0 or valid > features.shape[0]:
            raise ValueError("invalid audio feature mask")
        features = features[:valid]
        mask = np.ones((valid,), dtype=np.float32)
        np.save(feature_path, features)
        np.save(mask_path, mask)
        metadata["feature_length"] = valid
        metadata["audio_token_count"] = sum(token == 151646 for token in tokens)
    return metadata

emit({"ready":True})
for line in sys.stdin:
    if not line.strip():
        continue
    try:
        emit({"ok":True,"metadata":process(json.loads(line))})
    except Exception as error:
        emit({"ok":False,"error":"%s: %s" % (type(error).__name__, error)})
"#;

struct ProcessorWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ProcessorWorker {
    fn new(model_dir: &Path) -> Result<Self, String> {
        let mut child = Command::new("python3")
            .arg("-c")
            .arg(PROCESSOR_WORKER_SCRIPT)
            .arg(model_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("launch Omni processor worker: {error}"))?;
        let stdin = match child.stdin.take() {
            Some(stdin) => stdin,
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Omni processor worker stdin is unavailable".into());
            }
        };
        let stdout = match child.stdout.take() {
            Some(stdout) => BufReader::new(stdout),
            None => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("Omni processor worker stdout is unavailable".into());
            }
        };
        let mut worker = Self {
            child,
            stdin,
            stdout,
        };
        let ready = worker.read_response("startup")?;
        if ready.get("ready").and_then(Value::as_bool) != Some(true) {
            return Err(format!("Omni processor worker rejected startup: {ready}"));
        }
        Ok(worker)
    }

    fn read_response(&mut self, phase: &str) -> Result<Value, String> {
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .map_err(|error| format!("read Omni processor {phase}: {error}"))?;
        if bytes == 0 {
            let status = self
                .child
                .try_wait()
                .map_err(|error| format!("inspect Omni processor worker: {error}"))?;
            return Err(format!(
                "Omni processor worker closed stdout during {phase}; status={status:?}"
            ));
        }
        if line.len() > MAX_BODY_BYTES {
            return Err(format!("Omni processor {phase} response exceeds limit"));
        }
        serde_json::from_str(&line)
            .map_err(|error| format!("parse Omni processor {phase} response: {error}"))
    }

    fn preprocess(&mut self, body: &Value) -> Result<Prepared, String> {
        let suffix = REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        let prefix = std::env::temp_dir().join(format!(
            "apxinf-qwen25-omni-{}-{suffix}",
            std::process::id()
        ));
        let pixel_path = prefix.with_extension("pixels.npy");
        let feature_path = prefix.with_extension("audio.npy");
        let mask_path = prefix.with_extension("mask.npy");
        let request = json!({
            "body": body,
            "pixel_path": pixel_path,
            "feature_path": feature_path,
            "mask_path": mask_path
        });
        let result = (|| {
            serde_json::to_writer(&mut self.stdin, &request)
                .map_err(|error| format!("serialize Omni processor request: {error}"))?;
            self.stdin
                .write_all(b"\n")
                .and_then(|_| self.stdin.flush())
                .map_err(|error| format!("write Omni processor request: {error}"))?;
            let response = self.read_response("request")?;
            if response.get("ok").and_then(Value::as_bool) != Some(true) {
                return Err(response
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("processor returned an invalid response")
                    .to_string());
            }
            let metadata = response
                .get("metadata")
                .ok_or_else(|| "processor omitted metadata".to_string())?;
            let tokens = parse_u32_array(metadata, "tokens")?;
            let image = if metadata["has_image"].as_bool() == Some(true) {
                let grids = metadata["grids"]
                    .as_array()
                    .ok_or_else(|| "processor omitted image grids".to_string())?
                    .iter()
                    .map(|grid| {
                        let values = grid
                            .as_array()
                            .ok_or_else(|| "image grid must be an array".to_string())?;
                        if values.len() != 3 {
                            return Err("image grid must contain T,H,W".into());
                        }
                        Ok([
                            u32_value(&values[0], "grid T")?,
                            u32_value(&values[1], "grid H")?,
                            u32_value(&values[2], "grid W")?,
                        ])
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let (shape, values) = super::read_npy_f32_to_bf16(&pixel_path)?;
                Some(PreparedImage {
                    pixels: Tensor::from_bf16(shape, &values).map_err(|error| error.to_string())?,
                    grids,
                })
            } else {
                None
            };
            let audio = if metadata["has_audio"].as_bool() == Some(true) {
                let length = u32_value(&metadata["feature_length"], "feature_length")?;
                let count = u32_value(&metadata["audio_token_count"], "audio_token_count")?;
                let (feature_shape, features) = super::read_npy_f32_to_bf16(&feature_path)?;
                let (mask_shape, mask) = super::read_npy_f32_to_bf16(&mask_path)?;
                Some(PreparedAudio {
                    features: Tensor::from_bf16(feature_shape, &features)
                        .map_err(|error| error.to_string())?,
                    mask: Tensor::from_bf16(mask_shape, &mask)
                        .map_err(|error| error.to_string())?,
                    lengths: vec![length],
                    counts: vec![count],
                })
            } else {
                None
            };
            Ok(Prepared {
                tokens,
                image,
                audio,
            })
        })();
        cleanup(&[&pixel_path, &feature_path, &mask_path]);
        result
    }
}

impl Drop for ProcessorWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn parse_u32_array(value: &Value, key: &str) -> Result<Vec<u32>, String> {
    value[key]
        .as_array()
        .ok_or_else(|| format!("processor metadata missing {key}"))?
        .iter()
        .enumerate()
        .map(|(index, value)| u32_value(value, &format!("{key}[{index}]")))
        .collect()
}

fn u32_value(value: &Value, name: &str) -> Result<u32, String> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!("{name} must be a u32"))
}

fn cleanup(paths: &[&Path]) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

fn send_stream(
    stream: &mut TcpStream,
    id: &str,
    generation: &Generation,
    runtime: &Runtime,
) -> Result<(), String> {
    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream
        .write_all(headers.as_bytes())
        .map_err(|error| error.to_string())?;
    let mut accumulated = Vec::new();
    let mut previous = String::new();
    for &token in &generation.tokens {
        accumulated.push(token);
        let text = runtime
            .tokenizer
            .decode(&accumulated)
            .map_err(|error| error.to_string())?;
        let delta = text.strip_prefix(&previous).unwrap_or(&text);
        previous = text.clone();
        let event = json!({
            "id":id,"object":"chat.completion.chunk","model":MODEL_ID,
            "choices":[{"index":0,"delta":{"content":delta},"finish_reason":Value::Null}]
        });
        writeln!(stream, "data: {}\n", event).map_err(|error| error.to_string())?;
    }
    let final_event = json!({
        "id":id,"object":"chat.completion.chunk","model":MODEL_ID,
        "choices":[{"index":0,"delta":{},"finish_reason":"stop"}],
        "apxinf":{"fallback_active":false,"output_modalities":["text"]}
    });
    writeln!(stream, "data: {}\n\ndata: [DONE]\n", final_event)
        .map_err(|error| error.to_string())?;
    Ok(())
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8192];
    let header_end = loop {
        let count = stream
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            return Err("connection closed before request headers".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
        if bytes.len() > 128 * 1024 {
            return Err("request headers exceed 128 KiB".into());
        }
        if let Some(index) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| "request headers are not UTF-8".to_string())?;
    let mut lines = header.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| "missing method".to_string())?
        .to_owned();
    let path = request_parts
        .next()
        .ok_or_else(|| "missing path".to_string())?
        .to_owned();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse::<usize>())
        .transpose()
        .map_err(|_| "invalid Content-Length".to_string())?
        .unwrap_or(0);
    if content_length > MAX_BODY_BYTES {
        return Err(format!("request body exceeds {MAX_BODY_BYTES} bytes"));
    }
    while bytes.len() - header_end < content_length {
        let count = stream
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            return Err("connection closed before request body".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    Ok(HttpRequest {
        method,
        path,
        body: bytes[header_end..header_end + content_length].to_vec(),
    })
}

fn send_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        503 => "Service Unavailable",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .map_err(|error| error.to_string())?;
    stream.write_all(&body).map_err(|error| error.to_string())
}

fn request_id(prefix: &str) -> String {
    format!(
        "{prefix}-apxinf-{}",
        REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_sampling_and_unknown_fields() {
        let request = json!({"messages":[],"max_tokens":1,"temperature":0,"stream":false});
        assert_eq!(
            validate_generation_request(&request, true)
                .unwrap()
                .max_tokens,
            1
        );
        let sampled = json!({"messages":[],"max_tokens":1,"temperature":0.5});
        assert!(validate_generation_request(&sampled, true)
            .unwrap_err()
            .contains("temperature"));
        let unknown = json!({"messages":[],"max_tokens":1,"beam_search":true});
        assert!(validate_generation_request(&unknown, true)
            .unwrap_err()
            .contains("beam_search"));
        for request in [
            json!({"messages":[],"model":"wrong","max_tokens":1}),
            json!({"messages":[],"max_tokens":0}),
            json!({"messages":[],"max_tokens":129}),
            json!({"messages":[],"max_tokens":1,"top_p":0.9}),
            json!({"messages":[],"max_tokens":1,"top_k":1}),
            json!({"messages":[],"max_tokens":1,"presence_penalty":0.1}),
            json!({"messages":[],"max_tokens":1,"frequency_penalty":0.1}),
            json!({"messages":[],"max_tokens":1,"repetition_penalty":1.1}),
            json!({"messages":[],"max_tokens":1,"n":2}),
        ] {
            assert!(validate_generation_request(&request, true).is_err());
        }
    }

    #[test]
    fn rejects_combined_image_and_audio_before_preprocessing() {
        let request = json!({
            "messages":[{
                "role":"user",
                "content":[
                    {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},
                    {"type":"input_audio","input_audio":{"format":"wav","data":"AA=="}}
                ]
            }]
        });
        assert!(validate_chat_content(&request)
            .unwrap_err()
            .contains("simultaneous image and audio"));

        for request in [
            json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.invalid/image.png"}}]}]}),
            json!({"messages":[{"role":"user","content":[{"type":"video","video":"AA=="}]}]}),
            json!({"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"format":"mp3","data":"AA=="}}]}]}),
            json!({"messages":[{"role":"tool","content":"unsupported"}]}),
        ] {
            assert!(validate_chat_content(&request).is_err());
        }
    }
}

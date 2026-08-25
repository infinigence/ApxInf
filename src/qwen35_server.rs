use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use apxinf_core::{DType, Tensor};
use apxinf_cuda::CudaContext;
use apxinf_loader::safetensors;
use apxinf_model::qwen35::{
    compute_mrope_positions, HybridUnit, HybridUnitMode, Qwen35Config, Qwen35LmHead,
    Qwen35PrefillMode, Qwen35VisionEncoder,
};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static PROCESSOR_ID: AtomicU64 = AtomicU64::new(1);
const EVALUATION_CONTRACT: &str = "apxinf.qwen38_27b.inference_interface.v1";
const MODEL_REVISION: &str = "63768c10df38c0395e12ef49edac1bd539eaeeea";

pub fn serve(
    model_dir: &Path,
    host: &str,
    port: u16,
    max_model_len: usize,
    enable_marlin_m64: bool,
    enable_multimodal: bool,
) -> Result<(), String> {
    if max_model_len == 0 || max_model_len > 32768 {
        return Err("--max-model-len must be within 1..=32768".into());
    }
    let runtime = NativeRuntime::load(
        model_dir,
        max_model_len,
        enable_marlin_m64,
        enable_multimodal,
    )?;
    let listener =
        TcpListener::bind((host, port)).map_err(|error| format!("bind {host}:{port}: {error}"))?;
    println!(
        "ApxInf Qwen3.8 native server ready on http://{host}:{port} (max_model_len={max_model_len}, experimental_marlin_m64={enable_marlin_m64}, multimodal={enable_multimodal})"
    );
    for connection in listener.incoming() {
        match connection {
            Ok(mut stream) => {
                stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
                stream
                    .set_write_timeout(Some(Duration::from_secs(300)))
                    .ok();
                if let Err(error) = handle_connection(&runtime, &mut stream) {
                    eprintln!("request failed: {error}");
                    let _ = send_json(
                        &mut stream,
                        500,
                        &json!({"error":{"message":error,"type":"server_error"}}),
                    );
                }
            }
            Err(error) => eprintln!("accept failed: {error}"),
        }
    }
    Ok(())
}

struct NativeRuntime {
    embedding: Tensor,
    context: CudaContext,
    decoder: HybridUnit,
    lm_head: Qwen35LmHead,
    tokenizer: Tokenizer,
    max_model_len: usize,
    model_id: String,
    marlin_m64_enabled: bool,
    model_dir: PathBuf,
    vision: Option<Qwen35VisionEncoder>,
    processor_python: Option<String>,
    spatial_merge_size: u32,
    image_token_id: u32,
}

impl NativeRuntime {
    fn load(
        model_dir: &Path,
        max_model_len: usize,
        enable_marlin_m64: bool,
        enable_multimodal: bool,
    ) -> Result<Self, String> {
        let load_start = Instant::now();
        let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))
            .map_err(|error| error.to_string())?;
        let manifest = safetensors::inspect_path(model_dir)?;
        let context = CudaContext::new(0)?;
        let prefill_mode = if enable_marlin_m64 {
            Qwen35PrefillMode::MarlinM64
        } else {
            Qwen35PrefillMode::M8
        };
        let decoder = HybridUnit::load_all_with_prefill_mode(
            &manifest,
            &context,
            max_model_len,
            prefill_mode,
        )
        .map_err(|error| error.to_string())?;
        let lm_head = Qwen35LmHead::load(&manifest, &context).map_err(|error| error.to_string())?;
        let embedding_entry = manifest
            .tensor("model.language_model.embed_tokens.weight")
            .ok_or_else(|| "missing embedding table".to_string())?;
        if embedding_entry.dtype != DType::BF16 || embedding_entry.shape != [248_320, 5120] {
            return Err(format!(
                "Qwen3.8 embedding table must be BF16 [248320,5120], got {} {:?}",
                embedding_entry.dtype, embedding_entry.shape
            ));
        }
        let embedding = safetensors::load_manifest_tensor(embedding_entry)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| error.to_string())?;
        let (vision, processor_python) = if enable_multimodal {
            let python =
                std::env::var("APXINF_PROCESSOR_PYTHON").unwrap_or_else(|_| "python3".to_owned());
            validate_processor_python(&python, model_dir)?;
            let encoder =
                Qwen35VisionEncoder::load(model_dir, &config).map_err(|error| error.to_string())?;
            (Some(encoder), Some(python))
        } else {
            (None, None)
        };
        println!(
            "resident model loaded in {:.3}s",
            load_start.elapsed().as_secs_f64()
        );
        Ok(Self {
            embedding,
            context,
            decoder,
            lm_head,
            tokenizer,
            max_model_len,
            model_id: model_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Qwen3.8-27B-AWQ-INT4")
                .to_owned(),
            marlin_m64_enabled: enable_marlin_m64,
            model_dir: model_dir.to_owned(),
            vision,
            processor_python,
            spatial_merge_size: config.vision.spatial_merge_size as u32,
            image_token_id: config.image_token_id,
        })
    }

    fn encode_messages(&self, body: &Value) -> Result<Vec<u32>, String> {
        let messages = body
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| "messages must be an array".to_string())?;
        let mut parsed = Vec::with_capacity(messages.len());
        for message in messages {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| "message.role must be a string".to_string())?;
            let content = parse_text_content(
                message
                    .get("content")
                    .ok_or_else(|| "message.content is required".to_string())?,
            )?;
            parsed.push(ChatMessage {
                role: role.to_owned(),
                content,
            });
        }
        if self.tokenizer.has_chat_template() {
            self.tokenizer
                .encode_chat(&parsed)
                .map_err(|error| error.to_string())
        } else {
            let prompt = parsed
                .iter()
                .map(|message| format!("{}: {}", message.role, message.content))
                .collect::<Vec<_>>()
                .join("\n");
            self.tokenizer
                .encode(&prompt)
                .map_err(|error| error.to_string())
        }
    }

    fn load_embedding_tokens(&self, tokens: &[u32]) -> Result<Tensor, String> {
        const HIDDEN: usize = 5120;
        if !matches!(tokens.len(), 1 | 8 | 64) {
            return Err(format!(
                "Qwen3.8 embedding batch requires 1, 8, or 64 tokens, got {}",
                tokens.len()
            ));
        }
        let values = self.load_embedding_values(tokens)?;
        Tensor::from_bf16(vec![tokens.len(), HIDDEN], &values).map_err(|error| error.to_string())
    }

    fn load_embedding_values(&self, tokens: &[u32]) -> Result<Vec<half::bf16>, String> {
        const HIDDEN: usize = 5120;
        let table = self
            .embedding
            .as_bf16()
            .map_err(|error| error.to_string())?;
        let mut values = Vec::with_capacity(tokens.len() * HIDDEN);
        for &token in tokens {
            let token = token as usize;
            if token >= 248_320 {
                return Err(format!(
                    "Qwen3.8 token id {token} exceeds embedding vocabulary"
                ));
            }
            let start = token * HIDDEN;
            values.extend_from_slice(&table[start..start + HIDDEN]);
        }
        Ok(values)
    }

    fn generate<F>(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        eos_stop: bool,
        prefill_mode: Qwen35PrefillMode,
        mut on_delta: F,
    ) -> Result<Generation, String>
    where
        F: FnMut(&str, u32) -> Result<(), String>,
    {
        if prompt_tokens.is_empty() {
            return Err("prompt token sequence is empty".into());
        }
        if max_tokens == 0 || prompt_tokens.len() + max_tokens > self.max_model_len {
            return Err(format!(
                "prompt+output must fit max_model_len {} (got {}+{})",
                self.max_model_len,
                prompt_tokens.len(),
                max_tokens
            ));
        }
        let first_embedding = self.load_embedding_tokens(&prompt_tokens[..1])?;
        self.decoder
            .reset_text_request(&self.context, &first_embedding)
            .map_err(|error| error.to_string())?;
        let prefill_start = Instant::now();
        if prefill_mode == Qwen35PrefillMode::MarlinM64
            && (!self.marlin_m64_enabled
                || !self.decoder.has_marlin_prefill64()
                || prompt_tokens.len() < 64)
        {
            return Err(
                "marlin-m64 requires an enabled SM89 workspace and at least 64 prompt tokens"
                    .into(),
            );
        }
        let marlin_tokens = match prefill_mode {
            Qwen35PrefillMode::M8 => 0,
            Qwen35PrefillMode::MarlinM64 => prompt_tokens.len() / 64 * 64,
        };
        for position in (0..marlin_tokens).step_by(64) {
            let embedding = self.load_embedding_tokens(&prompt_tokens[position..position + 64])?;
            self.decoder
                .set_marlin_prefill64_input(&self.context, &embedding)
                .map_err(|error| error.to_string())?;
            self.decoder
                .forward_marlin_prefill64(&self.context, position, false)
                .map_err(|error| error.to_string())?;
        }
        let m8_tokens = (prompt_tokens.len() - marlin_tokens) / 8 * 8;
        let tiled_tokens = marlin_tokens + m8_tokens;
        for position in (marlin_tokens..tiled_tokens).step_by(8) {
            let embedding = self.load_embedding_tokens(&prompt_tokens[position..position + 8])?;
            self.decoder
                .set_prefill8_input(&self.context, &embedding)
                .map_err(|error| error.to_string())?;
            self.decoder
                .forward_prefill8(&self.context, position, false)
                .map_err(|error| error.to_string())?;
        }
        for (offset, &token) in prompt_tokens[tiled_tokens..].iter().enumerate() {
            let position = tiled_tokens + offset;
            if position > 0 || tiled_tokens > 0 {
                let embedding = self.load_embedding_tokens(std::slice::from_ref(&token))?;
                self.decoder
                    .set_token_input(&self.context, &embedding)
                    .map_err(|error| error.to_string())?;
            }
            let bucket = (position + 1).next_power_of_two().min(self.max_model_len);
            self.decoder
                .forward(
                    &self.context,
                    HybridUnitMode::ModelOptimized,
                    bucket,
                    position as u32,
                    false,
                )
                .map_err(|error| error.to_string())?;
        }
        if tiled_tokens == prompt_tokens.len() {
            if m8_tokens > 0 {
                self.decoder
                    .commit_prefill8_last(&self.context)
                    .map_err(|error| error.to_string())?;
            } else if marlin_tokens > 0 {
                self.decoder
                    .commit_marlin_prefill64_last(&self.context)
                    .map_err(|error| error.to_string())?;
            }
        }
        self.context.synchronize()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();

        let eos = eos_stop.then(|| self.tokenizer.eos_token_id()).flatten();
        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated = Vec::with_capacity(max_tokens);
        let mut output = String::new();
        let decode_start = Instant::now();
        let mut first_token_seconds = None;
        for step in 0..max_tokens {
            if step > 0 {
                let previous = generated[step - 1];
                let embedding = self.load_embedding_tokens(std::slice::from_ref(&previous))?;
                self.decoder
                    .set_token_input(&self.context, &embedding)
                    .map_err(|error| error.to_string())?;
                let position = prompt_tokens.len() + step - 1;
                let bucket = (position + 1).next_power_of_two().min(self.max_model_len);
                self.decoder
                    .forward(
                        &self.context,
                        HybridUnitMode::ModelOptimized,
                        bucket,
                        position as u32,
                        false,
                    )
                    .map_err(|error| error.to_string())?;
            }
            self.lm_head
                .forward(&self.context, self.decoder.normalized_output())
                .map_err(|error| error.to_string())?;
            let token = self
                .lm_head
                .argmax_cpu()
                .map_err(|error| error.to_string())?;
            if first_token_seconds.is_none() {
                first_token_seconds = Some(decode_start.elapsed().as_secs_f64());
            }
            generated.push(token);
            all_tokens.push(token);
            let decoded = self
                .tokenizer
                .decode(&all_tokens)
                .map_err(|error| error.to_string())?;
            let previous = self
                .tokenizer
                .decode(&all_tokens[..all_tokens.len() - 1])
                .unwrap_or_default();
            let delta = decoded.strip_prefix(&previous).unwrap_or(&decoded);
            output.push_str(delta);
            on_delta(delta, token)?;
            if eos == Some(token) {
                break;
            }
        }
        Ok(Generation {
            text: output,
            tokens: generated,
            prompt_tokens: prompt_tokens.len(),
            prefill_seconds,
            decode_seconds: decode_start.elapsed().as_secs_f64(),
            first_token_seconds: first_token_seconds.unwrap_or_default(),
            prefill_mode,
            marlin_tiles: marlin_tokens / 64,
            m8_tiles: m8_tokens / 8,
            m1_tokens: prompt_tokens.len() - tiled_tokens,
        })
    }

    fn generate_multimodal<F>(
        &self,
        prepared: &PreparedImage,
        max_tokens: usize,
        eos_stop: bool,
        mut on_delta: F,
    ) -> Result<Generation, String>
    where
        F: FnMut(&str, u32) -> Result<(), String>,
    {
        const HIDDEN: usize = 5120;
        const TILE: usize = 8;
        let prompt_tokens = &prepared.tokens;
        if prompt_tokens.is_empty()
            || prompt_tokens.len() != prepared.modality_types.len()
            || max_tokens == 0
            || prompt_tokens.len() + max_tokens > self.max_model_len
        {
            return Err(format!(
                "multimodal prompt and output must fit max_model_len {} (got {}+{})",
                self.max_model_len,
                prompt_tokens.len(),
                max_tokens,
            ));
        }
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| "native multimodal is not enabled".to_string())?;
        let prefill_start = Instant::now();
        let primary = vision
            .encode_cpu(&prepared.pixel_values, prepared.grid_thw)
            .map_err(|error| error.to_string())?;
        let image_positions = prepared
            .modality_types
            .iter()
            .enumerate()
            .filter_map(|(index, modality)| (*modality == 1).then_some(index))
            .collect::<Vec<_>>();
        if image_positions.len() != primary.shape().dims()[0]
            || image_positions
                .iter()
                .any(|index| prompt_tokens[*index] != self.image_token_id)
        {
            return Err(format!(
                "image placeholder/feature mismatch: placeholders={}, features={}",
                image_positions.len(),
                primary.shape().dims()[0],
            ));
        }
        let mut embeddings = self.load_embedding_values(prompt_tokens)?;
        let vision_values = primary.as_bf16().map_err(|error| error.to_string())?;
        for (vision_row, token_position) in image_positions.iter().copied().enumerate() {
            let destination = token_position * HIDDEN;
            let source = vision_row * HIDDEN;
            embeddings[destination..destination + HIDDEN]
                .copy_from_slice(&vision_values[source..source + HIDDEN]);
        }
        let mrope = compute_mrope_positions(
            &prepared.modality_types,
            &[prepared.grid_thw],
            self.spatial_merge_size,
        )
        .map_err(|error| error.to_string())?;
        let first_embedding = Tensor::from_bf16(vec![1, HIDDEN], &embeddings[..HIDDEN])
            .map_err(|error| error.to_string())?;
        self.decoder
            .reset_text_request(&self.context, &first_embedding)
            .map_err(|error| error.to_string())?;

        let tiled_tokens = prompt_tokens.len() / TILE * TILE;
        for position in (0..tiled_tokens).step_by(TILE) {
            let start = position * HIDDEN;
            let input = Tensor::from_bf16(
                vec![TILE, HIDDEN],
                &embeddings[start..start + TILE * HIDDEN],
            )
            .map_err(|error| error.to_string())?;
            let rope_positions: [[u32; 3]; TILE] = mrope.positions[position..position + TILE]
                .try_into()
                .map_err(|_| "multimodal mRoPE tile conversion failed".to_string())?;
            self.decoder
                .set_prefill8_input(&self.context, &input)
                .map_err(|error| error.to_string())?;
            self.decoder
                .forward_prefill8_with_mrope(&self.context, position, &rope_positions, false)
                .map_err(|error| error.to_string())?;
        }
        for position in tiled_tokens..prompt_tokens.len() {
            let start = position * HIDDEN;
            let input = Tensor::from_bf16(vec![1, HIDDEN], &embeddings[start..start + HIDDEN])
                .map_err(|error| error.to_string())?;
            self.decoder
                .set_token_input(&self.context, &input)
                .map_err(|error| error.to_string())?;
            let bucket = (position + 1).next_power_of_two().min(self.max_model_len);
            self.decoder
                .forward_with_mrope(
                    &self.context,
                    HybridUnitMode::ModelOptimized,
                    bucket,
                    position as u32,
                    mrope.positions[position],
                    false,
                )
                .map_err(|error| error.to_string())?;
        }
        if tiled_tokens == prompt_tokens.len() {
            self.decoder
                .commit_prefill8_last(&self.context)
                .map_err(|error| error.to_string())?;
        }
        self.context.synchronize()?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();

        let eos = eos_stop.then(|| self.tokenizer.eos_token_id()).flatten();
        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated = Vec::with_capacity(max_tokens);
        let mut output = String::new();
        let decode_start = Instant::now();
        let mut first_token_seconds = None;
        for step in 0..max_tokens {
            if step > 0 {
                let previous = generated[step - 1];
                let embedding = self.load_embedding_tokens(std::slice::from_ref(&previous))?;
                self.decoder
                    .set_token_input(&self.context, &embedding)
                    .map_err(|error| error.to_string())?;
                let cache_position = prompt_tokens.len() + step - 1;
                let rope_position = cache_position as i64 + mrope.decode_delta;
                if rope_position < 0 || rope_position > u32::MAX as i64 {
                    return Err(format!(
                        "multimodal decode mRoPE position {rope_position} is invalid"
                    ));
                }
                let bucket = (cache_position + 1)
                    .next_power_of_two()
                    .min(self.max_model_len);
                self.decoder
                    .forward_with_mrope(
                        &self.context,
                        HybridUnitMode::ModelOptimized,
                        bucket,
                        cache_position as u32,
                        [rope_position as u32; 3],
                        false,
                    )
                    .map_err(|error| error.to_string())?;
            }
            self.lm_head
                .forward(&self.context, self.decoder.normalized_output())
                .map_err(|error| error.to_string())?;
            let token = self
                .lm_head
                .argmax_cpu()
                .map_err(|error| error.to_string())?;
            if first_token_seconds.is_none() {
                first_token_seconds = Some(decode_start.elapsed().as_secs_f64());
            }
            generated.push(token);
            all_tokens.push(token);
            let decoded = self
                .tokenizer
                .decode(&all_tokens)
                .map_err(|error| error.to_string())?;
            let previous = self
                .tokenizer
                .decode(&all_tokens[..all_tokens.len() - 1])
                .unwrap_or_default();
            let delta = decoded.strip_prefix(&previous).unwrap_or(&decoded);
            output.push_str(delta);
            on_delta(delta, token)?;
            if eos == Some(token) {
                break;
            }
        }
        Ok(Generation {
            text: output,
            tokens: generated,
            prompt_tokens: prompt_tokens.len(),
            prefill_seconds,
            decode_seconds: decode_start.elapsed().as_secs_f64(),
            first_token_seconds: first_token_seconds.unwrap_or_default(),
            prefill_mode: Qwen35PrefillMode::M8,
            marlin_tiles: 0,
            m8_tiles: tiled_tokens / TILE,
            m1_tokens: prompt_tokens.len() - tiled_tokens,
        })
    }
}

struct Generation {
    text: String,
    tokens: Vec<u32>,
    prompt_tokens: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    first_token_seconds: f64,
    prefill_mode: Qwen35PrefillMode,
    marlin_tiles: usize,
    m8_tiles: usize,
    m1_tokens: usize,
}

struct PreparedImage {
    tokens: Vec<u32>,
    modality_types: Vec<u8>,
    pixel_values: Tensor,
    grid_thw: [u32; 3],
}

struct ImageChatInput {
    data_url: String,
    prompt: String,
}

fn handle_connection(runtime: &NativeRuntime, stream: &mut TcpStream) -> Result<(), String> {
    let request = read_request(stream)?;
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => send_json(
            stream,
            200,
            &json!({
                "status":"ok",
                "evaluation_contract":EVALUATION_CONTRACT,
                "model_revision":MODEL_REVISION,
                "max_model_len":runtime.max_model_len,
                "parallel_requests":1,
                "fallback_active":false,
                "capabilities":{
                    "pretokenized_input_ids":true,
                    "token_id_output":true,
                    "multimodal":runtime.vision.is_some()
                }
            }),
        ),
        ("GET", "/v1/models") => send_json(
            stream,
            200,
            &json!({
                "object":"list",
                "data":[{"id":runtime.model_id,"object":"model","owned_by":"apxinf"}]
            }),
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

fn parse_input_ids(body: &Value) -> Result<Vec<u32>, String> {
    let values = body
        .get("input_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "input_ids must be a non-empty integer array".to_string())?;
    if values.is_empty() {
        return Err("input_ids must be a non-empty integer array".into());
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let token = value
                .as_u64()
                .ok_or_else(|| format!("input_ids[{index}] must be a non-negative integer"))?;
            u32::try_from(token).map_err(|_| format!("input_ids[{index}] exceeds u32"))
        })
        .collect()
}

fn validate_evaluation_request_shape(body: &Value) -> Result<(), String> {
    let object = body
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    const ALLOWED_FIELDS: [&str; 5] = [
        "input_ids",
        "max_new_tokens",
        "temperature",
        "ignore_eos",
        "stream",
    ];
    if let Some(field) = object
        .keys()
        .find(|field| !ALLOWED_FIELDS.contains(&field.as_str()))
    {
        return Err(format!("unsupported evaluation field `{field}`"));
    }
    if !matches!(object.get("max_new_tokens"), Some(Value::Number(value)) if value.as_u64().is_some())
    {
        return Err("max_new_tokens must be a positive integer".into());
    }
    if !matches!(object.get("temperature"), Some(Value::Number(value)) if value.as_f64() == Some(0.0))
    {
        return Err("evaluation v1 requires numeric temperature=0".into());
    }
    if !matches!(object.get("ignore_eos"), Some(Value::Bool(_))) {
        return Err("ignore_eos must be boolean".into());
    }
    if !matches!(object.get("stream"), Some(Value::Bool(_))) {
        return Err("stream must be boolean".into());
    }
    Ok(())
}

fn handle_evaluation(
    runtime: &NativeRuntime,
    stream: &mut TcpStream,
    raw: &[u8],
) -> Result<(), String> {
    let body: Value = match serde_json::from_slice(raw) {
        Ok(body) => body,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":format!("invalid JSON: {error}"),"type":"invalid_request"}}),
            )
        }
    };
    if let Err(error) = validate_evaluation_request_shape(&body) {
        return send_json(
            stream,
            400,
            &json!({"error":{"message":error,"type":"invalid_request"}}),
        );
    }
    let prompt_tokens = match parse_input_ids(&body) {
        Ok(tokens) => tokens,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":error,"type":"invalid_request"}}),
            )
        }
    };
    let vocab_size = runtime.tokenizer.vocab_size();
    if let Some((index, token)) = prompt_tokens
        .iter()
        .enumerate()
        .find(|(_, token)| **token as usize >= vocab_size)
    {
        return send_json(
            stream,
            400,
            &json!({"error":{
                "message":format!(
                    "input_ids[{index}]={token} is outside tokenizer vocabulary {}",
                    vocab_size
                ),
                "type":"invalid_request"
            }}),
        );
    }
    let max_new_tokens = body
        .get("max_new_tokens")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    if max_new_tokens == 0 || prompt_tokens.len() + max_new_tokens > runtime.max_model_len {
        return send_json(
            stream,
            400,
            &json!({"error":{
                "message":format!(
                    "input and output must be non-empty and fit max_model_len {} (got {}+{})",
                    runtime.max_model_len,
                    prompt_tokens.len(),
                    max_new_tokens
                ),
                "type":"invalid_request"
            }}),
        );
    }
    let ignore_eos = body["ignore_eos"].as_bool().expect("validated boolean");
    let stream_mode = body["stream"].as_bool().expect("validated boolean");
    let id = format!("eval-apxinf-{}", REQUEST_ID.fetch_add(1, Ordering::Relaxed));
    let prefill_mode = if runtime.marlin_m64_enabled && prompt_tokens.len() >= 64 {
        Qwen35PrefillMode::MarlinM64
    } else {
        Qwen35PrefillMode::M8
    };

    if stream_mode {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )
        .map_err(|error| error.to_string())?;
        stream.flush().map_err(|error| error.to_string())?;
        let mut index = 0_usize;
        let generation = runtime.generate(
            &prompt_tokens,
            max_new_tokens,
            !ignore_eos,
            prefill_mode,
            |_delta, token| {
                let event = json!({
                    "type":"token",
                    "request_id":id,
                    "index":index,
                    "token_id":token
                });
                index += 1;
                write!(stream, "data: {}\n\n", event).map_err(|error| error.to_string())?;
                stream.flush().map_err(|error| error.to_string())
            },
        )?;
        let done = json!({
            "type":"done",
            "request_id":id,
            "usage":{
                "prompt_tokens":generation.prompt_tokens,
                "completion_tokens":generation.tokens.len(),
                "total_tokens":generation.prompt_tokens+generation.tokens.len()
            },
            "server_timing":{
                "prefill_s":generation.prefill_seconds,
                "first_token_s":generation.first_token_seconds,
                "decode_s":generation.decode_seconds
            }
        });
        write!(stream, "data: {}\n\ndata: [DONE]\n\n", done).map_err(|error| error.to_string())?;
        stream.flush().map_err(|error| error.to_string())?;
        log_generation(&id, &generation);
        Ok(())
    } else {
        let generation = runtime.generate(
            &prompt_tokens,
            max_new_tokens,
            !ignore_eos,
            prefill_mode,
            |_delta, _token| Ok(()),
        )?;
        let response = json!({
            "type":"result",
            "request_id":id,
            "output_ids":generation.tokens,
            "usage":{
                "prompt_tokens":generation.prompt_tokens,
                "completion_tokens":generation.tokens.len(),
                "total_tokens":generation.prompt_tokens+generation.tokens.len()
            },
            "server_timing":{
                "prefill_s":generation.prefill_seconds,
                "first_token_s":generation.first_token_seconds,
                "decode_s":generation.decode_seconds
            }
        });
        log_generation(&id, &generation);
        send_json(stream, 200, &response)
    }
}

fn handle_chat(runtime: &NativeRuntime, stream: &mut TcpStream, raw: &[u8]) -> Result<(), String> {
    let body: Value = match serde_json::from_slice(raw) {
        Ok(body) => body,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":format!("invalid JSON: {error}"),"type":"invalid_request"}}),
            )
        }
    };
    let image_request = match parse_image_chat_input(&body) {
        Ok(request) => request,
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":error,"type":"invalid_request"}}),
            )
        }
    };
    if let Some(image_request) = image_request {
        if runtime.vision.is_none() {
            return send_json(
                stream,
                501,
                &json!({"error":{"message":"native multimodal is not enabled","type":"unsupported_capability"}}),
            );
        }
        if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":"native multimodal v1 requires stream=false","type":"invalid_request"}}),
            );
        }
        if body
            .get("temperature")
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            != 0.0
        {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":"native multimodal v1 requires temperature=0","type":"invalid_request"}}),
            );
        }
        let python = runtime
            .processor_python
            .as_deref()
            .ok_or_else(|| "multimodal processor Python is unavailable".to_string())?;
        let prepared = preprocess_image_data_url(
            python,
            &runtime.model_dir,
            &image_request.data_url,
            &image_request.prompt,
        )?;
        let max_tokens = body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(16) as usize;
        let id = format!(
            "chatcmpl-apxinf-mm-{}",
            REQUEST_ID.fetch_add(1, Ordering::Relaxed)
        );
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let generation =
            runtime.generate_multimodal(&prepared, max_tokens, true, |_delta, _token| Ok(()))?;
        let response = json!({
            "id":id,"object":"chat.completion","created":created,"model":runtime.model_id,
            "choices":[{"index":0,"message":{"role":"assistant","content":generation.text},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":generation.prompt_tokens,"completion_tokens":generation.tokens.len(),"total_tokens":generation.prompt_tokens+generation.tokens.len()},
            "apxinf":generation_metadata(&generation)
        });
        log_generation(&id, &generation);
        return send_json(stream, 200, &response);
    }
    let prompt_tokens = match runtime.encode_messages(&body) {
        Ok(tokens) => tokens,
        Err(error) if error.contains("native multimodal is not ready") => {
            return send_json(
                stream,
                501,
                &json!({"error":{"message":error,"type":"unsupported_capability"}}),
            )
        }
        Err(error) => {
            return send_json(
                stream,
                400,
                &json!({"error":{"message":error,"type":"invalid_request"}}),
            )
        }
    };
    let prefill_mode = match body
        .get("apxinf_prefill_mode")
        .and_then(Value::as_str)
        .unwrap_or("m8")
    {
        "m8" => Qwen35PrefillMode::M8,
        "marlin-m64" if runtime.marlin_m64_enabled => Qwen35PrefillMode::MarlinM64,
        "marlin-m64" => {
            return Err(
                "marlin-m64 was requested but the server was not started with --enable-experimental-marlin-m64"
                    .into(),
            )
        }
        other => return Err(format!("unsupported apxinf_prefill_mode `{other}`")),
    };
    let max_tokens = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(16) as usize;
    let stream_mode = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let id = format!(
        "chatcmpl-apxinf-{}",
        REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    );
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    if stream_mode {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )
        .map_err(|error| error.to_string())?;
        stream.flush().map_err(|error| error.to_string())?;
        let generation = runtime.generate(
            &prompt_tokens,
            max_tokens,
            true,
            prefill_mode,
            |delta, _token| {
                let chunk = json!({
                    "id":id,"object":"chat.completion.chunk","created":created,
                    "model":runtime.model_id,
                    "choices":[{"index":0,"delta":{"content":delta},"finish_reason":Value::Null}],
                    "apxinf":{"prefill_mode":prefill_mode.as_str()}
                });
                write!(stream, "data: {}\n\n", chunk).map_err(|error| error.to_string())?;
                stream.flush().map_err(|error| error.to_string())
            },
        )?;
        let final_chunk = json!({
            "id":id,"object":"chat.completion.chunk","created":created,
            "model":runtime.model_id,
            "choices":[{"index":0,"delta":{},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":generation.prompt_tokens,"completion_tokens":generation.tokens.len(),"total_tokens":generation.prompt_tokens+generation.tokens.len()},
            "apxinf":generation_metadata(&generation)
        });
        write!(stream, "data: {}\n\ndata: [DONE]\n\n", final_chunk)
            .map_err(|error| error.to_string())?;
        stream.flush().map_err(|error| error.to_string())?;
        log_generation(&id, &generation);
        Ok(())
    } else {
        let generation = runtime.generate(
            &prompt_tokens,
            max_tokens,
            true,
            prefill_mode,
            |_delta, _token| Ok(()),
        )?;
        let response = json!({
            "id":id,"object":"chat.completion","created":created,"model":runtime.model_id,
            "choices":[{"index":0,"message":{"role":"assistant","content":generation.text},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":generation.prompt_tokens,"completion_tokens":generation.tokens.len(),"total_tokens":generation.prompt_tokens+generation.tokens.len()},
            "apxinf":generation_metadata(&generation)
        });
        log_generation(&id, &generation);
        send_json(stream, 200, &response)
    }
}

fn log_generation(id: &str, generation: &Generation) {
    println!(
        "{id}: prompt={} prefill_mode={} tiles=m64:{}/m8:{}/m1:{} prefill={:.3}s first_decode={:.3}s completion={} decode={:.3}s ({:.2} tok/s)",
        generation.prompt_tokens,
        generation.prefill_mode.as_str(),
        generation.marlin_tiles,
        generation.m8_tiles,
        generation.m1_tokens,
        generation.prefill_seconds,
        generation.first_token_seconds,
        generation.tokens.len(),
        generation.decode_seconds,
        generation.tokens.len() as f64 / generation.decode_seconds.max(1.0e-9),
    );
}

fn generation_metadata(generation: &Generation) -> Value {
    json!({
        "prefill_mode":generation.prefill_mode.as_str(),
        "marlin_m64_tiles":generation.marlin_tiles,
        "m8_tiles":generation.m8_tiles,
        "m1_tokens":generation.m1_tokens,
        "token_ids":generation.tokens,
    })
}

fn validate_processor_python(python: &str, model_dir: &Path) -> Result<(), String> {
    let script = r#"
import sys
from transformers import AutoProcessor
from PIL import Image
import numpy
AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(model_dir)
        .output()
        .map_err(|error| format!("launch multimodal processor Python `{python}`: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "multimodal processor Python validation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn parse_image_chat_input(body: &Value) -> Result<Option<ImageChatInput>, String> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "messages must be an array".to_string())?;
    let mut image_url = None;
    let mut prompt = String::new();
    let mut image_message_count = 0usize;
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| "message.role must be a string".to_string())?;
        let Some(parts) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        let mut message_has_image = false;
        let mut message_text = String::new();
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("image_url") => {
                    if image_url.is_some() {
                        return Err("native multimodal v1 accepts exactly one image".into());
                    }
                    let url = part
                        .get("image_url")
                        .and_then(|value| value.get("url"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| "image_url content part is missing url".to_string())?;
                    if !url.starts_with("data:image/png;base64,") {
                        return Err(
                            "native multimodal v1 requires a data:image/png;base64 URL".into()
                        );
                    }
                    image_url = Some(url.to_owned());
                    message_has_image = true;
                }
                Some("text") | Some("input_text") => {
                    message_text.push_str(
                        part.get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| "text content part is missing text".to_string())?,
                    );
                }
                Some(other) => {
                    return Err(format!(
                        "native multimodal v1 does not support content part `{other}`"
                    ))
                }
                None => return Err("content part is missing type".into()),
            }
        }
        if message_has_image {
            if role != "user" {
                return Err("native multimodal v1 requires the image in a user message".into());
            }
            image_message_count += 1;
            prompt = message_text;
        }
    }
    let Some(data_url) = image_url else {
        return Ok(None);
    };
    if image_message_count != 1 || prompt.trim().is_empty() {
        return Err("native multimodal v1 requires one image and non-empty text".into());
    }
    Ok(Some(ImageChatInput { data_url, prompt }))
}

fn preprocess_image_data_url(
    python: &str,
    model_dir: &Path,
    data_url: &str,
    prompt: &str,
) -> Result<PreparedImage, String> {
    let suffix = PROCESSOR_ID.fetch_add(1, Ordering::Relaxed);
    let pixel_path = std::env::temp_dir().join(format!(
        "apxinf-qwen35-mm-{}-{suffix}-pixels.npy",
        std::process::id()
    ));
    let metadata_path = std::env::temp_dir().join(format!(
        "apxinf-qwen35-mm-{}-{suffix}-metadata.json",
        std::process::id()
    ));
    let script = r#"
import base64
import io
import json
import sys
import numpy as np
from PIL import Image
from transformers import AutoProcessor

model_dir, data_url, prompt, pixel_path, metadata_path = sys.argv[1:]
prefix = "data:image/png;base64,"
if not data_url.startswith(prefix):
    raise ValueError("expected PNG data URL")
image = Image.open(io.BytesIO(base64.b64decode(data_url[len(prefix):], validate=True))).convert("RGB")
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ],
}]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
)
np.save(pixel_path, inputs["pixel_values"].cpu().numpy().astype(np.float32))
with open(metadata_path, "w") as output:
    json.dump({
        "grid": inputs["image_grid_thw"][0].cpu().tolist(),
        "tokens": inputs["input_ids"][0].cpu().tolist(),
        "modality_types": inputs["mm_token_type_ids"][0].cpu().tolist(),
    }, output)
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(model_dir)
        .arg(data_url)
        .arg(prompt)
        .arg(&pixel_path)
        .arg(&metadata_path)
        .output()
        .map_err(|error| format!("launch multimodal preprocessing: {error}"))?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&pixel_path);
        let _ = std::fs::remove_file(&metadata_path);
        return Err(format!(
            "multimodal preprocessing failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let result = (|| {
        let metadata: Value = serde_json::from_str(
            &std::fs::read_to_string(&metadata_path)
                .map_err(|error| format!("read processor metadata: {error}"))?,
        )
        .map_err(|error| format!("parse processor metadata: {error}"))?;
        let grid_values = metadata
            .get("grid")
            .and_then(Value::as_array)
            .ok_or_else(|| "processor metadata is missing grid".to_string())?;
        if grid_values.len() != 3 {
            return Err("processor grid must have three values".into());
        }
        let grid_thw = [
            parse_u32_value(&grid_values[0], "grid T")?,
            parse_u32_value(&grid_values[1], "grid H")?,
            parse_u32_value(&grid_values[2], "grid W")?,
        ];
        let tokens = parse_u32_array(&metadata, "tokens")?;
        let modality_types = metadata
            .get("modality_types")
            .and_then(Value::as_array)
            .ok_or_else(|| "processor metadata is missing modality_types".to_string())?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let number = value.as_u64().ok_or_else(|| {
                    format!("modality_types[{index}] must be a non-negative integer")
                })?;
                u8::try_from(number).map_err(|_| format!("modality_types[{index}] exceeds u8"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (pixel_shape, pixel_values) = read_npy_f32_to_bf16(&pixel_path)?;
        let pixel_values =
            Tensor::from_bf16(pixel_shape, &pixel_values).map_err(|error| error.to_string())?;
        Ok(PreparedImage {
            tokens,
            modality_types,
            pixel_values,
            grid_thw,
        })
    })();
    let _ = std::fs::remove_file(&pixel_path);
    let _ = std::fs::remove_file(&metadata_path);
    result
}

fn parse_u32_value(value: &Value, label: &str) -> Result<u32, String> {
    let number = value
        .as_u64()
        .ok_or_else(|| format!("{label} must be a non-negative integer"))?;
    u32::try_from(number).map_err(|_| format!("{label} exceeds u32"))
}

fn parse_u32_array(metadata: &Value, key: &str) -> Result<Vec<u32>, String> {
    metadata
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("processor metadata is missing {key}"))?
        .iter()
        .enumerate()
        .map(|(index, value)| parse_u32_value(value, &format!("{key}[{index}]")))
        .collect()
}

fn read_npy_f32_to_bf16(path: &Path) -> Result<(Vec<usize>, Vec<half::bf16>), String> {
    let mut file =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if buffer.len() < 10 || &buffer[..6] != b"\x93NUMPY" || buffer[6] != 1 {
        return Err(format!("{} is not a NumPy v1 array", path.display()));
    }
    let header_len = u16::from_le_bytes([buffer[8], buffer[9]]) as usize;
    let data_start = 10usize
        .checked_add(header_len)
        .ok_or_else(|| "NumPy header length overflow".to_string())?;
    if data_start > buffer.len() {
        return Err("NumPy header exceeds file length".into());
    }
    let header = std::str::from_utf8(&buffer[10..data_start])
        .map_err(|error| format!("invalid NumPy header: {error}"))?;
    if !header.contains("<f4") {
        return Err("processor pixel array is not little-endian f32".into());
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
        .map(|bytes| {
            half::bf16::from_f32(f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        })
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
    header[shape_start..shape_start + close_offset]
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid NumPy shape: {error}"))
        })
        .collect()
}

fn parse_text_content(content: &Value) -> Result<String, String> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_owned());
    }
    let parts = content
        .as_array()
        .ok_or_else(|| "message.content must be a string or text-part array".to_string())?;
    let mut text = String::new();
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("text") | Some("input_text") => {
                text.push_str(
                    part.get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "text content part is missing text".to_string())?,
                );
            }
            Some(other) => {
                return Err(format!(
                    "content part `{other}` is unsupported; native multimodal is not ready"
                ))
            }
            None => return Err("content part is missing type".into()),
        }
    }
    Ok(text)
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    const MAX_HEADER: usize = 64 * 1024;
    const MAX_BODY: usize = 8 * 1024 * 1024;
    let mut data = Vec::with_capacity(4096);
    let mut chunk = [0_u8; 4096];
    let header_end = loop {
        let read = stream.read(&mut chunk).map_err(|error| error.to_string())?;
        if read == 0 {
            return Err("connection closed before HTTP headers".into());
        }
        data.extend_from_slice(&chunk[..read]);
        if data.len() > MAX_HEADER {
            return Err("HTTP headers exceed 64 KiB".into());
        }
        if let Some(index) = data.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
    };
    let header = std::str::from_utf8(&data[..header_end])
        .map_err(|error| format!("HTTP header is not UTF-8: {error}"))?;
    let mut lines = header.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| "missing HTTP method".to_string())?
        .to_owned();
    let path = request_parts
        .next()
        .ok_or_else(|| "missing HTTP path".to_string())?
        .split('?')
        .next()
        .unwrap_or("/")
        .to_owned();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    if content_length > MAX_BODY {
        return Err("HTTP body exceeds 8 MiB".into());
    }
    while data.len() - header_end < content_length {
        let read = stream.read(&mut chunk).map_err(|error| error.to_string())?;
        if read == 0 {
            return Err("connection closed before HTTP body".into());
        }
        data.extend_from_slice(&chunk[..read]);
    }
    Ok(HttpRequest {
        method,
        path,
        body: data[header_end..header_end + content_length].to_vec(),
    })
}

fn send_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<(), String> {
    let body = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        501 => "Not Implemented",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .map_err(|error| error.to_string())?;
    stream.write_all(&body).map_err(|error| error.to_string())?;
    stream.flush().map_err(|error| error.to_string())
}

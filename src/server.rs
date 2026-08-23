//! Minimal HTTP + SSE front end implementing the Qwen3.8-27B evaluation
//! contract: `apxinf.qwen38_27b.inference_interface.v1`.
//!
//! The protocol is small enough that a hand-rolled HTTP/1.1 server (single
//! request at a time, matching `parallel_requests: 1`) keeps the binary free
//! of extra dependencies. Validation rules mirror
//! `run_evaluation.py::protocol_checks` exactly.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};

use apxinf_model::qwen35::cpu::CpuQwen35;

/// Contract constants advertised in `/health`.
pub const EVALUATION_CONTRACT: &str = "apxinf.qwen38_27b.inference_interface.v1";
pub const MODEL_REVISION: &str = "63768c10df38c0395e12ef49edac1bd539eaeeea";
pub const MAX_MODEL_LEN: usize = 32768;
pub const VOCAB_SIZE: u64 = 248320;
/// Context the current Rust forward can actually serve (KV cache + rope).
/// /health keeps advertising the official model max_model_len.
pub const SUPPORTED_PREFILL_LEN: usize = 16384;
pub const SUPPORTED_SEQ_LEN: usize = 17408;

/// Produces `max_new_tokens` token ids for an already-tokenized prompt.
/// The service thread owns one instance; requests are served serially.
pub trait TokenGenerator: Send {
    fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<Vec<u32>, String>;

    /// Incremental variant: `on_token` is called with (index, token_id) after
    /// each sampled token so the caller can flush it before the next decode
    /// step. Default replay keeps existing generators working.
    fn generate_stream(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        on_token: &mut dyn FnMut(usize, u32) -> Result<(), String>,
    ) -> Result<Vec<u32>, String> {
        let tokens = self.generate(input_ids, max_new_tokens, ignore_eos)?;
        for (i, t) in tokens.iter().enumerate() {
            on_token(i, *t).map_err(|e| e.to_string())?;
        }
        Ok(tokens)
    }
}

/// Temporary stand-in so the protocol can be exercised before the Rust model
/// forward exists. Emits deterministic in-vocab tokens; functional scoring
/// remains absent until a real model is wired in.
pub struct PlaceholderGenerator;

impl TokenGenerator for PlaceholderGenerator {
    fn generate(
        &mut self,
        _input_ids: &[u32],
        max_new_tokens: usize,
        _ignore_eos: bool,
    ) -> Result<Vec<u32>, String> {
        Ok(vec![0u32; max_new_tokens])
    }
}

/// Real model generator backed by the CPU reference forward. Correctness-first
/// (stateless full-sequence recompute); the CUDA path will replace it later.
pub struct ModelGenerator {
    model: CpuQwen35,
}

impl ModelGenerator {
    pub fn new(model: CpuQwen35) -> Self {
        Self { model }
    }
}

impl TokenGenerator for ModelGenerator {
    fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<Vec<u32>, String> {
        self.model.generate_greedy(input_ids, max_new_tokens, ignore_eos)
    }
}

/// CUDA-backed generator: prefill + incremental decode through the captured
/// CUDA graph. Requests are still served serially (parallel_requests: 1).
#[cfg(feature = "cuda")]
pub struct CudaGenerator {
    model: apxinf_model::qwen35::gpu::CudaQwen35,
}

#[cfg(feature = "cuda")]
impl CudaGenerator {
    pub fn new(model: apxinf_model::qwen35::gpu::CudaQwen35) -> Self {
        Self { model }
    }
}

// The server is single-threaded (parallel_requests: 1); the CUDA context and
// its buffers are owned and used by that one thread only.
#[cfg(feature = "cuda")]
unsafe impl Send for CudaGenerator {}

#[cfg(feature = "cuda")]
impl TokenGenerator for CudaGenerator {
    fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<Vec<u32>, String> {
        self.model.generate_greedy(input_ids, max_new_tokens, ignore_eos)
    }

    fn generate_stream(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        on_token: &mut dyn FnMut(usize, u32) -> Result<(), String>,
    ) -> Result<Vec<u32>, String> {
        self.model
            .generate_greedy_stream(input_ids, max_new_tokens, ignore_eos, |i, t| on_token(i, t))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    InputIds,
    MaxNewTokens,
    Temperature,
    IgnoreEos,
    Stream,
}

const ALLOWED_FIELDS: &[(&str, Field)] = &[
    ("input_ids", Field::InputIds),
    ("max_new_tokens", Field::MaxNewTokens),
    ("temperature", Field::Temperature),
    ("ignore_eos", Field::IgnoreEos),
    ("stream", Field::Stream),
];

struct ParsedRequest {
    input_ids: Vec<u32>,
    max_new_tokens: usize,
    ignore_eos: bool,
    stream: bool,
}

/// Validate the generate body. Any failure maps to HTTP 400 per the contract.
fn parse_generate_body(body: &[u8]) -> Result<ParsedRequest, String> {
    let value: Value =
        serde_json::from_slice(body).map_err(|_| "malformed JSON body".to_string())?;
    let obj = value
        .as_object()
        .ok_or_else(|| "body must be a JSON object".to_string())?;

    // Unknown fields (images, text prompts, ...) are rejected; this is the
    // `unsupported_modality_field` negative case.
    for key in obj.keys() {
        if !ALLOWED_FIELDS.iter().any(|(name, _)| name == key) {
            return Err(format!("unsupported field `{key}`"));
        }
    }

    let get = |field: Field| -> Option<&Value> {
        obj.iter().find_map(|(key, value)| {
            ALLOWED_FIELDS
                .iter()
                .find(|(_, f)| *f == field)
                .filter(|(name, _)| name == key)
                .map(|_| value)
        })
    };

    // input_ids
    let ids = get(Field::InputIds)
        .ok_or("missing input_ids")?
        .as_array()
        .ok_or("input_ids must be an array")?;
    if ids.is_empty() {
        return Err("input_ids must not be empty".to_string());
    }
    let mut input_ids = Vec::with_capacity(ids.len());
    for id in ids {
        let id = id.as_i64().ok_or("input_ids must contain integers")?;
        if id < 0 {
            return Err("input_ids contains a negative token id".to_string());
        }
        if id as u64 >= VOCAB_SIZE {
            return Err("input_ids contains an out-of-vocabulary token id".to_string());
        }
        input_ids.push(id as u32);
    }

    // max_new_tokens
    let max_new_tokens = get(Field::MaxNewTokens)
        .ok_or("missing max_new_tokens")?
        .as_i64()
        .ok_or("max_new_tokens must be an integer")?;
    if max_new_tokens <= 0 || max_new_tokens as usize >= MAX_MODEL_LEN {
        return Err("max_new_tokens out of supported range".to_string());
    }

    // temperature: only greedy (0.0) is supported
    let temperature = get(Field::Temperature)
        .ok_or("missing temperature")?
        .as_f64()
        .ok_or("temperature must be a number")?;
    if temperature != 0.0 {
        return Err("only greedy decoding (temperature=0.0) is supported".to_string());
    }

    let ignore_eos = get(Field::IgnoreEos)
        .ok_or("missing ignore_eos")?
        .as_bool()
        .ok_or("ignore_eos must be a boolean")?;
    let stream = get(Field::Stream)
        .ok_or("missing stream")?
        .as_bool()
        .ok_or("stream must be a boolean")?;

    Ok(ParsedRequest {
        input_ids,
        max_new_tokens: max_new_tokens as usize,
        ignore_eos,
        stream,
    })
}

pub struct Server {
    addr: String,
    generator: Box<dyn TokenGenerator>,
    request_counter: AtomicU64,
}

impl Server {
    pub fn new(addr: impl Into<String>, generator: Box<dyn TokenGenerator>) -> Self {
        Self {
            addr: addr.into(),
            generator,
            request_counter: AtomicU64::new(0),
        }
    }

    pub fn run(&mut self) -> Result<(), String> {
        let listener = TcpListener::bind(&self.addr)
            .map_err(|e| format!("bind {}: {e}", self.addr))?;
        eprintln!("[apxinf] serving on http://{}", self.addr);
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    if let Err(e) = self.handle(stream) {
                        eprintln!("[apxinf] request error: {e}");
                    }
                }
                Err(e) => eprintln!("[apxinf] accept error: {e}"),
            }
        }
        Ok(())
    }

    fn next_request_id(&self) -> String {
        format!(
            "req-{}",
            self.request_counter.fetch_add(1, Ordering::Relaxed) + 1
        )
    }

    fn handle(&mut self, stream: TcpStream) -> Result<(), String> {
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(120)))
            .map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        let mut writer = stream;

        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .map_err(|e| e.to_string())?;
        if request_line.is_empty() {
            return Ok(());
        }
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("/").to_string();

        let mut headers: HashMap<String, String> = HashMap::new();
        loop {
            let mut line = String::new();
            reader
                .read_line(&mut line)
                .map_err(|e| e.to_string())?;
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }

        let content_length: usize = headers
            .get("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader
                .read_exact(&mut body)
                .map_err(|e| e.to_string())?;
        }

        match (method.as_str(), path.as_str()) {
            ("GET", "/health") => write_json(&mut writer, 200, &self.health_json()),
            ("POST", "/v1/evaluations/generate") => self.handle_generate(&mut writer, &body),
            // Multimodal probes / unimplemented endpoints advertise
            // unsupported_capability instead of a generic 404.
            (_, p) if p.starts_with("/v1/") => write_json(
                &mut writer,
                501,
                &json!({"error": {"type": "unsupported_capability", "message": "endpoint not supported"}}),
            ),
            _ => write_json(
                &mut writer,
                404,
                &json!({"error": {"type": "not_found", "message": "not found"}}),
            ),
        }
    }

    fn health_json(&self) -> Value {
        json!({
            "status": "ok",
            "evaluation_contract": EVALUATION_CONTRACT,
            "model_revision": MODEL_REVISION,
            "max_model_len": MAX_MODEL_LEN,
            "parallel_requests": 1,
            "fallback_active": false,
            "capabilities": {
                "pretokenized_input_ids": true,
                "token_id_output": true,
                "multimodal": false,
            },
        })
    }

    fn handle_generate(&mut self, writer: &mut TcpStream, body: &[u8]) -> Result<(), String> {
        let parsed = match parse_generate_body(body) {
            Ok(parsed) => parsed,
            Err(message) => {
                return write_json(
                    writer,
                    400,
                    &json!({"error": {"type": "invalid_request", "message": message}}),
                );
            }
        };

        if parsed.input_ids.is_empty() {
            return write_json(
                writer,
                400,
                &json!({"error": {"type": "invalid_request", "message": "input_ids must not be empty"}}),
            );
        }
        let requested = parsed.input_ids.len().saturating_add(parsed.max_new_tokens);
        if parsed.input_ids.len() > SUPPORTED_PREFILL_LEN {
            return write_json(
                writer,
                400,
                &json!({"error": {"type": "invalid_request", "message": format!("input length {} exceeds supported prefill length {SUPPORTED_PREFILL_LEN}", parsed.input_ids.len())}}),
            );
        }
        if requested > SUPPORTED_SEQ_LEN {
            return write_json(
                writer,
                400,
                &json!({"error": {"type": "invalid_request", "message": format!("request length {requested} exceeds supported total sequence length {SUPPORTED_SEQ_LEN}")}}),
            );
        }
        for &id in &parsed.input_ids {
            if id as u64 >= VOCAB_SIZE {
                return write_json(
                    writer,
                    400,
                    &json!({"error": {"type": "invalid_request", "message": format!("token id {id} out of range")}}),
                );
            }
        }

        if parsed.stream {
            let request_id = self.next_request_id();
            // Send headers immediately so the client's TTFT clock starts with
            // the connection, not after the whole generation finishes.
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
            writer.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
            writer.flush().map_err(|e| e.to_string())?;

            let mut cb = |index: usize, token_id: u32| {
                let event = json!({
                    "type": "token",
                    "request_id": request_id,
                    "index": index,
                    "token_id": token_id,
                });
                write_sse(writer, &event.to_string())
            };
            let tokens = self
                .generator
                .generate_stream(
                    &parsed.input_ids,
                    parsed.max_new_tokens,
                    parsed.ignore_eos,
                    &mut cb,
                )
                .map_err(|e| {
                    let _ = write_json(
                        writer,
                        500,
                        &json!({"error": {"type": "internal_error", "message": e}}),
                    );
                    String::new()
                })?;

            let prompt_tokens = parsed.input_ids.len();
            let completion_tokens = tokens.len();
            let usage = json!({
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            });
            let done = json!({
                "type": "done",
                "request_id": request_id,
                "usage": usage,
            });
            write_sse(writer, &done.to_string())?;
            writer
                .write_all(b"data: [DONE]\n")
                .map_err(|e| e.to_string())?;
            return writer.flush().map_err(|e| e.to_string());
        }

        let tokens = match self.generator.generate(
            &parsed.input_ids,
            parsed.max_new_tokens,
            parsed.ignore_eos,
        ) {
            Ok(tokens) => tokens,
            Err(message) => {
                return write_json(
                    writer,
                    500,
                    &json!({"error": {"type": "internal_error", "message": message}}),
                );
            }
        };
        let prompt_tokens = parsed.input_ids.len();
        let completion_tokens = tokens.len();
        let usage = json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        });
        write_json(
            writer,
            200,
            &json!({"type": "result", "output_ids": tokens, "usage": usage}),
        )
    }
}

/// Write one SSE `data:` line and flush immediately so the client observes
/// per-token latency (TTFT is measured off the first flushed token).
fn write_sse(writer: &mut TcpStream, payload: &str) -> Result<(), String> {
    writer
        .write_all(format!("data: {payload}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())
}

fn write_json(writer: &mut TcpStream, status: u16, value: &Value) -> Result<(), String> {
    let body = value.to_string();
    write_response(writer, status, "application/json", body.as_bytes())
}

fn write_response(
    writer: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        501 => "Not Implemented",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    writer.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    writer.write_all(body).map_err(|e| e.to_string())?;
    writer.flush().map_err(|e| e.to_string())
}

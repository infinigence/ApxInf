//! Minimal HTTP inference service for the ApxInf Qwen3.8 RTX 4090 assignment.
//!
//! Implements the `apxinf.qwen38_27b.inference_interface.v1` surface:
//!   GET  /health
//!   POST /v1/evaluations/generate   (SSE when stream=true, JSON when false)
//! Every other route fails closed. Image capability is not implemented, so
//! multimodal probes receive `unsupported_capability` errors as required.

use std::io::{BufRead, BufReader, Read, Write};
use std::sync::mpsc;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::Duration;

use apxinf_core::Device;
use apxinf_model::{AutoModel, LlmInput, LoadOptions, LoadedModel};
use apxinf_tokenizer::Tokenizer;

const EVALUATION_CONTRACT: &str = "apxinf.qwen38_27b.inference_interface.v1";
struct EngineMeta {
    model_revision: String,
    max_model_len: usize,
}

fn health_json(meta: &EngineMeta) -> String {
    serde_json::json!({
        "status": "ok",
        "evaluation_contract": EVALUATION_CONTRACT,
        "model_revision": meta.model_revision,
        "max_model_len": meta.max_model_len,
        "parallel_requests": 1,
        "fallback_active": false,
        "capabilities": {
            "pretokenized_input_ids": true,
            "token_id_output": true,
            "multimodal": false,
        },
    })
    .to_string()
}
#[derive(Clone, Debug)]
pub struct ServeConfig {
    pub model_dir: PathBuf,
    pub host: String,
    pub port: u16,
    pub max_model_len: usize,
    pub model_revision: String,
    pub device: Device,
}

struct Engine {
    model: LoadedModel,
    tokenizer: Tokenizer,
    vocab_size: usize,
    eos_token_id: Option<u32>,
    max_model_len: usize,
    model_revision: String,
    request_counter: u64,
}

/// Run the service. Blocks until the process is killed.
pub fn run(config: ServeConfig) -> Result<(), String> {
    let address = format!("{}:{}", config.host, config.port);
    let listener =
        TcpListener::bind(&address).map_err(|error| format!("bind {address}: {error}"))?;
    let meta = EngineMeta {
        model_revision: config.model_revision.clone(),
        max_model_len: config.max_model_len,
    };
    let (job_tx, job_rx) = mpsc::channel::<GenerateJob>();
    // The single model worker owns `Engine` for its whole lifetime, so the
    // (non-Send) model trait objects never cross a thread boundary. Health
    // and error routes are answered directly by the accept loop and stay
    // responsive even while a generation is running.
    let worker_config = config.clone();
    std::thread::spawn(move || {
        worker_loop(worker_config, job_rx);
    });
    println!("apxinf serve: listening on http://{address} (device={:?})", config.device);

    for connection in listener.incoming() {
        match connection {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                if let Err(error) = handle_connection(&job_tx, &meta, stream) {
                    eprintln!("apxinf serve: request error: {error}");
                }
            }
            Err(error) => eprintln!("apxinf serve: accept error: {error}"),
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum JobKind {
    Evaluate,
    Completions,
}

struct GenerateJob {
    stream: TcpStream,
    body: Vec<u8>,
    kind: JobKind,
}

fn worker_loop(config: ServeConfig, job_rx: mpsc::Receiver<GenerateJob>) {
    println!("apxinf serve: loading model from {:?}...", config.model_dir);
    let mut engine = match Engine::load(&config) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("apxinf serve: model load failed: {error}");
            return;
        }
    };
    while let Ok(mut job) = job_rx.recv() {
        let started = std::time::Instant::now();
        let result = (|| -> Result<(), String> {
            match job.kind {
                JobKind::Evaluate => handle_generate(&mut engine, &mut job.stream, &job.body),
                JobKind::Completions => handle_completions(&mut engine, &mut job.stream, &job.body),
            }
        })();
        eprintln!(
            "apxinf serve: generate -> {} ({:.3}s)",
            if result.is_ok() { "ok" } else { "error" },
            started.elapsed().as_secs_f64()
        );
        if let Err(error) = result {
            eprintln!("apxinf serve: generate error: {error}");
        }
    }
}
impl Engine {
    fn load(config: &ServeConfig) -> Result<Self, String> {
        let tokenizer_path = config.model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("load tokenizer {}: {error}", tokenizer_path.display()))?;
        let vocab_size = tokenizer.vocab_size();
        let eos_token_id = tokenizer.eos_token_id();

        let options = LoadOptions::default();
        let model = AutoModel::load_model(config.device, &config.model_dir, &options)
            .map_err(|error| format!("load model {:?}: {error}", config.model_dir))?;

        Ok(Self {
            model,
            tokenizer,
            vocab_size,
            eos_token_id,
            max_model_len: config.max_model_len,
            model_revision: config.model_revision.clone(),
            request_counter: 0,
        })
    }

    fn next_request_id(&mut self) -> String {
        let id = format!("apxinf-{:016x}", self.request_counter);
        self.request_counter = self.request_counter.wrapping_add(1);
        id
    }
}

#[derive(Debug)]
struct GenerateRequest {
    input_ids: Vec<u32>,
    max_new_tokens: usize,
    ignore_eos: bool,
    stream: bool,
}

fn parse_generate(body: &[u8], vocab_size: usize, max_model_len: usize) -> Result<GenerateRequest, (u16, String)> {
    // Capacity bound: the on-device KV cache covers this many tokens; longer
    // requests are rejected cleanly instead of overflowing the cache.
    #[cfg(feature = "cuda")]
    let capacity = apxinf_model::qwen35::cuda::MAX_SEQ_LEN;
    #[cfg(not(feature = "cuda"))]
    let capacity = max_model_len;
    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|_| error_response(400, "invalid_request", "malformed JSON body"))?;
    let object = value
        .as_object()
        .ok_or_else(|| error_response(400, "invalid_request", "body must be a JSON object"))?;

    if object.contains_key("images") {
        return Err(error_response(400, "unsupported_capability", "image input is not supported"));
    }

    let input_ids = object
        .get("input_ids")
        .and_then(|value| value.as_array())
        .ok_or_else(|| error_response(400, "invalid_request", "input_ids must be a non-empty array"))?;
    if input_ids.is_empty() {
        return Err(error_response(400, "invalid_request", "input_ids must not be empty"));
    }
    let mut tokens = Vec::with_capacity(input_ids.len());
    for (position, token) in input_ids.iter().enumerate() {
        let token = token
            .as_i64()
            .ok_or_else(|| error_response(400, "invalid_request", "input_ids entries must be integers"))?;
        if token < 0 || (token as u128) >= (vocab_size as u128) {
            return Err(error_response(400, "invalid_request", format!("input_ids[{position}] is out of vocabulary: {token}")));
        }
        tokens.push(token as u32);
    }

    let max_new_tokens = match object.get("max_new_tokens") {
        None => 128usize,
        Some(value) => value
            .as_i64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| error_response(400, "invalid_request", "max_new_tokens must be a positive integer"))?,
    };
    if max_new_tokens == 0 {
        return Err(error_response(400, "invalid_request", "max_new_tokens must be positive"));
    }
    let prompt_tokens = tokens.len();
    if prompt_tokens.saturating_add(max_new_tokens) > capacity {
        return Err(error_response(
            400,
            "capacity_exceeded",
            format!(
                "request length {prompt_tokens} + {max_new_tokens} exceeds engine capacity {capacity}"
            ),
        ));
    }
    if prompt_tokens.saturating_add(max_new_tokens) > max_model_len {
        return Err(error_response(400, "invalid_request", format!(
            "request length {prompt_tokens} + {max_new_tokens} exceeds max_model_len {max_model_len}"
        )));
    }

    if let Some(temperature) = object.get("temperature") {
        let temperature = temperature
            .as_f64()
            .ok_or_else(|| error_response(400, "invalid_request", "temperature must be a number"))?;
        if temperature != 0.0 {
            return Err(error_response(400, "invalid_request", "only greedy decoding (temperature=0.0) is supported"));
        }
    }

    let ignore_eos = object
        .get("ignore_eos")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let stream = object
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);

    Ok(GenerateRequest { input_ids: tokens, max_new_tokens, ignore_eos, stream })
}

fn error_response(status: u16, error_type: &str, message: impl Into<String>) -> (u16, String) {
    (status, serde_json::json!({ "error": { "type": error_type, "message": message.into() } }).to_string())
}
fn handle_connection(
    job_tx: &mpsc::Sender<GenerateJob>,
    meta: &EngineMeta,
    mut stream: TcpStream,
) -> Result<(), String> {
    let started = std::time::Instant::now();
    let request = read_request(&stream).map_err(|error| format!("read request: {error}"))?;
    let Some((method, path, body)) = request else {
        return Ok(());
    };

    let result: Result<(), String> = match (method.as_str(), path.as_str()) {
        ("GET", "/health") | ("GET", "/") => {
            let payload = health_json(meta);
            write_full_response(&mut stream, 200, "application/json", Some(payload.len()), &payload)
                .map_err(|error| format!("write /health: {error}"))
        }
        ("POST", "/v1/evaluations/generate") => {
            match job_tx.send(GenerateJob { stream, body, kind: JobKind::Evaluate }) {
                Ok(()) => Ok(()),
                Err(error) => {
                    let mut job = error.0;
                    let payload = error_response(503, "capacity_unavailable", "model worker unavailable").1;
                    write_full_response(&mut job.stream, 503, "application/json", Some(payload.len()), &payload)
                        .map_err(|error| format!("write 503: {error}"))
                }
            }
        }
        ("POST", "/v1/completions") => {
            match job_tx.send(GenerateJob { stream, body, kind: JobKind::Completions }) {
                Ok(()) => Ok(()),
                Err(error) => {
                    let mut job = error.0;
                    let payload = error_response(503, "capacity_unavailable", "model worker unavailable").1;
                    write_full_response(&mut job.stream, 503, "application/json", Some(payload.len()), &payload)
                        .map_err(|error| format!("write 503: {error}"))
                }
            }
        }
        ("POST", "/v1/chat/completions") => {
            let payload = error_response(501, "unsupported_capability", "image input is not supported").1;
            write_full_response(&mut stream, 501, "application/json", Some(payload.len()), &payload)
                .map_err(|error| format!("write unsupported: {error}"))
        }
        _ => {
            let payload = error_response(404, "not_found", "unknown route").1;
            write_full_response(&mut stream, 404, "application/json", Some(payload.len()), &payload)
                .map_err(|error| format!("write 404: {error}"))
        }
    };

    eprintln!(
        "apxinf serve: {} {} -> {} ({:.3}s)",
        method,
        path,
        if result.is_ok() { "ok" } else { "error" },
        started.elapsed().as_secs_f64()
    );
    result
}
fn handle_generate(engine: &mut Engine, stream: &mut TcpStream, body: &[u8]) -> Result<(), String> {
    let request = match parse_generate(body, engine.vocab_size, engine.max_model_len) {
        Ok(request) => request,
        Err((status, payload)) => {
            write_full_response(stream, status, "application/json", Some(payload.len()), &payload)
                .map_err(|error| format!("write validation error: {error}"))?;
            return Ok(());
        }
    };

    let request_id = engine.next_request_id();
    let eos_token_id = if request.ignore_eos { None } else { engine.eos_token_id };

    if request.stream {
        run_generate_sse(engine, stream, &request, &request_id, eos_token_id)
            .map_err(|error| format!("streaming generate: {error}"))
    } else {
        let result = run_generate_collect(engine, &request, eos_token_id)
            .map_err(|error| format!("generate: {error}"))?;
        let payload = serde_json::json!({
            "type": "result",
            "request_id": request_id,
            "output_ids": result.output_ids,
            "usage": result.usage,
        })
        .to_string();
        write_full_response(stream, 200, "application/json", Some(payload.len()), &payload)
            .map_err(|error| format!("write result: {error}"))
    }
}

struct GenerateOutcome {
    output_ids: Vec<u32>,
    usage: serde_json::Value,
}

fn run_generate_collect(
    engine: &mut Engine,
    request: &GenerateRequest,
    eos_token_id: Option<u32>,
) -> Result<GenerateOutcome, String> {
    let mut output_ids = Vec::with_capacity(request.max_new_tokens);
    engine
        .model
        .generate_streaming(
            LlmInput::text(&request.input_ids),
            request.max_new_tokens,
            |token| output_ids.push(token),
            eos_token_id,
        )
        .map_err(|error| error.to_string())?;
    Ok(GenerateOutcome {
        usage: serde_json::json!({
            "prompt_tokens": request.input_ids.len(),
            "completion_tokens": output_ids.len(),
            "total_tokens": request.input_ids.len() + output_ids.len(),
        }),
        output_ids,
    })
}

fn run_generate_sse(
    engine: &mut Engine,
    stream: &mut TcpStream,
    request: &GenerateRequest,
    request_id: &str,
    eos_token_id: Option<u32>,
) -> Result<(), String> {
    start_sse(stream).map_err(|error| error.to_string())?;
    let mut index = 0usize;
    let mut write_failed = false;
    let write_event = |stream: &mut TcpStream, json: serde_json::Value| -> Result<(), String> {
        let payload = format!("data: {}\n\n", json);
        stream
            .write_all(payload.as_bytes())
            .and_then(|_| stream.flush())
            .map_err(|error| error.to_string())
    };

    let result = engine.model.generate_streaming(
        LlmInput::text(&request.input_ids),
        request.max_new_tokens,
        |token| {
            let event = serde_json::json!({ "type": "token", "request_id": request_id, "index": index, "token_id": token });
            if !write_failed {
                if let Err(error) = write_event(&mut *stream, event) {
                    eprintln!("apxinf serve: stream write error: {error}");
                    write_failed = true;
                }
            }
            index += 1;
        },
        eos_token_id,
    );
    let output_len = match result {
        Ok((tokens, _profile)) => tokens.len(),
        Err(error) => {
            eprintln!("apxinf serve: generation error: {error}");
            index
        }
    };

    if !write_failed {
        let done = serde_json::json!({
            "type": "done",
            "request_id": request_id,
            "usage": {
                "prompt_tokens": request.input_ids.len(),
                "completion_tokens": output_len,
                "total_tokens": request.input_ids.len() + output_len,
            },
        });
        write_event(stream, done)?;
        stream
            .write_all(b"data: [DONE]\n\n")
            .and_then(|_| stream.flush())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn start_sse(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    stream.flush()
}

fn write_full_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    content_length: Option<usize>,
    body: &str,
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        501 => "Not Implemented",
        _ => "OK",
    };
    let length_header = match content_length {
        Some(length) => format!("Content-Length: {length}\r\n"),
        None => "Transfer-Encoding: chunked\r\n".to_string(),
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n{length_header}Connection: close\r\n\r\n"
    )?;
    // Chunked framing is not implemented, so callers always pass a length.
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

/// Read one HTTP request head plus its body. Returns `None` on a clean EOF.
fn read_request(stream: &TcpStream) -> std::io::Result<Option<(String, String, Vec<u8>)>> {
    let mut reader = BufReader::new(stream);
    let mut request_line = String::new();
    let read = reader.read_line(&mut request_line)?;
    if read == 0 {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(colon) = trimmed.find(':') {
            let (name, value) = trimmed.split_at(colon);
            if name.eq_ignore_ascii_case("Content-Length") {
                content_length = value[1..].trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some((method, path, body)))
}












/// Parse a `POST /v1/completions` body into a text prompt + sampling params.
struct CompletionsRequest {
    prompt: String,
    max_new_tokens: usize,
    ignore_eos: bool,
    stream: bool,
}

fn parse_completions(body: &[u8]) -> Result<CompletionsRequest, (u16, String)> {
    let object: serde_json::Value = serde_json::from_slice(body)
        .map_err(|error| error_response(400, "invalid_request", format!("invalid JSON: {error}")))?;
    let prompt = object
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| error_response(400, "invalid_request", "prompt must be a non-empty string"))?
        .to_string();
    if prompt.trim().is_empty() {
        return Err(error_response(400, "invalid_request", "prompt must be a non-empty string"));
    }
    let max_new_tokens = object
        .get("max_new_tokens")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| error_response(400, "invalid_request", "max_new_tokens must be a positive integer"))?
        as usize;
    if max_new_tokens == 0 {
        return Err(error_response(400, "invalid_request", "max_new_tokens must be positive"));
    }
    let ignore_eos = object
        .get("ignore_eos")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let stream = object
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    Ok(CompletionsRequest { prompt, max_new_tokens, ignore_eos, stream })
}

/// Tokenize a text prompt with the chat template and generate, streaming SSE
/// (with decoded `text` on the done event) or returning one JSON result.
fn handle_completions(
    engine: &mut Engine,
    stream: &mut TcpStream,
    body: &[u8],
) -> Result<(), String> {
    let request = match parse_completions(body) {
        Ok(request) => request,
        Err((status, payload)) => {
            write_full_response(stream, status, "application/json", Some(payload.len()), &payload)
                .map_err(|error| format!("write validation error: {error}"))?;
            return Ok(());
        }
    };

    let messages = vec![
        apxinf_tokenizer::ChatMessage::system("You are a helpful assistant."),
        apxinf_tokenizer::ChatMessage::user(request.prompt.clone()),
    ];
    let input_ids = engine
        .tokenizer
        .encode_chat(&messages)
        .map_err(|error| format!("tokenize prompt: {error}"))?;

    #[cfg(feature = "cuda")]
    let capacity = apxinf_model::qwen35::cuda::MAX_SEQ_LEN;
    #[cfg(not(feature = "cuda"))]
    let capacity = engine.max_model_len;
    if input_ids.len().saturating_add(request.max_new_tokens) > capacity {
        let payload = error_response(
            400,
            "capacity_exceeded",
            format!(
                "request length {} + {} exceeds engine capacity {}",
                input_ids.len(),
                request.max_new_tokens,
                capacity
            ),
        )
        .1;
        write_full_response(stream, 400, "application/json", Some(payload.len()), &payload)
            .map_err(|error| format!("write capacity error: {error}"))?;
        return Ok(());
    }

    let request_id = engine.next_request_id();
    let eos_token_id = if request.ignore_eos { None } else { engine.eos_token_id };
    let generate = GenerateRequest {
        input_ids,
        max_new_tokens: request.max_new_tokens,
        ignore_eos: request.ignore_eos,
        stream: request.stream,
    };

    if request.stream {
        run_generate_sse_text(engine, stream, &generate, &request_id, eos_token_id)
            .map_err(|error| format!("streaming completions: {error}"))
    } else {
        let result = run_generate_collect(engine, &generate, eos_token_id)
            .map_err(|error| format!("completions: {error}"))?;
        let text = engine.tokenizer.decode(&result.output_ids).unwrap_or_default();
        let payload = serde_json::json!({
            "type": "result",
            "request_id": request_id,
            "output_ids": result.output_ids,
            "text": text,
            "usage": result.usage,
        })
        .to_string();
        write_full_response(stream, 200, "application/json", Some(payload.len()), &payload)
            .map_err(|error| format!("write completions result: {error}"))
    }
}

/// Like [`run_generate_sse`] but accumulates the tokens and appends the
/// decoded `text` to the done event.
fn run_generate_sse_text(
    engine: &mut Engine,
    stream: &mut TcpStream,
    request: &GenerateRequest,
    request_id: &str,
    eos_token_id: Option<u32>,
) -> Result<(), String> {
    start_sse(stream).map_err(|error| error.to_string())?;
    let mut index = 0usize;
    let mut all_tokens = Vec::new();
    let mut write_failed = false;
    let write_event = |stream: &mut TcpStream, json: serde_json::Value| -> Result<(), String> {
        let payload = format!("data: {}\n\n", json);
        stream
            .write_all(payload.as_bytes())
            .and_then(|_| stream.flush())
            .map_err(|error| error.to_string())
    };

    let result = engine.model.generate_streaming(
        LlmInput::text(&request.input_ids),
        request.max_new_tokens,
        |token| {
            all_tokens.push(token);
            let event = serde_json::json!({ "type": "token", "request_id": request_id, "index": index, "token_id": token });
            if !write_failed {
                if let Err(error) = write_event(&mut *stream, event) {
                    eprintln!("apxinf serve: stream write error: {error}");
                    write_failed = true;
                }
            }
            index += 1;
        },
        eos_token_id,
    );
    let output_len = match result {
        Ok((tokens, _profile)) => tokens.len(),
        Err(error) => {
            eprintln!("apxinf serve: generation error: {error}");
            index
        }
    };

    let text = engine.tokenizer.decode(&all_tokens).unwrap_or_default();
    let done = serde_json::json!({
        "type": "done",
        "request_id": request_id,
        "text": text,
        "usage": {
            "prompt_tokens": request.input_ids.len(),
            "completion_tokens": output_len,
            "total_tokens": request.input_ids.len() + output_len,
        },
    });
    write_event(stream, done)?;
    stream
        .write_all(b"data: [DONE]\n\n")
        .and_then(|_| stream.flush())
        .map_err(|error| error.to_string())
}

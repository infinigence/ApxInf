use std::convert::Infallible;
use std::future::IntoFuture;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use apxinf_model::GenerationOptions;
use axum::{
    body::{Body, Bytes},
    extract::{rejection::JsonRejection, DefaultBodyLimit, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::anthropic::{
    empty_options, response_content, stop_reason as anthropic_stop_reason, MessagesApiError,
    MessagesRequest, MessagesResponse, MessagesUsage,
};
use crate::engine::{ApxInfChatEngine, ApxInfEngineConfig, EngineRequest};
use crate::error::ApiError;
use crate::protocol::{
    AssistantMessage, ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
    ModelCard, ModelList, PromptTokensDetails, Usage,
};
use crate::tool_calling::{parse_qwen3_xml, validate_tool_choice, ToolRegistry};
use crate::worker::{EngineStreamEvent, WorkerClient};

static REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub struct ApiConfig {
    pub served_model_name: String,
    pub max_output_tokens: usize,
}

pub struct ServeConfig {
    pub host: IpAddr,
    pub port: u16,
    pub queue_capacity: usize,
    pub media_limits: crate::media::MediaLimits,
    pub log_timing: bool,
    pub api: ApiConfig,
    pub engine: ApxInfEngineConfig,
}

#[derive(Clone)]
struct AppState {
    api: ApiConfig,
    worker: WorkerClient,
    created: u64,
    media_limits: crate::media::MediaLimits,
    decode_slots: Arc<tokio::sync::Semaphore>,
}

pub fn router(api: ApiConfig, worker: WorkerClient) -> Router {
    router_with_limits(api, worker, crate::media::MediaLimits::default())
}

pub fn router_with_limits(
    api: ApiConfig,
    worker: WorkerClient,
    limits: crate::media::MediaLimits,
) -> Router {
    let state = AppState {
        api,
        worker,
        created: unix_seconds(),
        media_limits: limits,
        decode_slots: Arc::new(tokio::sync::Semaphore::new(2)),
    };
    Router::new()
        .route("/health", get(health))
        .route("/healthz", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .layer(DefaultBodyLimit::max(limits.body_bytes))
        .with_state(Arc::new(state))
}

pub async fn serve(config: ServeConfig) -> Result<(), String> {
    let address = SocketAddr::new(config.host, config.port);
    // Register handlers before starting model loading or accepting requests.
    let signal = shutdown_signal()?;
    config.media_limits.validate()?;
    eprintln!("serving limits: body_bytes={} image_bytes={} image_dimension={} decoded_pixels={} max_input_tokens={} max_output_tokens={} queue_capacity={} log_timing={}",
        config.media_limits.body_bytes,config.media_limits.image_bytes,config.media_limits.image_dimension,config.media_limits.decoded_pixels,
        config.engine.max_input_tokens,config.api.max_output_tokens,config.queue_capacity,config.log_timing);
    let engine = config.engine;
    let worker =
        WorkerClient::spawn_with_timing(config.queue_capacity, config.log_timing, move || {
            ApxInfChatEngine::load(engine)
                .map(|engine| Box::new(engine) as Box<dyn crate::ChatEngine>)
        });
    let app = router_with_limits(config.api, worker, config.media_limits);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .map_err(|error| format!("bind {address}: {error}"))?;
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = stop_rx.await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => {
            result.map_err(|error| format!("serve {address}: {error}"))
        }
        () = signal => {
            let _ = stop_tx.send(());
            // Bound HTTP draining so idle clients cannot prevent termination.
            match tokio::time::timeout(Duration::from_secs(5), &mut server).await {
                Ok(result) => result.map_err(|error| format!("serve {address}: {error}")),
                Err(_) => Ok(()),
            }
        }
    }
}

fn shutdown_signal() -> Result<impl std::future::Future<Output = ()>, String> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut terminate = signal(SignalKind::terminate()).map_err(|e| e.to_string())?;
        let mut interrupt = signal(SignalKind::interrupt()).map_err(|e| e.to_string())?;
        Ok(async move {
            tokio::select! {
                _ = terminate.recv() => {},
                _ = interrupt.recv() => {},
            }
        })
    }
    #[cfg(not(unix))]
    {
        Ok(async {
            let _ = tokio::signal::ctrl_c().await;
        })
    }
}

pub fn serve_blocking(config: ServeConfig) -> Result<(), String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("create server runtime: {error}"))?
        .block_on(serve(config))
}

#[derive(Serialize)]
struct Health<'a> {
    status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn health(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Health<'static>>) {
    let status = state.worker.status_message();
    let code = if state.worker.ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        code,
        Json(Health {
            status,
            error: state.worker.startup_error(),
        }),
    )
}

async fn models(State(state): State<Arc<AppState>>) -> Result<Json<ModelList>, ApiError> {
    if !state.worker.ready() {
        return Err(ApiError::unavailable(format!(
            "model is {}",
            state.worker.status_message()
        )));
    }
    Ok(Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.api.served_model_name.clone(),
            object: "model",
            created: state.created,
            owned_by: "apxinf",
        }],
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(mut request) = payload.map_err(|error| {
        ApiError::invalid(
            format!("invalid JSON request: {}", error.body_text()),
            Some("body".to_owned()),
        )
    })?;
    validate_request(&request, &state.api)?;
    let (messages, images) = decode_request_messages(&state, request.messages).await?;
    request.messages = messages;
    let registry = ToolRegistry::validate(&request.tools)?;
    let tool_choice = validate_tool_choice(request.tool_choice.as_ref(), &registry)?;
    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let parallel_tool_calls = request.parallel_tool_calls;
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or_else(|| {
            state
                .api
                .max_output_tokens
                .min(GenerationOptions::DEFAULT_MAX_NEW_TOKENS)
        });
    let engine_request = EngineRequest {
        messages: request.messages,
        images,
        tools: request.tools,
        tool_choice: tool_choice.clone(),
        chat_template_kwargs: request.chat_template_kwargs,
        enable_thinking: request.enable_thinking.unwrap_or(false),
        max_tokens,
        temperature: request.temperature,
        top_k: request.top_k,
        top_p: request.top_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        seed: request.seed,
        stop_after_tool_call: !parallel_tool_calls,
    };
    let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let id = format!("chatcmpl-{sequence:016x}");
    let created = unix_seconds();
    if stream {
        let receiver = state
            .worker
            .submit_stream_for_request(engine_request, Some(sequence))?;
        return Ok(streaming_response(
            receiver,
            registry,
            tool_choice,
            parallel_tool_calls,
            include_usage,
            sequence,
            id,
            created,
            state.api.served_model_name.clone(),
        ));
    }
    let output = state.worker.complete(engine_request).await?;
    let prepared = prepare_completion(
        output,
        &registry,
        &tool_choice,
        parallel_tool_calls,
        sequence,
    )?;
    Ok(Json(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: state.api.served_model_name.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: prepared.content,
                tool_calls: (!prepared.tool_calls.is_empty()).then_some(prepared.tool_calls),
            },
            finish_reason: prepared.finish_reason,
        }],
        usage: prepared.usage,
    })
    .into_response())
}

async fn messages(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Response, MessagesApiError> {
    let Json(request) = payload.map_err(|error| {
        MessagesApiError(ApiError::invalid(
            format!("invalid JSON request: {}", error.body_text()),
            Some("body".to_owned()),
        ))
    })?;
    let converted = request.into_openai()?;
    let mut openai_request = ChatCompletionRequest {
        model: converted.model,
        messages: converted.messages,
        tools: converted.tools,
        tool_choice: converted.tool_choice,
        parallel_tool_calls: converted.parallel_tool_calls,
        stream: converted.stream,
        stream_options: None,
        max_tokens: Some(converted.max_tokens),
        max_completion_tokens: None,
        temperature: converted.temperature,
        top_k: None,
        top_p: converted.top_p,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        n: None,
        stop: None,
        logprobs: None,
        response_format: None,
        enable_thinking: Some(converted.enable_thinking),
        disable_spec_decode: false,
        chat_template_kwargs: empty_options(),
    };
    validate_request(&openai_request, &state.api)?;
    let (messages, images) = decode_request_messages(&state, openai_request.messages).await?;
    openai_request.messages = messages;
    let registry = ToolRegistry::validate(&openai_request.tools)?;
    let tool_choice = validate_tool_choice(openai_request.tool_choice.as_ref(), &registry)?;
    let parallel_tool_calls = openai_request.parallel_tool_calls;
    let stream = openai_request.stream;
    let max_tokens = openai_request
        .max_tokens
        .expect("Anthropic max_tokens was validated as required");
    let engine_request = EngineRequest {
        messages: openai_request.messages,
        images,
        tools: openai_request.tools,
        tool_choice: tool_choice.clone(),
        chat_template_kwargs: openai_request.chat_template_kwargs,
        enable_thinking: openai_request.enable_thinking.unwrap_or(false),
        max_tokens,
        temperature: openai_request.temperature,
        top_k: openai_request.top_k,
        top_p: openai_request.top_p,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        stop_after_tool_call: !parallel_tool_calls,
    };
    let sequence = REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let id = format!("msg-{sequence:016x}");
    if stream {
        let receiver = state
            .worker
            .submit_stream_for_request(engine_request, Some(sequence))?;
        return Ok(anthropic_streaming_response(
            receiver,
            registry,
            tool_choice,
            parallel_tool_calls,
            sequence,
            id,
            state.api.served_model_name.clone(),
        ));
    }
    let output = state.worker.complete(engine_request).await?;
    let prepared = prepare_completion(
        output,
        &registry,
        &tool_choice,
        parallel_tool_calls,
        sequence,
    )?;
    let stop_reason = anthropic_stop_reason(&prepared.finish_reason).to_owned();
    let content = response_content(prepared.content, prepared.tool_calls)?;
    Ok(Json(MessagesResponse {
        id,
        kind: "message",
        role: "assistant",
        content,
        model: state.api.served_model_name.clone(),
        stop_reason,
        stop_sequence: None,
        usage: prepared.usage.into(),
    })
    .into_response())
}

struct PreparedCompletion {
    content: Option<String>,
    tool_calls: Vec<crate::protocol::ToolCall>,
    finish_reason: String,
    usage: Usage,
}

fn prepare_completion(
    output: crate::EngineOutput,
    registry: &ToolRegistry,
    tool_choice: &crate::tool_calling::ToolChoice,
    parallel_tool_calls: bool,
    sequence: u64,
) -> Result<PreparedCompletion, ApiError> {
    let mut parsed = parse_qwen3_xml(&output.text, registry, tool_choice)?;
    if !parallel_tool_calls && parsed.tool_calls.len() > 1 {
        return Err(ApiError::internal(
            "model emitted multiple calls while parallel_tool_calls was false",
        ));
    }
    for (index, call) in parsed.tool_calls.iter_mut().enumerate() {
        call.id = format!("call-{sequence:016x}-{index}");
    }
    let finish_reason = if !parsed.tool_calls.is_empty() {
        "tool_calls"
    } else if output.hit_length {
        "length"
    } else {
        "stop"
    };
    let usage = Usage {
        prompt_tokens: output.prompt_tokens,
        completion_tokens: output.completion_tokens,
        total_tokens: output.prompt_tokens + output.completion_tokens,
        prompt_tokens_details: Some(PromptTokensDetails {
            cached_tokens: output.reused_prompt_tokens,
        }),
    };
    Ok(PreparedCompletion {
        content: parsed.content,
        tool_calls: parsed.tool_calls,
        finish_reason: finish_reason.into(),
        usage,
    })
}

#[allow(clippy::too_many_arguments)]
fn streaming_response(
    mut receiver: mpsc::Receiver<Result<EngineStreamEvent, ApiError>>,
    registry: ToolRegistry,
    tool_choice: crate::tool_calling::ToolChoice,
    parallel_tool_calls: bool,
    include_usage: bool,
    sequence: u64,
    id: String,
    created: u64,
    model: String,
) -> Response {
    let (sender, body_receiver) = mpsc::channel::<Result<Bytes, Infallible>>(128);
    tokio::spawn(async move {
        let envelope = StreamEnvelope {
            id: &id,
            created,
            model: &model,
        };
        let mut state = ProtocolStreamState::new();
        let mut role_sent = false;

        loop {
            // Returning the HTTP response before inference starts lets a
            // client close the body while its request is queued. Dropping the
            // receiver makes the worker skip that job; closing it during
            // inference also stops producer backpressure promptly.
            let event = tokio::select! {
                _ = sender.closed() => return,
                event = receiver.recv() => event,
            };
            let Some(event) = event else {
                send_stream_error(
                    &sender,
                    ApiError::unavailable("model worker stopped before responding"),
                )
                .await;
                return;
            };
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    send_stream_error(&sender, error).await;
                    return;
                }
            };
            match event {
                EngineStreamEvent::TextDelta(delta) => {
                    let updates = match state.push(
                        &delta,
                        &registry,
                        &tool_choice,
                        parallel_tool_calls,
                        sequence,
                    ) {
                        Ok(updates) => updates,
                        Err(error) => {
                            send_stream_error(&sender, error).await;
                            return;
                        }
                    };
                    for update in updates {
                        if !role_sent {
                            if !send_stream_role(&sender, &envelope).await {
                                return;
                            }
                            role_sent = true;
                        }
                        if !send_stream_update(&sender, &envelope, update).await {
                            return;
                        }
                    }
                }
                EngineStreamEvent::Complete(output) => {
                    if state.raw != output.text {
                        send_stream_error(
                            &sender,
                            ApiError::internal(
                                "streamed text did not match the completed generation",
                            ),
                        )
                        .await;
                        return;
                    }
                    let prepared = match prepare_completion(
                        output,
                        &registry,
                        &tool_choice,
                        parallel_tool_calls,
                        sequence,
                    ) {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            send_stream_error(&sender, error).await;
                            return;
                        }
                    };
                    let updates = match state.finish(&prepared) {
                        Ok(updates) => updates,
                        Err(error) => {
                            send_stream_error(&sender, error).await;
                            return;
                        }
                    };
                    if !role_sent && !send_stream_role(&sender, &envelope).await {
                        return;
                    }
                    for update in updates {
                        if !send_stream_update(&sender, &envelope, update).await {
                            return;
                        }
                    }
                    if !send_sse_json(
                        &sender,
                        &envelope.chunk(json!([{
                            "index": 0,
                            "delta": {},
                            "logprobs": null,
                            "finish_reason": prepared.finish_reason
                        }])),
                    )
                    .await
                    {
                        return;
                    }
                    if include_usage
                        && !send_sse_json(
                            &sender,
                            &json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [],
                                "usage": prepared.usage,
                            }),
                        )
                        .await
                    {
                        return;
                    }
                    let _ = sender
                        .send(Ok(Bytes::from_static(b"data: [DONE]\n\n")))
                        .await;
                    return;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(ReceiverStream::new(body_receiver)))
        .expect("static SSE response headers are valid")
}

fn anthropic_streaming_response(
    mut receiver: mpsc::Receiver<Result<EngineStreamEvent, ApiError>>,
    registry: ToolRegistry,
    tool_choice: crate::tool_calling::ToolChoice,
    parallel_tool_calls: bool,
    sequence: u64,
    id: String,
    model: String,
) -> Response {
    let (sender, body_receiver) = mpsc::channel::<Result<Bytes, Infallible>>(128);
    tokio::spawn(async move {
        if !send_anthropic_sse(
            &sender,
            "message_start",
            &json!({
                "type":"message_start",
                "message":{
                    "id":id,
                    "type":"message",
                    "role":"assistant",
                    "content":[],
                    "model":model,
                    "stop_reason":null,
                    "stop_sequence":null,
                    "usage":{"input_tokens":0,"output_tokens":0},
                }
            }),
        )
        .await
        {
            return;
        }

        let mut protocol = ProtocolStreamState::new();
        let mut wire = AnthropicWireState::new();
        loop {
            let event = tokio::select! {
                _ = sender.closed() => return,
                event = receiver.recv() => event,
            };
            let Some(event) = event else {
                send_anthropic_stream_error(
                    &sender,
                    ApiError::unavailable("model worker stopped before responding"),
                )
                .await;
                return;
            };
            let event = match event {
                Ok(event) => event,
                Err(error) => {
                    send_anthropic_stream_error(&sender, error).await;
                    return;
                }
            };
            match event {
                EngineStreamEvent::TextDelta(delta) => {
                    let updates = match protocol.push(
                        &delta,
                        &registry,
                        &tool_choice,
                        parallel_tool_calls,
                        sequence,
                    ) {
                        Ok(updates) => updates,
                        Err(error) => {
                            send_anthropic_stream_error(&sender, error).await;
                            return;
                        }
                    };
                    if !wire.send_updates(&sender, updates).await {
                        return;
                    }
                }
                EngineStreamEvent::Complete(output) => {
                    if protocol.raw != output.text {
                        send_anthropic_stream_error(
                            &sender,
                            ApiError::internal(
                                "streamed text did not match the completed generation",
                            ),
                        )
                        .await;
                        return;
                    }
                    let prepared = match prepare_completion(
                        output,
                        &registry,
                        &tool_choice,
                        parallel_tool_calls,
                        sequence,
                    ) {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            send_anthropic_stream_error(&sender, error).await;
                            return;
                        }
                    };
                    let updates = match protocol.finish(&prepared) {
                        Ok(updates) => updates,
                        Err(error) => {
                            send_anthropic_stream_error(&sender, error).await;
                            return;
                        }
                    };
                    if !wire.send_updates(&sender, updates).await || !wire.close_text(&sender).await
                    {
                        return;
                    }
                    if !send_anthropic_sse(
                        &sender,
                        "message_delta",
                        &json!({
                            "type":"message_delta",
                            "delta":{
                                "stop_reason":anthropic_stop_reason(&prepared.finish_reason),
                                "stop_sequence":null,
                            },
                            "usage":MessagesUsage::from(prepared.usage),
                        }),
                    )
                    .await
                    {
                        return;
                    }
                    let _ = send_anthropic_sse(
                        &sender,
                        "message_stop",
                        &json!({"type":"message_stop"}),
                    )
                    .await;
                    return;
                }
            }
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(ReceiverStream::new(body_receiver)))
        .expect("static SSE response headers are valid")
}

struct AnthropicWireState {
    next_index: usize,
    text_index: Option<usize>,
    text_open: bool,
}

impl AnthropicWireState {
    fn new() -> Self {
        Self {
            next_index: 0,
            text_index: None,
            text_open: false,
        }
    }

    async fn send_updates(
        &mut self,
        sender: &mpsc::Sender<Result<Bytes, Infallible>>,
        updates: Vec<StreamUpdate>,
    ) -> bool {
        for update in updates {
            match update {
                StreamUpdate::Content(content) => {
                    let index = match self.text_index {
                        Some(index) => index,
                        None => {
                            let index = self.next_index;
                            self.next_index += 1;
                            self.text_index = Some(index);
                            self.text_open = true;
                            if !send_anthropic_sse(
                                sender,
                                "content_block_start",
                                &json!({
                                    "type":"content_block_start",
                                    "index":index,
                                    "content_block":{"type":"text","text":""},
                                }),
                            )
                            .await
                            {
                                return false;
                            }
                            index
                        }
                    };
                    if !send_anthropic_sse(
                        sender,
                        "content_block_delta",
                        &json!({
                            "type":"content_block_delta",
                            "index":index,
                            "delta":{"type":"text_delta","text":content},
                        }),
                    )
                    .await
                    {
                        return false;
                    }
                }
                StreamUpdate::ToolCall { call, .. } => {
                    if !self.close_text(sender).await {
                        return false;
                    }
                    let index = self.next_index;
                    self.next_index += 1;
                    if !send_anthropic_sse(
                        sender,
                        "content_block_start",
                        &json!({
                            "type":"content_block_start",
                            "index":index,
                            "content_block":{
                                "type":"tool_use",
                                "id":call.id,
                                "name":call.function.name,
                                "input":{},
                            },
                        }),
                    )
                    .await
                        || !send_anthropic_sse(
                            sender,
                            "content_block_delta",
                            &json!({
                                "type":"content_block_delta",
                                "index":index,
                                "delta":{
                                    "type":"input_json_delta",
                                    "partial_json":call.function.arguments,
                                },
                            }),
                        )
                        .await
                        || !send_anthropic_sse(
                            sender,
                            "content_block_stop",
                            &json!({"type":"content_block_stop","index":index}),
                        )
                        .await
                    {
                        return false;
                    }
                }
            }
        }
        true
    }

    async fn close_text(&mut self, sender: &mpsc::Sender<Result<Bytes, Infallible>>) -> bool {
        if !self.text_open {
            return true;
        }
        self.text_open = false;
        send_anthropic_sse(
            sender,
            "content_block_stop",
            &json!({
                "type":"content_block_stop",
                "index":self.text_index.expect("open text has an index"),
            }),
        )
        .await
    }
}

async fn send_anthropic_sse(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    event: &str,
    value: &Value,
) -> bool {
    let data = format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(value).expect("serde_json::Value is serializable")
    );
    sender.send(Ok(Bytes::from(data))).await.is_ok()
}

async fn send_anthropic_stream_error(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    error: ApiError,
) {
    let _ = send_anthropic_sse(
        sender,
        "error",
        &json!({
            "type":"error",
            "error":{"type":error.error_type,"message":error.message},
        }),
    )
    .await;
}

struct StreamEnvelope<'a> {
    id: &'a str,
    created: u64,
    model: &'a str,
}

impl StreamEnvelope<'_> {
    fn chunk(&self, choices: Value) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        })
    }
}

enum StreamUpdate {
    Content(String),
    ToolCall {
        index: usize,
        call: crate::protocol::ToolCall,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProtocolStreamMode {
    Undecided,
    Text,
    Tool,
}

struct ProtocolStreamState {
    raw: String,
    mode: ProtocolStreamMode,
    sent_tool_calls: usize,
    streamed_content: String,
}

impl ProtocolStreamState {
    fn new() -> Self {
        Self {
            raw: String::new(),
            mode: ProtocolStreamMode::Undecided,
            sent_tool_calls: 0,
            streamed_content: String::new(),
        }
    }

    fn push(
        &mut self,
        delta: &str,
        registry: &ToolRegistry,
        tool_choice: &crate::tool_calling::ToolChoice,
        parallel_tool_calls: bool,
        sequence: u64,
    ) -> Result<Vec<StreamUpdate>, ApiError> {
        const MARKER: &str = "<tool_call>";
        self.raw.push_str(delta);
        let mut updates = Vec::new();
        if self.mode != ProtocolStreamMode::Tool {
            if let Some(marker) = self.raw.find(MARKER) {
                self.mode = ProtocolStreamMode::Tool;
                let content = self.raw[..marker].to_owned();
                if !content.trim().is_empty() {
                    if let Some(content) = self.content_update(&content)? {
                        updates.push(StreamUpdate::Content(content));
                    }
                }
            } else if self.mode == ProtocolStreamMode::Undecided
                && MARKER.starts_with(self.raw.trim_start())
            {
                return Ok(Vec::new());
            } else {
                self.mode = ProtocolStreamMode::Text;
                let content = self.raw.as_str();
                let marker_prefix = (1..MARKER.len())
                    .rev()
                    .find(|length| content.ends_with(&MARKER[..*length]))
                    .unwrap_or(0);
                let stable_end = content.len() - marker_prefix;
                let stable = content[..stable_end].to_owned();
                if let Some(content) = self.content_update(&stable)? {
                    updates.push(StreamUpdate::Content(content));
                }
                return Ok(updates);
            }
        }

        let Some(end) = self.raw.rfind("</tool_call>") else {
            return Ok(updates);
        };
        let end = end + "</tool_call>".len();
        let mut parsed = parse_qwen3_xml(&self.raw[..end], registry, tool_choice)?;
        if !parallel_tool_calls && parsed.tool_calls.len() > 1 {
            return Err(ApiError::internal(
                "model emitted multiple calls while parallel_tool_calls was false",
            ));
        }
        if let Some(content) = parsed.content.take() {
            if let Some(content) = self.content_update(&content)? {
                updates.push(StreamUpdate::Content(content));
            }
        }
        for (index, mut call) in parsed
            .tool_calls
            .into_iter()
            .enumerate()
            .skip(self.sent_tool_calls)
        {
            call.id = format!("call-{sequence:016x}-{index}");
            updates.push(StreamUpdate::ToolCall { index, call });
            self.sent_tool_calls += 1;
        }
        Ok(updates)
    }

    fn finish(&mut self, prepared: &PreparedCompletion) -> Result<Vec<StreamUpdate>, ApiError> {
        let mut updates = Vec::new();
        if let Some(content) = prepared.content.as_deref() {
            if let Some(content) = self.content_update(content)? {
                updates.push(StreamUpdate::Content(content));
            }
        }
        for (index, call) in prepared
            .tool_calls
            .iter()
            .cloned()
            .enumerate()
            .skip(self.sent_tool_calls)
        {
            updates.push(StreamUpdate::ToolCall { index, call });
        }
        Ok(updates)
    }

    fn content_update(&mut self, content: &str) -> Result<Option<String>, ApiError> {
        let delta = content
            .strip_prefix(&self.streamed_content)
            .ok_or_else(|| {
                ApiError::internal("streamed content changed before generation completed")
            })?
            .to_owned();
        self.streamed_content = content.to_owned();
        Ok((!delta.is_empty()).then_some(delta))
    }
}

async fn send_stream_role(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    envelope: &StreamEnvelope<'_>,
) -> bool {
    send_sse_json(
        sender,
        &envelope.chunk(json!([{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "logprobs": null,
            "finish_reason": null
        }])),
    )
    .await
}

async fn send_stream_update(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    envelope: &StreamEnvelope<'_>,
    update: StreamUpdate,
) -> bool {
    let delta = match update {
        StreamUpdate::Content(content) => json!({"content": content}),
        StreamUpdate::ToolCall { index, call } => json!({
            "tool_calls": [{
                "index": index,
                "id": call.id,
                "type": call.kind,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments
                }
            }]
        }),
    };
    send_sse_json(
        sender,
        &envelope.chunk(json!([{
            "index": 0,
            "delta": delta,
            "logprobs": null,
            "finish_reason": null
        }])),
    )
    .await
}

async fn send_sse_json(sender: &mpsc::Sender<Result<Bytes, Infallible>>, value: &Value) -> bool {
    let data = format!(
        "data: {}\n\n",
        serde_json::to_string(value).expect("serde_json::Value is serializable")
    );
    sender.send(Ok(Bytes::from(data))).await.is_ok()
}

async fn send_stream_error(sender: &mpsc::Sender<Result<Bytes, Infallible>>, error: ApiError) {
    let value = json!({
        "error": {
            "message": error.message,
            "type": error.error_type,
            "param": error.param,
            "code": error.code,
        }
    });
    if send_sse_json(sender, &value).await {
        let _ = sender
            .send(Ok(Bytes::from_static(b"data: [DONE]\n\n")))
            .await;
    }
}

async fn decode_request_messages(
    state: &AppState,
    messages: Vec<crate::protocol::ChatCompletionMessage>,
) -> Result<
    (
        Vec<crate::protocol::ChatCompletionMessage>,
        Vec<apxinf_model::vision_preprocess::DecodedImage>,
    ),
    ApiError,
> {
    if let Some(contract) = state.worker.vision_contract() {
        let count = messages
            .iter()
            .filter_map(|m| m.content.as_ref().and_then(Value::as_array))
            .flatten()
            .filter(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))
            .count();
        if count > contract.max_images {
            return Err(ApiError::invalid(
                format!(
                    "request has {count} images; model '{}' declares max_images={}",
                    contract.model, contract.max_images
                ),
                Some("messages".into()),
            ));
        }
    }
    let has_images = messages.iter().any(|m| {
        m.content
            .as_ref()
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts
                    .iter()
                    .any(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))
            })
    });
    if !has_images {
        return crate::media::decode_messages(messages, state.media_limits);
    }
    // A canceled HTTP task leaves its permit with the blocking decoder until it
    // finishes. Decoding never consumes the model's dedicated thread.
    let permit = state
        .decode_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::queue_full())?;
    let limits = state.media_limits;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        crate::media::decode_messages(messages, limits)
    })
    .await
    .map_err(|e| ApiError::internal(format!("image decoding failed: {e}")))?
}

fn validate_request(request: &ChatCompletionRequest, config: &ApiConfig) -> Result<(), ApiError> {
    if request
        .model
        .as_ref()
        .is_some_and(|model| model != &config.served_model_name)
    {
        return Err(ApiError::invalid(
            format!(
                "model must be '{}' for this deployment",
                config.served_model_name
            ),
            Some("model".to_owned()),
        ));
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid(
            "messages must not be empty",
            Some("messages".to_owned()),
        ));
    }
    if request.stream_options.is_some() && !request.stream {
        return Err(ApiError::invalid(
            "stream_options requires stream=true",
            Some("stream_options".to_owned()),
        ));
    }
    if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
        return Err(ApiError::invalid(
            "max_tokens and max_completion_tokens cannot both be set",
            Some("max_tokens".to_owned()),
        ));
    }
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) || max_tokens.is_some_and(|value| value > config.max_output_tokens) {
        return Err(ApiError::invalid(
            format!(
                "requested output tokens must be in 1..={}",
                config.max_output_tokens
            ),
            Some("max_tokens".to_owned()),
        ));
    }
    if request.disable_spec_decode {
        return Err(ApiError::invalid(
            "disable_spec_decode requires a model runtime extension not included in this server",
            Some("disable_spec_decode".to_owned()),
        ));
    }
    if request.n.unwrap_or(1) != 1 {
        return Err(ApiError::invalid(
            "only n=1 is supported",
            Some("n".to_owned()),
        ));
    }
    if request.stop.is_some() {
        return Err(ApiError::invalid(
            "stop strings are not implemented",
            Some("stop".to_owned()),
        ));
    }
    if request.logprobs == Some(true) {
        return Err(ApiError::invalid(
            "wire logprobs are not implemented",
            Some("logprobs".to_owned()),
        ));
    }
    if let Some(response_format) = &request.response_format {
        if response_format
            .get("type")
            .and_then(serde_json::Value::as_str)
            != Some("text")
        {
            return Err(ApiError::invalid(
                "only response_format.type='text' is supported",
                Some("response_format".to_owned()),
            ));
        }
    }
    validate_finite_range(request.temperature, 0.0, 2.0, "temperature")?;
    if request.top_k.is_some_and(|top_k| top_k < 0) {
        return Err(ApiError::invalid(
            "top_k must be non-negative",
            Some("top_k".to_owned()),
        ));
    }
    validate_finite_range(request.top_p, f32::MIN_POSITIVE, 1.0, "top_p")?;
    validate_finite_range(request.frequency_penalty, -2.0, 2.0, "frequency_penalty")?;
    validate_finite_range(request.presence_penalty, -2.0, 2.0, "presence_penalty")?;
    Ok(())
}

fn validate_finite_range(
    value: Option<f32>,
    minimum: f32,
    maximum: f32,
    name: &str,
) -> Result<(), ApiError> {
    if value.is_some_and(|value| !value.is_finite() || value < minimum || value > maximum) {
        return Err(ApiError::invalid(
            format!("{name} must be finite and in [{minimum}, {maximum}]"),
            Some(name.to_owned()),
        ));
    }
    Ok(())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

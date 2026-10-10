//! Public API profiles for local text and tool requests.
use crate::{
    output::{anthropic_blocks, Part},
    supervisor::{ApiError, Begin, Event as ModelEvent, Request, Service},
};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::Stream;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::mpsc;

#[derive(Clone, Copy, PartialEq)]
enum Api {
    Anthropic,
    OpenAi,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(json!({"type":"error","error":{"type":public_error_type(self.status),"code":self.code,"message":self.message}}))).into_response()
    }
}

fn public_error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ => "api_error",
    }
}

pub fn router(service: Arc<Service>) -> Router {
    Router::new()
        .route(
            "/healthz",
            get(|| async { Json(json!({"status":"alive"})) }),
        )
        .route("/readyz", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/chat/completions", post(chat))
        .route("/metrics", get(metrics))
        .layer(DefaultBodyLimit::max(1_000_000))
        .with_state(service)
}
async fn ready(State(service): State<Arc<Service>>) -> Response {
    let (admission_allowed, host_pressure) = service.admission_snapshot();
    let worker_available = service.available.load(Ordering::Acquire);
    let status = if worker_available && admission_allowed {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(json!({"ready":status==StatusCode::OK,"worker_available":worker_available,"host_pressure":host_pressure,"model":service.model_id,"worker":service.ready_snapshot(),"profile":"apxinf-serial-v0.1","public_capabilities":{"anthropic_messages":true,"openai_chat_completions":true,"tools":"qwen_xml_or_json","tool_choice":["auto","none"],"thinking":false,"sampling":false,"images":false,"sessions":false,"batching":false,"prompt_cache":false,"output_effort":"advisory_only"}}))).into_response()
}
async fn models(State(service): State<Arc<Service>>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":service.model_id,"object":"model","created":0,"owned_by":"apxinf","model_revision":service.ready["model_revision"]}],"has_more":false,"first_id":service.model_id,"last_id":service.model_id}),
    )
}
async fn metrics(State(service): State<Arc<Service>>) -> impl IntoResponse {
    (
        [("content-type", "text/plain; version=0.0.4")],
        service.metrics_text(),
    )
}
async fn messages(
    State(service): State<Arc<Service>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(service, headers, body, Api::Anthropic, false)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}
async fn chat(State(service): State<Arc<Service>>, headers: HeaderMap, body: Bytes) -> Response {
    handle(service, headers, body, Api::OpenAi, false)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}
async fn count_tokens(
    State(service): State<Arc<Service>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(service, headers, body, Api::Anthropic, true)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

struct CancelGuard(Arc<AtomicBool>);
impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

async fn handle(
    service: Arc<Service>,
    _headers: HeaderMap,
    body: Bytes,
    api: Api,
    count_only: bool,
) -> Result<Response, ApiError> {
    let value = crate::contracts::parse_document(&body)
        .map_err(|e| ApiError::invalid(format!("The request is not valid JSON: {e}")))?;
    let (request, stream) = normalize(&value, api, count_only, &service)?;
    let preparation = json!({"protocol":crate::supervisor::PROTOCOL,"kind":"prepare_input","worker_epoch":service.ready["worker_epoch"],"command_id":uuid::Uuid::new_v4().to_string(),"model_revision":service.ready["model_revision"],"messages":request.messages,"tools":request.tools,"template_options":{"enable_thinking":false}});
    crate::contracts::encode_frame(&preparation)
        .map_err(|e| ApiError::invalid(format!("The normalized input is invalid: {e}")))?;
    let ticket = service.enqueue(request)?;
    let guard = CancelGuard(ticket.cancel);
    let begin = ticket.begin.await.map_err(|_| {
        ApiError::new(
            503,
            "worker_lost",
            "The worker stopped before request admission.",
        )
    })??;
    if count_only {
        return Ok(Json(json!({"input_tokens":begin.input_tokens})).into_response());
    }
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", begin.request_id.parse().unwrap());
    headers.insert(
        "x-apxinf-model-revision",
        service.ready["model_revision"]
            .as_str()
            .unwrap_or("")
            .parse()
            .unwrap(),
    );
    headers.insert("x-apxinf-profile", "apxinf-serial-v0.1".parse().unwrap());
    if stream {
        let include_usage = value["stream_options"]["include_usage"] == true;
        let stream = PublicStream::new(
            ticket.output,
            guard,
            api,
            begin,
            service.model_id.clone(),
            include_usage,
        );
        Ok((
            headers,
            Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(5))),
        )
            .into_response())
    } else {
        let mut receiver = ticket.output;
        let mut parts = Vec::new();
        while let Some(event) = receiver.recv().await {
            match event {
                ModelEvent::Part(part) => parts.push(part),
                ModelEvent::Error(error) => return Err(error),
                ModelEvent::Done {
                    cause,
                    input_tokens,
                    output_tokens,
                    matched_stop,
                } => {
                    return Ok((
                        headers,
                        Json(response_body(
                            api,
                            &begin,
                            &service.model_id,
                            &parts,
                            &cause,
                            input_tokens,
                            output_tokens,
                            matched_stop,
                        )),
                    )
                        .into_response());
                }
            }
        }
        drop(guard);
        Err(ApiError::new(
            503,
            "worker_lost",
            "The output stream ended without a terminal result.",
        ))
    }
}

fn unknown(value: &Value, allowed: &[&str], label: &str) -> Result<(), ApiError> {
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::invalid(format!("{label} must be an object.")))?;
    for field in object.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                format!("{label} field {field} is unsupported."),
            ));
        }
    }
    Ok(())
}
fn text(value: &Value, label: &str) -> Result<String, ApiError> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| ApiError::invalid(format!("{label} must be a string.")))
}
fn array<'a>(value: &'a Value, label: &str) -> Result<&'a Vec<Value>, ApiError> {
    value
        .as_array()
        .ok_or_else(|| ApiError::invalid(format!("{label} must be an array.")))
}

fn normalize(
    value: &Value,
    api: Api,
    count_only: bool,
    service: &Service,
) -> Result<(Request, bool), ApiError> {
    let allowed = if api == Api::Anthropic {
        vec![
            "model",
            "max_tokens",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "stream",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "metadata",
            "thinking",
            "output_config",
            "service_tier",
            "cache_control",
        ]
    } else {
        vec![
            "model",
            "max_tokens",
            "max_completion_tokens",
            "messages",
            "tools",
            "tool_choice",
            "stream",
            "temperature",
            "top_p",
            "stop",
            "stream_options",
            "user",
        ]
    };
    unknown(value, &allowed, "Request")?;
    if value["model"].as_str() != Some(&service.model_id) {
        return Err(ApiError::new(
            404,
            "model_not_found",
            "Select the model listed by /v1/models.",
        ));
    }
    for (key, expected) in [("temperature", 0.0), ("top_p", 1.0)] {
        if let Some(v) = value.get(key) {
            if v.as_f64() != Some(expected) {
                return Err(ApiError::new(
                    400,
                    "unsupported_feature",
                    format!("This profile requires {key}={expected}."),
                ));
            }
        }
    }
    if value.get("top_k").is_some() {
        return Err(ApiError::new(
            400,
            "unsupported_feature",
            "This profile does not support top_k.",
        ));
    }
    if let Some(thinking) = value.get("thinking") {
        if thinking != &json!({"type":"disabled"}) {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "This profile requires thinking to be disabled.",
            ));
        }
    }
    if let Some(output) = value.get("output_config") {
        unknown(output, &["effort"], "output_config")?;
        if let Some(effort) = output.get("effort") {
            if !["low", "medium", "high", "xhigh", "max"].contains(&effort.as_str().unwrap_or("")) {
                return Err(ApiError::invalid("The output effort value is invalid."));
            }
        }
    }
    if let Some(tier) = value.get("service_tier") {
        if tier != "auto" && tier != "standard_only" {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "This profile supports only the standard service tier.",
            ));
        }
    }
    if let Some(options) = value.get("stream_options") {
        unknown(options, &["include_usage"], "stream_options")?;
        if options
            .get("include_usage")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(ApiError::invalid("include_usage must be a Boolean."));
        }
    }
    let stream = match value.get("stream") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| ApiError::invalid("stream must be a Boolean."))?,
    };
    if !stream && value.get("stream_options").is_some() {
        return Err(ApiError::invalid("stream_options requires stream=true."));
    }
    if value.get("max_tokens").is_some() && value.get("max_completion_tokens").is_some() {
        return Err(ApiError::invalid(
            "Set either max_tokens or max_completion_tokens.",
        ));
    }
    let max_value = value
        .get("max_tokens")
        .or_else(|| value.get("max_completion_tokens"));
    let max_tokens = if count_only {
        0
    } else {
        max_value
            .map_or(Some(512), Value::as_u64)
            .ok_or_else(|| ApiError::invalid("max_tokens must be a non-negative integer."))?
            as usize
    };
    if max_tokens as u64
        > service.ready["capabilities"]["max_output_tokens"]
            .as_u64()
            .unwrap_or(0)
    {
        return Err(ApiError::invalid(
            "max_tokens exceeds the deployment output limit.",
        ));
    }
    let stops = value.get(if api == Api::Anthropic {
        "stop_sequences"
    } else {
        "stop"
    });
    let stops = match stops {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) if api == Api::OpenAi => vec![s.clone()],
        Some(v) => array(v, "Stop sequences")?
            .iter()
            .map(|v| text(v, "Stop sequence"))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if stops.len() > 16 || stops.iter().any(|s| s.is_empty() || s.len() > 1024) {
        return Err(ApiError::invalid(
            "Use at most 16 non-empty stop sequences of at most 1024 bytes.",
        ));
    }
    let mut tools = Vec::new();
    if let Some(items) = value.get("tools") {
        for tool in array(items, "Tools")? {
            if api == Api::Anthropic {
                unknown(
                    tool,
                    &[
                        "name",
                        "description",
                        "input_schema",
                        "cache_control",
                        "type",
                    ],
                    "Tool",
                )?;
                if tool.get("type").is_some_and(|v| v != "custom") {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "Server tools are unsupported.",
                    ));
                }
                let name = text(&tool["name"], "Tool name")?;
                if !tool["input_schema"].is_object() {
                    return Err(ApiError::invalid("Tool input_schema must be an object."));
                }
                let mut function = json!({"name":name,"parameters":tool["input_schema"]});
                if let Some(description) = tool.get("description") {
                    function["description"] = json!(text(description, "Tool description")?);
                }
                tools.push(json!({"type":"function","function":function}));
            } else {
                if tool["function"]["strict"] == true {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "This profile does not provide strict schema generation.",
                    ));
                }
                tools.push(tool.clone());
            }
        }
    }
    if tools.len() > 256 {
        return Err(ApiError::invalid("The tool count exceeds 256."));
    }
    if let Some(choice) = value.get("tool_choice") {
        let kind = if api == Api::Anthropic {
            unknown(
                choice,
                &["type", "disable_parallel_tool_use"],
                "tool_choice",
            )?;
            choice["type"].as_str()
        } else {
            choice.as_str()
        };
        match kind {
            Some("auto") => {}
            Some("none") => tools.clear(),
            _ => {
                return Err(ApiError::new(
                    400,
                    "unsupported_feature",
                    "This profile supports only auto or none tool choice.",
                ))
            }
        }
        if choice
            .get("disable_parallel_tool_use")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(ApiError::invalid(
                "disable_parallel_tool_use must be a Boolean.",
            ));
        }
        if choice["disable_parallel_tool_use"] == true {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "This profile cannot enforce one tool call per generation.",
            ));
        }
    }
    let mut messages = Vec::new();
    if let Some(system) = value.get("system") {
        messages.push(json!({"role":"system","content":content_text(system)?}));
    }
    for message in array(&value["messages"], "Messages")? {
        if api == Api::OpenAi {
            unknown(
                message,
                &["role", "content", "tool_calls", "tool_call_id", "name"],
                "Message",
            )?;
            let mut message = message.clone();
            if message["content"].is_null() && message.get("tool_calls").is_some() {
                message["content"] = json!("");
            }
            if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    if let Some(s) = call["function"]["arguments"].as_str() {
                        call["function"]["arguments"] =
                            crate::contracts::parse_document(s.as_bytes()).map_err(|_| {
                                ApiError::invalid(
                                "Tool arguments must contain valid JSON without duplicate keys.",
                            )
                            })?;
                    }
                }
            }
            messages.push(message);
            continue;
        }
        unknown(message, &["role", "content"], "Message")?;
        let role = text(&message["role"], "Message role")?;
        if role != "user" && role != "assistant" {
            return Err(ApiError::invalid(
                "Messages require user or assistant roles.",
            ));
        }
        if let Some(content) = message["content"].as_str() {
            messages.push(json!({"role":role,"content":content}));
            continue;
        }
        let mut text_parts = String::new();
        let mut calls = Vec::new();
        for block in array(&message["content"], "Message content")? {
            match block["type"].as_str() {
                Some("text") => {
                    unknown(block, &["type", "text", "cache_control"], "Text block")?;
                    text_parts.push_str(&text(&block["text"], "Text block text")?);
                }
                Some("tool_use") if role == "assistant" => {
                    unknown(
                        block,
                        &["type", "id", "name", "input", "cache_control"],
                        "Tool use",
                    )?;
                    if !block["input"].is_object() {
                        return Err(ApiError::invalid("Tool input must be an object."));
                    }
                    calls.push(json!({"id":text(&block["id"],"Tool ID")?,"type":"function","function":{"name":text(&block["name"],"Tool name")?,"arguments":block["input"]}}));
                }
                Some("tool_result") if role == "user" => {
                    unknown(
                        block,
                        &[
                            "type",
                            "tool_use_id",
                            "content",
                            "is_error",
                            "cache_control",
                        ],
                        "Tool result",
                    )?;
                    if !text_parts.is_empty() {
                        return Err(ApiError::invalid(
                            "Tool results must precede text within the user message.",
                        ));
                    }
                    if block.get("is_error").is_some_and(|v| !v.is_boolean()) {
                        return Err(ApiError::invalid("Tool result is_error must be a Boolean."));
                    }
                    let mut content = if block.get("content").is_none() {
                        String::new()
                    } else {
                        content_text(&block["content"])?
                    };
                    if block["is_error"] == true {
                        content = format!("[tool_result is_error=true]\n{content}");
                    }
                    messages.push(json!({"role":"tool","tool_call_id":text(&block["tool_use_id"],"Tool result ID")?,"content":content}));
                }
                _ => {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "This profile supports text, tool_use, and tool_result blocks.",
                    ))
                }
            }
        }
        if !text_parts.is_empty() || !calls.is_empty() {
            let mut normalized = json!({"role":role,"content":text_parts});
            if !calls.is_empty() {
                normalized["tool_calls"] = json!(calls);
            }
            messages.push(normalized);
        }
    }
    if messages.is_empty() || messages.len() > 256 {
        return Err(ApiError::invalid(
            "Use between 1 and 256 normalized messages.",
        ));
    }
    Ok((
        Request {
            messages,
            tools,
            max_tokens,
            stops,
            count_only,
        },
        stream,
    ))
}

fn content_text(content: &Value) -> Result<String, ApiError> {
    if let Some(s) = content.as_str() {
        return Ok(s.into());
    }
    let mut result = String::new();
    for block in array(content, "Text content")? {
        unknown(
            block,
            &["type", "text", "cache_control"],
            "Text content block",
        )?;
        if block["type"] != "text" {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "Only text content is supported.",
            ));
        }
        result.push_str(&text(&block["text"], "Content text")?);
    }
    Ok(result)
}

fn finish_reason(api: Api, cause: &str) -> &'static str {
    match (api, cause) {
        (Api::Anthropic, "tool_calls") => "tool_use",
        (Api::Anthropic, "length") => "max_tokens",
        (Api::Anthropic, "stop_sequence") => "stop_sequence",
        (Api::Anthropic, _) => "end_turn",
        (Api::OpenAi, "tool_calls") => "tool_calls",
        (Api::OpenAi, "length") => "length",
        _ => "stop",
    }
}
fn response_body(
    api: Api,
    begin: &Begin,
    model: &str,
    parts: &[Part],
    cause: &str,
    input: u64,
    output: u64,
    stop: Option<String>,
) -> Value {
    if api == Api::Anthropic {
        json!({"id":format!("msg_{}",begin.request_id),"type":"message","role":"assistant","model":model,"content":anthropic_blocks(parts),"stop_reason":finish_reason(api,cause),"stop_sequence":stop,"usage":{"input_tokens":input,"output_tokens":output}})
    } else {
        let mut text = String::new();
        let mut tools = Vec::new();
        for part in parts {
            match part { Part::Text(t)=>text.push_str(t),Part::Tool{id,name,input}=>tools.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":input.to_string()}})) }
        }
        let mut message = json!({"role":"assistant","content":text});
        if !tools.is_empty() {
            message["tool_calls"] = json!(tools);
        }
        json!({"id":format!("chatcmpl-{}",begin.request_id),"object":"chat.completion","created":0,"model":model,"choices":[{"index":0,"message":message,"finish_reason":finish_reason(api,cause)}],"usage":{"prompt_tokens":input,"completion_tokens":output,"total_tokens":input+output}})
    }
}

struct PublicStream {
    receiver: mpsc::Receiver<ModelEvent>,
    _guard: CancelGuard,
    api: Api,
    id: String,
    model: String,
    pending: std::collections::VecDeque<Event>,
    index: usize,
    text_open: bool,
    done: bool,
    tools: usize,
    include_usage: bool,
}
impl PublicStream {
    fn new(
        receiver: mpsc::Receiver<ModelEvent>,
        guard: CancelGuard,
        api: Api,
        begin: Begin,
        model: String,
        include_usage: bool,
    ) -> Self {
        let mut stream = Self {
            receiver,
            _guard: guard,
            api,
            id: begin.request_id.clone(),
            model,
            pending: Default::default(),
            index: 0,
            text_open: false,
            done: false,
            tools: 0,
            include_usage,
        };
        if api == Api::Anthropic {
            stream.emit("message_start",json!({"type":"message_start","message":{"id":format!("msg_{}",begin.request_id),"type":"message","role":"assistant","content":[],"model":stream.model,"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":begin.input_tokens,"output_tokens":0}}}));
        } else {
            stream.chunk(json!({"role":"assistant","content":""}), Value::Null);
        }
        stream
    }
    fn emit(&mut self, name: &str, value: Value) {
        self.pending
            .push_back(Event::default().event(name).data(value.to_string()));
    }
    fn chunk(&mut self, delta: Value, finish: Value) {
        let mut v = json!({"id":format!("chatcmpl-{}",self.id),"object":"chat.completion.chunk","created":0,"model":self.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
        if self.include_usage {
            v["usage"] = Value::Null;
        }
        self.pending.push_back(Event::default().data(v.to_string()));
    }
    fn close_text(&mut self) {
        if self.text_open {
            self.emit(
                "content_block_stop",
                json!({"type":"content_block_stop","index":self.index}),
            );
            self.index += 1;
            self.text_open = false;
        }
    }
    fn accept(&mut self, event: ModelEvent) {
        match event {
            ModelEvent::Part(Part::Text(text)) => {
                if !text.is_empty() {
                    if self.api == Api::OpenAi {
                        self.chunk(json!({"content":text}), Value::Null);
                    } else {
                        if !self.text_open {
                            self.emit("content_block_start",json!({"type":"content_block_start","index":self.index,"content_block":{"type":"text","text":""}}));
                            self.text_open = true;
                        }
                        self.emit("content_block_delta",json!({"type":"content_block_delta","index":self.index,"delta":{"type":"text_delta","text":text}}));
                    }
                }
            }
            ModelEvent::Part(Part::Tool { id, name, input }) => {
                if self.api == Api::OpenAi {
                    self.chunk(json!({"tool_calls":[{"index":self.tools,"id":id,"type":"function","function":{"name":name,"arguments":input.to_string()}}]}),Value::Null);
                } else {
                    self.close_text();
                    self.emit("content_block_start",json!({"type":"content_block_start","index":self.index,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}}));
                    self.emit("content_block_delta",json!({"type":"content_block_delta","index":self.index,"delta":{"type":"input_json_delta","partial_json":input.to_string()}}));
                    self.emit(
                        "content_block_stop",
                        json!({"type":"content_block_stop","index":self.index}),
                    );
                    self.index += 1;
                }
                self.tools += 1;
            }
            ModelEvent::Done {
                cause,
                input_tokens,
                output_tokens,
                matched_stop,
            } => {
                if self.api == Api::Anthropic {
                    self.close_text();
                    self.emit("message_delta",json!({"type":"message_delta","delta":{"stop_reason":finish_reason(self.api,&cause),"stop_sequence":matched_stop},"usage":{"output_tokens":output_tokens}}));
                    self.emit("message_stop", json!({"type":"message_stop"}));
                } else {
                    self.chunk(json!({}), json!(finish_reason(self.api, &cause)));
                    if self.include_usage {
                        let usage = json!({"id":format!("chatcmpl-{}",self.id),"object":"chat.completion.chunk","created":0,"model":self.model,"choices":[],
                            "usage":{"prompt_tokens":input_tokens,"completion_tokens":output_tokens,"total_tokens":input_tokens+output_tokens}});
                        self.pending
                            .push_back(Event::default().data(usage.to_string()));
                    }
                    self.pending.push_back(Event::default().data("[DONE]"));
                }
                self.done = true;
            }
            ModelEvent::Error(error) => {
                self.emit("error",json!({"type":"error","error":{"type":public_error_type(error.status),"code":error.code,"message":error.message}}));
                self.done = true;
            }
        }
    }
}
impl Stream for PublicStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            if self.done {
                return Poll::Ready(None);
            }
            match self.receiver.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(event)) => self.accept(event),
                Poll::Ready(None) => self.accept(ModelEvent::Error(ApiError::new(
                    503,
                    "worker_lost",
                    "The worker ended the stream without a terminal result.",
                ))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::gateway_test_service;

    #[tokio::test]
    async fn readiness_separates_host_pressure_from_worker_availability() {
        use crate::host_pressure::{HostPressure, HostPressurePolicy, Pressure};
        let pressure = Arc::new(HostPressure::new(HostPressurePolicy::Macos));
        pressure.record(Ok(Pressure::Warning), std::time::Instant::now());
        let (service, mut observed) =
            crate::supervisor::gateway_test_service_with_pressure(pressure.clone());
        let response = ready(State(service.clone())).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 10000)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["ready"], false);
        assert_eq!(body["worker_available"], true);
        assert_eq!(body["host_pressure"]["policy"], "macos");
        assert_eq!(body["host_pressure"]["state"], "warning");
        assert_eq!(
            body["worker"]["worker_epoch"],
            service.ready_snapshot()["worker_epoch"]
        );
        let error = request_error(
            service.clone(),
            openai_request(json!([
                {"role":"user","content":"original pressure request"}
            ])),
            Api::OpenAi,
        )
        .await;
        assert_eq!(
            (error.status, error.code.as_str()),
            (503, "capacity_unavailable")
        );
        assert!(observed.try_recv().is_err());
        service.shutdown();
        let now = std::time::Instant::now();
        pressure.record(Ok(Pressure::Normal), now - Duration::from_secs(1));
        pressure.record(Ok(Pressure::Normal), now);
        let response = ready(State(service)).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    fn openai_request(messages: Value) -> Value {
        json!({"model": "test-model", "max_tokens": 4, "messages": messages})
    }

    async fn request_error(service: Arc<Service>, value: Value, api: Api) -> ApiError {
        tokio::time::timeout(
            Duration::from_secs(1),
            handle(
                service,
                HeaderMap::new(),
                Bytes::from(value.to_string()),
                api,
                false,
            ),
        )
        .await
        .expect("The request must not wait for admission.")
        .expect_err("The request must fail.")
    }

    #[tokio::test]
    async fn invalid_normalized_inputs_never_reach_admission() {
        let (service, mut observed) = gateway_test_service();
        let cases = [
            openai_request(json!([{"role": 17, "content": "hello"}])),
            openai_request(json!([{"role": "user", "content": 17}])),
            openai_request(json!([{"role": "user"}])),
            openai_request(json!([{"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read", "arguments": {}}, "unknown": true}
            ]}])),
            openai_request(json!([{"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read", "arguments": []}}
            ]}])),
            json!({"model": "test-model", "max_tokens": 4, "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "read", "parameters": [], "extra": true}}]}),
        ];
        for value in cases {
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(error.status, 400, "{}", error.message);
            assert_eq!(error.code, "invalid_request");
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
            assert!(service.available.load(Ordering::Acquire));
            assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
            assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
        }

        // This control proves that the observation queue is live.
        let valid = openai_request(json!([{"role": "user", "content": "hello"}]));
        let error = request_error(service.clone(), valid, Api::OpenAi).await;
        assert_eq!(error.message, "Test coordinator observed admission.");
        let admitted = observed
            .try_recv()
            .expect("Valid input must reach admission.");
        assert_eq!(
            admitted.messages,
            vec![json!({"role": "user", "content": "hello"})]
        );
        assert!(service.available.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn conflicting_output_limits_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
        value["max_completion_tokens"] = json!(1);
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "invalid_request");
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn parallel_tool_option_requires_a_boolean() {
        let (service, mut observed) = gateway_test_service();
        for invalid in [json!("false"), json!(0), json!(null)] {
            let value = json!({"model":"test-model", "max_tokens":4,
                "messages":[{"role":"user", "content":"hello"}],
                "tool_choice":{"type":"auto", "disable_parallel_tool_use":invalid}});
            let error = request_error(service.clone(), value, Api::Anthropic).await;
            assert_eq!(error.status, 400);
            assert_eq!(error.code, "invalid_request");
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }

    #[tokio::test]
    async fn strict_tools_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
        value["tools"] = json!([{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object"}, "strict": true
        }}]);
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "unsupported_feature");
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(service.available.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn duplicate_keys_inside_argument_strings_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        for arguments in [
            r#"{"path":"A","path":"B"}"#,
            r#"{"config":{"mode":1,"mode":2}}"#,
        ] {
            let value = openai_request(json!([{
                "role": "assistant", "content": null,
                "tool_calls": [{"id": "call-a", "type": "function", "function": {"name": "read", "arguments": arguments}}]
            }]));
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(error.status, 400);
            assert!(error.message.contains("duplicate keys"));
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
        assert!(service.available.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn openai_tool_history_preserves_typed_arguments() {
        let (service, _) = gateway_test_service();
        let value = openai_request(json!([
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"001\",\"count\":2,\"enabled\":false}"}}
            ]},
            {"role": "tool", "tool_call_id": "call-a", "content": "file text"}
        ]));
        let (request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(request.messages[0]["content"], "");
        assert_eq!(
            request.messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({"path": "001", "count": 2, "enabled": false})
        );
        assert_eq!(
            request.messages[1],
            json!({"role": "tool", "tool_call_id": "call-a", "content": "file text"})
        );
    }

    #[tokio::test]
    async fn anthropic_history_preserves_error_results_and_message_order() {
        let (service, _) = gateway_test_service();
        let value = json!({
            "model": "test-model", "max_tokens": 4,
            "system": [{"type": "text", "text": "Read local files."}],
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Inspect both files."},
                    {"type": "tool_use", "id": "call-a", "name": "read", "input": {"path": "A"}},
                    {"type": "tool_use", "id": "call-b", "name": "read", "input": {"path": "B"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call-a", "is_error": true,
                        "content": [{"type": "text", "text": "Access denied."}]},
                    {"type": "tool_result", "tool_use_id": "call-b", "is_error": false, "content": "File B text."},
                    {"type": "text", "text": "Continue with file B."}
                ]}
            ]
        });
        let (request, _) = normalize(&value, Api::Anthropic, false, &service).unwrap();
        assert_eq!(request.messages.len(), 5);
        assert_eq!(
            request.messages[0],
            json!({"role": "system", "content": "Read local files."})
        );
        assert_eq!(request.messages[1]["content"], "Inspect both files.");
        assert_eq!(
            request.messages[1]["tool_calls"][0],
            json!({"id": "call-a", "type": "function", "function": {"name": "read", "arguments": {"path": "A"}}})
        );
        assert_eq!(request.messages[1]["tool_calls"][1]["id"], "call-b");
        assert_eq!(
            request.messages[2],
            json!({"role": "tool", "tool_call_id": "call-a", "content": "[tool_result is_error=true]\nAccess denied."})
        );
        assert_eq!(
            request.messages[3],
            json!({"role": "tool", "tool_call_id": "call-b", "content": "File B text."})
        );
        assert_eq!(
            request.messages[4],
            json!({"role": "user", "content": "Continue with file B."})
        );
    }

    #[tokio::test]
    async fn anthropic_invalid_result_order_and_error_flag_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        for content in [
            json!([{"type": "text", "text": "Continue."}, {"type": "tool_result", "tool_use_id": "call-a", "content": "text"}]),
            json!([{"type": "tool_result", "tool_use_id": "call-a", "is_error": "true", "content": "text"}]),
        ] {
            let value = json!({"model": "test-model", "max_tokens": 4, "messages": [{"role": "user", "content": content}]});
            let error = request_error(service.clone(), value, Api::Anthropic).await;
            assert_eq!(error.status, 400);
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }

    fn completed(cause: &str) -> ModelEvent {
        ModelEvent::Done {
            cause: cause.into(),
            input_tokens: 7,
            output_tokens: 9,
            matched_stop: None,
        }
    }

    async fn collect_stream(api: Api, events: Vec<ModelEvent>) -> (String, Arc<AtomicBool>) {
        collect_stream_with_usage(api, events, false).await
    }

    async fn collect_stream_with_usage(
        api: Api,
        events: Vec<ModelEvent>,
        include_usage: bool,
    ) -> (String, Arc<AtomicBool>) {
        let (sender, receiver) = mpsc::channel(32);
        for event in events {
            assert!(sender.try_send(event).is_ok());
        }
        drop(sender);
        let cancelled = Arc::new(AtomicBool::new(false));
        let stream = PublicStream::new(
            receiver,
            CancelGuard(cancelled.clone()),
            api,
            Begin {
                request_id: "request-a".into(),
                input_tokens: 7,
            },
            "test-model".into(),
            include_usage,
        );
        let body = Sse::new(stream).into_response().into_body();
        let bytes = tokio::time::timeout(Duration::from_secs(1), axum::body::to_bytes(body, 65536))
            .await
            .expect("The stream must end.")
            .unwrap();
        (String::from_utf8(bytes.to_vec()).unwrap(), cancelled)
    }

    fn decode_sse(text: &str) -> Vec<(String, Value)> {
        text.split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let name = block
                    .lines()
                    .find_map(|line| line.strip_prefix("event:"))
                    .unwrap_or("")
                    .trim()
                    .to_owned();
                let data = block
                    .lines()
                    .find_map(|line| line.strip_prefix("data:"))
                    .expect("SSE data is required.")
                    .trim();
                let value = if data == "[DONE]" {
                    json!(data)
                } else {
                    serde_json::from_str(data).unwrap()
                };
                (name, value)
            })
            .collect()
    }

    #[tokio::test]
    async fn anthropic_stream_closes_each_block_before_the_final_result() {
        let (text, cancelled) = collect_stream(
            Api::Anthropic,
            vec![
                ModelEvent::Part(Part::Text("A".into())),
                ModelEvent::Part(Part::Text("B".into())),
                ModelEvent::Part(Part::Tool {
                    id: "call-a".into(),
                    name: "read".into(),
                    input: json!({"path": "A"}),
                }),
                ModelEvent::Part(Part::Tool {
                    id: "call-b".into(),
                    name: "read".into(),
                    input: json!({"path": "B"}),
                }),
                ModelEvent::Part(Part::Text("C".into())),
                completed("tool_calls"),
            ],
        )
        .await;
        let events = decode_sse(&text);
        let names: Vec<_> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        for (start, stop, index) in [(1, 4, 0), (5, 7, 1), (8, 10, 2), (11, 13, 3)] {
            assert_eq!(events[start].1["index"], index);
            assert_eq!(events[stop].1["index"], index);
        }
        assert_eq!(events[5].1["content_block"]["id"], "call-a");
        assert_eq!(events[8].1["content_block"]["id"], "call-b");
        let arguments: Value =
            serde_json::from_str(events[6].1["delta"]["partial_json"].as_str().unwrap()).unwrap();
        assert_eq!(arguments, json!({"path": "A"}));
        assert_eq!(
            events[0].1["message"]["usage"],
            json!({"input_tokens": 7, "output_tokens": 0})
        );
        assert_eq!(events[14].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(events[14].1["usage"]["output_tokens"], 9);
        assert!(cancelled.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn openai_stream_has_stable_tool_indices_and_one_final_result() {
        let (text, _) = collect_stream_with_usage(
            Api::OpenAi,
            vec![
                ModelEvent::Part(Part::Text("Inspect.".into())),
                ModelEvent::Part(Part::Tool {
                    id: "call-a".into(),
                    name: "read".into(),
                    input: json!({"path": "A"}),
                }),
                ModelEvent::Part(Part::Tool {
                    id: "call-b".into(),
                    name: "read".into(),
                    input: json!({"path": "B"}),
                }),
                completed("tool_calls"),
            ],
            true,
        )
        .await;
        let events = decode_sse(&text);
        assert_eq!(events.len(), 7);
        assert_eq!(events[0].1["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            events[2].1["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            events[3].1["choices"][0]["delta"]["tool_calls"][0]["index"],
            1
        );
        assert_eq!(events[4].1["choices"][0]["finish_reason"], "tool_calls");
        assert!(events[..5]
            .iter()
            .all(|(_, value)| value.get("usage") == Some(&Value::Null)));
        assert_eq!(events[5].1["choices"], json!([]));
        assert_eq!(
            events[5].1["usage"],
            json!({"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16})
        );
        assert_eq!(events[5].1["id"], events[0].1["id"]);
        assert_eq!(events[6].1, "[DONE]");
        assert!(events[..4]
            .iter()
            .all(|(_, value)| value["choices"][0]["finish_reason"].is_null()));
    }

    #[tokio::test]
    async fn openai_stream_omits_usage_when_not_requested() {
        let (text, _) = collect_stream(
            Api::OpenAi,
            vec![
                ModelEvent::Part(Part::Text("Answer.".into())),
                completed("length"),
            ],
        )
        .await;
        let events = decode_sse(&text);
        assert_eq!(events.len(), 4);
        assert!(events.iter().all(|(_, value)| value.get("usage").is_none()));
        assert!(events[..3]
            .iter()
            .all(|(_, value)| value["choices"].as_array().unwrap().len() == 1));
        assert_eq!(events[2].1["choices"][0]["finish_reason"], "length");
        assert_eq!(events[3].1, "[DONE]");
    }

    #[tokio::test]
    async fn stream_options_require_streaming_and_a_boolean_usage_flag() {
        let (service, mut observed) = gateway_test_service();
        for (stream, options) in [
            (false, json!({"include_usage": true})),
            (true, json!({"include_usage": 1})),
        ] {
            let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
            value["stream"] = json!(stream);
            value["stream_options"] = options;
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(error.status, 400);
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
        for options in [
            None,
            Some(json!({})),
            Some(json!({"include_usage": false})),
            Some(json!({"include_usage": true})),
        ] {
            let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
            value["stream"] = json!(true);
            if let Some(options) = options {
                value["stream_options"] = options;
            }
            assert!(normalize(&value, Api::OpenAi, false, &service).unwrap().1);
        }
    }

    #[tokio::test]
    async fn stream_error_suppresses_a_queued_success_result() {
        for api in [Api::Anthropic, Api::OpenAi] {
            let (text, _) = collect_stream_with_usage(
                api,
                vec![
                    ModelEvent::Part(Part::Text("Partial text.".into())),
                    ModelEvent::Error(ApiError::new(500, "internal_error", "Execution failed.")),
                    completed("eos"),
                ],
                true,
            )
            .await;
            let events = decode_sse(&text);
            assert_eq!(events.iter().filter(|(name, _)| name == "error").count(), 1);
            assert_eq!(events.last().unwrap().1["error"]["code"], "internal_error");
            assert!(!events.iter().any(|(name, value)| name == "message_stop"
                || name == "message_delta"
                || value == "[DONE]"));
            assert!(events
                .iter()
                .all(|(_, value)| value["choices"][0]["finish_reason"].is_null()));
            if api == Api::OpenAi {
                assert!(events.iter().all(|(_, value)| !value["usage"].is_object()));
            }
        }
    }

    #[tokio::test]
    async fn stream_eof_without_terminal_is_one_worker_error() {
        for api in [Api::Anthropic, Api::OpenAi] {
            let (text, _) = collect_stream(
                api,
                vec![ModelEvent::Part(Part::Text("Partial text.".into()))],
            )
            .await;
            let events = decode_sse(&text);
            assert_eq!(events.last().unwrap().0, "error");
            assert_eq!(events.last().unwrap().1["error"]["code"], "worker_lost");
            assert_eq!(events.iter().filter(|(name, _)| name == "error").count(), 1);
            assert!(!text.contains("message_stop") && !text.contains("[DONE]"));
        }
    }

    #[tokio::test]
    async fn dropping_an_unpolled_stream_cancels_the_request() {
        let (_sender, receiver) = mpsc::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let stream = PublicStream::new(
            receiver,
            CancelGuard(cancelled.clone()),
            Api::Anthropic,
            Begin {
                request_id: "request-a".into(),
                input_tokens: 7,
            },
            "test-model".into(),
            false,
        );
        assert!(!cancelled.load(Ordering::Acquire));
        drop(stream);
        assert!(cancelled.load(Ordering::Acquire));
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use apxinf_serving::tool_calling::ToolFormat;
use apxinf_serving::{
    router, ApiConfig, ApiError, ChatEngine, EngineOutput, EngineRequest, WorkerClient,
};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

const TEXT: &str = "hello from fake engine";
const TOOL_CALL: &str = "<tool_call>\n<function=weather.get>\n<parameter=city>\n上海\n</parameter>\n</function>\n</tool_call>";
const REASONING: &str = concat!(
    "<|message_model|><|content_thinking|>Let me think.<|end_message|>",
    "<|message_model|><|content_text|>Hello there<|end_message|>",
    "<|content_model_end_sampling|>"
);

#[derive(Clone, Copy)]
enum Script {
    Text,
    ToolCall,
    Reasoning,
}

struct ScriptedEngine {
    script: Script,
    calls: Arc<AtomicUsize>,
}

impl ChatEngine for ScriptedEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let text = match self.script {
            Script::Text => TEXT,
            Script::ToolCall => TOOL_CALL,
            Script::Reasoning => REASONING,
        };
        Ok(EngineOutput {
            text: text.to_owned(),
            prompt_tokens: 11,
            reused_prompt_tokens: 3,
            completion_tokens: 7,
            hit_length: false,
            ttft_ms: None,
            tpot_ms: None,
            generation_tps: None,
            total_latency_ms: None,
        })
    }
}

struct RecordingEngine {
    requests: Sender<EngineRequest>,
}

impl ChatEngine for RecordingEngine {
    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        let _ = self.requests.send(request);
        Ok(EngineOutput {
            text: "ok".into(),
            prompt_tokens: 1,
            reused_prompt_tokens: 0,
            completion_tokens: 1,
            hit_length: false,
            ttft_ms: None,
            tpot_ms: None,
            generation_tps: None,
            total_latency_ms: None,
        })
    }
}

async fn wait_until_ready(worker: &WorkerClient) {
    for _ in 0..1_000 {
        if worker.ready() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
        tokio::task::yield_now().await;
    }
    panic!(
        "worker did not become ready: {} {:?}",
        worker.status_message(),
        worker.startup_error()
    );
}

async fn app_with(script: Script, tool_format: ToolFormat) -> (axum::Router, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let worker_calls = calls.clone();
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(ScriptedEngine {
            script,
            calls: worker_calls,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    (
        router(
            ApiConfig {
                served_model_name: "fake-model".into(),
                max_output_tokens: 128,
                tool_format,
                ..Default::default()
            },
            worker,
        ),
        calls,
    )
}

async fn recording_app() -> (axum::Router, Receiver<EngineRequest>) {
    let (sender, receiver) = channel();
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(RecordingEngine { requests: sender }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    (
        router(
            ApiConfig {
                served_model_name: "fake-model".into(),
                max_output_tokens: 128,
                ..Default::default()
            },
            worker,
        ),
        receiver,
    )
}

async fn json_response(app: axum::Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn sse_response(app: axum::Router, body: Value) -> (StatusCode, Vec<(String, Value)>) {
    let response = app
        .oneshot(
            Request::post("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert_eq!(
        response.headers()["content-type"],
        "text/event-stream",
        "unexpected response headers"
    );
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    (status, parse_sse(&text))
}

/// Reads both the `event:` name and the JSON `data:` payload of every frame.
fn parse_sse(text: &str) -> Vec<(String, Value)> {
    text.split("\n\n")
        .filter(|block| !block.is_empty())
        .map(|block| {
            let mut name = None;
            let mut data = None;
            for line in block.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    name = Some(rest.to_owned());
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = Some(serde_json::from_str::<Value>(rest).unwrap());
                }
            }
            (name.expect("event line"), data.expect("data line"))
        })
        .collect()
}

fn function_tool(name: &str) -> Value {
    json!({
        "type": "function",
        "name": name,
        "description": format!("{name} tool"),
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
        "strict": false,
    })
}

fn captured_tools() -> Value {
    json!([
        function_tool("exec_command"),
        function_tool("write_stdin"),
        function_tool("request_user_input"),
        function_tool("view_image"),
        function_tool("get_goal"),
        function_tool("create_goal"),
        function_tool("update_goal"),
        {
            "type": "namespace",
            "name": "multi_agent_v1",
            "description": "Multi-agent tools",
            "tools": [
                function_tool("close_agent"),
                function_tool("resume_agent"),
                function_tool("send_input"),
                function_tool("spawn_agent"),
                function_tool("wait_agent"),
            ],
        },
        {"type": "web_search", "external_web_access": false},
    ])
}

fn captured_input() -> Value {
    json!([
        {"type": "message", "id": "msg_a", "role": "developer", "content": [
            {"type": "input_text", "text": "system one"}
        ]},
        {"type": "message", "id": "msg_b", "role": "developer", "content": [
            {"type": "input_text", "text": "system two a"},
            {"type": "input_text", "text": "system two b"}
        ]},
        {"type": "message", "id": "msg_c", "role": "user", "content": [
            {"type": "input_text", "text": "hello"}
        ]},
        {"type": "message", "id": "msg_d", "role": "user", "content": [
            {"type": "input_text", "text": "world"}
        ]}
    ])
}

#[tokio::test]
async fn captured_codex_request_converts_to_messages_and_tools() {
    let (app, requests) = recording_app().await;
    let body = json!({
        "model": "fake-model",
        "stream": false,
        "input": captured_input(),
        "tools": captured_tools(),
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "reasoning": {"effort": "low", "summary": "auto"},
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "01a21bf4-cdca-79b2-8c71-70135089d9e2",
        "client_metadata": {"x-codex-window-id": "abc:0"},
    });
    let (status, response) = json_response(app, body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();

    let roles: Vec<&str> = request.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(
        roles,
        ["system", "system", "user", "user"],
        "the two developer items stay separate system messages, in order"
    );
    assert_eq!(
        request.messages[0].content,
        Some(Value::String("system one".into()))
    );
    assert_eq!(
        request.messages[1].content,
        Some(Value::String("system two a\nsystem two b".into()))
    );

    assert_eq!(
        request.tools.len(),
        12,
        "7 top-level functions plus 5 hoisted from multi_agent_v1; web_search dropped"
    );
    assert!(request.tools.iter().all(|tool| tool.kind == "function"));
    let names: Vec<&str> = request
        .tools
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect();
    assert!(names.contains(&"exec_command"));
    assert!(names.contains(&"close_agent"));
    assert!(!names.contains(&"web_search"));
}

#[tokio::test]
async fn function_call_and_output_survive_the_round_trip() {
    let (app, requests) = recording_app().await;
    let body = json!({
        "model": "fake-model",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "run ls"}
            ]},
            {"type": "function_call", "call_id": "call_abc", "name": "exec_command",
             "arguments": "{\"cmd\":\"ls\"}"},
            {"type": "function_call_output", "call_id": "call_abc", "output": "file.txt\n"}
        ]
    });
    let (status, response) = json_response(app, body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(request.messages.len(), 3);

    assert_eq!(request.messages[1].role, "assistant");
    let calls = request.messages[1].tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_abc");
    assert_eq!(calls[0].kind, "function");
    assert_eq!(calls[0].function.name, "exec_command");
    assert_eq!(calls[0].function.arguments, "{\"cmd\":\"ls\"}");

    assert_eq!(request.messages[2].role, "tool");
    assert_eq!(
        request.messages[2].tool_call_id.as_deref(),
        Some("call_abc")
    );
    assert_eq!(
        request.messages[2].content,
        Some(Value::String("file.txt\n".into()))
    );
}

#[tokio::test]
async fn unknown_items_and_fields_are_ignored() {
    let (app, requests) = recording_app().await;
    let body = json!({
        "model": "fake-model",
        "unknown_top_level": {"a": 1},
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hi"}
            ]},
            {"type": "brand_new_item", "payload": {"x": 1}},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "role": "user", "content": "bare string content"}
        ]
    });
    let (status, response) = json_response(app, body).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(request.messages.len(), 2);
    assert_eq!(
        request.messages[0].content,
        Some(Value::String("hi".into()))
    );
    assert_eq!(
        request.messages[1].content,
        Some(Value::String("bare string content".into()))
    );
}

#[tokio::test]
async fn previous_response_id_is_rejected() {
    let (app, calls) = app_with(Script::Text, ToolFormat::Auto).await;
    let (status, body) = json_response(
        app,
        json!({
            "model": "fake-model",
            "input": "hi",
            "previous_response_id": "resp_123"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "previous_response_id");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn clamps_max_output_tokens_above_the_cap_and_rejects_zero() {
    let (app, requests) = recording_app().await;
    let (status, body) = json_response(
        app.clone(),
        json!({"model": "fake-model", "input": "hi", "max_output_tokens": 100000}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(
        request.max_tokens, 128,
        "engine must receive the clamped output cap"
    );

    let (status, body) = json_response(
        app,
        json!({"model": "fake-model", "input": "hi", "max_output_tokens": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["param"], "max_tokens");
}

#[tokio::test]
async fn non_streaming_text_turn_returns_the_response_object() {
    let (app, _) = app_with(Script::Text, ToolFormat::Auto).await;
    let (status, body) = json_response(
        app,
        json!({"model": "fake-model", "input": "hello", "max_output_tokens": 32}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert!(body["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["role"], "assistant");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], TEXT);
    assert_eq!(body["usage"]["input_tokens"], 11);
    assert_eq!(body["usage"]["output_tokens"], 7);
    assert_eq!(body["usage"]["total_tokens"], 18);
    assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 3);
}

#[tokio::test]
async fn streams_text_turn_event_sequence() {
    let (app, _) = app_with(Script::Text, ToolFormat::Auto).await;
    let (status, events) = sse_response(
        app,
        json!({"model": "fake-model", "stream": true, "input": "hello"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    for (index, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["type"].as_str(), Some(name.as_str()));
        assert_eq!(data["sequence_number"].as_u64(), Some(index as u64));
    }
    assert_eq!(events[4].1["delta"], TEXT);
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["output"][0]["type"], "message");
    assert_eq!(completed["output"][0]["status"], "completed");
    assert_eq!(completed["output"][0]["content"][0]["text"], TEXT);
    assert_eq!(completed["usage"]["input_tokens"], 11);
    assert_eq!(
        completed["usage"]["input_tokens_details"]["cached_tokens"],
        3
    );
    assert_eq!(
        completed["usage"]["output_tokens_details"]["reasoning_tokens"],
        0
    );
}

#[tokio::test]
async fn streams_tool_call_event_sequence() {
    let (app, _) = app_with(Script::ToolCall, ToolFormat::Auto).await;
    let (status, events) = sse_response(
        app,
        json!({
            "model": "fake-model",
            "stream": true,
            "input": "weather?",
            "tools": [function_tool("weather.get")],
            "tool_choice": "auto"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    for (index, (name, data)) in events.iter().enumerate() {
        assert_eq!(data["type"].as_str(), Some(name.as_str()));
        assert_eq!(data["sequence_number"].as_u64(), Some(index as u64));
    }

    let added = &events[2].1;
    assert_eq!(added["item"]["type"], "function_call");
    assert_eq!(added["item"]["name"], "weather.get");
    let call_id = added["item"]["call_id"].as_str().unwrap().to_owned();
    let item_id = added["item"]["id"].as_str().unwrap().to_owned();

    let delta = &events[3].1;
    assert_eq!(delta["item_id"].as_str(), Some(item_id.as_str()));
    let arguments: Value = serde_json::from_str(delta["delta"].as_str().unwrap()).unwrap();
    assert_eq!(arguments, json!({"city": "上海"}));

    let done = &events[4].1;
    assert_eq!(done["item_id"].as_str(), Some(item_id.as_str()));
    assert_eq!(done["arguments"], delta["delta"]);

    let output = &events.last().unwrap().1["response"]["output"];
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["call_id"].as_str(), Some(call_id.as_str()));
    assert_eq!(output[0]["name"], "weather.get");
    assert_eq!(output[0]["arguments"], delta["delta"]);
}

#[tokio::test]
async fn streams_reasoning_items_only_when_thinking_is_present() {
    let (app, _) = app_with(Script::Reasoning, ToolFormat::Inkling2Json).await;
    let (status, events) = sse_response(
        app,
        json!({"model": "fake-model", "stream": true, "input": "think"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        names,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );

    let reasoning = &events[2].1["item"];
    assert_eq!(reasoning["type"], "reasoning");
    assert_eq!(events[4].1["delta"], "Let me think.");
    assert_eq!(events[5].1["text"], "Let me think.");

    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["output"][0]["type"], "reasoning");
    assert_eq!(
        completed["output"][0]["summary"][0]["text"],
        "Let me think."
    );
    assert_eq!(completed["output"][1]["type"], "message");
    assert_eq!(completed["output"][1]["content"][0]["text"], "Hello there");
}

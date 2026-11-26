use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::time::Duration;

use apxinf_serving::{protocol::ChatCompletionMessage, tool_calling::ToolChoice};
use apxinf_serving::{
    router, ApiConfig, ApiError, ChatEngine, EngineOutput, EngineRequest, WorkerClient,
};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tokio_stream::StreamExt;
use tower::ServiceExt;

struct FakeEngine {
    calls: Arc<AtomicUsize>,
}

#[tokio::test]
async fn openai_image_requests_reach_the_engine_in_plain_and_stream_modes() {
    let (app, calls) = app().await;
    let body = serde_json::json!({"model":"fake-model", "messages":[{"role":"user", "content":[
        {"type":"text","text":"Describe this image"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}}
    ]}]});
    let (status, plain) = json_response(app.clone(), body.clone()).await;
    assert_eq!(status, StatusCode::OK, "{plain}");
    let mut stream = body;
    stream["stream"] = Value::Bool(true);
    let (status, chunks, done) = sse_response(app, stream).await;
    assert_eq!(status, StatusCode::OK);
    assert!(done);
    assert!(chunks
        .iter()
        .any(|chunk| chunk["choices"][0]["delta"]["content"].is_string()));
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[tokio::test]
async fn invalid_image_urls_fail_before_engine_admission() {
    let (app, calls) = app().await;
    for url in [
        "https://example.com/a.png",
        "file:///etc/passwd",
        "data:image/png;base64,!",
    ] {
        let (status, body) = json_response(
            app.clone(),
            serde_json::json!({"messages":[{"role":"user", "content":[
                {"type":"image_url","image_url":{"url":url}}
            ]}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

async fn wait_until_ready(worker: &WorkerClient) {
    for _ in 0..1_000 {
        if worker.ready() {
            return;
        }
        // Worker construction happens on a native thread. A bounded sleep is
        // required here: repeatedly yielding only the Tokio task does not
        // guarantee that the OS schedules that thread on slower ARM hosts.
        std::thread::sleep(Duration::from_millis(1));
        tokio::task::yield_now().await;
    }
    panic!(
        "worker did not become ready: {} {:?}",
        worker.status_message(),
        worker.startup_error()
    );
}

impl ChatEngine for FakeEngine {
    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if request
            .messages
            .iter()
            .any(|message| message.content.as_ref().and_then(Value::as_str) == Some("fail"))
        {
            return Err(ApiError::internal("fake generation failure"));
        }
        let text = if request.messages.iter().any(|message| {
            message.content.as_ref().and_then(Value::as_str) == Some("inspect tool stop policy")
        }) {
            request.stop_after_tool_call.to_string()
        } else if request.tools.is_empty() {
            "你好，我是离线 fake engine。".to_owned()
        } else {
            "<tool_call>\n<function=weather.get>\n<parameter=city>\n上海\n</parameter>\n<parameter=date>\n2026-10-22\n</parameter>\n</function>\n</tool_call>".to_owned()
        };
        Ok(EngineOutput {
            text,
            prompt_tokens: 11,
            reused_prompt_tokens: 3,
            completion_tokens: 7,
            hit_length: false,
            ttft_ms: Some(10.0),
            tpot_ms: Some(5.0),
            generation_tps: Some(200.0),
            total_latency_ms: Some(40.0),
        })
    }
}

async fn app() -> (axum::Router, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let worker_calls = calls.clone();
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(FakeEngine {
            calls: worker_calls,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    (
        router(
            ApiConfig {
                served_model_name: "fake-model".into(),
                max_output_tokens: 128,
            },
            worker,
        ),
        calls,
    )
}

async fn json_response(app: axum::Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
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

async fn sse_response(app: axum::Router, body: Value) -> (StatusCode, Vec<Value>, bool) {
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
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
    let mut values = Vec::new();
    let mut done = false;
    for event in text.split("\n\n").filter(|event| !event.is_empty()) {
        let data = event.strip_prefix("data: ").expect("SSE data field");
        if data == "[DONE]" {
            done = true;
        } else {
            values.push(serde_json::from_str(data).unwrap());
        }
    }
    (status, values, done)
}

#[tokio::test]
async fn serves_health_models_and_plain_chat() {
    let (app, calls) = app().await;
    let health = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let models = app
        .clone()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(models.status(), StatusCode::OK);

    let (status, response) = json_response(
        app,
        serde_json::json!({
            "model":"fake-model",
            "messages":[{"role":"user","content":"你好"}],
            "temperature":0,
            "top_k":20,
            "max_tokens":32
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["choices"][0]["message"]["role"], "assistant");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["total_tokens"], 18);
    assert_eq!(
        response["usage"]["prompt_tokens_details"]["cached_tokens"],
        3
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn converts_qwen_xml_to_openai_tool_calls() {
    let (app, _) = app().await;
    let (status, response) = json_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"明天上海天气怎么样"}],
            "tools":[{"type":"function","function":{
                "name":"weather.get",
                "parameters":{
                    "type":"object",
                    "properties":{"city":{"type":"string"},"date":{"type":"string"}},
                    "required":["city","date"],
                    "additionalProperties":false
                }
            }}],
            "tool_choice":"required",
            "parallel_tool_calls":false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    let call = &response["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "weather.get");
    let arguments: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(
        arguments,
        serde_json::json!({"city":"上海","date":"2026-10-22"})
    );
}

#[tokio::test]
async fn single_tool_requests_enable_native_tool_termination() {
    let (app, _) = app().await;
    for parallel in [None, Some(true), Some(false)] {
        let mut body = serde_json::json!({
            "messages":[{"role":"user","content":"inspect tool stop policy"}],
            "tools":[{"type":"function","function":{
                "name":"weather.get","parameters":{"type":"object"}
            }}]
        });
        if let Some(parallel) = parallel {
            body["parallel_tool_calls"] = Value::Bool(parallel);
        }
        let (status, response) = json_response(app.clone(), body).await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(
            response["choices"][0]["message"]["content"],
            (parallel == Some(false)).to_string()
        );
    }
}

#[tokio::test]
async fn rejects_invalid_stream_options_and_bad_tool_schema_before_inference() {
    let (app, calls) = app().await;
    let (status, _) = json_response(
        app.clone(),
        serde_json::json!({
            "messages":[{"role":"user","content":"x"}],
            "stream_options":{"include_usage":true}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, response) = json_response(
        app.clone(),
        serde_json::json!({
            "messages":[{"role":"user","content":"x"}],
            "temperature":1.0,
            "top_k":-1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(response["error"]["param"], "top_k");

    let (status, response) = json_response(
        app.clone(),
        serde_json::json!({
            "messages":[{"role":"user","content":"x"}],
            "disable_spec_decode":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(response["error"]["param"], "disable_spec_decode");

    let (status, _) = json_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"x"}],
            "tools":[{"type":"function","function":{"name":"bad name","parameters":{}}}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn streams_text_finish_usage_and_done_in_openai_sse_shape() {
    let (app, calls) = app().await;
    let (status, chunks, done) = sse_response(
        app,
        serde_json::json!({
            "model":"fake-model",
            "messages":[{"role":"user","content":"你好"}],
            "stream":true,
            "stream_options":{"include_usage":true}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(done);
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"].get(0))
        .filter_map(|choice| choice["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "你好，我是离线 fake engine。");
    assert!(chunks
        .iter()
        .any(|chunk| { chunk["choices"][0]["finish_reason"].as_str() == Some("stop") }));
    let usage = chunks
        .iter()
        .find(|chunk| chunk["choices"].as_array().is_some_and(Vec::is_empty))
        .expect("terminal usage chunk");
    assert_eq!(usage["usage"]["total_tokens"], 18);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

struct IncrementalEngine {
    release: Receiver<()>,
}

impl ChatEngine for IncrementalEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        panic!("streaming requests must use complete_stream")
    }

    fn complete_stream(
        &mut self,
        _request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        if !on_text("first") {
            return Err(ApiError::internal("stream consumer closed"));
        }
        self.release.recv().unwrap();
        let _ = on_text(" second");
        Ok(EngineOutput {
            text: "first second".into(),
            prompt_tokens: 1,
            reused_prompt_tokens: 0,
            completion_tokens: 2,
            hit_length: false,
            ttft_ms: Some(1.0),
            tpot_ms: Some(2.0),
            generation_tps: Some(500.0),
            total_latency_ms: Some(3.0),
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delivers_content_before_generation_completes() {
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(IncrementalEngine {
            release: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let app = router(
        ApiConfig {
            served_model_name: "incremental-model".into(),
            max_output_tokens: 16,
        },
        worker,
    );
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model":"incremental-model",
                        "messages":[{"role":"user","content":"x"}],
                        "stream":true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut before_completion = String::new();
    while !before_completion.contains(r#""content":"first""#) {
        let bytes = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .expect("first token delta was not delivered before completion")
            .expect("SSE body ended before the first token delta")
            .expect("SSE body error");
        before_completion.push_str(std::str::from_utf8(&bytes).unwrap());
    }

    // The engine is still blocked here. Releasing it only after observing the
    // first delta proves the response was not synthesized from the final text.
    release_sender.send(()).unwrap();
    let mut after_completion = String::new();
    while let Some(bytes) = body.next().await {
        after_completion.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
    }
    assert!(after_completion.contains(r#""content":" second""#));
    assert!(after_completion.contains("data: [DONE]"));
}

#[tokio::test]
async fn streams_tool_call_delta_and_tool_calls_finish_reason() {
    let (app, _) = app().await;
    let (status, chunks, done) = sse_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"明天上海天气怎么样"}],
            "tools":[{"type":"function","function":{
                "name":"weather.get",
                "parameters":{
                    "type":"object",
                    "properties":{"city":{"type":"string"},"date":{"type":"string"}},
                    "required":["city","date"],
                    "additionalProperties":false
                }
            }}],
            "tool_choice":"required",
            "parallel_tool_calls":false,
            "stream":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(done);
    let tool_delta = chunks
        .iter()
        .find_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].as_array())
        .expect("tool call delta");
    assert_eq!(tool_delta[0]["type"], "function");
    assert_eq!(tool_delta[0]["function"]["name"], "weather.get");
    let arguments: Value =
        serde_json::from_str(tool_delta[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(
        arguments,
        serde_json::json!({"city":"上海","date":"2026-10-22"})
    );
    assert!(chunks
        .iter()
        .any(|chunk| { chunk["choices"][0]["finish_reason"].as_str() == Some("tool_calls") }));
}

struct SplitToolEngine {
    before_close: SyncSender<()>,
    release_close: Receiver<()>,
}

impl ChatEngine for SplitToolEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        panic!("streaming requests must use complete_stream")
    }

    fn complete_stream(
        &mut self,
        _request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        let fragments = [
            "<to",
            "ol_call>\n<function=weather.get>\n<parameter=city>\n上",
            "海\n</parameter>\n<parameter=date>\n2026-",
            "10-22\n</parameter>\n</function>\n",
        ];
        for fragment in fragments {
            if !on_text(fragment) {
                return Err(ApiError::internal("stream consumer closed"));
            }
        }
        self.before_close.send(()).unwrap();
        self.release_close.recv().unwrap();
        let closing = "</tool_call>";
        let _ = on_text(closing);
        Ok(EngineOutput {
            text: format!("{}{}", fragments.concat(), closing),
            prompt_tokens: 9,
            reused_prompt_tokens: 0,
            completion_tokens: 10,
            hit_length: false,
            ttft_ms: None,
            tpot_ms: None,
            generation_tps: None,
            total_latency_ms: None,
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn buffers_split_tool_xml_until_it_can_emit_valid_openai_json() {
    let (before_close_sender, before_close_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(SplitToolEngine {
            before_close: before_close_sender,
            release_close: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let app = router(
        ApiConfig {
            served_model_name: "split-tool-model".into(),
            max_output_tokens: 32,
        },
        worker,
    );
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model":"split-tool-model",
                        "messages":[{"role":"user","content":"天气"}],
                        "tools":[{"type":"function","function":{
                            "name":"weather.get",
                            "parameters":{
                                "type":"object",
                                "properties":{
                                    "city":{"type":"string"},
                                    "date":{"type":"string"}
                                },
                                "required":["city","date"],
                                "additionalProperties":false
                            }
                        }}],
                        "tool_choice":"required",
                        "parallel_tool_calls":false,
                        "stream":true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let mut body = response.into_body().into_data_stream();
    before_close_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("all incomplete XML fragments reached the callback");
    assert!(
        tokio::time::timeout(Duration::from_millis(50), body.next())
            .await
            .is_err(),
        "incomplete tool XML must not leak into an SSE delta"
    );

    release_sender.send(()).unwrap();
    let mut wire = String::new();
    while let Some(bytes) = body.next().await {
        wire.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
    }
    assert!(!wire.contains("<tool_call>"));
    assert!(!wire.contains("<parameter="));
    let tool_delta = wire
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .find(|chunk| chunk["choices"][0]["delta"]["tool_calls"].is_array())
        .expect("validated tool-call delta");
    let arguments = tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(arguments).unwrap(),
        serde_json::json!({"city":"上海","date":"2026-10-22"})
    );
}

#[tokio::test]
async fn reports_post_start_generation_errors_as_sse_error_then_done() {
    let (app, _) = app().await;
    let (status, chunks, done) = sse_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"fail"}],
            "stream":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(done);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0]["error"]["type"], "server_error");
    assert!(chunks[0]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("fake generation failure"));
}

struct BlockingEngine {
    calls: Arc<AtomicUsize>,
    started: SyncSender<()>,
    release: Receiver<()>,
}

impl ChatEngine for BlockingEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let _ = self.started.send(());
        self.release.recv().unwrap();
        Ok(EngineOutput {
            text: "done".into(),
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

fn empty_engine_request() -> EngineRequest {
    EngineRequest {
        images: vec![],
        messages: vec![ChatCompletionMessage {
            role: "user".into(),
            content: Some(Value::String("x".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: ToolChoice::None,
        chat_template_kwargs: serde_json::Map::new(),
        enable_thinking: false,
        max_tokens: 1,
        temperature: None,
        top_k: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        stop_after_tool_call: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canceled_queued_request_is_skipped_by_the_single_worker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine_calls = calls.clone();
    let (started_sender, started_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(BlockingEngine {
            calls: engine_calls,
            started: started_sender,
            release: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let first = worker.submit(empty_engine_request()).unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("first request started");
    let canceled = worker.submit(empty_engine_request()).unwrap();
    drop(canceled);
    release_sender.send(()).unwrap();
    first.await.unwrap().unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canceled_queued_stream_is_skipped_by_the_single_worker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine_calls = calls.clone();
    let (started_sender, started_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(BlockingEngine {
            calls: engine_calls,
            started: started_sender,
            release: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let first = worker.submit(empty_engine_request()).unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("first request started");
    let canceled = worker.submit_stream(empty_engine_request()).unwrap();
    drop(canceled);
    release_sender.send(()).unwrap();
    first.await.unwrap().unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bounded_queue_rejects_overflow_then_recovers_after_cancellation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine_calls = calls.clone();
    let (started_sender, started_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(1, move || {
        Ok(Box::new(BlockingEngine {
            calls: engine_calls,
            started: started_sender,
            release: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let first = worker.submit(empty_engine_request()).unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    let queued = worker.submit(empty_engine_request()).unwrap();
    let error = worker.submit(empty_engine_request()).unwrap_err();
    assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(error.code, Some("queue_full"));
    drop(queued);
    release_sender.send(()).unwrap();
    first.await.unwrap().unwrap();
    // A canceled job still occupies its queue slot until the worker consumes
    // it. Retry only the short admission check, not a second inference.
    let mut recovered = None;
    for _ in 0..1_000 {
        match worker.submit(empty_engine_request()) {
            Ok(receiver) => {
                recovered = Some(receiver);
                break;
            }
            Err(error) if error.status == StatusCode::TOO_MANY_REQUESTS => {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            Err(error) => panic!("unexpected worker error: {error:?}"),
        }
    }
    let recovered = recovered.expect("queue capacity recovered");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    release_sender.send(()).unwrap();
    recovered.await.unwrap().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_and_admission_report_loading_and_startup_failure() {
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(1, move || {
        release_receiver.recv().unwrap();
        Err(ApiError::unavailable("checkpoint failed to load"))
    });
    let app = router(
        ApiConfig {
            served_model_name: "failed-model".into(),
            max_output_tokens: 8,
        },
        worker.clone(),
    );
    let loading = app
        .clone()
        .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(loading.status(), StatusCode::SERVICE_UNAVAILABLE);
    let error = worker.submit(empty_engine_request()).unwrap_err();
    assert_eq!(error.status, StatusCode::SERVICE_UNAVAILABLE);
    release_sender.send(()).unwrap();
    for _ in 0..1_000 {
        if worker.startup_error().is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let health = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(health.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["error"], "checkpoint failed to load");
    let (status, _) = json_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"hello"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

struct CancellationAwareEngine {
    started: SyncSender<()>,
    release: Receiver<()>,
    callback_result: SyncSender<bool>,
}

impl ChatEngine for CancellationAwareEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        panic!("streaming requests must use complete_stream")
    }

    fn complete_stream(
        &mut self,
        _request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        self.started.send(()).unwrap();
        self.release.recv().unwrap();
        self.callback_result.send(on_text("x")).unwrap();
        Ok(EngineOutput {
            text: "x".into(),
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_stream_disconnect_reaches_the_engine_callback() {
    let (started_sender, started_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let (result_sender, result_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(1, move || {
        Ok(Box::new(CancellationAwareEngine {
            started: started_sender,
            release: release_receiver,
            callback_result: result_sender,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let stream = worker.submit_stream(empty_engine_request()).unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("stream request started");
    drop(stream);
    release_sender.send(()).unwrap();
    assert!(!result_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("engine observed callback result"));
}

#[tokio::test]
async fn wraps_json_extraction_errors_in_an_openai_error_envelope() {
    let (app, calls) = app().await;
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages": ["#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "body");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

struct PanickingEngine {
    owner: std::thread::ThreadId,
    dropped: SyncSender<std::thread::ThreadId>,
    _thread_bound: std::rc::Rc<()>,
}

impl ChatEngine for PanickingEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        panic!("inference failed");
    }
}

impl Drop for PanickingEngine {
    fn drop(&mut self) {
        assert_eq!(std::thread::current().id(), self.owner);
        let _ = self.dropped.send(self.owner);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inference_panic_revokes_readiness_before_response_and_drops_model_on_owner() {
    for stream in [false, true] {
        let (dropped_sender, dropped_receiver) = sync_channel(1);
        let worker = WorkerClient::spawn(1, move || {
            Ok(Box::new(PanickingEngine {
                owner: std::thread::current().id(),
                dropped: dropped_sender,
                _thread_bound: std::rc::Rc::new(()),
            }))
        });
        wait_until_ready(&worker).await;
        let app = router(
            ApiConfig {
                served_model_name: "panic".into(),
                max_output_tokens: 8,
            },
            worker.clone(),
        );
        let request =
            serde_json::json!({"messages":[{"role":"user","content":"x"}],"stream":stream});
        if stream {
            let (_, chunks, done) = sse_response(app.clone(), request).await;
            assert!(done);
            assert_eq!(chunks.last().unwrap()["error"]["code"], "model_unavailable");
        } else {
            let (status, body) = json_response(app.clone(), request).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error"]["code"], "model_unavailable");
        }
        // No polling: failure must be visible before the client sees the error.
        assert!(!worker.ready());
        assert_eq!(worker.status_message(), "failed");
        assert!(worker.startup_error().unwrap().contains("restart required"));
        for path in ["/health", "/healthz", "/v1/models"] {
            let response = app
                .clone()
                .oneshot(Request::get(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{path}");
        }
        assert_eq!(
            worker.submit(empty_engine_request()).unwrap_err().status,
            StatusCode::SERVICE_UNAVAILABLE
        );
        dropped_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("failed model dropped on its worker thread");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn factory_panic_reports_failure_instead_of_remaining_loading() {
    let worker = WorkerClient::spawn(1, || panic!("factory failed"));
    for _ in 0..1_000 {
        if worker.status_message() == "failed" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(worker.status_message(), "failed");
    assert!(!worker.ready());
    assert!(worker.startup_error().unwrap().contains("restart required"));
    assert_eq!(
        worker.submit(empty_engine_request()).unwrap_err().status,
        StatusCode::SERVICE_UNAVAILABLE
    );
}

struct EchoEngine;
impl ChatEngine for EchoEngine {
    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        Ok(EngineOutput {
            text: request.messages[0]
                .content
                .as_ref()
                .unwrap()
                .as_str()
                .unwrap()
                .into(),
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
    fn complete_stream(
        &mut self,
        request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        let output = self.complete(request)?;
        // Deliberately split indentation, Unicode, trailing whitespace and XML
        // markers across callbacks rather than sending a single finished string.
        for character in output.text.chars() {
            if !on_text(character.encode_utf8(&mut [0; 4])) {
                break;
            }
        }
        Ok(output)
    }
}

#[tokio::test]
async fn http_replies_preserve_text_and_string_tool_arguments_across_protocols() {
    let worker = WorkerClient::spawn(1, || Ok(Box::new(EchoEngine)));
    wait_until_ready(&worker).await;
    let app = router(
        ApiConfig {
            served_model_name: "echo".into(),
            max_output_tokens: 32,
        },
        worker,
    );
    let call =
        "<tool_call><function=weather.get><parameter=city>10001</parameter></function></tool_call>";
    let with_prefix = format!(" \tchecking\n{call}");
    let tool_only = format!(" \t\n{call}");
    for (text, expected_text, has_tools) in [
        ("    print(1)\n", Some("    print(1)\n"), false),
        ("\t\n  ", Some("\t\n  "), false),
        ("\n中文 \t", Some("\n中文 \t"), false),
        (with_prefix.as_str(), Some(" \tchecking\n"), true),
        (tool_only.as_str(), None, true),
    ] {
        for anthropic in [false, true] {
            for stream in [false, true] {
                let mut body = serde_json::json!({"messages":[{"role":"user","content":text}],"max_tokens":32,"stream":stream});
                if has_tools {
                    let schema = serde_json::json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false});
                    body["tools"] = if anthropic {
                        serde_json::json!([{"name":"weather.get","input_schema":schema}])
                    } else {
                        serde_json::json!([{"type":"function","function":{"name":"weather.get","parameters":schema}}])
                    };
                }
                let path = if anthropic {
                    "/v1/messages"
                } else {
                    "/v1/chat/completions"
                };
                let response = app
                    .clone()
                    .oneshot(
                        Request::post(path)
                            .header("content-type", "application/json")
                            .body(Body::from(body.to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::OK,
                    "{path}, stream={stream}, {text:?}"
                );
                let wire = String::from_utf8(
                    to_bytes(response.into_body(), usize::MAX)
                        .await
                        .unwrap()
                        .to_vec(),
                )
                .unwrap();
                let (actual_text, arguments) = if stream {
                    let mut content = String::new();
                    let mut arguments = String::new();
                    for event in wire.split("\n\n") {
                        let Some(data) = event.lines().find_map(|line| line.strip_prefix("data: "))
                        else {
                            continue;
                        };
                        if data == "[DONE]" {
                            continue;
                        }
                        let event: Value = serde_json::from_str(data).unwrap();
                        assert!(event.get("error").is_none(), "{wire}");
                        if anthropic {
                            if let Some(text) = event["delta"]["text"].as_str() {
                                content.push_str(text);
                            }
                            if let Some(json) = event["delta"]["partial_json"].as_str() {
                                arguments.push_str(json);
                            }
                        } else {
                            let delta = &event["choices"][0]["delta"];
                            if let Some(text) = delta["content"].as_str() {
                                content.push_str(text);
                            }
                            if let Some(json) =
                                delta["tool_calls"][0]["function"]["arguments"].as_str()
                            {
                                arguments.push_str(json);
                            }
                        }
                    }
                    assert!(
                        wire.contains(if anthropic {
                            "event: message_stop"
                        } else {
                            "data: [DONE]"
                        }),
                        "{wire}"
                    );
                    assert_eq!(
                        content,
                        expected_text.unwrap_or_default(),
                        "{path}, {text:?}"
                    );
                    ((!content.is_empty()).then_some(content), arguments)
                } else {
                    let body: Value = serde_json::from_str(&wire).unwrap();
                    if anthropic {
                        let blocks = body["content"].as_array().unwrap();
                        let content = blocks
                            .iter()
                            .find(|b| b["type"] == "text")
                            .and_then(|b| b["text"].as_str())
                            .map(str::to_owned);
                        let arguments = blocks
                            .iter()
                            .find(|b| b["type"] == "tool_use")
                            .map(|b| b["input"].to_string())
                            .unwrap_or_default();
                        (content, arguments)
                    } else {
                        let message = &body["choices"][0]["message"];
                        (
                            message["content"].as_str().map(str::to_owned),
                            message["tool_calls"][0]["function"]["arguments"]
                                .as_str()
                                .unwrap_or_default()
                                .into(),
                        )
                    }
                };
                assert_eq!(actual_text.as_deref(), expected_text, "{path}, {text:?}");
                if has_tools {
                    assert_eq!(
                        serde_json::from_str::<Value>(&arguments).unwrap(),
                        serde_json::json!({"city":"10001"})
                    );
                }
            }
        }
    }
}

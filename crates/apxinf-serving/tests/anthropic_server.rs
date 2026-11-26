use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

use apxinf_serving::{
    router, ApiConfig, ApiError, ChatEngine, EngineOutput, EngineRequest, WorkerClient,
};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tower::ServiceExt;

struct CaptureEngine {
    requests: Sender<EngineRequest>,
}

#[tokio::test]
async fn anthropic_images_preserve_block_order_and_history_in_both_modes() {
    for stream in [false, true] {
        let (app, requests) = app().await;
        let (status, body) = response(app, serde_json::json!({"model":"fake-model", "max_tokens":16, "stream":stream,
            "messages":[{"role":"user", "content":[
                {"type":"text","text":"before"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"}},
                {"type":"text","text":"after"}
            ]},{"role":"assistant","content":"It is red."},{"role":"user","content":"What color was it?"}]
        })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
        let parts = request.messages[0].content.as_ref().unwrap();
        assert_eq!(parts[0]["text"], "before");
        assert_eq!(parts[1]["type"], "image");
        assert_eq!(request.images.len(), 1);
        assert_eq!(request.images[0].rgb, [255, 0, 0]);
        assert_eq!(parts[2]["text"], "after");
        assert_eq!(
            request.messages[2].content,
            Some(Value::String("What color was it?".into()))
        );
        if stream {
            assert!(body.contains("message_stop"));
        }
    }
}

impl ChatEngine for CaptureEngine {
    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        let has_tools = !request.tools.is_empty();
        let _ = self.requests.send(request);
        Ok(EngineOutput {
            text: if has_tools {
                "<tool_call>\n<function=weather.get>\n<parameter=city>\n上海\n</parameter>\n</function>\n</tool_call>".into()
            } else {
                "你好，我是 Anthropic 兼容接口。".into()
            },
            prompt_tokens: 12,
            reused_prompt_tokens: 5,
            completion_tokens: 8,
            hit_length: false,
            ttft_ms: None,
            tpot_ms: None,
            generation_tps: None,
            total_latency_ms: None,
        })
    }
}

async fn app() -> (axum::Router, Receiver<EngineRequest>) {
    let (sender, receiver) = channel();
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(CaptureEngine { requests: sender }) as Box<dyn ChatEngine>)
    });
    for _ in 0..1_000 {
        if worker.ready() {
            return (
                router(
                    ApiConfig {
                        served_model_name: "fake-model".into(),
                        max_output_tokens: 128,
                    },
                    worker,
                ),
                receiver,
            );
        }
        std::thread::sleep(Duration::from_millis(1));
        tokio::task::yield_now().await;
    }
    panic!("worker did not become ready")
}

async fn response(app: axum::Router, body: Value) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

#[tokio::test]
async fn serves_anthropic_text_messages_without_removing_openai_routes() {
    let (app, requests) = app().await;
    let (status, body) = response(
        app.clone(),
        serde_json::json!({
            "model":"fake-model",
            "max_tokens":32,
            "system":[{"type":"text","text":"简短回答"}],
            "messages":[{"role":"user","content":"你好"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["usage"]["input_tokens"], 12);
    assert_eq!(body["usage"]["cache_read_input_tokens"], 5);

    let request = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(request.messages[0].role, "system");
    assert_eq!(request.messages[1].role, "user");

    let models = app
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(models.status(), StatusCode::OK);
}

#[tokio::test]
async fn converts_anthropic_tools_and_tool_results() {
    let (app, requests) = app().await;
    let (status, body) = response(
        app.clone(),
        serde_json::json!({
            "model":"fake-model",
            "max_tokens":32,
            "messages":[{"role":"user","content":"上海天气"}],
            "tools":[{
                "name":"weather.get",
                "description":"查询天气",
                "input_schema":{
                    "type":"object",
                    "properties":{"city":{"type":"string"}},
                    "required":["city"]
                }
            }],
            "tool_choice":{"type":"any","disable_parallel_tool_use":true}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["stop_reason"], "tool_use");
    assert_eq!(body["content"][0]["type"], "tool_use");
    assert_eq!(body["content"][0]["name"], "weather.get");
    assert_eq!(body["content"][0]["input"]["city"], "上海");
    let first = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(!first.tools.is_empty());

    let (status, body) = response(
        app,
        serde_json::json!({
            "model":"fake-model",
            "max_tokens":32,
            "messages":[
                {"role":"user","content":"上海天气"},
                {"role":"assistant","content":[{
                    "type":"tool_use","id":"toolu_1","name":"weather.get",
                    "input":{"city":"上海"}
                }]},
                {"role":"user","content":[{
                    "type":"tool_result","tool_use_id":"toolu_1",
                    "content":[{"type":"text","text":"晴，20℃"}]
                }]}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let second = requests.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(second.messages[1].role, "assistant");
    assert_eq!(
        second.messages[1].tool_calls.as_ref().unwrap()[0].id,
        "toolu_1"
    );
    assert_eq!(second.messages[2].role, "tool");
    assert_eq!(second.messages[2].tool_call_id.as_deref(), Some("toolu_1"));
}

#[tokio::test]
async fn streams_anthropic_sse_event_sequence() {
    let (app, _) = app().await;
    let (status, body) = response(
        app,
        serde_json::json!({
            "model":"fake-model",
            "max_tokens":32,
            "messages":[{"role":"user","content":"你好"}],
            "stream":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let starts = body.find("event: message_start").unwrap();
    let block_start = body.find("event: content_block_start").unwrap();
    let block_delta = body.find("event: content_block_delta").unwrap();
    let block_stop = body.find("event: content_block_stop").unwrap();
    let message_delta = body.find("event: message_delta").unwrap();
    let message_stop = body.find("event: message_stop").unwrap();
    assert!(starts < block_start);
    assert!(block_start < block_delta);
    assert!(block_delta < block_stop);
    assert!(block_stop < message_delta);
    assert!(message_delta < message_stop);
    assert!(body.contains(r#""type":"text_delta""#));
    assert!(body.contains(r#""stop_reason":"end_turn""#));
}

#[tokio::test]
async fn uses_anthropic_error_envelope() {
    let (app, _) = app().await;
    let (status, body) = response(
        app,
        serde_json::json!({
            "model":"fake-model",
            "messages":[{"role":"user","content":"missing max_tokens"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

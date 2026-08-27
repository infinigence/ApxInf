use std::sync::atomic::{AtomicU64, Ordering};

use apxinf_model::{CancellationToken, RuntimeCapabilities, RuntimeError, RuntimeRequest};
use serde_json::json;

use crate::server::response::{
    error_json, result_json, sse_done_frame, sse_done_sentinel, sse_token_frame, EOS_TOKEN_IDS,
};
use crate::server::schema::{
    parse_generate_request, GenerateRequest, ProtocolError, MODEL_VOCAB_SIZE,
};

pub const EVALUATION_CONTRACT: &str = "apxinf.qwen38_27b.inference_interface.v1";
pub const MODEL_REVISION: &str = "63768c10df38c0395e12ef49edac1bd539eaeeea";

pub trait TokenStream: Send {
    fn next_token(&mut self) -> Result<Option<u32>, RuntimeError>;
    fn cancel(&self);
}

pub trait ProtocolRuntime: Send + Sync {
    fn capabilities(&self) -> RuntimeCapabilities;
    fn start(&self, request: RuntimeRequest) -> Result<Box<dyn TokenStream>, RuntimeError>;

    /// Execute a small request through the production execution path. The
    /// default keeps model-neutral fixtures usable; real runtimes override it
    /// so readiness proves the worker and device can actually execute.
    fn warmup(&self) -> Result<(), RuntimeError> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

pub struct ActiveGeneration {
    request_id: String,
    prompt_tokens: usize,
    max_new_tokens: usize,
    ignore_eos: bool,
    output_ids: Vec<u32>,
    request_cancel: CancellationToken,
    stream: Box<dyn TokenStream>,
    finished: bool,
}

impl ActiveGeneration {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub fn output_ids(&self) -> &[u32] {
        &self.output_ids
    }

    pub fn next_output_token(&mut self) -> Result<Option<u32>, RuntimeError> {
        if self.finished {
            return Ok(None);
        }
        let token_id = match self.stream.next_token() {
            Ok(Some(token_id)) => token_id,
            Ok(None) => {
                self.finished = true;
                return Ok(None);
            }
            Err(error) => {
                self.cancel();
                return Err(error);
            }
        };
        self.output_ids.push(token_id);
        let hit_eos = !self.ignore_eos && EOS_TOKEN_IDS.contains(&token_id);
        let hit_budget = self.output_ids.len() >= self.max_new_tokens;
        if hit_eos || hit_budget {
            self.cancel_runtime();
            self.finished = true;
        }
        Ok(Some(token_id))
    }

    pub fn cancel(&mut self) {
        if !self.finished {
            self.cancel_runtime();
            self.finished = true;
        }
    }

    fn cancel_runtime(&self) {
        self.request_cancel.cancel();
        self.stream.cancel();
    }
}

impl Drop for ActiveGeneration {
    fn drop(&mut self) {
        self.cancel();
    }
}

enum SsePhase {
    Tokens,
    Sentinel,
    Complete,
}

pub struct SseGeneration {
    generation: ActiveGeneration,
    phase: SsePhase,
    readiness: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl SseGeneration {
    pub fn request_id(&self) -> &str {
        self.generation.request_id()
    }

    pub fn next_frame(&mut self) -> Result<Option<Vec<u8>>, RuntimeError> {
        match self.phase {
            SsePhase::Tokens => {
                let next = match self.generation.next_output_token() {
                    Ok(next) => next,
                    Err(error) => {
                        if is_fatal_runtime_error(&error) {
                            self.readiness.store(false, Ordering::Release);
                        }
                        self.phase = SsePhase::Complete;
                        return Err(error);
                    }
                };
                if let Some(token_id) = next {
                    let index = self.generation.output_ids().len() - 1;
                    return Ok(Some(sse_token_frame(self.request_id(), index, token_id)));
                }
                self.phase = SsePhase::Sentinel;
                Ok(Some(sse_done_frame(
                    self.request_id(),
                    self.generation.prompt_tokens(),
                    self.generation.output_ids(),
                )))
            }
            SsePhase::Sentinel => {
                self.phase = SsePhase::Complete;
                Ok(Some(sse_done_sentinel()))
            }
            SsePhase::Complete => Ok(None),
        }
    }

    pub fn cancel(&mut self) {
        self.generation.cancel();
        self.phase = SsePhase::Complete;
    }
}

pub struct ProtocolService<R> {
    runtime: R,
    next_request_id: AtomicU64,
    capabilities: RuntimeCapabilities,
    stub: bool,
    ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    recovery_lock: std::sync::Mutex<()>,
    /// Present only when the multimodal capability is enabled; its absence
    /// makes `/v1/chat/completions` fail closed with `unsupported_capability`
    /// and keeps `/health.capabilities.multimodal` false.
    chat: Option<std::sync::Arc<crate::server::chat::ChatPreprocessor>>,
}

impl<R: ProtocolRuntime> ProtocolService<R> {
    pub fn new(runtime: R, stub: bool) -> Self {
        Self::with_readiness(runtime, stub, true)
    }

    pub fn new_unready(runtime: R, stub: bool) -> Self {
        Self::with_readiness(runtime, stub, false)
    }

    /// Enable the multimodal chat endpoint. Only called when
    /// `APXINF_ENABLE_MULTIMODAL=1` and the vision tower is resident.
    pub fn with_chat(mut self, chat: crate::server::chat::ChatPreprocessor) -> Self {
        self.chat = Some(std::sync::Arc::new(chat));
        self
    }

    fn with_readiness(runtime: R, stub: bool, ready: bool) -> Self {
        let capabilities = runtime.capabilities();
        assert_eq!(
            capabilities.vocab_size, MODEL_VOCAB_SIZE as usize,
            "protocol runtime vocab_size must match the model config"
        );
        Self {
            runtime,
            next_request_id: AtomicU64::new(1),
            capabilities,
            stub,
            ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(ready)),
            recovery_lock: std::sync::Mutex::new(()),
            chat: None,
        }
    }

    pub fn health_json(&self) -> Vec<u8> {
        self.health_json_inner()
    }

    fn health_json_inner(&self) -> Vec<u8> {
        let ready = self.ready.load(Ordering::Acquire);
        serde_json::to_vec(&json!({
            "status": if ready { "ok" } else { "unhealthy" },
            "evaluation_contract": EVALUATION_CONTRACT,
            "model_revision": MODEL_REVISION,
            "vocab_size": MODEL_VOCAB_SIZE,
            "max_model_len": self.capabilities.max_model_len,
            "parallel_requests": self.capabilities.parallel_requests,
            "fallback_active": false,
            "capabilities": {
                "pretokenized_input_ids": true,
                "token_id_output": true,
                "multimodal": self.chat.is_some(),
            },
            "stub": self.stub,
        }))
        .expect("JSON serialization is infallible")
    }

    /// Return the health response, attempting one serialized worker warmup if
    /// the previous request left the service unhealthy.
    pub fn health_response(&self) -> HttpResponse {
        if !self.ready.load(Ordering::Acquire) {
            let _guard = self.recovery_guard();
            if !self.ready.load(Ordering::Acquire) {
                let _ = self.warmup_inner();
            }
        }
        let ready = self.ready.load(Ordering::Acquire);
        HttpResponse {
            status: if ready { 200 } else { 503 },
            content_type: "application/json",
            body: self.health_json_inner(),
        }
    }

    pub fn warmup(&self) -> Result<(), RuntimeError> {
        let _guard = self.recovery_guard();
        self.warmup_inner()
    }

    fn recovery_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        match self.recovery_lock.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn warmup_inner(&self) -> Result<(), RuntimeError> {
        self.ready.store(false, Ordering::Release);
        match self.runtime.warmup() {
            Ok(()) => {
                self.ready.store(true, Ordering::Release);
                Ok(())
            }
            Err(error) => {
                self.ready.store(false, Ordering::Release);
                Err(error)
            }
        }
    }

    pub fn mark_unhealthy(&self) {
        self.ready.store(false, Ordering::Release);
    }

    pub fn mark_ready(&self) {
        self.ready.store(true, Ordering::Release);
    }

    pub fn handle_non_stream(&self, raw: &[u8]) -> HttpResponse {
        self.handle_non_stream_with_cancel(raw, CancellationToken::new())
    }

    /// Non-streaming request path with an externally owned cancellation
    /// token, so the HTTP layer can abort the generation when the client
    /// disconnects before the result is written.
    pub fn handle_non_stream_with_cancel(
        &self,
        raw: &[u8],
        cancel: CancellationToken,
    ) -> HttpResponse {
        let request = match parse_generate_request(raw, self.capabilities.max_model_len) {
            Ok(request) => request,
            Err(error) => return invalid_response(error),
        };
        if request.stream {
            return invalid_response(ProtocolError::invalid(
                "stream=true requires the streaming response path",
            ));
        }
        let mut generation = match self.start(request, cancel) {
            Ok(generation) => generation,
            Err(error) => {
                self.note_runtime_error(&error);
                return runtime_response(error);
            }
        };
        loop {
            match generation.next_output_token() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => {
                    self.note_runtime_error(&error);
                    return runtime_response(error);
                }
            }
        }
        HttpResponse {
            status: 200,
            content_type: "application/json",
            body: result_json(
                generation.request_id(),
                generation.prompt_tokens(),
                generation.output_ids(),
            ),
        }
    }

    pub fn start_stream(&self, raw: &[u8]) -> Result<SseGeneration, HttpResponse> {
        self.start_stream_with_cancel(raw, CancellationToken::new())
    }

    /// Streaming request path with an externally owned cancellation token,
    /// so the HTTP layer can abort the generation while it is still inside
    /// prefill and no frame has been produced yet.
    pub fn start_stream_with_cancel(
        &self,
        raw: &[u8],
        cancel: CancellationToken,
    ) -> Result<SseGeneration, HttpResponse> {
        let request = parse_generate_request(raw, self.capabilities.max_model_len)
            .map_err(invalid_response)?;
        if !request.stream {
            return Err(invalid_response(ProtocolError::invalid(
                "stream=false requires the non-streaming response path",
            )));
        }
        let generation = self.start(request, cancel).map_err(|error| {
            self.note_runtime_error(&error);
            runtime_response(error)
        })?;
        Ok(SseGeneration {
            generation,
            phase: SsePhase::Tokens,
            readiness: std::sync::Arc::clone(&self.ready),
        })
    }

    fn start(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ActiveGeneration, RuntimeError> {
        if !self.ready.load(Ordering::Acquire) {
            return Err(RuntimeError::Unhealthy);
        }
        let mut runtime_request =
            RuntimeRequest::new(request.input_ids.clone(), request.max_new_tokens);
        runtime_request.cancel = cancel;
        let request_cancel = runtime_request.cancel.clone();
        let stream = self.runtime.start(runtime_request)?;
        Ok(ActiveGeneration {
            request_id: self.next_id(),
            prompt_tokens: request.input_ids.len(),
            max_new_tokens: request.max_new_tokens,
            ignore_eos: request.ignore_eos,
            output_ids: Vec::with_capacity(request.max_new_tokens),
            request_cancel,
            stream,
            finished: false,
        })
    }

    /// `/v1/chat/completions`: the multimodal image endpoint. Fails closed
    /// with `unsupported_capability` whenever the capability is not enabled,
    /// per the multimodal contract's unsupported rule (HTTP 400 and never
    /// 200/500, no fallback, images are never ignored).
    pub fn handle_chat_completions(&self, raw: &[u8], cancel: CancellationToken) -> HttpResponse {
        let Some(chat) = self.chat.as_ref() else {
            return HttpResponse {
                status: 400,
                content_type: "application/json",
                body: br#"{"error":{"type":"unsupported_capability","message":"multimodal image input is not enabled on this service"}}"#
                    .to_vec(),
            };
        };
        let prepared = match chat.prepare(raw) {
            Ok(prepared) => prepared,
            Err(error) => {
                return invalid_response(ProtocolError::invalid(&error.0));
            }
        };
        if !self.ready.load(Ordering::Acquire) {
            return runtime_response(RuntimeError::Unhealthy);
        }
        let prompt_tokens = prepared.input_ids.len();
        let max_new_tokens = prepared.max_new_tokens;
        let mut runtime_request = RuntimeRequest::new(prepared.input_ids, max_new_tokens);
        runtime_request.cancel = cancel;
        runtime_request.multimodal = Some(prepared.multimodal);
        let request_cancel = runtime_request.cancel.clone();
        let stream = match self.runtime.start(runtime_request) {
            Ok(stream) => stream,
            Err(error) => {
                self.note_runtime_error(&error);
                return runtime_response(error);
            }
        };
        let mut generation = ActiveGeneration {
            request_id: self.next_id(),
            prompt_tokens,
            max_new_tokens,
            ignore_eos: false,
            output_ids: Vec::with_capacity(max_new_tokens),
            request_cancel,
            stream,
            finished: false,
        };
        loop {
            match generation.next_output_token() {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => {
                    self.note_runtime_error(&error);
                    return runtime_response(error);
                }
            }
        }
        let output_ids = generation.output_ids();
        let hit_eos = output_ids
            .last()
            .is_some_and(|token| EOS_TOKEN_IDS.contains(token));
        let content = match chat.decode(output_ids) {
            Ok(content) => content,
            Err(error) => {
                return runtime_response(RuntimeError::Execution(error));
            }
        };
        HttpResponse {
            status: 200,
            content_type: "application/json",
            body: crate::server::chat::chat_completion_json(
                generation.request_id(),
                content.trim(),
                prompt_tokens,
                output_ids.len(),
                if hit_eos { "stop" } else { "length" },
            ),
        }
    }

    fn note_runtime_error(&self, error: &RuntimeError) {
        if is_fatal_runtime_error(error) {
            self.mark_unhealthy();
        }
    }

    fn next_id(&self) -> String {
        format!(
            "req-{}",
            self.next_request_id.fetch_add(1, Ordering::Relaxed)
        )
    }
}

fn invalid_response(error: ProtocolError) -> HttpResponse {
    HttpResponse {
        status: 400,
        content_type: "application/json",
        body: error_json(&error),
    }
}

pub(crate) fn runtime_response(error: RuntimeError) -> HttpResponse {
    let (status, error_type) = match error {
        RuntimeError::Capacity => (503, "capacity"),
        RuntimeError::QueueFull => (503, "capacity"),
        RuntimeError::WorkerStopped => (503, "unavailable"),
        RuntimeError::Admission(_) => (400, "invalid_request"),
        RuntimeError::Cancelled => (499, "cancelled"),
        RuntimeError::Unhealthy => (503, "unhealthy"),
        RuntimeError::Execution(_) => (500, "runtime_error"),
    };
    let protocol_error = ProtocolError {
        error_type,
        message: error.to_string(),
    };
    HttpResponse {
        status,
        content_type: "application/json",
        body: error_json(&protocol_error),
    }
}

fn is_fatal_runtime_error(error: &RuntimeError) -> bool {
    matches!(
        error,
        RuntimeError::Execution(_) | RuntimeError::WorkerStopped
    )
}

#[cfg(test)]
mod tests {
    use super::{
        ProtocolRuntime, ProtocolService, TokenStream, EVALUATION_CONTRACT, MODEL_REVISION,
    };
    use apxinf_model::{CancellationToken, RuntimeCapabilities, RuntimeError, RuntimeRequest};
    use serde_json::Value;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    enum StartResult {
        Tokens(Vec<u32>),
        TokensThenError(Vec<u32>, RuntimeError),
        Error(RuntimeError),
    }

    struct FakeStream {
        tokens: VecDeque<u32>,
        terminal_error: Option<RuntimeError>,
        cancelled: Arc<Mutex<bool>>,
    }

    impl TokenStream for FakeStream {
        fn next_token(&mut self) -> Result<Option<u32>, RuntimeError> {
            if let Some(token) = self.tokens.pop_front() {
                return Ok(Some(token));
            }
            if let Some(error) = self.terminal_error.take() {
                return Err(error);
            }
            Ok(None)
        }

        fn cancel(&self) {
            *self.cancelled.lock().unwrap() = true;
        }
    }

    struct FakeRuntime {
        starts: Mutex<VecDeque<StartResult>>,
        cancelled: Arc<Mutex<bool>>,
        capabilities: RuntimeCapabilities,
        request_cancel: Arc<Mutex<Option<CancellationToken>>>,
    }

    impl FakeRuntime {
        fn new(starts: Vec<StartResult>) -> Self {
            Self {
                starts: Mutex::new(starts.into()),
                cancelled: Arc::new(Mutex::new(false)),
                capabilities: RuntimeCapabilities::frozen_qwen35(32, 0),
                request_cancel: Arc::new(Mutex::new(None)),
            }
        }

        fn with_vocab_size(mut self, vocab_size: usize) -> Self {
            self.capabilities.vocab_size = vocab_size;
            self
        }
    }

    impl ProtocolRuntime for FakeRuntime {
        fn capabilities(&self) -> RuntimeCapabilities {
            self.capabilities
        }

        fn start(&self, request: RuntimeRequest) -> Result<Box<dyn TokenStream>, RuntimeError> {
            *self.request_cancel.lock().unwrap() = Some(request.cancel.clone());
            match self.starts.lock().unwrap().pop_front().unwrap() {
                StartResult::Tokens(tokens) => Ok(Box::new(FakeStream {
                    tokens: tokens.into(),
                    terminal_error: None,
                    cancelled: Arc::clone(&self.cancelled),
                })),
                StartResult::TokensThenError(tokens, error) => Ok(Box::new(FakeStream {
                    tokens: tokens.into(),
                    terminal_error: Some(error),
                    cancelled: Arc::clone(&self.cancelled),
                })),
                StartResult::Error(error) => Err(error),
            }
        }
    }

    fn body(ignore_eos: bool, stream: bool) -> Vec<u8> {
        body_with_budget(ignore_eos, stream, 3)
    }

    fn body_with_budget(ignore_eos: bool, stream: bool, max_new_tokens: usize) -> Vec<u8> {
        format!(
            r#"{{"input_ids":[1,2,3,4,5,6,7,8],"max_new_tokens":{max_new_tokens},"temperature":0,"ignore_eos":{ignore_eos},"stream":{stream}}}"#
        )
        .into_bytes()
    }

    #[test]
    fn health_has_frozen_contract_identity() {
        let service = ProtocolService::new(FakeRuntime::new(vec![]), true);
        let value: Value = serde_json::from_slice(&service.health_json()).unwrap();
        assert_eq!(value["status"], "ok");
        assert_eq!(value["evaluation_contract"], EVALUATION_CONTRACT);
        assert_eq!(value["model_revision"], MODEL_REVISION);
        assert_eq!(value["vocab_size"], 248_320);
        assert_eq!(value["max_model_len"], 32);
        assert_eq!(value["parallel_requests"], 1);
        assert_eq!(value["fallback_active"], false);
        assert_eq!(value["capabilities"]["pretokenized_input_ids"], true);
        assert_eq!(value["capabilities"]["token_id_output"], true);
        assert_eq!(value["capabilities"]["multimodal"], false);
        assert_eq!(value["stub"], true);
    }

    #[test]
    fn non_stream_consumes_incremental_tokens_with_usage() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![StartResult::Tokens(vec![7, 8, 9])]),
            true,
        );
        let response = service.handle_non_stream(&body(true, false));
        assert_eq!(response.status, 200);
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["type"], "result");
        assert_eq!(value["output_ids"], serde_json::json!([7, 8, 9]));
        assert_eq!(value["usage"]["prompt_tokens"], 8);
        assert_eq!(value["usage"]["completion_tokens"], 3);
    }

    #[test]
    fn eos_is_included_then_cancels_remaining_generation() {
        let runtime = FakeRuntime::new(vec![StartResult::Tokens(vec![7, 248_046, 9])]);
        let cancelled = Arc::clone(&runtime.cancelled);
        let service = ProtocolService::new(runtime, true);
        let response = service.handle_non_stream(&body(false, false));
        assert_eq!(response.status, 200);
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["output_ids"], serde_json::json!([7, 248_046]));
        assert!(*cancelled.lock().unwrap());
    }

    #[test]
    fn capacity_error_maps_to_503_and_service_recovers() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![
                StartResult::Error(RuntimeError::Capacity),
                StartResult::Tokens(vec![7, 7, 7]),
            ]),
            true,
        );
        let failed = service.handle_non_stream(&body(true, false));
        assert_eq!(failed.status, 503);
        let value: Value = serde_json::from_slice(&failed.body).unwrap();
        assert_eq!(value["error"]["type"], "capacity");
        assert_eq!(
            serde_json::from_slice::<Value>(&service.health_json()).unwrap()["status"],
            "ok"
        );
        assert_eq!(service.handle_non_stream(&body(true, false)).status, 200);
    }

    #[test]
    fn queue_full_maps_to_capacity_and_service_recovers() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![
                StartResult::Error(RuntimeError::QueueFull),
                StartResult::Tokens(vec![7]),
            ]),
            true,
        );
        let failed = service.handle_non_stream(&body(true, false));
        assert_eq!(failed.status, 503);
        let value: Value = serde_json::from_slice(&failed.body).unwrap();
        assert_eq!(value["error"]["type"], "capacity");
        assert_eq!(service.handle_non_stream(&body(true, false)).status, 200);
    }

    #[test]
    fn execution_error_rejects_requests_until_health_warmup_recovers() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![
                StartResult::TokensThenError(vec![], RuntimeError::Execution("boom".into())),
                StartResult::Tokens(vec![7]),
            ]),
            true,
        );
        assert_eq!(service.handle_non_stream(&body(true, false)).status, 500);
        assert_eq!(service.handle_non_stream(&body(true, false)).status, 503);
        assert_eq!(service.health_response().status, 200);
        assert_eq!(service.handle_non_stream(&body(true, false)).status, 200);
    }

    #[test]
    fn execution_error_marks_health_unhealthy_until_warmup_recovers() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![
                StartResult::TokensThenError(vec![], RuntimeError::Execution("boom".into())),
                StartResult::Tokens(vec![7]),
            ]),
            false,
        );

        assert_eq!(service.handle_non_stream(&body(true, false)).status, 500);
        service.mark_unhealthy();
        assert_eq!(
            serde_json::from_slice::<Value>(&service.health_json()).unwrap()["status"],
            "unhealthy"
        );
        let unhealthy: Value = serde_json::from_slice(&service.health_json()).unwrap();
        assert_eq!(unhealthy["status"], "unhealthy");

        service.mark_ready();
        assert_eq!(service.health_response().status, 200);
        let healthy: Value = serde_json::from_slice(&service.health_json()).unwrap();
        assert_eq!(healthy["status"], "ok");
    }

    #[test]
    fn protocol_errors_map_to_400_without_starting_runtime() {
        let service = ProtocolService::new(FakeRuntime::new(vec![]), true);
        let response = service.handle_non_stream(b"{not-json");
        assert_eq!(response.status, 400);
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request");
    }

    #[test]
    fn service_rejects_stream_mode_mismatch() {
        let service =
            ProtocolService::new(FakeRuntime::new(vec![StartResult::Tokens(vec![7])]), true);
        let response = service.handle_non_stream(&body(true, true));
        assert_eq!(response.status, 400);
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request");
    }

    #[test]
    fn budget_exhaustion_cancels_real_stream() {
        let runtime = FakeRuntime::new(vec![StartResult::Tokens(vec![7, 8, 9])]);
        let cancelled = Arc::clone(&runtime.cancelled);
        let service = ProtocolService::new(runtime, true);
        let response = service.handle_non_stream(&body(true, false));
        assert_eq!(response.status, 200);
        assert!(*cancelled.lock().unwrap());
    }

    #[test]
    fn stream_response_consumes_one_token_per_step_and_observes_cancel() {
        let runtime = FakeRuntime::new(vec![StartResult::Tokens(vec![7, 8, 9])]);
        let cancelled = Arc::clone(&runtime.cancelled);
        let service = ProtocolService::new(runtime, true);
        let mut generation = service.start_stream(&body(true, true)).unwrap();
        assert_eq!(generation.next_frame().unwrap().unwrap()[0], b'd');
        assert_eq!(generation.next_frame().unwrap().unwrap()[0], b'd');
        assert!(!*cancelled.lock().unwrap());
        generation.cancel();
        assert!(*cancelled.lock().unwrap());
    }

    #[test]
    fn health_uses_model_config_vocab_not_tokenizer_vocab() {
        let result = std::panic::catch_unwind(|| {
            ProtocolService::new(FakeRuntime::new(vec![]).with_vocab_size(248_044), true)
        });
        assert!(result.is_err());
    }

    #[test]
    fn worker_stopped_maps_to_unavailable_503() {
        let response = super::runtime_response(RuntimeError::WorkerStopped);
        assert_eq!(response.status, 503);
        let value: Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["error"]["type"], "unavailable");
    }

    #[test]
    fn poisoned_recovery_mutex_still_provides_a_serialized_guard() {
        let service = ProtocolService::new(FakeRuntime::new(vec![]), true);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = service.recovery_lock.lock().unwrap();
            panic!("poison recovery lock");
        }));
        assert!(result.is_err());
        let guard = service.recovery_guard();
        drop(guard);
    }

    #[test]
    fn stream_error_cancels_both_handles_and_never_emits_done() {
        let runtime = FakeRuntime::new(vec![StartResult::TokensThenError(
            vec![7],
            RuntimeError::Execution("boom".to_owned()),
        )]);
        let stream_cancelled = Arc::clone(&runtime.cancelled);
        let request_cancel = Arc::clone(&runtime.request_cancel);
        let service = ProtocolService::new(runtime, true);
        let mut generation = service.start_stream(&body(true, true)).unwrap();
        assert!(generation.next_frame().unwrap().is_some());
        assert_eq!(
            generation.next_frame().unwrap_err(),
            RuntimeError::Execution("boom".to_owned())
        );
        assert!(generation.next_frame().unwrap().is_none());
        assert!(*stream_cancelled.lock().unwrap());
        assert!(request_cancel
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .is_cancelled());
    }

    #[test]
    fn stream_frames_keep_request_identity_and_contiguous_indexes() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![StartResult::Tokens(vec![7, 8, 9])]),
            true,
        );
        let mut generation = service
            .start_stream(&body_with_budget(true, true, 3))
            .unwrap();
        let frames = [
            generation.next_frame().unwrap().unwrap(),
            generation.next_frame().unwrap().unwrap(),
        ];
        let request_id = generation.request_id().to_owned();
        for (expected_index, frame) in frames.iter().enumerate() {
            let value: Value = serde_json::from_str(
                String::from_utf8(frame.clone())
                    .unwrap()
                    .trim_start_matches("data: ")
                    .trim(),
            )
            .unwrap();
            assert_eq!(value["request_id"], request_id);
            assert_eq!(value["index"], expected_index);
        }
    }

    #[test]
    fn separate_requests_receive_non_overlapping_ids() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![
                StartResult::Tokens(vec![7]),
                StartResult::Tokens(vec![8]),
            ]),
            true,
        );
        let first = service.start_stream(&body(true, true)).unwrap();
        let first_id = first.request_id().to_owned();
        drop(first);
        let second = service.start_stream(&body(true, true)).unwrap();
        assert_ne!(first_id, second.request_id());
    }

    #[test]
    fn stream_eos_is_emitted_once_then_done_and_sentinel() {
        let service = ProtocolService::new(
            FakeRuntime::new(vec![StartResult::Tokens(vec![7, 248_044, 9])]),
            true,
        );
        let mut generation = service.start_stream(&body(false, true)).unwrap();
        let first = generation.next_frame().unwrap().unwrap();
        let eos = generation.next_frame().unwrap().unwrap();
        let done = generation.next_frame().unwrap().unwrap();
        let sentinel = generation.next_frame().unwrap().unwrap();
        assert_eq!(generation.next_frame().unwrap(), None);
        assert!(String::from_utf8(first).unwrap().contains("\"token_id\":7"));
        assert!(String::from_utf8(eos)
            .unwrap()
            .contains("\"token_id\":248044"));
        let done: Value = serde_json::from_str(
            String::from_utf8(done)
                .unwrap()
                .trim_start_matches("data: ")
                .trim(),
        )
        .unwrap();
        assert_eq!(done["type"], "done");
        assert_eq!(done["usage"]["completion_tokens"], 2);
        assert_eq!(sentinel, b"data: [DONE]\n\n");
    }

    #[test]
    fn natural_eof_finishes_without_cancelling_runtime() {
        let runtime = FakeRuntime::new(vec![StartResult::Tokens(vec![7])]);
        let cancelled = Arc::clone(&runtime.cancelled);
        let service = ProtocolService::new(runtime, true);
        let mut generation = service.start_stream(&body(true, true)).unwrap();
        assert!(generation.next_frame().unwrap().is_some());
        assert!(generation.next_frame().unwrap().is_some());
        assert!(generation.next_frame().unwrap().is_some());
        assert_eq!(generation.next_frame().unwrap(), None);
        assert!(!*cancelled.lock().unwrap());
    }

    #[test]
    fn dropping_unfinished_stream_cancels_both_handles() {
        let runtime = FakeRuntime::new(vec![StartResult::Tokens(vec![7, 8, 9])]);
        let stream_cancelled = Arc::clone(&runtime.cancelled);
        let request_cancel = Arc::clone(&runtime.request_cancel);
        let service = ProtocolService::new(runtime, true);
        let generation = service
            .start_stream(&body_with_budget(true, true, 3))
            .unwrap();
        drop(generation);
        assert!(*stream_cancelled.lock().unwrap());
        assert!(request_cancel
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .is_cancelled());
    }

    #[test]
    fn stream_path_rejects_non_stream_request() {
        let service = ProtocolService::new(FakeRuntime::new(vec![]), true);
        match service.start_stream(&body(true, false)) {
            Ok(_) => panic!("non-stream request unexpectedly accepted by SSE path"),
            Err(response) => assert_eq!(response.status, 400),
        }
    }
}

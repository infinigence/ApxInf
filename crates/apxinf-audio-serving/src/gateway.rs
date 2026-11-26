use crate::{
    config::sha256,
    contracts::{self, AttemptKey, AudioLimits, AudioOperation, AudioTask, MediaGrant},
    service::{AudioError, AudioService, Output},
    wav,
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full, Limited};
use hyper::{body::Incoming, header, Request, Response};
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    sync::{atomic::Ordering, Arc},
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

pub type AudioResponse = Response<Full<Bytes>>;
#[derive(Clone)]
pub struct Gateway {
    pub service: AudioService,
    pub uploads: Arc<Semaphore>,
}
fn response(status: u16, content_type: &str, body: Bytes) -> AudioResponse {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, body.len())
        .header(header::CONNECTION, "close")
        .body(Full::new(body))
        .unwrap()
}
fn json(status: u16, value: serde_json::Value) -> AudioResponse {
    response(
        status,
        "application/json",
        Bytes::from(serde_json::to_vec(&value).unwrap()),
    )
}
pub fn error(error: AudioError) -> AudioResponse {
    json(
        error.status(),
        serde_json::json!({"error":{"code":error.code,"message":error.message}}),
    )
}
fn invalid() -> AudioError {
    AudioError::new("invalid_request")
}

pub async fn handle(
    request: Request<Incoming>,
    gateway: Gateway,
    received: Instant,
) -> Result<AudioResponse, Infallible> {
    let result = route(request, &gateway, received)
        .await
        .unwrap_or_else(error);
    Ok(result)
}

async fn route(
    request: Request<Incoming>,
    gateway: &Gateway,
    received: Instant,
) -> Result<AudioResponse, AudioError> {
    let service = &gateway.service;
    let model = &service.config.manifest;
    let obs = &service.observations;
    if request.uri().query().is_some() {
        return Err(invalid());
    }
    if request.method() == hyper::Method::GET {
        let value = match request.uri().path() {
            "/healthz" => serde_json::json!({"status":"ok"}),
            "/readyz" => {
                let ready = obs.ready.load(Ordering::SeqCst);
                let diagnostics: Vec<u8> =
                    obs.diagnostics.lock().unwrap().iter().copied().collect();
                return Ok(json(
                    if ready { 200 } else { 503 },
                    serde_json::json!({"ready":ready,"task":model.task,
                    "model":model.model,"provider":model.provider,"worker_epoch":service.epoch,
                    "model_revision":service.config.revision,"capability_revision":service.config.capability_revision,
                    "limits":AudioLimits::default(),"pressure_policy":if service.config.pressure { "macos_normal" } else { "disabled_cpu_fixture" },
                    "qualification":"open","reaped":obs.reaped.load(Ordering::SeqCst),
                    "failure":*obs.failure.lock().unwrap(),"diagnostics":String::from_utf8_lossy(&diagnostics)}),
                ));
            }
            "/v1/models" => {
                serde_json::json!({"object":"list","data":[{"id":model.model,"object":"model","owned_by":"apxinf",
                "audio":{"task":model.task,"model_revision":service.config.revision,"languages":model.languages,
                    "voices":model.voices.keys().collect::<Vec<_>>(),"streaming":false}}]})
            }
            "/metrics" => {
                let text = format!("apxinf_audio_ready {}\napxinf_audio_active {}\napxinf_audio_queued {}\napxinf_audio_media_reserved_bytes {}\napxinf_audio_completed_total {}\napxinf_audio_faults_total {}\napxinf_audio_worker_reaped {}\n",
                    usize::from(obs.ready.load(Ordering::SeqCst)),obs.active.load(Ordering::SeqCst),obs.queued.load(Ordering::SeqCst),
                    service.media.reserved.load(Ordering::SeqCst),obs.completed.load(Ordering::SeqCst),obs.faults.load(Ordering::SeqCst),usize::from(obs.reaped.load(Ordering::SeqCst)));
                return Ok(response(
                    200,
                    "text/plain; version=0.0.4",
                    Bytes::from(text),
                ));
            }
            _ => {
                return Ok(json(
                    404,
                    serde_json::json!({"error":{"code":"unsupported_feature","message":"The audio route is unavailable."}}),
                ))
            }
        };
        return Ok(json(200, value));
    }
    if request.method() != hyper::Method::POST {
        return Err(AudioError::new("unsupported_feature"));
    }
    let task = match request.uri().path() {
        "/v1/audio/transcriptions" => AudioTask::Asr,
        "/v1/audio/speech" => AudioTask::Tts,
        _ => return Err(AudioError::new("unsupported_feature")),
    };
    if task != model.task {
        return Err(AudioError::new("task_mismatch"));
    }
    if !obs.ready.load(Ordering::SeqCst) {
        return Err(AudioError::new("model_unavailable"));
    }
    if request
        .headers()
        .get_all(header::CONTENT_LENGTH)
        .iter()
        .count()
        != 1
        || request.headers().contains_key(header::TRANSFER_ENCODING)
        || request.headers().contains_key(header::CONTENT_ENCODING)
        || request
            .headers()
            .get_all(header::CONTENT_TYPE)
            .iter()
            .count()
            != 1
    {
        return Err(invalid());
    }
    let content_length = request
        .headers()
        .get(header::CONTENT_LENGTH)
        .unwrap()
        .to_str()
        .map_err(|_| invalid())?
        .parse::<usize>()
        .map_err(|_| invalid())?;
    let cap = if task == AudioTask::Asr {
        contracts::MAX_UPLOAD + 65536
    } else {
        32768
    };
    if content_length == 0 || content_length > cap {
        return Err(AudioError::new("payload_too_large"));
    }
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .map_err(|_| invalid())?
        .to_owned();
    let _upload = gateway
        .uploads
        .clone()
        .try_acquire_owned()
        .map_err(|_| AudioError::new("capacity_unavailable"))?;
    let lease = service
        .media
        .reserve(content_length.checked_mul(3).ok_or_else(invalid)?)
        .map_err(|_| AudioError::new("capacity_unavailable"))?;
    let ingress_left = Duration::from_secs(15).saturating_sub(received.elapsed());
    let body = tokio::time::timeout(
        ingress_left,
        Limited::new(request.into_body(), cap).collect(),
    )
    .await
    .map_err(|_| AudioError::new("deadline_exceeded"))?
    .map_err(|_| invalid())?
    .to_bytes();
    if body.len() != content_length {
        return Err(invalid());
    }
    let key = AttemptKey {
        request_id: uuid::Uuid::new_v4().to_string(),
        attempt: 1,
    };
    let operation = if task == AudioTask::Asr {
        let fields = transcription(body, &content_type, model).await?;
        let info =
            wav::inspect(&fields.audio, true).map_err(|_| AudioError::new("invalid_audio"))?;
        let handle = uuid::Uuid::new_v4().to_string();
        lease
            .publish(&handle, &fields.audio)
            .map_err(|_| AudioError::new("capacity_unavailable"))?;
        AudioOperation::Asr {
            input: MediaGrant {
                handle,
                worker_epoch: service.epoch.clone(),
                key: key.clone(),
                bytes: fields.audio.len() as u64,
                sha256: sha256(&fields.audio),
                frames: info.frames,
                sample_rate: info.sample_rate as u64,
            },
            language: fields.language,
        }
    } else {
        if content_type.split(';').next() != Some("application/json") {
            return Err(invalid());
        }
        speech(&body, model)?
    };
    drop(_upload);
    let deadline = received + Duration::from_secs(120);
    let (result, _cancellation) = service.submit(operation, lease, key, deadline)?;
    let output = tokio::time::timeout(deadline.saturating_duration_since(Instant::now()), result)
        .await
        .map_err(|_| AudioError::new("deadline_exceeded"))?
        .map_err(|_| AudioError::new("worker_lost"))??;
    match output {
        Output::Transcript(text) => Ok(json(200, serde_json::json!({"text":text}))),
        Output::Speech(media) => Ok(response(200, "audio/wav", Bytes::from_owner(media))),
    }
}

struct Transcription {
    audio: Bytes,
    language: String,
}
async fn transcription(
    body: Bytes,
    content_type: &str,
    manifest: &crate::config::Manifest,
) -> Result<Transcription, AudioError> {
    let boundary = multer::parse_boundary(content_type).map_err(|_| invalid())?;
    if boundary.len() > 70 {
        return Err(invalid());
    }
    let stream = futures_util::stream::once(async move { Ok::<Bytes, std::io::Error>(body) });
    let constraints = multer::Constraints::new().size_limit(
        multer::SizeLimit::new()
            .whole_stream((contracts::MAX_UPLOAD + 65536) as u64)
            .per_field(contracts::MAX_UPLOAD as u64),
    );
    let mut multipart = multer::Multipart::with_constraints(stream, boundary, constraints);
    let mut seen = HashSet::new();
    let mut values = HashMap::new();
    let mut audio = None;
    while let Some(field) = multipart.next_field().await.map_err(|_| invalid())? {
        let name = field.name().ok_or_else(invalid)?.to_owned();
        if !seen.insert(name.clone()) {
            return Err(invalid());
        }
        if [
            "prompt",
            "temperature",
            "stream",
            "timestamp_granularities[]",
            "timestamp_granularities",
            "chunking_strategy",
            "stream_format",
        ]
        .contains(&name.as_str())
        {
            return Err(AudioError::new("unsupported_feature"));
        }
        if name == "file" {
            audio = Some(
                field
                    .bytes()
                    .await
                    .map_err(|_| AudioError::new("payload_too_large"))?,
            );
        } else {
            if !["model", "language", "response_format"].contains(&name.as_str())
                || field.file_name().is_some()
            {
                return Err(invalid());
            }
            let bytes = field.bytes().await.map_err(|_| invalid())?;
            if bytes.len() > 128 {
                return Err(invalid());
            }
            values.insert(
                name,
                String::from_utf8(bytes.to_vec()).map_err(|_| invalid())?,
            );
        }
    }
    if values.get("model") != Some(&manifest.model) {
        return Err(if values.contains_key("model") {
            AudioError::new("model_not_found")
        } else {
            invalid()
        });
    }
    if values.get("response_format").is_some_and(|s| s != "json") {
        return Err(AudioError::new("unsupported_feature"));
    }
    let language = values
        .remove("language")
        .unwrap_or_else(|| manifest.languages[0].clone());
    if !manifest.languages.contains(&language) {
        return Err(AudioError::new("unsupported_feature"));
    }
    let audio = audio.ok_or_else(invalid)?;
    if audio.len() > contracts::MAX_UPLOAD {
        return Err(AudioError::new("payload_too_large"));
    }
    Ok(Transcription { audio, language })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(default)]
    response_format: Option<String>,
}
fn speech(body: &[u8], manifest: &crate::config::Manifest) -> Result<AudioOperation, AudioError> {
    let value = contracts::strict_json(body).map_err(|_| invalid())?;
    if value.as_object().is_some_and(|o| {
        o.keys().any(|name| {
            [
                "speed",
                "stream",
                "stream_format",
                "instructions",
                "reference_audio",
                "language",
            ]
            .contains(&name.as_str())
        })
    }) {
        return Err(AudioError::new("unsupported_feature"));
    }
    if value
        .get("response_format")
        .is_some_and(serde_json::Value::is_null)
    {
        return Err(invalid());
    }
    let request: SpeechRequest = serde_json::from_value(value).map_err(|_| invalid())?;
    if request.model != manifest.model {
        return Err(AudioError::new("model_not_found"));
    }
    if !contracts::text(&request.input) {
        return Err(AudioError::new("payload_too_large"));
    }
    if request.response_format.is_some_and(|s| s != "wav")
        || !manifest.voices.contains_key(&request.voice)
    {
        return Err(AudioError::new("unsupported_feature"));
    }
    Ok(AudioOperation::Tts {
        text: request.input,
        voice: request.voice,
        language: manifest.languages[0].clone(),
        output_handle: uuid::Uuid::new_v4().to_string(),
    })
}

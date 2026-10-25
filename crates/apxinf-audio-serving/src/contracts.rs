use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
use serde_json::Value;
use std::{collections::HashSet, fmt};

pub const PROTOCOL: &str = "apxinf-audio-worker/1.0";
pub const MAX_FRAME: usize = 65536;
pub const SAFE_INTEGER: u64 = 9007199254740991;
pub const MAX_UPLOAD: usize = 2 * 1024 * 1024;
pub const MAX_OUTPUT: usize = 8 * 1024 * 1024;
pub const MEDIA_QUOTA: usize = 32 * 1024 * 1024;
pub const QUEUE_LIMIT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTask {
    Asr,
    Tts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptKey {
    pub request_id: String,
    pub attempt: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MediaGrant {
    pub handle: String,
    pub worker_epoch: String,
    pub key: AttemptKey,
    pub bytes: u64,
    pub sha256: String,
    pub frames: u64,
    pub sample_rate: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioLimits {
    pub upload_bytes: u64,
    pub asr_frames: u64,
    pub text_scalars: u64,
    pub text_bytes: u64,
    pub output_bytes: u64,
    pub output_seconds: u64,
    pub queue: u64,
    pub uploads: u64,
    pub media_bytes: u64,
}

impl Default for AudioLimits {
    fn default() -> Self {
        Self {
            upload_bytes: MAX_UPLOAD as u64,
            asr_frames: 480000,
            text_scalars: 4096,
            text_bytes: 16384,
            output_bytes: MAX_OUTPUT as u64,
            output_seconds: 60,
            queue: 4,
            uploads: 2,
            media_bytes: MEDIA_QUOTA as u64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case", deny_unknown_fields)]
pub enum AudioOperation {
    Asr {
        input: MediaGrant,
        language: String,
    },
    Tts {
        text: String,
        voice: String,
        language: String,
        output_handle: String,
    },
}
impl AudioOperation {
    pub fn task(&self) -> AudioTask {
        match self {
            Self::Asr { .. } => AudioTask::Asr,
            Self::Tts { .. } => AudioTask::Tts,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum AudioResult {
    Transcript { text: String },
    Speech { output: MediaGrant },
    Cancelled {},
    Failed { code: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Message {
    Ready {
        task: AudioTask,
        model: String,
        model_revision: String,
        capability_revision: String,
        manifest_revision: String,
        provider: String,
        languages: Vec<String>,
        voices: Vec<String>,
        cancellation: String,
        limits: AudioLimits,
    },
    Submit {
        command_id: String,
        key: AttemptKey,
        lease_id: String,
        model_revision: String,
        timeout_ms: u64,
        operation: AudioOperation,
    },
    Cancel {
        command_id: String,
        key: AttemptKey,
    },
    Shutdown {
        command_id: String,
    },
    Accepted {
        key: AttemptKey,
        event_seq: u64,
    },
    Terminal {
        key: AttemptKey,
        event_seq: u64,
        result: AudioResult,
    },
    ResourcesReleased {
        key: AttemptKey,
        event_seq: u64,
        lease_id: String,
    },
    ControlAck {
        command_id: String,
        status: String,
    },
    Stopped {},
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    pub protocol: String,
    pub worker_epoch: String,
    pub message: Message,
}

pub fn uuid(value: &str) -> bool {
    uuid::Uuid::parse_str(value)
        .map(|v| v.hyphenated().to_string() == value)
        .unwrap_or(false)
}
pub fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(&v))
}
pub fn text(value: &str) -> bool {
    value.chars().any(|c| !white_space(c)) && value.len() <= 16384 && value.chars().count() <= 4096
}
fn white_space(c: char) -> bool {
    matches!(c, '\u{0009}'..='\u{000d}' | '\u{0020}' | '\u{0085}' | '\u{00a0}' | '\u{1680}'
        | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}
fn key(value: &AttemptKey) -> bool {
    uuid(&value.request_id) && (1..=SAFE_INTEGER).contains(&value.attempt)
}
fn grant(value: &MediaGrant) -> bool {
    uuid(&value.handle)
        && uuid(&value.worker_epoch)
        && key(&value.key)
        && digest(&value.sha256)
        && value.bytes > 0
        && value.bytes <= MAX_OUTPUT as u64
        && value.frames > 0
        && (8000..=48000).contains(&value.sample_rate)
        && value.frames <= value.sample_rate * 60
}
fn label(value: &str) -> bool {
    !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_control)
}

impl Frame {
    pub fn new(worker_epoch: &str, message: Message) -> Self {
        Self {
            protocol: PROTOCOL.into(),
            worker_epoch: worker_epoch.into(),
            message,
        }
    }
    pub fn validate(&self) -> bool {
        if self.protocol != PROTOCOL || !uuid(&self.worker_epoch) {
            return false;
        }
        match &self.message {
            Message::Ready {
                model,
                model_revision,
                capability_revision,
                manifest_revision,
                provider,
                languages,
                voices,
                cancellation,
                limits,
                task,
            } => {
                label(model)
                    && digest(model_revision)
                    && digest(capability_revision)
                    && digest(manifest_revision)
                    && ["mlx_audio", "cpu_fixture"].contains(&provider.as_str())
                    && !languages.is_empty()
                    && languages.len() <= 32
                    && languages.iter().all(|s| label(s))
                    && voices.len() <= 32
                    && voices.iter().all(|s| label(s))
                    && (*task != AudioTask::Tts || !voices.is_empty())
                    && cancellation == "provider_boundary_or_process_reap"
                    && *limits == AudioLimits::default()
            }
            Message::Submit {
                command_id,
                key: k,
                lease_id,
                model_revision,
                timeout_ms,
                operation,
            } => {
                uuid(command_id)
                    && key(k)
                    && uuid(lease_id)
                    && digest(model_revision)
                    && (1..=120000).contains(timeout_ms)
                    && match operation {
                        AudioOperation::Asr { input, language } => {
                            grant(input)
                                && input.key == *k
                                && input.worker_epoch == self.worker_epoch
                                && input.sample_rate == 16000
                                && input.frames <= 480000
                                && input.bytes <= MAX_UPLOAD as u64
                                && label(language)
                        }
                        AudioOperation::Tts {
                            text: t,
                            voice,
                            language,
                            output_handle,
                        } => text(t) && label(voice) && label(language) && uuid(output_handle),
                    }
            }
            Message::Cancel { command_id, key: k } => uuid(command_id) && key(k),
            Message::Shutdown { command_id } => uuid(command_id),
            Message::Accepted { key: k, event_seq } => key(k) && *event_seq <= SAFE_INTEGER,
            Message::Terminal {
                key: k,
                event_seq,
                result,
            } => {
                key(k)
                    && *event_seq <= SAFE_INTEGER
                    && match result {
                        AudioResult::Transcript { text } => text.len() <= 32768,
                        AudioResult::Speech { output } => {
                            grant(output)
                                && output.key == *k
                                && output.worker_epoch == self.worker_epoch
                        }
                        AudioResult::Cancelled {} => true,
                        AudioResult::Failed { code, message } => {
                            error_code(code) && !message.is_empty() && message.len() <= 512
                        }
                    }
            }
            Message::ResourcesReleased {
                key: k,
                event_seq,
                lease_id,
            } => key(k) && *event_seq <= SAFE_INTEGER && uuid(lease_id),
            Message::ControlAck { command_id, status } => {
                uuid(command_id)
                    && ["cancel_requested", "already_terminal", "shutdown_requested"]
                        .contains(&status.as_str())
            }
            Message::Stopped {} => true,
        }
    }
}

pub fn error_code(value: &str) -> bool {
    [
        "invalid_request",
        "unsupported_feature",
        "task_mismatch",
        "invalid_audio",
        "payload_too_large",
        "queue_full",
        "queue_timeout",
        "deadline_exceeded",
        "cancelled",
        "worker_fault",
        "not_ready",
        "media_quota",
        "provider_error",
        "output_limit",
        "pressure",
        "missing_resource",
    ]
    .contains(&value)
}

struct StrictValue(Value);
impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = StrictValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("strict JSON")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Bool(v)))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::String(v.into())))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                if v.unsigned_abs() > SAFE_INTEGER {
                    return Err(E::custom("unsafe integer"));
                }
                Ok(StrictValue(Value::from(v)))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                if v > SAFE_INTEGER {
                    return Err(E::custom("unsafe integer"));
                }
                Ok(StrictValue(Value::from(v)))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| StrictValue(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite number"))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(v) = a.next_element::<StrictValue>()? {
                    if out.len() == 256 {
                        return Err(de::Error::custom("array limit"));
                    }
                    out.push(v.0);
                }
                Ok(StrictValue(Value::Array(out)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Self::Value, A::Error> {
                let mut out = serde_json::Map::new();
                while let Some((k, v)) = a.next_entry::<String, StrictValue>()? {
                    if out.len() == 64 || out.insert(k, v.0).is_some() {
                        return Err(de::Error::custom("duplicate key or object limit"));
                    }
                }
                Ok(StrictValue(Value::Object(out)))
            }
        }
        d.deserialize_any(StrictVisitor)
    }
}

pub fn strict_json(bytes: &[u8]) -> Result<Value, String> {
    if bytes.len() > MAX_FRAME {
        return Err("frame limit".into());
    }
    let mut parser = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue::deserialize(&mut parser)
        .map_err(|e| e.to_string())?
        .0;
    parser.end().map_err(|e| e.to_string())?;
    fn depth(v: &Value, n: usize) -> bool {
        n <= 12
            && match v {
                Value::Array(a) => a.iter().all(|v| depth(v, n + 1)),
                Value::Object(o) => o.values().all(|v| depth(v, n + 1)),
                _ => true,
            }
    }
    if !depth(&value, 0) {
        return Err("nesting limit".into());
    }
    Ok(value)
}
pub fn parse_frame(bytes: &[u8]) -> Result<Frame, String> {
    let frame: Frame = serde_json::from_value(strict_json(bytes)?).map_err(|e| e.to_string())?;
    if !frame.validate() {
        return Err("invalid audio frame".into());
    }
    Ok(frame)
}

#[derive(Debug, PartialEq, Eq)]
pub enum AttemptPhase {
    Submitted,
    Accepted,
    Terminal,
    Settled,
}
pub struct AttemptState {
    pub key: AttemptKey,
    pub lease_id: String,
    pub phase: AttemptPhase,
    next_seq: u64,
    seen_leases: HashSet<String>,
}
impl AttemptState {
    pub fn new(key: AttemptKey, lease_id: String) -> Self {
        Self {
            key,
            lease_id,
            phase: AttemptPhase::Submitted,
            next_seq: 0,
            seen_leases: HashSet::new(),
        }
    }
    pub fn event(&mut self, message: &Message) -> Result<(), String> {
        let (k, seq) = match message {
            Message::Accepted { key, event_seq }
            | Message::Terminal { key, event_seq, .. }
            | Message::ResourcesReleased { key, event_seq, .. } => (key, event_seq),
            _ => return Err("not an attempt event".into()),
        };
        if *k != self.key || *seq != self.next_seq {
            return Err("execution identity or sequence mismatch".into());
        }
        match message {
            Message::Accepted { .. } if self.phase == AttemptPhase::Submitted => {
                self.phase = AttemptPhase::Accepted
            }
            Message::Terminal { .. } if self.phase == AttemptPhase::Accepted => {
                self.phase = AttemptPhase::Terminal
            }
            Message::ResourcesReleased { lease_id, .. }
                if self.phase == AttemptPhase::Terminal
                    && *lease_id == self.lease_id
                    && self.seen_leases.insert(lease_id.clone()) =>
            {
                self.phase = AttemptPhase::Settled
            }
            _ => return Err("invalid attempt transition".into()),
        }
        self.next_seq += 1;
        Ok(())
    }
}

//! Handwritten wire checks for the serial apxinf-worker/2.0 profile.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

pub const PROTOCOL: &str = "apxinf-worker/2.0";
pub const MAX_FRAME_BYTES: usize = 1_048_576;
pub const MAX_DEPTH: usize = 16;
pub const MAX_COMMANDS: u64 = 10_000;
pub const MAX_PROMPT_TOKENS: u64 = 131_072;
pub const MAX_OUTPUT_TOKENS: u64 = 65_536;
pub const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const BASE: &[&str] = &["protocol", "kind", "worker_epoch"];
const COMMAND: &[&str] = &["protocol", "kind", "worker_epoch", "command_id"];
const REQUEST: &[&str] = &[
    "protocol",
    "kind",
    "worker_epoch",
    "command_id",
    "request_id",
    "attempt",
];
const EVENT: &[&str] = &[
    "protocol",
    "kind",
    "worker_epoch",
    "request_id",
    "attempt",
    "event_seq",
];
const ERRORS: &[&str] = &[
    "invalid_request",
    "unsupported_feature",
    "context_limit",
    "model_not_found",
    "session_not_found",
    "session_conflict",
    "prefix_mismatch",
    "queue_full",
    "quota_exceeded",
    "capacity_unavailable",
    "model_unavailable",
    "worker_lost",
    "deadline_exceeded",
    "protocol_fault",
    "internal_error",
    "state_invalid",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError(pub String);

impl fmt::Display for ContractError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(&self.0)
    }
}
impl std::error::Error for ContractError {}
type Result<T> = std::result::Result<T, ContractError>;

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ContractError(message.into()))
    }
}
fn object(value: &Value) -> Result<&Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| ContractError("expected object".into()))
}
fn string(value: &Value) -> Result<&str> {
    value
        .as_str()
        .ok_or_else(|| ContractError("expected string".into()))
}
fn integer(value: &Value, min: u64, max: u64) -> Result<u64> {
    let number = value
        .as_u64()
        .ok_or_else(|| ContractError("expected non-negative integer".into()))?;
    require(
        number >= min && number <= max,
        "integer outside allowed range",
    )?;
    Ok(number)
}
fn choice(value: &Value, values: &[&str]) -> Result<()> {
    require(values.contains(&string(value)?), "unsupported enum value")
}
fn fields(value: &Value, base: &[&str], required: &[&str], optional: &[&str]) -> Result<()> {
    let map = object(value)?;
    require(
        base.iter()
            .chain(required)
            .all(|key| map.contains_key(*key)),
        "missing fields",
    )?;
    require(
        map.keys().all(|key| {
            base.contains(&key.as_str())
                || required.contains(&key.as_str())
                || optional.contains(&key.as_str())
        }),
        "unexpected fields",
    )
}
fn uuid(value: &Value) -> Result<()> {
    let bytes = string(value)?.as_bytes();
    require(
        bytes.len() == 36
            && bytes.iter().enumerate().all(|(index, b)| {
                if [8, 13, 18, 23].contains(&index) {
                    *b == b'-'
                } else {
                    b.is_ascii_digit() || (b'a'..=b'f').contains(b)
                }
            }),
        "invalid lowercase UUID",
    )
}
fn digest(value: &Value) -> Result<()> {
    let bytes = string(value)?.as_bytes();
    require(
        bytes.len() == 64
            && bytes
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b)),
        "invalid SHA-256 digest",
    )
}
fn array(value: &Value, min: usize, max: usize) -> Result<&Vec<Value>> {
    let items = value
        .as_array()
        .ok_or_else(|| ContractError("expected array".into()))?;
    require(
        items.len() >= min && items.len() <= max,
        "invalid array length",
    )?;
    Ok(items)
}
fn tokens(value: &Value, min: usize, max: usize) -> Result<()> {
    for token in array(value, min, max)? {
        integer(token, 0, 2_147_483_647)?;
    }
    Ok(())
}
fn leases(value: &Value) -> Result<BTreeSet<String>> {
    let mut ids = BTreeSet::new();
    for id in array(value, 0, 64)? {
        uuid(id)?;
        require(ids.insert(string(id)?.to_owned()), "duplicate lease ID")?;
    }
    Ok(ids)
}
fn counts(value: &Value, keys: &[&str]) -> Result<()> {
    fields(value, &[], keys, &[])?;
    for count in object(value)?.values() {
        integer(count, 0, MAX_SAFE_INTEGER)?;
    }
    Ok(())
}
fn tree(value: &Value, depth: usize, identity: bool) -> Result<()> {
    require(depth <= MAX_DEPTH, "maximum JSON depth exceeded")?;
    match value {
        Value::Object(map) => {
            require(map.len() <= MAX_PROMPT_TOKENS as usize, "object too large")?;
            for (key, child) in map {
                require(!identity || key.is_ascii(), "identity keys must be ASCII")?;
                tree(child, depth + 1, identity)?;
            }
        }
        Value::Array(items) => {
            require(items.len() <= MAX_PROMPT_TOKENS as usize, "array too large")?;
            for child in items {
                tree(child, depth + 1, identity)?;
            }
        }
        Value::Number(number) => {
            if identity {
                integer(value, 0, MAX_SAFE_INTEGER)?;
            } else if let Some(value) = number.as_u64() {
                require(value <= MAX_SAFE_INTEGER, "unsafe integer")?;
            } else if let Some(value) = number.as_i64() {
                require(value >= -(MAX_SAFE_INTEGER as i64), "unsafe integer")?;
            } else {
                require(
                    number.as_f64().is_some_and(f64::is_finite),
                    "non-finite JSON number",
                )?;
            }
        }
        Value::Null => {
            require(!identity, "identity cannot contain null")?;
        }
        _ => {}
    }
    Ok(())
}
fn error_record(value: &Value) -> Result<()> {
    fields(
        value,
        &[],
        &["code", "message", "scope", "state_validity"],
        &[],
    )?;
    choice(&value["code"], ERRORS)?;
    string(&value["message"])?;
    choice(
        &value["scope"],
        &["request", "session", "worker", "service"],
    )?;
    choice(
        &value["state_validity"],
        &["unchanged", "committed", "invalid", "none"],
    )
}
fn messages(value: &Value, tool_values: &Value) -> Result<()> {
    for message in array(value, 1, 256)? {
        fields(
            message,
            &[],
            &["role", "content"],
            &["tool_call_id", "name", "tool_calls"],
        )?;
        choice(&message["role"], &["system", "user", "assistant", "tool"])?;
        string(&message["content"])?;
        for key in ["tool_call_id", "name"] {
            if let Some(value) = message.get(key) {
                string(value)?;
            }
        }
        if let Some(calls) = message.get("tool_calls") {
            for call in array(calls, 0, 256)? {
                fields(call, &[], &["id", "type", "function"], &[])?;
                string(&call["id"])?;
                choice(&call["type"], &["function"])?;
                fields(&call["function"], &[], &["name", "arguments"], &[])?;
                string(&call["function"]["name"])?;
                require(
                    call["function"]["arguments"].is_object()
                        || call["function"]["arguments"].is_string(),
                    "invalid tool arguments",
                )?;
            }
        }
    }
    for tool in array(tool_values, 0, 256)? {
        fields(tool, &[], &["type", "function"], &[])?;
        choice(&tool["type"], &["function"])?;
        let function = &tool["function"];
        fields(
            function,
            &[],
            &["name"],
            &["description", "parameters", "strict"],
        )?;
        string(&function["name"])?;
        if let Some(value) = function.get("description") {
            string(value)?;
        }
        if let Some(value) = function.get("parameters") {
            object(value)?;
        }
        if let Some(value) = function.get("strict") {
            require(value.is_boolean(), "strict must be Boolean")?;
        }
    }
    Ok(())
}
fn capabilities(value: &Value) -> Result<()> {
    fields(
        value,
        &[],
        &[
            "text",
            "image",
            "greedy",
            "sampling",
            "token_events",
            "cancel_granularity",
            "max_active_sequences",
            "max_context",
            "max_output_tokens",
            "exact_append",
            "state_schema",
        ],
        &[],
    )?;
    for (key, expected) in [
        ("text", true),
        ("image", false),
        ("greedy", true),
        ("sampling", false),
        ("token_events", true),
        ("exact_append", false),
    ] {
        require(
            value[key].as_bool() == Some(expected),
            "unsupported serial capability",
        )?;
    }
    choice(
        &value["cancel_granularity"],
        &["token", "token_or_prefill_chunk"],
    )?;
    integer(&value["max_active_sequences"], 1, 1)?;
    integer(&value["max_context"], 1, MAX_PROMPT_TOKENS)?;
    integer(&value["max_output_tokens"], 0, MAX_OUTPUT_TOKENS)?;
    choice(&value["state_schema"], &["none"])
}
fn is_event(kind: &str) -> bool {
    [
        "accepted",
        "rejected",
        "prefill_progress",
        "tokens",
        "terminal",
        "resources_released",
    ]
    .contains(&kind)
}

/// Check safe routing and settlement fields, without accepting payload values.
pub fn validate_command_envelope(frame: &Value) -> Result<()> {
    tree(frame, 1, false)?;
    object(frame)?;
    require(frame["protocol"] == PROTOCOL, "unsupported protocol")?;
    uuid(&frame["worker_epoch"])?;
    uuid(&frame["command_id"])?;
    let kind = string(&frame["kind"])?;
    match kind {
        "prepare_input" => fields(
            frame,
            COMMAND,
            &["model_revision", "messages", "tools", "template_options"],
            &[],
        )?,
        "submit" => fields(
            frame,
            REQUEST,
            &[
                "model_revision",
                "capability_revision",
                "token_ids",
                "max_tokens",
                "remaining_timeout_ms",
                "eos_token_ids",
                "capacity_lease_ids",
            ],
            &[],
        )?,
        "cancel_request" => fields(frame, REQUEST, &["reason"], &[])?,
        "stop_generation" => fields(
            frame,
            REQUEST,
            &["cause", "output_token_count", "text_byte_cutoff"],
            &[],
        )?,
        "query_stats" | "drain" | "shutdown" => fields(frame, COMMAND, &[], &[])?,
        _ => return Err(ContractError("unknown command kind".into())),
    }
    if ["submit", "cancel_request", "stop_generation"].contains(&kind) {
        uuid(&frame["request_id"])?;
        integer(&frame["attempt"], 1, MAX_SAFE_INTEGER)?;
    }
    if kind == "submit" {
        leases(&frame["capacity_lease_ids"])?;
    }
    Ok(())
}

/// Check a frame against the serial profile without changing the frame.
pub fn validate_frame(frame: &Value) -> Result<()> {
    tree(frame, 1, false)?;
    object(frame)?;
    require(frame["protocol"] == PROTOCOL, "unsupported protocol")?;
    uuid(&frame["worker_epoch"])?;
    let kind = string(&frame["kind"])?;
    if let Some(id) = frame.get("command_id") {
        uuid(id)?;
    }
    if is_event(kind) || ["submit", "cancel_request", "stop_generation"].contains(&kind) {
        uuid(&frame["request_id"])?;
        integer(&frame["attempt"], 1, MAX_SAFE_INTEGER)?;
    }
    if is_event(kind) {
        integer(&frame["event_seq"], 0, MAX_SAFE_INTEGER)?;
    }
    match kind {
        "ready" => {
            fields(
                frame,
                BASE,
                &[
                    "model_revision",
                    "capability_revision",
                    "model_manifest",
                    "capabilities",
                    "limits",
                    "model_path",
                    "eos_token_ids",
                    "vocab_size",
                    "runtime",
                    "memory",
                ],
                &[],
            )?;
            capabilities(&frame["capabilities"])?;
            counts(
                &frame["limits"],
                &["max_frame_bytes", "max_depth", "max_commands"],
            )?;
            require(
                frame["limits"]["max_frame_bytes"].as_u64() == Some(MAX_FRAME_BYTES as u64)
                    && frame["limits"]["max_depth"].as_u64() == Some(MAX_DEPTH as u64)
                    && frame["limits"]["max_commands"].as_u64() == Some(MAX_COMMANDS),
                "incompatible transport limits",
            )?;
            fields(&frame["runtime"], &[], &["python", "mlx", "mlx_lm"], &[])?;
            for value in object(&frame["runtime"])?.values() {
                string(value)?;
            }
            counts(
                &frame["memory"],
                &["active_bytes", "peak_bytes", "cache_bytes"],
            )?;
            require(
                string(&frame["model_path"])?.starts_with('/'),
                "model path must be absolute",
            )?;
            let vocabulary = integer(&frame["vocab_size"], 1, 2_147_483_648)?;
            tokens(&frame["eos_token_ids"], 0, 256)?;
            require(
                frame["eos_token_ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|token| token.as_u64().unwrap() < vocabulary),
                "EOS outside vocabulary",
            )?;
            object(&frame["model_manifest"])?;
            require(
                frame["model_revision"]
                    == canonical_identity_digest(
                        "apxinf-model-identity-v1",
                        &frame["model_manifest"],
                    )?,
                "model revision mismatch",
            )?;
            require(
                frame["capability_revision"]
                    == canonical_identity_digest("apxinf-capability-v1", &frame["capabilities"])?,
                "capability revision mismatch",
            )?;
        }
        "prepare_input" => {
            fields(
                frame,
                COMMAND,
                &["model_revision", "messages", "tools", "template_options"],
                &[],
            )?;
            digest(&frame["model_revision"])?;
            messages(&frame["messages"], &frame["tools"])?;
            fields(&frame["template_options"], &[], &["enable_thinking"], &[])?;
            require(
                frame["template_options"]["enable_thinking"].is_boolean(),
                "enable_thinking must be Boolean",
            )?;
        }
        "prepared_input" => {
            fields(
                frame,
                COMMAND,
                &[
                    "model_revision",
                    "token_ids",
                    "effective_input_tokens",
                    "prompt_digest",
                ],
                &[],
            )?;
            digest(&frame["model_revision"])?;
            tokens(&frame["token_ids"], 1, MAX_PROMPT_TOKENS as usize)?;
            integer(&frame["effective_input_tokens"], 1, MAX_PROMPT_TOKENS)?;
            require(
                frame["effective_input_tokens"].as_u64()
                    == Some(frame["token_ids"].as_array().unwrap().len() as u64),
                "prepared input count mismatch",
            )?;
            let ids: Vec<u32> = frame["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect();
            require(
                frame["prompt_digest"] == token_prefix_digest(&ids)?,
                "prepared input digest mismatch",
            )?;
        }
        "submit" => {
            fields(
                frame,
                REQUEST,
                &[
                    "model_revision",
                    "capability_revision",
                    "token_ids",
                    "max_tokens",
                    "remaining_timeout_ms",
                    "eos_token_ids",
                    "capacity_lease_ids",
                ],
                &[],
            )?;
            digest(&frame["model_revision"])?;
            digest(&frame["capability_revision"])?;
            tokens(&frame["token_ids"], 1, MAX_PROMPT_TOKENS as usize)?;
            tokens(&frame["eos_token_ids"], 0, 256)?;
            integer(&frame["max_tokens"], 0, MAX_OUTPUT_TOKENS)?;
            integer(&frame["remaining_timeout_ms"], 1, 3_600_000)?;
            leases(&frame["capacity_lease_ids"])?;
        }
        "cancel_request" => {
            fields(frame, REQUEST, &["reason"], &[])?;
            choice(
                &frame["reason"],
                &[
                    "user_cancel",
                    "client_disconnect",
                    "slow_consumer",
                    "deadline_exceeded",
                ],
            )?;
        }
        "stop_generation" => {
            fields(
                frame,
                REQUEST,
                &["cause", "output_token_count", "text_byte_cutoff"],
                &[],
            )?;
            choice(&frame["cause"], &["stop_sequence", "tool_calls"])?;
            integer(&frame["output_token_count"], 0, MAX_OUTPUT_TOKENS)?;
            integer(&frame["text_byte_cutoff"], 0, MAX_SAFE_INTEGER)?;
        }
        "query_stats" | "drain" | "shutdown" => fields(frame, COMMAND, &[], &[])?,
        "accepted" => fields(frame, EVENT, &[], &[])?,
        "rejected" => {
            fields(frame, EVENT, &["error"], &[])?;
            error_record(&frame["error"])?;
        }
        "prefill_progress" => {
            fields(frame, EVENT, &["consumed_tokens"], &[])?;
            integer(&frame["consumed_tokens"], 0, MAX_PROMPT_TOKENS)?;
        }
        "tokens" => {
            fields(
                frame,
                EVENT,
                &["output_index", "token_ids", "text_delta"],
                &[],
            )?;
            integer(&frame["output_index"], 0, MAX_OUTPUT_TOKENS)?;
            tokens(&frame["token_ids"], 1, 256)?;
            string(&frame["text_delta"])?;
        }
        "terminal" => {
            fields(
                frame,
                EVENT,
                &[
                    "status",
                    "cause",
                    "usage",
                    "state_result",
                    "text_delta",
                    "metrics",
                ],
                &["error"],
            )?;
            let status = string(&frame["status"])?;
            let causes: &[&str] = match status {
                "completed" => &["eos", "length", "stop_sequence", "tool_calls"],
                "cancelled" => &["user_cancel", "client_disconnect", "slow_consumer"],
                "expired" => &["deadline_exceeded"],
                "failed" => ERRORS,
                _ => return Err(ContractError("unsupported terminal status".into())),
            };
            choice(&frame["cause"], causes)?;
            require(
                frame.get("error").is_some() == (status == "failed"),
                "invalid terminal error presence",
            )?;
            if let Some(error) = frame.get("error") {
                error_record(error)?;
                require(
                    error["code"] == frame["cause"],
                    "terminal cause differs from error",
                )?;
            }
            counts(&frame["usage"], &["input_tokens", "output_tokens"])?;
            let state = &frame["state_result"];
            fields(
                state,
                &[],
                &["validity", "consumed_position", "history_count"],
                &[],
            )?;
            choice(&state["validity"], &["none"])?;
            let history = integer(&state["history_count"], 0, MAX_SAFE_INTEGER)?;
            integer(&state["consumed_position"], 0, history)?;
            string(&frame["text_delta"])?;
            counts(
                &frame["metrics"],
                &["elapsed_ns", "ttft_ns", "peak_memory_bytes"],
            )?;
            require(
                frame["metrics"]["ttft_ns"].as_u64() <= frame["metrics"]["elapsed_ns"].as_u64(),
                "TTFT exceeds elapsed time",
            )?;
        }
        "resources_released" => {
            fields(
                frame,
                EVENT,
                &["released_lease_ids", "retained_lease_ids"],
                &[],
            )?;
            leases(&frame["released_lease_ids"])?;
            require(
                leases(&frame["retained_lease_ids"])?.is_empty(),
                "stateless profile cannot retain leases",
            )?;
        }
        "command_result" => {
            fields(frame, COMMAND, &["status"], &["error"])?;
            choice(
                &frame["status"],
                &[
                    "accepted",
                    "already_registered",
                    "already_terminal",
                    "error",
                ],
            )?;
            require(
                frame.get("error").is_some() == (frame["status"] == "error"),
                "invalid command error presence",
            )?;
            if let Some(error) = frame.get("error") {
                error_record(error)?;
            }
        }
        "stats" => {
            fields(
                frame,
                COMMAND,
                &[
                    "active_requests",
                    "completed_requests",
                    "cancelled_requests",
                ],
                &[],
            )?;
            integer(&frame["active_requests"], 0, 1)?;
            integer(&frame["completed_requests"], 0, MAX_SAFE_INTEGER)?;
            integer(&frame["cancelled_requests"], 0, MAX_SAFE_INTEGER)?;
        }
        "worker_fault" => {
            fields(frame, BASE, &["error"], &[])?;
            error_record(&frame["error"])?;
        }
        "drained" => fields(frame, BASE, &[], &[])?,
        _ => return Err(ContractError("unknown frame kind".into())),
    }
    Ok(())
}

struct CheckedJson(Value);
impl<'de> Deserialize<'de> for CheckedJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = CheckedJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value without duplicate keys")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(CheckedJson(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(CheckedJson(Value::Number(v.into())))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(CheckedJson(Value::Number(v.into())))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                Number::from_f64(v)
                    .map(|n| CheckedJson(Value::Number(n)))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(CheckedJson(Value::String(v.into())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Self::Value, E> {
                Ok(CheckedJson(Value::String(v)))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(CheckedJson(Value::Null))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut access: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(CheckedJson(value)) = access.next_element()? {
                    values.push(value);
                }
                Ok(CheckedJson(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut access: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = access.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom("duplicate JSON key"));
                    }
                    let CheckedJson(value) = access.next_value()?;
                    values.insert(key, value);
                }
                Ok(CheckedJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(JsonVisitor)
    }
}

fn validate_integer_literals(payload: &[u8]) -> Result<()> {
    // The JSON parser checks syntax before this scan checks the original number spelling.
    let maximum = MAX_SAFE_INTEGER.to_string();
    let mut cursor = 0;
    while cursor < payload.len() {
        match payload[cursor] {
            b'"' => {
                cursor += 1;
                while cursor < payload.len() {
                    match payload[cursor] {
                        b'\\' => cursor += 2,
                        b'"' => {
                            cursor += 1;
                            break;
                        }
                        _ => cursor += 1,
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = cursor;
                while payload.get(cursor).is_some_and(|byte| {
                    matches!(byte, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                }) {
                    cursor += 1;
                }
                let literal = &payload[start..cursor];
                if !literal
                    .iter()
                    .any(|byte| matches!(byte, b'.' | b'e' | b'E'))
                {
                    let digits = literal.strip_prefix(b"-").unwrap_or(literal);
                    require(
                        digits.len() < maximum.len()
                            || (digits.len() == maximum.len() && digits <= maximum.as_bytes()),
                        "unsafe integer",
                    )?;
                }
            }
            _ => cursor += 1,
        }
    }
    Ok(())
}

/// Parse bounded UTF-8 JSON without accepting duplicate keys or unsafe integers.
/// This operation does not impose a worker-frame schema or JSONL framing.
pub fn parse_document(payload: &[u8]) -> Result<Value> {
    require(
        payload.len() <= MAX_FRAME_BYTES,
        "document exceeds byte limit",
    )?;
    let frame: CheckedJson = serde_json::from_slice(payload)
        .map_err(|error| ContractError(format!("invalid JSON: {error}")))?;
    validate_integer_literals(payload)?;
    tree(&frame.0, 1, false)?;
    Ok(frame.0)
}

pub fn decode_frame(line: &[u8]) -> Result<Value> {
    require(line.len() <= MAX_FRAME_BYTES, "frame exceeds byte limit")?;
    let payload = line.strip_suffix(b"\n").unwrap_or(line);
    require(
        !payload.contains(&b'\n') && !payload.contains(&b'\r'),
        "frame contains a line break",
    )?;
    let frame = parse_document(payload)?;
    validate_frame(&frame)?;
    Ok(frame)
}

pub fn encode_frame(frame: &Value) -> Result<Vec<u8>> {
    validate_frame(frame)?;
    let mut bytes = serde_json::to_vec(frame).map_err(|error| ContractError(error.to_string()))?;
    bytes.push(b'\n');
    require(bytes.len() <= MAX_FRAME_BYTES, "frame exceeds byte limit")?;
    Ok(bytes)
}

fn canonical_json(value: &Value, output: &mut String) {
    match value {
        Value::String(text) => {
            output.push('"');
            for ch in text.chars() {
                match ch {
                    '"' | '\\' => {
                        output.push('\\');
                        output.push(ch);
                    }
                    ch if ch < '\u{20}' => {
                        output.push_str(&format!("\\u{:04x}", ch as u32));
                    }
                    ch => output.push(ch),
                }
            }
            output.push('"');
        }
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&number.to_string()),
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                canonical_json(item, output);
            }
            output.push(']');
        }
        Value::Object(map) => {
            output.push('{');
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                canonical_json(&Value::String(key.clone()), output);
                output.push(':');
                canonical_json(&map[key], output);
            }
            output.push('}');
        }
        Value::Null => unreachable!("identity validator rejects null"),
    }
}

pub fn canonical_identity_digest(domain: &str, document: &Value) -> Result<String> {
    require(
        domain.is_ascii() && !domain.contains('\0'),
        "invalid identity domain",
    )?;
    tree(document, 1, true)?;
    let mut encoded = String::new();
    canonical_json(document, &mut encoded);
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    hasher.update([0]);
    hasher.update(encoded.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn token_prefix_digest(tokens: &[u32]) -> Result<String> {
    require(
        tokens.len() <= (MAX_PROMPT_TOKENS + MAX_OUTPUT_TOKENS) as usize,
        "token digest too long",
    )?;
    require(
        tokens.iter().all(|id| *id <= 2_147_483_647),
        "token digest ID out of range",
    )?;
    let mut hasher = Sha256::new();
    hasher.update(b"apxinf-token-prefix-v1\0");
    hasher.update((tokens.len() as u64).to_le_bytes());
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Clone)]
pub struct AttemptTracker {
    key: (String, String, u64),
    input_tokens: u64,
    max_tokens: u64,
    leases: BTreeSet<String>,
    pub next_seq: u64,
    pub output_tokens: u64,
    prefill_tokens: u64,
    pub phase: &'static str,
}

impl AttemptTracker {
    pub fn new(submit: &Value) -> Result<Self> {
        validate_frame(submit)?;
        require(submit["kind"] == "submit", "tracker requires submit")?;
        Ok(Self {
            key: (
                string(&submit["worker_epoch"])?.into(),
                string(&submit["request_id"])?.into(),
                submit["attempt"].as_u64().unwrap(),
            ),
            input_tokens: submit["token_ids"].as_array().unwrap().len() as u64,
            max_tokens: submit["max_tokens"].as_u64().unwrap(),
            leases: leases(&submit["capacity_lease_ids"])?,
            next_seq: 0,
            output_tokens: 0,
            prefill_tokens: 0,
            phase: "new",
        })
    }
    pub fn done(&self) -> bool {
        self.phase == "released"
    }
    pub fn observe(&mut self, event: &Value) -> Result<()> {
        validate_frame(event)?;
        let kind = string(&event["kind"])?;
        require(is_event(kind), "not an attempt event")?;
        require(
            event["worker_epoch"] == self.key.0
                && event["request_id"] == self.key.1
                && event["attempt"].as_u64() == Some(self.key.2),
            "attempt identity mismatch",
        )?;
        require(
            event["event_seq"].as_u64() == Some(self.next_seq),
            "event sequence mismatch",
        )?;
        let mut phase = self.phase;
        let mut output = self.output_tokens;
        let mut prefill = self.prefill_tokens;
        match (self.phase, kind) {
            ("new", "accepted") => phase = "accepted",
            ("new", "rejected") => phase = "rejected",
            ("accepted", "prefill_progress") => {
                let consumed = event["consumed_tokens"].as_u64().unwrap();
                require(
                    output == 0 && consumed >= prefill && consumed <= self.input_tokens,
                    "invalid prefill progress",
                )?;
                prefill = consumed;
            }
            ("accepted", "tokens") => {
                require(
                    event["output_index"].as_u64() == Some(output),
                    "non-contiguous token range",
                )?;
                output += event["token_ids"].as_array().unwrap().len() as u64;
                require(output <= self.max_tokens, "output exceeds allowance")?;
            }
            ("accepted" | "rejected", "terminal") => {
                require(
                    phase != "rejected" || event["status"] == "failed",
                    "rejection requires failed terminal",
                )?;
                require(
                    event["usage"]["input_tokens"].as_u64() == Some(self.input_tokens)
                        && event["usage"]["output_tokens"].as_u64() == Some(output),
                    "terminal usage mismatch",
                )?;
                require(
                    event["state_result"]["history_count"].as_u64()
                        == Some(self.input_tokens + output),
                    "history count mismatch",
                )?;
                require(
                    output != 0 || event["metrics"]["ttft_ns"].as_u64() == Some(0),
                    "empty output must have zero TTFT",
                )?;
                if event["status"] == "completed" && event["cause"] == "length" {
                    require(
                        output == self.max_tokens,
                        "length completion requires full allowance",
                    )?;
                }
                phase = "terminal";
            }
            ("terminal", "resources_released") => {
                require(
                    leases(&event["released_lease_ids"])? == self.leases,
                    "lease settlement mismatch",
                )?;
                phase = "released";
            }
            _ => {
                return Err(ContractError(
                    "event is invalid in current attempt phase".into(),
                ))
            }
        }
        self.phase = phase;
        self.output_tokens = output;
        self.prefill_tokens = prefill;
        self.next_seq += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::process::{Command, Stdio};

    fn corpus() -> Value {
        serde_json::from_str(include_str!(
            "../../../tests/fixtures/serving/serial-v0.1.json"
        ))
        .unwrap()
    }

    fn fixture(corpus: &Value, case: &Value) -> Value {
        let mut value = corpus["templates"][case["template"].as_str().unwrap()].clone();
        if let Some(changes) = case.get("set") {
            for (pointer, replacement) in changes.as_object().unwrap() {
                let (parent, key) = pointer.rsplit_once('/').unwrap();
                value
                    .pointer_mut(parent)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert(key.to_owned(), replacement.clone());
            }
        }
        if let Some(removes) = case.get("remove") {
            for key in removes.as_array().unwrap() {
                value.as_object_mut().unwrap().remove(key.as_str().unwrap());
            }
        }
        value
    }

    #[test]
    fn shared_document_cases() {
        let corpus = corpus();
        for case in corpus["document_cases"].as_array().unwrap() {
            let result = parse_document(case["raw"].as_str().unwrap().as_bytes());
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}: {:?}",
                case["name"],
                result
            );
            if let Ok(value) = result {
                assert_eq!(value, case["value"], "{}", case["name"]);
            }
        }
    }

    #[test]
    fn shared_wire_cases() {
        let corpus = corpus();
        for case in corpus["frame_cases"].as_array().unwrap() {
            let raw = case
                .get("raw")
                .map(|v| v.as_str().unwrap().as_bytes().to_vec())
                .unwrap_or_else(|| serde_json::to_vec(&fixture(&corpus, case)).unwrap());
            let result = decode_frame(&raw);
            assert_eq!(
                result.is_ok(),
                case["valid"].as_bool().unwrap(),
                "{}: {:?}",
                case["name"],
                result
            );
            if let Ok(frame) = result {
                assert_eq!(decode_frame(&encode_frame(&frame).unwrap()).unwrap(), frame);
            }
        }
    }

    #[test]
    fn shared_event_transcripts() {
        let corpus = corpus();
        for case in corpus["stream_cases"].as_array().unwrap() {
            let mut tracker = AttemptTracker::new(&corpus["templates"]["submit"]).unwrap();
            let mut failure = None;
            for (index, selected) in case["events"].as_array().unwrap().iter().enumerate() {
                let before = format!("{tracker:?}");
                if tracker.observe(&fixture(&corpus, selected)).is_err() {
                    assert_eq!(
                        format!("{tracker:?}"),
                        before,
                        "invalid event changed state"
                    );
                    failure = Some(index as u64);
                    break;
                }
            }
            assert_eq!(failure, case["failure_at"].as_u64(), "{}", case["name"]);
            assert_eq!(
                tracker.done(),
                case["done"].as_bool().unwrap(),
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn shared_digest_vectors() {
        let corpus = corpus();
        for vector in corpus["token_digests"].as_array().unwrap() {
            let tokens: Vec<u32> = vector["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(
                token_prefix_digest(&tokens).unwrap(),
                vector["digest"].as_str().unwrap()
            );
        }
        for vector in corpus["identity_digests"].as_array().unwrap() {
            assert_eq!(
                canonical_identity_digest(vector["domain"].as_str().unwrap(), &vector["value"])
                    .unwrap(),
                vector["digest"].as_str().unwrap()
            );
        }
    }

    #[test]
    fn envelope_distinguishes_content_from_routing() {
        let mut frame = corpus()["templates"]["submit"].clone();
        frame["max_tokens"] = Value::Bool(true);
        assert!(validate_command_envelope(&frame).is_ok());
        assert!(validate_frame(&frame).is_err());
        frame["capacity_lease_ids"] = serde_json::json!(["untrusted"]);
        assert!(validate_command_envelope(&frame).is_err());
    }

    #[test]
    fn rust_frames_pass_python_and_return_as_valid_rust_frames() {
        let corpus = corpus();
        let frames: Vec<Value> = corpus["templates"]
            .as_object()
            .unwrap()
            .values()
            .cloned()
            .collect();
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/python/test_serving_contracts.py");
        let mut child = Command::new("python3")
            .arg(script)
            .arg("--peer")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        for frame in &frames {
            input.write_all(&encode_frame(frame).unwrap()).unwrap();
        }
        drop(input);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let returned: Vec<Value> = output
            .stdout
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| decode_frame(line).unwrap())
            .collect();
        assert_eq!(returned, frames);
    }
}

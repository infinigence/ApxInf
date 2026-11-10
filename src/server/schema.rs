use serde_json::Value;
use std::collections::BTreeSet;

pub const MODEL_VOCAB_SIZE: u64 = 248_320;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub input_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub ignore_eos: bool,
    pub stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolError {
    pub error_type: &'static str,
    pub message: String,
}

impl ProtocolError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            error_type: "invalid_request",
            message: message.into(),
        }
    }
}

pub fn parse_generate_request(
    raw: &[u8],
    max_model_len: usize,
) -> Result<GenerateRequest, ProtocolError> {
    let value: Value = serde_json::from_slice(raw)
        .map_err(|error| ProtocolError::invalid(format!("malformed JSON: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| ProtocolError::invalid("request body must be a JSON object"))?;
    const REQUIRED: [&str; 5] = [
        "input_ids",
        "max_new_tokens",
        "temperature",
        "ignore_eos",
        "stream",
    ];
    let allowed = REQUIRED.iter().copied().collect::<BTreeSet<_>>();
    for field in object.keys() {
        if !allowed.contains(field.as_str()) {
            let message = if field == "images" {
                format!("unsupported field `{field}`")
            } else {
                format!("unknown field `{field}`")
            };
            return Err(ProtocolError::invalid(message));
        }
    }
    for field in REQUIRED {
        if !object.contains_key(field) {
            return Err(ProtocolError::invalid(format!("missing field `{field}`")));
        }
    }

    let input_values = object["input_ids"]
        .as_array()
        .ok_or_else(|| ProtocolError::invalid("input_ids must be an array of integers"))?;
    if input_values.is_empty() {
        return Err(ProtocolError::invalid("input_ids must not be empty"));
    }
    let mut input_ids = Vec::with_capacity(input_values.len());
    for value in input_values {
        let token_id = value.as_i64().ok_or_else(|| {
            ProtocolError::invalid("input_ids must contain only signed integer values")
        })?;
        if token_id < 0 || token_id as u64 >= MODEL_VOCAB_SIZE {
            return Err(ProtocolError::invalid(format!(
                "token id {token_id} is outside model vocabulary [0, {MODEL_VOCAB_SIZE})"
            )));
        }
        input_ids.push(token_id as u32);
    }

    let max_new_tokens = object["max_new_tokens"]
        .as_u64()
        .ok_or_else(|| ProtocolError::invalid("max_new_tokens must be a positive integer"))?;
    let max_new_tokens = usize::try_from(max_new_tokens)
        .map_err(|_| ProtocolError::invalid("max_new_tokens is too large"))?;
    if max_new_tokens == 0 {
        return Err(ProtocolError::invalid("max_new_tokens must be positive"));
    }

    let temperature = object["temperature"]
        .as_f64()
        .ok_or_else(|| ProtocolError::invalid("temperature must be a number"))?;
    if temperature != 0.0 {
        return Err(ProtocolError::invalid(
            "temperature must be exactly 0 for greedy generation",
        ));
    }

    let ignore_eos = object["ignore_eos"]
        .as_bool()
        .ok_or_else(|| ProtocolError::invalid("ignore_eos must be a boolean"))?;
    let stream = object["stream"]
        .as_bool()
        .ok_or_else(|| ProtocolError::invalid("stream must be a boolean"))?;

    let total = input_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| ProtocolError::invalid("request exceeds max_model_len"))?;
    if total > max_model_len {
        return Err(ProtocolError::invalid(format!(
            "prompt tokens {} plus max_new_tokens {} exceed max_model_len {}",
            input_ids.len(),
            max_new_tokens,
            max_model_len
        )));
    }

    Ok(GenerateRequest {
        input_ids,
        max_new_tokens,
        temperature,
        ignore_eos,
        stream,
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_generate_request, ProtocolError, MODEL_VOCAB_SIZE};

    const MAX_MODEL_LEN: usize = 32;

    fn valid_json() -> Vec<u8> {
        br#"{
            "input_ids": [1, 248056],
            "max_new_tokens": 2,
            "temperature": 0,
            "ignore_eos": true,
            "stream": false
        }"#
        .to_vec()
    }

    fn assert_invalid(raw: &[u8], expected: &str) {
        let error = parse_generate_request(raw, MAX_MODEL_LEN).unwrap_err();
        assert_eq!(error.error_type, "invalid_request");
        assert!(
            error.message.contains(expected),
            "expected {expected:?} in {:?}",
            error.message
        );
    }

    #[test]
    fn accepts_model_vocab_and_image_token() {
        let request = parse_generate_request(&valid_json(), MAX_MODEL_LEN).unwrap();
        assert_eq!(MODEL_VOCAB_SIZE, 248_320);
        assert_eq!(request.input_ids, vec![1, 248_056]);
        assert_eq!(request.max_new_tokens, 2);
        assert_eq!(request.temperature, 0.0);
        assert!(request.ignore_eos);
        assert!(!request.stream);
    }

    #[test]
    fn rejects_malformed_json() {
        assert_invalid(b"{not-json", "malformed JSON");
    }

    #[test]
    fn rejects_empty_input_ids() {
        assert_invalid(
            br#"{"input_ids":[],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false}"#,
            "must not be empty",
        );
    }

    #[test]
    fn rejects_negative_and_out_of_vocab_tokens() {
        assert_invalid(
            br#"{"input_ids":[-1],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false}"#,
            "outside model vocabulary",
        );
        assert_invalid(
            br#"{"input_ids":[4294967295],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false}"#,
            "outside model vocabulary",
        );
    }

    #[test]
    fn rejects_unsupported_temperature_and_zero_budget() {
        assert_invalid(
            br#"{"input_ids":[1],"max_new_tokens":1,"temperature":0.1,"ignore_eos":true,"stream":false}"#,
            "temperature",
        );
        assert_invalid(
            br#"{"input_ids":[1],"max_new_tokens":0,"temperature":0,"ignore_eos":true,"stream":false}"#,
            "max_new_tokens",
        );
    }

    #[test]
    fn rejects_total_budget_overflow() {
        assert_invalid(
            br#"{"input_ids":[1],"max_new_tokens":32,"temperature":0,"ignore_eos":true,"stream":false}"#,
            "max_model_len",
        );
    }

    #[test]
    fn rejects_images_unknown_fields_and_missing_fields() {
        assert_invalid(
            br#"{"input_ids":[1],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false,"images":["x"]}"#,
            "unsupported field `images`",
        );
        assert_invalid(
            br#"{"input_ids":[1],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false,"extra":1}"#,
            "unknown field `extra`",
        );
        assert_invalid(
            br#"{"input_ids":[1],"max_new_tokens":1,"temperature":0,"ignore_eos":true}"#,
            "missing field `stream`",
        );
    }

    #[test]
    fn rejects_non_integer_and_non_boolean_types() {
        assert_invalid(
            br#"{"input_ids":[1.5],"max_new_tokens":1,"temperature":0,"ignore_eos":true,"stream":false}"#,
            "input_ids",
        );
        assert_invalid(
            br#"{"input_ids":[1],"max_new_tokens":1,"temperature":0,"ignore_eos":"yes","stream":false}"#,
            "ignore_eos",
        );
    }

    #[test]
    fn protocol_error_constructor_is_stable() {
        let error = ProtocolError::invalid("bad request");
        assert_eq!(error.error_type, "invalid_request");
        assert_eq!(error.message, "bad request");
    }
}

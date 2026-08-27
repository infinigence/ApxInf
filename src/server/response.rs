use serde_json::json;

use crate::server::schema::ProtocolError;

pub const EOS_TOKEN_IDS: [u32; 2] = [248_046, 248_044];

pub fn apply_eos(output_ids: Vec<u32>, ignore_eos: bool) -> Vec<u32> {
    if ignore_eos {
        return output_ids;
    }
    let mut result = Vec::with_capacity(output_ids.len());
    for token_id in output_ids {
        result.push(token_id);
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
    }
    result
}

fn usage(prompt_tokens: usize, output_ids: &[u32]) -> serde_json::Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_ids.len(),
        "total_tokens": prompt_tokens + output_ids.len(),
    })
}

pub fn result_json(request_id: &str, prompt_tokens: usize, output_ids: &[u32]) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "type": "result",
        "request_id": request_id,
        "output_ids": output_ids,
        "usage": usage(prompt_tokens, output_ids),
    }))
    .expect("JSON serialization is infallible")
}

pub fn error_json(error: &ProtocolError) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "error": {
            "type": error.error_type,
            "message": error.message,
        }
    }))
    .expect("JSON serialization is infallible")
}

pub fn sse_frames(request_id: &str, prompt_tokens: usize, output_ids: &[u32]) -> Vec<Vec<u8>> {
    let mut frames = output_ids
        .iter()
        .enumerate()
        .map(|(index, token_id)| sse_token_frame(request_id, index, *token_id))
        .collect::<Vec<_>>();
    frames.push(sse_done_frame(request_id, prompt_tokens, output_ids));
    frames.push(sse_done_sentinel());
    frames
}

pub fn sse_token_frame(request_id: &str, index: usize, token_id: u32) -> Vec<u8> {
    let body = serde_json::to_string(&json!({
        "type": "token",
        "request_id": request_id,
        "index": index,
        "token_id": token_id,
    }))
    .expect("JSON serialization is infallible");
    format!("data: {body}\n\n").into_bytes()
}

pub fn sse_done_frame(request_id: &str, prompt_tokens: usize, output_ids: &[u32]) -> Vec<u8> {
    let done = serde_json::to_string(&json!({
        "type": "done",
        "request_id": request_id,
        "usage": usage(prompt_tokens, output_ids),
    }))
    .expect("JSON serialization is infallible");
    format!("data: {done}\n\n").into_bytes()
}

pub fn sse_done_sentinel() -> Vec<u8> {
    b"data: [DONE]\n\n".to_vec()
}

pub fn sse_error_frame(request_id: &str, error: &apxinf_model::RuntimeError) -> Vec<u8> {
    let error_type = match error {
        apxinf_model::RuntimeError::Capacity | apxinf_model::RuntimeError::QueueFull => "capacity",
        apxinf_model::RuntimeError::WorkerStopped => "unavailable",
        apxinf_model::RuntimeError::Admission(_) => "invalid_request",
        apxinf_model::RuntimeError::Cancelled => "cancelled",
        apxinf_model::RuntimeError::Unhealthy => "unhealthy",
        apxinf_model::RuntimeError::Execution(_) => "runtime_error",
    };
    let body = serde_json::to_string(&json!({
        "type": "error",
        "request_id": request_id,
        "error": {
            "type": error_type,
            "message": error.to_string(),
        },
    }))
    .expect("JSON serialization is infallible");
    format!("data: {body}\n\n").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::{apply_eos, error_json, result_json, sse_frames, EOS_TOKEN_IDS};
    use crate::server::schema::ProtocolError;
    use serde_json::Value;

    #[test]
    fn eos_stops_only_when_ignore_eos_is_false() {
        assert_eq!(EOS_TOKEN_IDS, [248_046, 248_044]);
        assert_eq!(
            apply_eos(vec![7, 248_046, 8, 248_044], false),
            vec![7, 248_046]
        );
        assert_eq!(
            apply_eos(vec![7, 248_046, 8, 248_044], true),
            vec![7, 248_046, 8, 248_044]
        );
    }

    #[test]
    fn non_stream_result_has_actual_usage() {
        let value: Value = serde_json::from_slice(&result_json("req-4", 8, &[7])).unwrap();
        assert_eq!(value["type"], "result");
        assert_eq!(value["request_id"], "req-4");
        assert_eq!(value["output_ids"], serde_json::json!([7]));
        assert_eq!(
            value["usage"],
            serde_json::json!({
                "prompt_tokens": 8,
                "completion_tokens": 1,
                "total_tokens": 9,
            })
        );
    }

    #[test]
    fn sse_has_contiguous_indexes_done_usage_and_sentinel() {
        let frames = sse_frames("req-2", 3, &[11, 12]);
        assert_eq!(frames.len(), 4);
        let first = std::str::from_utf8(&frames[0]).unwrap();
        let second = std::str::from_utf8(&frames[1]).unwrap();
        let done = std::str::from_utf8(&frames[2]).unwrap();
        assert!(first.starts_with("data: "));
        assert!(second.starts_with("data: "));
        let first: Value = serde_json::from_str(first.trim_start_matches("data: ").trim()).unwrap();
        let second: Value =
            serde_json::from_str(second.trim_start_matches("data: ").trim()).unwrap();
        let done: Value = serde_json::from_str(done.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(first["index"], 0);
        assert_eq!(second["index"], 1);
        assert_eq!(first["request_id"], "req-2");
        assert_eq!(second["request_id"], "req-2");
        assert_eq!(done["type"], "done");
        assert_eq!(done["usage"]["completion_tokens"], 2);
        assert_eq!(frames[3], b"data: [DONE]\n\n");
    }

    #[test]
    fn error_body_has_stable_shape() {
        let value: Value =
            serde_json::from_slice(&error_json(&ProtocolError::invalid("bad"))).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"error":{"type":"invalid_request","message":"bad"}})
        );
    }

    #[test]
    fn sse_runtime_error_preserves_capacity_classification() {
        let frame = super::sse_error_frame("req-9", &apxinf_model::RuntimeError::QueueFull);
        let text = std::str::from_utf8(&frame).unwrap();
        let value: Value = serde_json::from_str(text.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["request_id"], "req-9");
        assert_eq!(value["error"]["type"], "capacity");
    }

    #[test]
    fn sse_worker_stopped_maps_to_unavailable() {
        let frame = super::sse_error_frame("req-10", &apxinf_model::RuntimeError::WorkerStopped);
        let text = std::str::from_utf8(&frame).unwrap();
        let value: Value = serde_json::from_str(text.trim_start_matches("data: ").trim()).unwrap();
        assert_eq!(value["error"]["type"], "unavailable");
    }
}

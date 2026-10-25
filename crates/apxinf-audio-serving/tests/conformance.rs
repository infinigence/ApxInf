use apxinf_audio_serving::{
    contracts::{parse_frame, AttemptKey, AttemptState},
    wav,
};
use serde_json::Value;
use std::collections::HashMap;

#[test]
fn shared_original_corpus() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/audio-serving/cases.json"
    ))
    .unwrap();
    let mut frames = HashMap::new();
    for case in corpus["frames"].as_array().unwrap() {
        let encoded = serde_json::to_string(&case["frame"]).unwrap();
        let raw = case["raw"].as_str().unwrap_or(&encoded);
        let result = parse_frame(raw.as_bytes());
        assert_eq!(
            result.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}: {:?}",
            case["name"],
            result
        );
        if let Ok(frame) = result {
            frames.insert(case["name"].as_str().unwrap(), frame);
        }
    }
    for case in corpus["wav"].as_array().unwrap() {
        let hex = case["hex"].as_str().unwrap();
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(
            wav::inspect(&bytes, case["asr"].as_bool().unwrap()).is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}",
            case["name"]
        );
    }
    for case in corpus["states"].as_array().unwrap() {
        let mut state = AttemptState::new(
            AttemptKey {
                request_id: "22222222-2222-4222-8222-222222222222".into(),
                attempt: 1,
            },
            "33333333-3333-4333-8333-333333333333".into(),
        );
        let valid = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .all(|name| state.event(&frames[name.as_str().unwrap()].message).is_ok());
        assert_eq!(valid, case["valid"].as_bool().unwrap(), "{}", case["name"]);
    }
}

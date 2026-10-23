use super::*;
use crate::decision::{parse_decision_request, project_response};

fn corpus() -> Value {
    serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/serving/clef-v0.1.json"
    ))
    .unwrap()
}
fn expand(corpus: &Value, value: &Value, stack: &mut Vec<String>) -> Value {
    if let Some(reference) = value
        .as_object()
        .filter(|object| object.len() == 1)
        .and_then(|object| object.get("$ref"))
        .and_then(Value::as_str)
    {
        assert!(!stack.iter().any(|item| item == reference));
        stack.push(reference.into());
        let (section, key) = reference.split_once('.').unwrap();
        let result = expand(corpus, &corpus[section][key], stack);
        stack.pop();
        return result;
    }
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), expand(corpus, value, stack)))
                .collect(),
        ),
        Value::Array(array) => Value::Array(
            array
                .iter()
                .map(|value| expand(corpus, value, stack))
                .collect(),
        ),
        _ => value.clone(),
    }
}
fn fixture(corpus: &Value, section: &str, case: &Value) -> Value {
    let mut value = expand(
        corpus,
        &corpus[section][case["template"].as_str().unwrap()],
        &mut Vec::new(),
    );
    for pointer in case["remove"].as_array().into_iter().flatten() {
        let (parent, key) = pointer.as_str().unwrap().rsplit_once('/').unwrap();
        let target = value.pointer_mut(parent).unwrap();
        match target {
            Value::Object(map) => {
                map.remove(key);
            }
            Value::Array(array) => {
                array.remove(key.parse().unwrap());
            }
            _ => panic!("invalid fixture removal"),
        }
    }
    for (pointer, replacement) in case["set"].as_object().into_iter().flatten() {
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        let replacement = expand(corpus, replacement, &mut Vec::new());
        let target = value.pointer_mut(parent).unwrap();
        match target {
            Value::Object(map) => {
                map.insert(key.into(), replacement);
            }
            Value::Array(array) => array[key.parse::<usize>().unwrap()] = replacement,
            _ => panic!("invalid fixture replacement"),
        }
    }
    value
}
#[test]
fn shared_clef_requests() {
    let corpus = corpus();
    for case in corpus["requests"].as_array().unwrap() {
        let raw = case["raw"]
            .as_str()
            .map(str::as_bytes)
            .map(Vec::from)
            .unwrap_or_else(|| serde_json::to_vec(&fixture(&corpus, "objects", case)).unwrap());
        let result = parse_decision_request(&raw);
        assert_eq!(
            result.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}: {result:?}",
            case["name"]
        );
        if let Some(expected) = case.get("canonical") {
            assert_eq!(
                serde_json::to_value(result.unwrap().record).unwrap(),
                expand(&corpus, expected, &mut Vec::new())
            );
        }
    }
}
#[test]
fn shared_clef_encodings_and_projection() {
    let corpus = corpus();
    let record = validate_record(&expand(
        &corpus,
        &corpus["objects"]["record"],
        &mut Vec::new(),
    ))
    .unwrap();
    for case in corpus["encodings"].as_array().unwrap() {
        let value = fixture(&corpus, "objects", case);
        let result = validate_encoding(&value, &record, 4096, 248320);
        assert_eq!(
            result.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}: {result:?}",
            case["name"]
        );
    }
    for case in corpus["probabilities"].as_array().unwrap() {
        let result = if let Some(raw) = case["raw"].as_str() {
            parse_document(raw.as_bytes()).and_then(|value| validate_probabilities(&value, &record))
        } else {
            validate_probabilities(&fixture(&corpus, "objects", case), &record)
        };
        assert_eq!(
            result.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}: {result:?}",
            case["name"]
        );
        if let Ok(probabilities) = result {
            let response = project_response(
                "local-clef",
                &record,
                &DecisionResult {
                    probabilities,
                    input_tokens: 12,
                },
            );
            if let Some(expected) = case.get("response") {
                assert_eq!(response, expand(&corpus, expected, &mut Vec::new()));
            }
            if let Some(choice) = case.get("choice") {
                assert_eq!(response["answers"]["color"]["choice"], *choice);
            }
        }
    }
}
#[test]
fn shared_clef_digests_and_frames() {
    let corpus = corpus();
    for vector in corpus["digests"].as_array().unwrap() {
        assert_eq!(
            canonical_identity_digest(
                vector["domain"].as_str().unwrap(),
                &expand(&corpus, &vector["document"], &mut Vec::new())
            )
            .unwrap(),
            vector["expected"]
        );
    }
    for frame in corpus["frames"].as_object().unwrap().values() {
        validate_decision_frame(&expand(&corpus, frame, &mut Vec::new())).unwrap();
    }
    for case in corpus["frame_checks"].as_array().unwrap() {
        let frame = fixture(&corpus, "frames", case);
        let result = validate_decision_frame(&frame);
        assert_eq!(
            result.is_ok(),
            case["valid"].as_bool().unwrap(),
            "{}: {result:?}",
            case["name"]
        );
        if result.is_ok() {
            assert_eq!(decode_frame(&encode_frame(&frame).unwrap()).unwrap(), frame);
        }
    }
}
#[test]
fn shared_clef_stream_settlement() {
    let corpus = corpus();
    let prepare = expand(&corpus, &corpus["frames"]["prepare"], &mut Vec::new());
    for case in corpus["streams"].as_array().unwrap() {
        let mut tracker = DecisionTracker::new(&prepare).unwrap();
        let mut valid = true;
        for event in case["events"].as_array().unwrap() {
            let event = if let Some(name) = event.as_str() {
                expand(&corpus, &corpus["frames"][name], &mut Vec::new())
            } else {
                fixture(&corpus, "frames", event)
            };
            let before = format!("{tracker:?}");
            if tracker.observe(&event).is_err() {
                assert_eq!(format!("{tracker:?}"), before);
                valid = false;
                break;
            }
        }
        assert!(tracker.result().is_none(), "result exposed before reaping");
        assert!(!tracker.resources_released());
        if let Some(reap) = case.get("reap") {
            let result = tracker.reaped(
                reap["exit_code"].as_i64().unwrap() as i32,
                reap["eof"].as_bool().unwrap(),
            );
            valid &= result.is_ok();
        }
        assert_eq!(valid, case["valid"].as_bool().unwrap(), "{}", case["name"]);
        assert_eq!(
            tracker.result().is_some(),
            case["public_success"].as_bool().unwrap(),
            "{}",
            case["name"]
        );
        assert_eq!(
            tracker.resources_released(),
            case["resources_released"].as_bool().unwrap(),
            "{}",
            case["name"]
        );
    }
}

#[test]
fn clef_ready_rejects_unapproved_identity_shapes() {
    let corpus = corpus();
    let record = |path: &str| json!({"path":path,"sha256":"a".repeat(64)});
    let sources: Map<String, Value> = SOURCE_NAMES
        .iter()
        .map(|name| ((*name).into(), record(&format!("/adapter/{name}"))))
        .collect();
    let modules: Map<String, Value> = [
        "mlx_core",
        "mlx_lm",
        "transformers",
        "numpy",
        "pillow",
        "tokenizers",
    ]
    .iter()
    .map(|name| ((*name).into(), record(&format!("/env/{name}"))))
    .collect();
    let manifest = json!({"checkpoint":"mlx-community/clef-flash-4bit","checkpoint_revision":"b5adbe2c3a95200d4bf04a0727c8ec457157d5e1","adapter_revision":"apxinf-python-clef-text-v1","platform":"macos-metal","artifacts":approved_artifacts(),"loader":{"path":"/loader.py","sha256":LOADER_SHA256},"runtime":corpus["objects"]["runtime"],"sources":sources,"python":record("/env/bin/python"),"runtime_files":{"modules":modules,"libraries":[record("/env/libmlx.dylib")]},"execution":{"provider":"python-clef","precision":"backbone-4bit-head-bf16","backend":"lm","max_context":4096,"max_output_tokens":0,"idle_cache_limit_bytes":0,"memory_limit_bytes":9000000000u64,"truncate":false,"kv_reuse":false,"worker_lifetime":"one-attempt"}});
    let mut ready = json!({"protocol":PROTOCOL,"kind":"ready","worker_epoch":"00000000-0000-4000-8000-000000000001","model_manifest":manifest,"capabilities":corpus["objects"]["capabilities"],"limits":{"max_frame_bytes":MAX_FRAME_BYTES,"max_depth":MAX_DEPTH,"max_commands":MAX_COMMANDS},"model_path":"/model","runtime":corpus["objects"]["runtime"],"memory":{"active_bytes":0,"peak_bytes":0,"cache_bytes":0}});
    let refresh = |frame: &mut Value| {
        frame["model_revision"] =
            canonical_identity_digest("apxinf-model-identity-v1", &frame["model_manifest"])
                .unwrap()
                .into();
        frame["capability_revision"] =
            canonical_identity_digest("apxinf-capability-v1", &frame["capabilities"])
                .unwrap()
                .into();
    };
    refresh(&mut ready);
    validate_frame(&ready).unwrap();
    assert!(validate_command_envelope(&ready).is_err());
    for (pointer, value) in [
        ("/model_manifest/loader/sha256", json!("b".repeat(64))),
        ("/model_manifest/artifacts/0/sha256", json!("b".repeat(64))),
        (
            "/model_manifest/sources/clef_worker.py/path",
            json!("/wrong.py"),
        ),
        ("/runtime/python", json!("3.12.11")),
        ("/capabilities/token_events", json!(true)),
        ("/model_manifest/execution/max_output_tokens", json!(1)),
    ] {
        let mut invalid = ready.clone();
        *invalid.pointer_mut(pointer).unwrap() = value;
        refresh(&mut invalid);
        assert!(validate_frame(&invalid).is_err(), "{pointer}");
    }
    let mut unknown = ready.clone();
    unknown["model_manifest"]["sources"]["extra.py"] = record("/adapter/extra.py");
    refresh(&mut unknown);
    assert!(validate_frame(&unknown).is_err());
}

use super::*;
use crate::{host_pressure::HostPressurePolicy, supervisor::WorkerRuntime};
use std::{path::PathBuf, time::Duration};

struct Files(PathBuf);

impl Files {
    fn new() -> Self {
        let root =
            std::env::temp_dir().join(format!("apxinf-vision-identity-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(root.join("pyvenv.cfg"), "home = fixture\n").unwrap();
        Self(root)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        std::fs::canonicalize(path).unwrap()
    }

    fn config(&self) -> Config {
        Config {
            runtime: WorkerRuntime::PythonMiniCpmVision,
            python: self.0.join("bin/python"),
            worker: self.0.join("vision_worker.py"),
            model: self.0.clone(),
            model_id: "vision-test".into(),
            max_context: 16384,
            max_output: 8192,
            prefill_step_size: 256,
            output_batch_tokens: 1,
            queue_capacity: 1,
            timeout: Duration::from_secs(1),
            memory_budget: 4194304,
            sequence_reservation: 1024,
            host_pressure_policy: HostPressurePolicy::Disabled,
        }
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn profile() -> Value {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/serving/minicpm-vision-v0.1.json"
    ))
    .unwrap();
    let mut manifest = corpus["templates"]["ready"]["model_manifest"].clone();
    manifest["platform"] = json!("macos-metal");
    manifest["packages"] = json!({"mlx":"0.32.4","mlx-metal":"0.32.4","mlx-vlm":"0.7.10","transformers":"5.18.0", "numpy":"2.5.4","Pillow":"13.0.0","torch":"2.12.1","torchvision":"0.27.1", "tokenizers":"0.23.3","safetensors":"0.8.0","huggingface-hub":"2.2.0"});
    manifest["artifacts"] = json!(ARTIFACTS
        .iter()
        .map(|(name, size, digest)| json!({"path":name,"size_bytes":size,"sha256":digest}))
        .collect::<Vec<_>>());
    manifest["processor"] = json!({
        "class":"mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor",
        "runtime":manifest["runtime"],
        "artifacts":manifest["artifacts"].as_array().unwrap().iter().filter(|artifact| artifact["path"] != "model.safetensors").collect::<Vec<_>>(),
        "options":{"return_tensors":"np","downsample_mode":"16x","max_slice_nums":1,"padding":true,"add_special_tokens":false},
        "template_options":{"enable_thinking":false,"tokenize":false,"add_generation_prompt":true}
    });
    manifest["processor_revision"] = json!(contracts::canonical_identity_digest(
        "apxinf-vision-processor-v1",
        &manifest["processor"]
    )
    .unwrap());
    manifest["request_state"] = json!({"revision":"minicpmv47-transient-state-v1","max_context":16384,"position_fields":["_position_ids","_rope_deltas"],"convolution_cache_field":"_qwen3_5_decode_conv_weight","convolution_module_count":18,"rotary_cache_field":"_compiled_apply","rotary_module_count":6});
    manifest
}

#[test]
fn selected_profile_refuses_mutated_runtime_processor_state_and_execution() {
    let files = Files::new();
    let config = files.config();
    let manifest = profile();
    verify_profile(&manifest, &config).unwrap();
    for (pointer, replacement) in [
        ("/packages/Pillow", json!("13.0.1")),
        ("/runtime/mlx", json!("0.32.1")),
        ("/processor/options/max_slice_nums", json!(2)),
        ("/processor/template_options/tokenize", json!(true)),
        ("/processor_revision", json!("0".repeat(64))),
        ("/artifacts/0/sha256", json!("0".repeat(64))),
        ("/request_state/rotary_module_count", json!(5)),
        ("/request_state/convolution_module_count", json!(17)),
        (
            "/request_state/position_fields",
            json!(["_rope_deltas", "_position_ids"]),
        ),
        ("/execution/max_context", json!(8192)),
        ("/execution/max_output_tokens", json!(4096)),
        ("/execution/memory_limit_bytes", json!(8388608)),
        ("/execution/selection/top_k", json!(99)),
    ] {
        let mut altered = manifest.clone();
        *altered.pointer_mut(pointer).unwrap() = replacement;
        assert!(verify_profile(&altered, &config).is_err(), "{pointer}");
    }
    let mut extra = manifest.clone();
    extra["packages"]["unqualified"] = json!("1");
    assert!(verify_profile(&extra, &config).is_err());
    for (context, output, chunk, batch) in [
        (16385, 8192, 256, 1),
        (0, 8192, 256, 1),
        (16384, 8193, 256, 1),
        (16384, 8192, 128, 1),
        (16384, 8192, 256, 2),
    ] {
        let mut altered = config.clone();
        altered.max_context = context;
        altered.max_output = output;
        altered.prefill_step_size = chunk;
        altered.output_batch_tokens = batch;
        assert!(altered.validate_memory().is_err());
    }
}

#[test]
fn every_first_party_source_and_selected_interpreter_is_checked() {
    let files = Files::new();
    let config = files.config();
    let mut manifest = json!({"sources":{}});
    for name in SOURCES {
        let path = files.write(name, name.as_bytes());
        manifest["sources"][name] = json!(file_digest(&path).unwrap().1);
    }
    let python = files.write("bin/python", b"selected interpreter");
    manifest["python_sha256"] = json!(file_digest(&python).unwrap().1);
    manifest["adapter_sha256"] = manifest["sources"]["vision_worker.py"].clone();
    verify_sources(&manifest, &config).unwrap();
    for name in SOURCES.into_iter().chain(["bin/python"]) {
        let path = files.0.join(name);
        let original = std::fs::read(&path).unwrap();
        std::fs::write(&path, b"changed source").unwrap();
        assert!(verify_sources(&manifest, &config).is_err(), "{name}");
        std::fs::write(path, original).unwrap();
    }
    manifest["sources"]["unexpected.py"] = json!("0".repeat(64));
    assert!(verify_sources(&manifest, &config).is_err());
}

#[test]
fn runtime_records_bind_selected_paths_content_and_pinned_state_sources() {
    let files = Files::new();
    let path = files.write(
        "lib/python3.14/site-packages/numpy/__init__.py",
        b"runtime fixture",
    );
    let root = std::fs::canonicalize(&files.0).unwrap();
    let digest = file_digest(&path).unwrap().1;
    let record = json!({"path":path,"sha256":digest});
    verify_runtime_record(&record, &path, &root, None).unwrap();
    verify_runtime_record(&record, &path, &root, Some(&digest)).unwrap();
    assert!(verify_runtime_record(&record, &path, &root, Some(LIBRARY_SHA256)).is_err());
    for (_, _, pinned) in MODULES.iter().filter(|(_, _, pinned)| pinned.is_some()) {
        assert!(verify_runtime_record(&record, &path, &root, *pinned).is_err());
    }
    let other = files.write("lib/python3.14/site-packages/other.py", b"runtime fixture");
    assert!(verify_runtime_record(&record, &other, &root, None).is_err());
    assert!(
        verify_runtime_record(&record, &path, &root.join("another-environment"), None).is_err()
    );
    std::fs::write(&path, b"changed runtime").unwrap();
    assert!(verify_runtime_record(&record, &path, &root, None).is_err());
    let mut modules = json!({});
    for (name, _, _) in MODULES {
        modules[name] = record.clone();
    }
    for (name, _, _) in MODULES {
        let mut incomplete = modules.clone();
        incomplete.as_object_mut().unwrap().remove(name);
        let error = verify_runtime_files(
            &json!({"modules":incomplete,"libraries":[record]}),
            &files.config().python,
        )
        .unwrap_err();
        assert!(error.contains("fields"), "{name}: {error}");
    }
}

#[test]
fn prepared_replay_preserves_order_repeats_and_selected_processor() {
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/serving/minicpm-vision-v0.1.json"
    ))
    .unwrap();
    let ready = &corpus["templates"]["ready"];
    let mut prepared = corpus["templates"]["prepared"].clone();
    let first = prepared["replay"]["media"][0].clone();
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/serving/owned-media-v0.1.json"
    ))
    .unwrap();
    let second = corpus["images"]["rgba"].clone();
    let messages = vec![
        json!({"role":"user","content":[{"type":"image","media":first},{"type":"image","media":second}]}),
        json!({"role":"tool","tool_call_id":"call_a","content":[{"type":"image","media":first}]}),
    ];
    prepared["replay"]["media"] = json!([first, second, first]);
    verify_prepared(&prepared, ready, &messages).unwrap();
    for media in [
        json!([first, first, second]),
        json!([first, second]),
        json!([]),
    ] {
        let mut altered = prepared.clone();
        altered["replay"]["media"] = media;
        assert!(verify_prepared(&altered, ready, &messages).is_err());
    }
    for key in ["processor_revision", "capability_revision"] {
        let mut altered = prepared.clone();
        altered["replay"][key] = json!("0".repeat(64));
        assert!(verify_prepared(&altered, ready, &messages).is_err());
    }
}

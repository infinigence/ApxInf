//! End-to-end Ascend NPU inference test with synthetic weights.
//!
//! Requires a real Ascend NPU device. Run with:
//!   APXINF_ASCEND_TEST=1 cargo test --features ascend --test ascend_e2e -- --nocapture --ignored

#![cfg(any(feature = "ascend", feature = "cuda"))]

use apxinf_core::{Device, RngKey};
use apxinf_model::auto::{AutoModel, LoadOptions, ModelPrecision, SyntheticWeights};
use apxinf_model::vla::{ImageLayout, Observation, VisionObservation, VlaRequest};

#[test]
#[ignore]
fn ascend_synthetic_e2e_inference() {
    if std::env::var("APXINF_ASCEND_TEST").is_err() {
        eprintln!("set APXINF_ASCEND_TEST=1 to run this test on a real NPU");
        return;
    }

    // Use synthetic weights — no checkpoint needed.
    let options = LoadOptions {
        model_name: Some("pi05".into()),
        model_variant: Some("bf16".into()),
        precision: ModelPrecision::Auto,
        synthetic: Some(SyntheticWeights { seed: 42 }),
        ..LoadOptions::default()
    };

    // load_model needs a path for config.json; use a temp dir with default config.
    let tmp = std::env::temp_dir().join("apxinf_ascend_e2e_test");
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    // No config.json → Pi05Config::default() will be used.

    eprintln!("[test] Loading synthetic PI0.5 BF16 model on NPU:0...");
    let load_start = std::time::Instant::now();
    let loaded = AutoModel::load_model(Device::Npu(0), &tmp, &options)
        .expect("load_model should succeed with synthetic weights");
    eprintln!("[test] Model loaded successfully in {:.3}s", load_start.elapsed().as_secs_f64());

    let vla = loaded.vla().expect("loaded model should be VLA");
    let contract = vla.contract();
    eprintln!(
        "[test] VLA contract: action_shape={:?}",
        contract.action_shape
    );

    // Construct a synthetic observation.
    // Use the model's contract to determine image dimensions.
    let image_bytes = contract.num_views * contract.image_size * contract.image_size * 3;
    eprintln!(
        "[test] Image config: num_views={}, image_size={}, image_bytes={}",
        contract.num_views, contract.image_size, image_bytes
    );
    let rgb = vec![128u8; image_bytes]; // grey images
    let token_ids = vec![1u32, 2, 3, 4, 5]; // dummy tokens

    let observation = Observation {
        vision: VisionObservation::RgbU8 {
            bytes: rgb,
            layout: ImageLayout::Nhwc,
        },
        token_ids,
        state: None,
        action_mask: None,
    };

    let rng = RngKey::new(42, 0, 0);
    let request = VlaRequest::generated(&observation, rng);

    eprintln!("[test] Running inference (prefill + denoise)...");
    let infer_start = std::time::Instant::now();
    let actions = vla
        .infer_host_f32(&request)
        .expect("inference should produce actions");
    eprintln!("[test] Inference completed in {:.3}s, {} action values", infer_start.elapsed().as_secs_f64(), actions.len());

    // Verify output
    let [horizon, action_dim] = vla.action_shape();
    let expected_len = horizon * action_dim;
    assert_eq!(
        actions.len(),
        expected_len,
        "action count should be horizon * action_dim"
    );

    // Check all values are finite
    let non_finite = actions
        .iter()
        .filter(|x| !x.is_finite())
        .count();
    assert_eq!(non_finite, 0, "all action values should be finite");

    // Check not all zeros (synthetic weights are random, so output should be non-trivial)
    let all_zero = actions.iter().all(|x| *x == 0.0);
    assert!(!all_zero, "actions should not be all zeros");

    eprintln!(
        "[test] PASS: {} actions, all finite, non-trivial. Sample: {:?}",
        actions.len(),
        &actions[..actions.len().min(10)]
    );
}

/// Real-weights end-to-end inference test.
///
/// Set `APXINF_WEIGHTS_PATH` to the directory containing `model.safetensors`.
/// Set `APXINF_REAL_TEST=1` to run.
///
/// On NPU:  `APXINF_REAL_TEST=1 APXINF_WEIGHTS_PATH=/path/to/pi05_libero cargo test --features ascend --test ascend_e2e real_weights -- --nocapture --ignored`
/// On CUDA: `APXINF_REAL_TEST=1 APXINF_WEIGHTS_PATH=/path/to/pi05_libero cargo test --features cuda --test ascend_e2e real_weights -- --nocapture --ignored`
#[test]
#[ignore]
fn real_weights_e2e_inference() {
    if std::env::var("APXINF_REAL_TEST").is_err() {
        eprintln!("set APXINF_REAL_TEST=1 to run this test with real weights");
        return;
    }

    let weights_path = std::env::var("APXINF_WEIGHTS_PATH")
        .expect("APXINF_WEIGHTS_PATH must point to the directory containing model.safetensors");
    let checkpoint = std::path::PathBuf::from(&weights_path);

    assert!(
        checkpoint.join("model.safetensors").exists(),
        "model.safetensors not found in {weights_path}"
    );

    let device = if cfg!(feature = "ascend") {
        Device::Npu(0)
    } else if cfg!(feature = "cuda") {
        Device::Cuda(0)
    } else {
        panic!("either 'ascend' or 'cuda' feature must be enabled");
    };

    let options = LoadOptions {
        model_name: Some("pi05".into()),
        model_variant: Some("bf16".into()),
        precision: ModelPrecision::Auto,
        synthetic: None, // use real weights
        ..LoadOptions::default()
    };

    eprintln!("[test] Loading real PI0.5 weights from {weights_path} on {device:?}...");
    let load_start = std::time::Instant::now();
    let loaded = AutoModel::load_model(device, &checkpoint, &options)
        .expect("load_model should succeed with real weights");
    eprintln!("[test] Model loaded successfully in {:.3}s", load_start.elapsed().as_secs_f64());

    let vla = loaded.vla().expect("loaded model should be VLA");
    let contract = vla.contract();
    eprintln!(
        "[test] VLA contract: action_shape={:?}",
        contract.action_shape
    );

    // Construct a deterministic observation (same for both NPU and CUDA)
    // Use the model's own config to determine image dimensions.
    let image_bytes = contract.num_views * contract.image_size * contract.image_size * 3;
    eprintln!(
        "[test] Image config: num_views={}, image_size={}, image_bytes={}",
        contract.num_views, contract.image_size, image_bytes
    );
    let rgb = vec![128u8; image_bytes]; // grey images
    let token_ids = vec![1u32, 2, 3, 4, 5]; // dummy tokens

    let observation = Observation {
        vision: VisionObservation::RgbU8 {
            bytes: rgb,
            layout: ImageLayout::Nhwc,
        },
        token_ids,
        state: None,
        action_mask: None,
    };

    // Fixed seed for reproducibility
    let rng = RngKey::new(42, 0, 0);
    let request = VlaRequest::generated(&observation, rng);

    eprintln!("[test] Running inference (prefill + denoise)...");
    let infer_start = std::time::Instant::now();
    let actions = vla
        .infer_host_f32(&request)
        .expect("inference should produce actions");
    eprintln!("[test] Inference completed in {:.3}s, {} action values", infer_start.elapsed().as_secs_f64(), actions.len());

    // Verify output
    let [horizon, action_dim] = vla.action_shape();
    let expected_len = horizon * action_dim;
    assert_eq!(
        actions.len(),
        expected_len,
        "action count should be horizon * action_dim"
    );

    // Check all values are finite
    let non_finite = actions
        .iter()
        .filter(|x| !x.is_finite())
        .count();
    assert_eq!(non_finite, 0, "all action values should be finite");

    // Print full output for cross-device comparison
    eprintln!("[test] OUTPUT_ACTIONS_JSON: {actions:?}");

    // Print summary stats
    let min = actions.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = actions.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = actions.iter().sum::<f32>() / actions.len() as f32;
    let var = actions.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / actions.len() as f32;
    let std = var.sqrt();
    eprintln!(
        "[test] STATS: len={} min={:.6} max={:.6} mean={:.6} std={:.6}",
        actions.len(), min, max, mean, std
    );
    eprintln!("[test] FIRST_10: {:?}", &actions[..actions.len().min(10)]);
    eprintln!("[test] LAST_10: {:?}", &actions[actions.len().saturating_sub(10)..]);
}

/// Benchmark: 10 consecutive inferences to measure steady-state latency.
///
/// First inference includes prepare (kernel compilation). Subsequent inferences
/// with the same spec should hit the prepared cache and only pay run cost.
#[test]
#[ignore]
fn real_weights_benchmark_10() {
    if std::env::var("APXINF_REAL_TEST").is_err() {
        eprintln!("set APXINF_REAL_TEST=1 to run this benchmark");
        return;
    }

    let weights_path = std::env::var("APXINF_WEIGHTS_PATH")
        .expect("APXINF_WEIGHTS_PATH must point to the directory containing model.safetensors");
    let checkpoint = std::path::PathBuf::from(&weights_path);
    assert!(checkpoint.join("model.safetensors").exists());

    let device = Device::Npu(0);
    let options = LoadOptions {
        model_name: Some("pi05".into()),
        model_variant: Some("bf16".into()),
        precision: ModelPrecision::Auto,
        synthetic: None,
        ..LoadOptions::default()
    };

    eprintln!("[bench] Loading real PI0.5 weights from {weights_path}...");
    let load_start = std::time::Instant::now();
    let loaded = AutoModel::load_model(device, &checkpoint, &options)
        .expect("load_model should succeed");
    eprintln!("[bench] Model loaded in {:.3}s", load_start.elapsed().as_secs_f64());

    let vla = loaded.vla().expect("loaded model should be VLA");
    let contract = vla.contract();
    let image_bytes = contract.num_views * contract.image_size * contract.image_size * 3;
    let rgb = vec![128u8; image_bytes];
    let token_ids = vec![1u32, 2, 3, 4, 5];

    let observation = Observation {
        vision: VisionObservation::RgbU8 {
            bytes: rgb,
            layout: ImageLayout::Nhwc,
        },
        token_ids,
        state: None,
        action_mask: None,
    };

    // Run 10 consecutive inferences with the same spec
    let mut latencies = Vec::with_capacity(10);
    for i in 0..10 {
        let rng = RngKey::new(42, i as u64, 0);
        let request = VlaRequest::generated(&observation, rng);
        let t0 = std::time::Instant::now();
        let actions = vla.infer_host_f32(&request).expect("inference should succeed");
        let elapsed = t0.elapsed().as_secs_f64();
        let non_finite = actions.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(non_finite, 0, "iteration {i}: non-finite actions");
        eprintln!("[bench] iter {i}: {:.3}s ({} actions)", elapsed, actions.len());
        latencies.push(elapsed);
    }

    let warm: Vec<f64> = latencies[1..].to_vec();
    let warm_avg = warm.iter().sum::<f64>() / warm.len() as f64;
    let warm_min = warm.iter().cloned().fold(f64::INFINITY, f64::min);
    let warm_max = warm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[bench] SUMMARY: first={:.3}s | warm avg={:.3}s min={:.3}s max={:.3}s (9 iters)",
        latencies[0], warm_avg, warm_min, warm_max,
    );
}


/// Repeated synthetic inference — A1 leak reproduction / regression check.
///
/// Forces eager mode (`APXINF_NO_GRAPH=1`, no RI graph) and runs
/// `APXINF_E2E_ITERS` (default 6) consecutive inferences against the same
/// prepared spec. Under the workspace-pool overflow (A1) leak every eager
/// forward that exceeds the pool capacity individually mallocs an output
/// that is never freed, so device free memory drops monotonically. The fix
/// keeps device free bounded across iterations.
#[test]
#[ignore]
fn ascend_synthetic_e2e_inference_repeated() {
    if std::env::var("APXINF_ASCEND_TEST").is_err() {
        eprintln!("set APXINF_ASCEND_TEST=1 to run this test on a real NPU");
        return;
    }
    // Force eager so the session output-path runs on every iteration.
    std::env::set_var("APXINF_NO_GRAPH", "1");
    let iterations: usize = std::env::var("APXINF_E2E_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    let options = LoadOptions {
        model_name: Some("pi05".into()),
        model_variant: Some("bf16".into()),
        precision: ModelPrecision::Auto,
        synthetic: Some(SyntheticWeights { seed: 42 }),
        ..LoadOptions::default()
    };
    let tmp = std::env::temp_dir().join("apxinf_ascend_e2e_repeated");
    std::fs::create_dir_all(&tmp).expect("create temp dir");

    eprintln!("[test] Loading synthetic PI0.5 BF16 on NPU:0 (repeated, eager, iters={iterations})...");
    let loaded = AutoModel::load_model(Device::Npu(0), &tmp, &options)
        .expect("load_model should succeed with synthetic weights");
    let vla = loaded.vla().expect("loaded model should be VLA");
    let contract = vla.contract();

    let image_bytes = contract.num_views * contract.image_size * contract.image_size * 3;
    let rgb = vec![128u8; image_bytes];
    let token_ids = vec![1u32, 2, 3, 4, 5];
    let observation = Observation {
        vision: VisionObservation::RgbU8 {
            bytes: rgb,
            layout: ImageLayout::Nhwc,
        },
        token_ids,
        state: None,
        action_mask: None,
    };

    for i in 0..iterations {
        let rng = RngKey::new(42, i as u64, 0);
        let request = VlaRequest::generated(&observation, rng);
        let t0 = std::time::Instant::now();
        let actions = vla
            .infer_host_f32(&request)
            .expect("inference should succeed");
        let elapsed = t0.elapsed().as_secs_f64();
        let non_finite = actions.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(non_finite, 0, "iteration {i}: non-finite actions");
        eprintln!("[test] iter {i}: {elapsed:.3}s actions={} non_finite={non_finite}", actions.len());
    }
    eprintln!("[test] REPEATED PASS: {iterations} inferences OK");
}

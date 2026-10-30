//! CPU FP32 vs CUDA precision and eager vs graph regression, using pinned inputs.
//! See tests/pi05_reference/README.md. Measurement mode never reports a passing gate.
use apxinf_core::{Device, Tensor};
use apxinf_model::{
    AutoModel, ImageLayout, LoadOptions, Observation, Pi05Config, VisionObservation, VlaRequest,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    checkpoint_sha256: String,
    num_views: usize,
    action_horizon: usize,
    num_flow_steps: usize,
    cases: Vec<Case>,
    // Pin measured limits for this corpus, not arbitrary synthetic thresholds.
    limits: BTreeMap<String, Limits>,
    calibration_sha256: Option<String>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    images: PathBuf,
    images_sha256: String,
    token_ids: Vec<u32>,
    latent: Vec<f32>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Limits {
    min_cosine: f64,
    max_relative_l2: f64,
    max_abs: f64,
    eager_graph_max_abs: f64,
}
#[cfg(any(feature = "cuda", test))]
impl Limits {
    fn accepts_reference(&self, metrics: &serde_json::Value) -> bool {
        metrics["cosine"].as_f64().unwrap() >= self.min_cosine
            && metrics["relative_l2"].as_f64().unwrap() <= self.max_relative_l2
            && metrics["max_abs"].as_f64().unwrap() <= self.max_abs
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CpuReference {
    kind: String,
    input_sha256: String,
    actions: Vec<Vec<f32>>,
}
fn digest(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn verify(path: &Path, expected: &str) -> Result<()> {
    if digest(path)? != expected {
        return Err(format!("SHA256 mismatch: {}", path.display()).into());
    }
    Ok(())
}
#[cfg(any(feature = "cuda", test))]
fn metrics(actual: &[f32], expected: &[f32]) -> Result<serde_json::Value> {
    if actual.is_empty()
        || actual.len() != expected.len()
        || actual.iter().chain(expected).any(|x| !x.is_finite())
    {
        return Err("non-finite or mismatched action tensors".into());
    }
    let (mut dot, mut aa, mut ee, mut err2, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &e) in actual.iter().zip(expected) {
        let (a, e) = (a as f64, e as f64);
        dot += a * e;
        aa += a * a;
        ee += e * e;
        err2 += (a - e).powi(2);
        max_abs = max_abs.max((a - e).abs());
    }
    // Zero reference outputs cannot establish the relative-error gate.
    if aa == 0. || ee == 0. {
        return Err("zero-norm action tensor".into());
    }
    Ok(
        serde_json::json!({"cosine": dot / (aa*ee).sqrt(), "relative_l2": (err2/ee).sqrt(), "max_abs": max_abs}),
    )
}
fn run(args: Vec<String>) -> Result<()> {
    if args.len() < 4 {
        return Err("usage: pi05_reference_gate CHECKPOINT.safetensors MANIFEST.json cpu|bf16|fp8_static|int8_dynamic REPORT.json [--calibration FILE] [--cpu-reference FILE] [--tuning DIR] [--autotune] [--measure]".into());
    }
    let checkpoint = Path::new(&args[0]);
    let manifest_path = Path::new(&args[1]);
    let variant = args[2].as_str();
    if !["cpu", "bf16", "fp8_static", "int8_dynamic"].contains(&variant) {
        return Err("unsupported variant".into());
    }
    #[cfg(not(feature = "cuda"))]
    if variant != "cpu" {
        return Err("GPU comparison requires --features cuda,openblas".into());
    }
    let mut measure = false;
    let mut calibration = None;
    let mut cpu_reference = None;
    let mut tuning = None;
    let mut autotune = false;
    let mut flags = args[4..].iter();
    while let Some(flag) = flags.next() {
        match flag.as_str() {
            "--measure" => measure = true,
            "--autotune" => autotune = true,
            "--tuning" => {
                tuning = Some(PathBuf::from(
                    flags.next().ok_or("missing tuning directory")?,
                ))
            }
            "--calibration" => {
                calibration = Some(PathBuf::from(
                    flags.next().ok_or("missing calibration path")?,
                ))
            }
            "--cpu-reference" => {
                cpu_reference = Some(PathBuf::from(
                    flags.next().ok_or("missing CPU reference path")?,
                ))
            }
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if autotune && tuning.is_none() {
        return Err(
            "--autotune requires an explicit --tuning directory for generated recipes".into(),
        );
    }
    if variant == "cpu" && (tuning.is_some() || autotune) {
        return Err("CPU does not use CUDA tuning".into());
    }
    if variant == "cpu" && cpu_reference.is_some() {
        return Err("cpu mode must compute its reference".into());
    }
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(manifest_path)?)?;
    if manifest.cases.is_empty() {
        return Err("empty fixture corpus".into());
    }
    verify(checkpoint, &manifest.checkpoint_sha256)?;
    if variant == "fp8_static" {
        verify(
            calibration
                .as_deref()
                .ok_or("FP8 requires explicit calibration")?,
            manifest
                .calibration_sha256
                .as_deref()
                .ok_or("missing calibration SHA256")?,
        )?;
    } else if calibration.is_some() {
        return Err("calibration is only accepted for FP8".into());
    }
    let limits = manifest.limits.get(variant);
    if variant != "cpu" && !measure && limits.is_none() {
        return Err("missing measured limits; use --measure to collect evidence first".into());
    }
    if let Some(l) = limits {
        if !l.min_cosine.is_finite()
            || !(0.0..=1.0).contains(&l.min_cosine)
            || [l.max_relative_l2, l.max_abs, l.eager_graph_max_abs]
                .iter()
                .any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err("invalid acceptance limits".into());
        }
    }
    let config = Pi05Config {
        num_views: manifest.num_views,
        action_horizon: manifest.action_horizon,
        num_flow_steps: manifest.num_flow_steps,
        ..Pi05Config::default()
    };
    let root = manifest_path.parent().unwrap_or(Path::new("."));
    let mut inputs = Vec::new();
    for case in &manifest.cases {
        let path = root.join(&case.images);
        verify(&path, &case.images_sha256)?;
        inputs.push((
            Observation {
                vision: VisionObservation::RgbU8 {
                    bytes: std::fs::read(path)?,
                    layout: ImageLayout::Nhwc,
                },
                token_ids: case.token_ids.clone(),
                state: None,
                action_mask: None,
            },
            Tensor::from_f32(vec![config.action_horizon, config.action_dim], &case.latent)?,
        ));
    }
    let input_identity = serde_json::json!({"checkpoint_sha256":manifest.checkpoint_sha256,
        "num_views":manifest.num_views, "action_horizon":manifest.action_horizon,
        "num_flow_steps":manifest.num_flow_steps, "cases":manifest.cases});
    let input_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&input_identity)?));
    let expected: Vec<Vec<f32>> = if let Some(path) = cpu_reference {
        let saved: CpuReference = serde_json::from_slice(&std::fs::read(path)?)?;
        if saved.input_sha256 != input_sha256
            || saved.kind != "apxinf_pi05_cpu_fp32_v1"
            || saved.actions.len() != manifest.cases.len()
            || saved.actions.iter().any(|a| {
                a.len() != config.action_horizon * config.action_dim
                    || a.iter().any(|x| !x.is_finite())
            })
        {
            return Err("CPU reference identity or output geometry mismatch".into());
        }
        saved.actions
    } else {
        let cpu = AutoModel::load_model(
            Device::Cpu,
            checkpoint,
            &LoadOptions {
                model_name: Some("pi05".into()),
                model_variant: Some("fp32_reference".into()),
                config: Some(config.clone()),
                ..Default::default()
            },
        )?;
        let mut outputs = Vec::new();
        for (case, (obs, noise)) in manifest.cases.iter().zip(&inputs) {
            eprintln!("CPU reference: {}", case.name);
            outputs.push(cpu.infer_host_f32(&VlaRequest::provided(obs, noise))?);
        }
        outputs
    };
    // Preserve the expensive reference even if subsequent GPU execution fails.
    let reference = CpuReference {
        kind: "apxinf_pi05_cpu_fp32_v1".into(),
        input_sha256,
        actions: expected.clone(),
    };
    let reference_path = format!("{}.cpu.json", args[3]);
    std::fs::write(
        &reference_path,
        serde_json::to_string_pretty(&reference)? + "\n",
    )?;
    let mut report = serde_json::json!({"manifest_sha256": digest(manifest_path)?, "checkpoint_sha256": manifest.checkpoint_sha256,
        "variant": variant, "mode": if variant == "cpu" {"reference_only"} else if measure {"measurement_only"} else {"check"}, "cases": []});
    report["tuning"] = serde_json::json!({"directory": tuning, "autotune": autotune});
    report["cpu_reference_sha256"] = serde_json::json!(digest(Path::new(&reference_path))?);
    let mut rows = Vec::new();
    #[cfg(feature = "cuda")]
    let mut passed = true;
    #[cfg(not(feature = "cuda"))]
    let passed = true;
    if variant == "cpu" {
        for (case, actions) in manifest.cases.iter().zip(&expected) {
            rows.push(serde_json::json!({"name":case.name, "actions": actions}));
        }
    } else {
        #[cfg(not(feature = "cuda"))]
        return Err("GPU comparison requires --features cuda,openblas".into());
        #[cfg(feature = "cuda")]
        {
            use apxinf_model::{ExecutionMode, ExecutionPolicy, PreparationStatus};
            let gpu = AutoModel::load_model(
                Device::Cuda(0),
                checkpoint,
                &LoadOptions {
                    model_name: Some("pi05".into()),
                    model_variant: Some(variant.into()),
                    config: Some(config),
                    calibration_path: calibration,
                    tuning_path: tuning,
                    autotune,
                    ..Default::default()
                },
            )?;
            for ((case, (obs, noise)), reference) in
                manifest.cases.iter().zip(&inputs).zip(&expected)
            {
                eprintln!("GPU {variant}: {}", case.name);
                let req = VlaRequest::provided(obs, noise);
                eprintln!("  preparing eager");
                let eager = gpu.prepare_for(&req, ExecutionPolicy::Eager)?;
                if !matches!(
                    eager.status(),
                    PreparationStatus::Ready {
                        mode: ExecutionMode::Eager,
                        ..
                    }
                ) {
                    return Err("expected eager plan".into());
                }
                let a =
                    apxinf_cuda_new::transfers::to_cpu(eager.run(&req)?.tensor())?.to_f32_vec()?;
                drop(eager);
                eprintln!("  preparing graph");
                let graph = gpu.prepare_for(&req, ExecutionPolicy::RequireGraph)?;
                if !matches!(
                    graph.status(),
                    PreparationStatus::Ready {
                        mode: ExecutionMode::Graph,
                        ..
                    }
                ) {
                    return Err("expected captured graph".into());
                }
                let b =
                    apxinf_cuda_new::transfers::to_cpu(graph.run(&req)?.tensor())?.to_f32_vec()?;
                let repeat =
                    apxinf_cuda_new::transfers::to_cpu(graph.run(&req)?.tensor())?.to_f32_vec()?;
                let eager_cpu = metrics(&a, reference)?;
                let graph_cpu = metrics(&b, reference)?;
                let eager_graph = metrics(&b, &a)?;
                let stable = b == repeat;
                let accepted = limits.map(|l| {
                    stable
                        && [&eager_cpu, &graph_cpu]
                            .iter()
                            .all(|m| l.accepts_reference(m))
                        && eager_graph["cosine"].as_f64().unwrap() >= 0.999_999
                        && eager_graph["max_abs"].as_f64().unwrap() <= l.eager_graph_max_abs
                });
                passed &= accepted.unwrap_or(false);
                rows.push(serde_json::json!({"name": case.name, "eager_vs_cpu":eager_cpu, "graph_vs_cpu":graph_cpu,
                    "eager_vs_graph":eager_graph, "repeat_bitwise_equal":stable, "accepted":if measure {None} else {accepted}}));
            }
        }
    }
    report["cases"] = serde_json::json!(rows);
    report["passed"] = if variant == "cpu" || measure {
        serde_json::Value::Null
    } else {
        serde_json::json!(passed)
    };
    std::fs::write(&args[3], serde_json::to_string_pretty(&report)? + "\n")?;
    if variant != "cpu" && !measure && !passed {
        return Err("CPU/GPU reference gate failed; see report".into());
    }
    Ok(())
}

/// External checkpoint and GPU execution are explicitly requested by run.py.
#[test]
#[ignore = "requires external checkpoint; use tests/pi05_reference/run.py on Thor/Orin"]
fn full_checkpoint_reference() {
    let args = std::env::var("APXINF_PI05_GATE_ARGS")
        .expect("use tests/pi05_reference/run.py to configure the full-checkpoint test");
    let args: Vec<String> = serde_json::from_str(&args).expect("JSON argument array");
    run(args).expect("full-checkpoint reference execution failed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracked_diagnostic_requests_have_pinned_inputs() {
        let manifest: Manifest =
            serde_json::from_str(include_str!("fixtures/pi05_gpu/manifest.json")).unwrap();
        let mut rgb = Vec::new();
        for view in 0..2 {
            for y in 0..224 {
                for x in 0..224 {
                    rgb.extend([
                        ((x + 41 * view) % 256) as u8,
                        ((3 * y + 29 * view) % 256) as u8,
                        ((x + 2 * y) % 256) as u8,
                    ]);
                }
            }
        }
        let hash = format!("{:x}", Sha256::digest(&rgb));
        assert_eq!(manifest.cases.len(), 2);
        for (case, tokens) in manifest.cases.iter().zip([10, 21]) {
            assert_eq!(case.images_sha256, hash);
            assert_eq!(case.token_ids.len(), tokens);
            assert_eq!(case.latent.len(), 50 * 32);
            assert!(case.latent.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn rejects_non_finite_and_wrong_shapes() {
        assert!(metrics(&[], &[]).is_err());
        assert!(metrics(&[1.], &[1., 2.]).is_err());
        assert!(metrics(&[f32::NAN], &[1.]).is_err());
        assert!(metrics(&[1.], &[f32::INFINITY]).is_err());
        assert!(metrics(&[1.], &[0.]).is_err());
    }

    #[test]
    fn localized_error_can_pass_aggregate_limits() {
        let expected = vec![1.; 1600];
        let mut actual = expected.clone();
        actual[21] += 1.;
        let m = metrics(&actual, &expected).unwrap();
        assert!(m["cosine"].as_f64().unwrap() > 0.999);
        assert!(m["relative_l2"].as_f64().unwrap() < 0.05);
        assert_eq!(m["max_abs"], 1.);
        let limits = Limits {
            min_cosine: 0.999,
            max_relative_l2: 0.05,
            max_abs: 0.125,
            eager_graph_max_abs: 0.01,
        };
        assert!(!limits.accepts_reference(&m));
        assert!(limits.accepts_reference(&metrics(&expected, &expected).unwrap()));
    }
}

//! CPU-only PI0.5 public loading, request and preparation contracts.
use apxinf_core::{DType, Device, RngKey, Tensor};
use apxinf_model::pi05::GemmaVariantConfig;
use apxinf_model::{
    AutoModel, ExecutionMode, ExecutionPolicy, ImageLayout, LoadOptions, Observation, Pi05Config,
    PreparationStatus, SyntheticWeights, VisionObservation, VlaRequest,
};

fn config() -> Pi05Config {
    let language = GemmaVariantConfig {
        width: 8,
        depth: 2,
        mlp_dim: 12,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
    };
    Pi05Config {
        action_dim: 3,
        action_horizon: 3,
        max_token_len: 4,
        num_flow_steps: 3,
        num_views: 2,
        image_size: 4,
        patch_size: 2,
        vision_width: 8,
        vision_depth: 2,
        vision_mlp_dim: 12,
        vision_heads: 2,
        vision_head_dim: 4,
        vocab_size: 16,
        language,
        action_expert: GemmaVariantConfig {
            width: 6,
            ..language
        },
        ..Pi05Config::default()
    }
}
fn options() -> LoadOptions {
    LoadOptions {
        model_name: Some("pi05".into()),
        model_variant: Some("fp32_reference".into()),
        config: Some(config()),
        synthetic: Some(SyntheticWeights { seed: 17 }),
        ..LoadOptions::default()
    }
}
fn observation() -> Observation {
    Observation {
        vision: VisionObservation::RgbU8 {
            bytes: (0..96).map(|x| (x * 7 % 256) as u8).collect(),
            layout: ImageLayout::Nhwc,
        },
        token_ids: vec![1, 3, 5],
        state: None,
        action_mask: None,
    }
}
#[test]
fn eager_cpu_loads_and_executes_without_cuda() {
    let model = AutoModel::load_model(Device::Cpu, ".", &options()).unwrap();
    let vla = model.vla().unwrap();
    assert_eq!(vla.model_variant(), Some("fp32_reference"));
    assert_eq!(vla.execution_mode(), "eager");
    let obs = observation();
    let latent =
        Tensor::from_f32(vec![3, 3], &[0.1, 0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8, 0.9]).unwrap();
    let req = VlaRequest::provided(&obs, &latent);
    let action = vla.infer(&req).unwrap();
    assert_eq!(action.tensor().device(), Device::Cpu);
    assert_eq!(action.tensor().dtype(), DType::F32);
    assert_eq!(action.tensor().shape().dims(), [3, 3]);
    let values = action.tensor().to_f32_vec().unwrap();
    assert!(values.iter().all(|x| x.is_finite()));
    assert_ne!(values, latent.to_f32_vec().unwrap());
    assert_eq!(values, vla.infer_host_f32(&req).unwrap());
    let generated = VlaRequest::generated(&obs, RngKey::new(7, 2, 3));
    let a = vla.infer_host_f32(&generated).unwrap();
    assert_eq!(a, vla.infer_host_f32(&generated).unwrap());
    assert_ne!(
        a,
        vla.infer_host_f32(&VlaRequest::generated(&obs, RngKey::new(7, 2, 4)))
            .unwrap()
    );
    assert_eq!(values, action.tensor().to_f32_vec().unwrap()); // later calls do not overwrite outputs
}
#[test]
fn prepared_cpu_is_eager_rebinds_inputs_and_retains_model() {
    let model = AutoModel::load_model(Device::Cpu, ".", &options()).unwrap();
    let vla = model.vla().unwrap();
    let obs = observation();
    let spec = obs.inference_spec();
    assert!(vla
        .prepare_with_policy(&spec, ExecutionPolicy::RequireGraph)
        .is_err());
    let plan = vla
        .prepare_with_policy(&spec, ExecutionPolicy::PreferGraph)
        .unwrap();
    assert!(matches!(
        plan.status(),
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: Some(_)
        }
    ));
    let a = plan
        .run(&VlaRequest::generated(&obs, RngKey::new(1, 0, 0)))
        .unwrap()
        .tensor()
        .to_f32_vec()
        .unwrap();
    let b = plan
        .run(&VlaRequest::generated(&obs, RngKey::new(2, 0, 0)))
        .unwrap()
        .tensor()
        .to_f32_vec()
        .unwrap();
    assert_ne!(a, b);
    let mut changed = obs.clone();
    changed.token_ids.pop();
    assert!(plan
        .run(&VlaRequest::generated(&changed, RngKey::new(1, 0, 0)))
        .is_err());
    vla.clear_prepared().unwrap();
    drop(model);
    assert_eq!(
        a,
        plan.run(&VlaRequest::generated(&obs, RngKey::new(1, 0, 0)))
            .unwrap()
            .tensor()
            .to_f32_vec()
            .unwrap()
    );
}
#[test]
fn rgb_layout_and_patch_input_are_equivalent() {
    let model = AutoModel::load_model(Device::Cpu, ".", &options()).unwrap();
    let vla = model.vla().unwrap();
    let obs = observation();
    let VisionObservation::RgbU8 { bytes, .. } = &obs.vision else {
        unreachable!()
    };
    let mut nchw = vec![0u8; 96];
    let mut patches = Vec::new();
    for view in 0..2 {
        for ch in 0..3 {
            for y in 0..4 {
                for x in 0..4 {
                    nchw[((view * 3 + ch) * 4 + y) * 4 + x] =
                        bytes[((view * 4 + y) * 4 + x) * 3 + ch];
                }
            }
        }
    }
    for view in 0..2 {
        for y in (0..4).step_by(2) {
            for x in (0..4).step_by(2) {
                for ch in 0..3 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            patches.push(
                                bytes[((view * 4 + y + dy) * 4 + x + dx) * 3 + ch] as f32
                                    * (2. / 255.)
                                    - 1.,
                            );
                        }
                    }
                }
            }
        }
    }
    let rng = RngKey::new(1, 2, 3);
    let a = vla
        .infer_host_f32(&VlaRequest::generated(&obs, rng))
        .unwrap();
    for vision in [
        VisionObservation::RgbU8 {
            bytes: nchw,
            layout: ImageLayout::Nchw,
        },
        VisionObservation::Patches(Tensor::from_f32(vec![8, 12], &patches).unwrap()),
    ] {
        let other = Observation {
            vision,
            ..obs.clone()
        };
        assert_eq!(
            a,
            vla.infer_host_f32(&VlaRequest::generated(&other, rng))
                .unwrap()
        );
    }
}
#[test]
fn cpu_rejects_invalid_variants_and_requests() {
    for variant in ["bf16", "fp8_static", "int8_dynamic"] {
        let mut opts = options();
        opts.model_variant = Some(variant.into());
        assert!(AutoModel::load_model(Device::Cpu, ".", &opts).is_err());
    }
    let model = AutoModel::load_model(Device::Cpu, ".", &options()).unwrap();
    let vla = model.vla().unwrap();
    let noise = Tensor::zeros(vec![3, 3], DType::F32);
    let mut obs = observation();
    obs.token_ids = vec![16];
    assert!(vla.infer(&VlaRequest::provided(&obs, &noise)).is_err());
    obs = observation();
    obs.vision = VisionObservation::RgbU8 {
        bytes: vec![0; 95],
        layout: ImageLayout::Nhwc,
    };
    assert!(vla.infer(&VlaRequest::provided(&obs, &noise)).is_err());
    obs = observation();
    let nan = Tensor::from_f32(vec![3, 3], &[f32::NAN; 9]).unwrap();
    assert!(vla.infer(&VlaRequest::provided(&obs, &nan)).is_err());
    obs.state = Some(noise.clone());
    assert!(vla.infer(&VlaRequest::provided(&obs, &noise)).is_err());
    // Frontend overrides are converted at AutoModel's boundary; validation
    // belongs to the independent CPU config, not the CUDA config type.
    let mut opts = options();
    opts.config.as_mut().unwrap().patch_size = 0;
    assert!(AutoModel::load_model(Device::Cpu, ".", &opts).is_err());
    opts.config = Some(config());
    opts.config.as_mut().unwrap().language.num_kv_heads = 0;
    assert!(AutoModel::load_model(Device::Cpu, ".", &opts).is_err());
}

/// Tracked tiny checkpoint and golden outputs from the independent PyTorch oracle.
/// See fixtures/pi05_cpu/README.md for provenance and regeneration.
#[test]
fn cpu_checkpoint_matches_independent_reference() {
    let dir = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/pi05_cpu"
    ));
    let mut opts = options();
    opts.synthetic = None;
    opts.model_name = None; // exercise detection
    let model = AutoModel::load_model(Device::Cpu, &dir, &opts).unwrap();
    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let obs = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: serde_json::from_value(case["rgb"].clone()).unwrap(),
                layout: ImageLayout::Nhwc,
            },
            token_ids: serde_json::from_value(case["tokens"].clone()).unwrap(),
            state: None,
            action_mask: None,
        };
        let noise: Vec<f32> = serde_json::from_value(case["noise"].clone()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(case["expected"].clone()).unwrap();
        let noise = Tensor::from_f32(vec![3, 3], &noise).unwrap();
        let request = VlaRequest::provided(&obs, &noise);
        let actual = model.vla().unwrap().infer_host_f32(&request).unwrap();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("CPU independent reference max_abs={max_error:e}");
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() <= 2e-5 + 2e-5 * b.abs(), "{a} != {b}");
        }
    }
}

//! Real-checkpoint acceptance. No captured tensor or checkpoint is vendored.
use super::*;
use crate::vla::{HistoryContext, Observation, VlaMetadata};
use std::path::{Path, PathBuf};

fn read(path: &Path) -> std::collections::HashMap<String, Tensor> {
    apxinf_loader::safetensors::load_native(path).unwrap().0
}
fn host(runner: &DreamZeroModelRunner, value: &Tensor) -> Vec<f32> {
    runner.model.ctx.synchronize().unwrap();
    transfers::to_cpu(value).unwrap().to_f32_vec().unwrap()
}
fn metric(name: &str, actual: &[f32], expected: &Tensor, actions: bool) -> bool {
    let expected = expected.to_f32_vec().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|x| x.is_finite()));
    let (mut error, mut norm, mut max) = (0f64, 0f64, 0f64);
    for (&a, &b) in actual.iter().zip(&expected) {
        let d = f64::from(a) - f64::from(b);
        error += d * d;
        norm += f64::from(b).powi(2);
        max = max.max(d.abs());
    }
    let relative = (error / norm.max(1e-30)).sqrt();
    let rmse = (error / actual.len() as f64).sqrt();
    eprintln!(
        "{}",
        serde_json::json!({"case":name,"relative_l2":relative,"rmse":rmse,"max_abs":max})
    );
    // Both modalities retain the original fixed-noise relative-L2 gate.
    relative <= 0.02 && (!actions || (rmse <= 0.02 && max <= 0.1))
}

#[test]
#[ignore = "requires pinned complete checkpoint and all official joint inference captures"]
fn checkpoint_joint_history_three_graphs_and_lifecycle() {
    joint_checkpoint(CheckpointCases::Full);
}

#[test]
#[ignore = "requires complete checkpoint and fixed official standalone captures"]
fn checkpoint_joint_standalone_noise_and_public_input() {
    joint_checkpoint(CheckpointCases::Standalone);
}

#[test]
#[ignore = "requires complete checkpoint and official episode/nine-frame captures"]
fn checkpoint_joint_nine_frame_history_precision() {
    joint_checkpoint(CheckpointCases::NineFrameHistory);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckpointCases {
    Full,
    Standalone,
    NineFrameHistory,
}

// Keep numerical diagnosis independently runnable before the longer history,
// RNG and resource-lifetime acceptance. The full test always exercises all cases.
fn joint_checkpoint(selection: CheckpointCases) {
    let checkpoint = PathBuf::from(std::env::var("APXINF_DREAMZERO_CHECKPOINT").unwrap());
    let fixtures = PathBuf::from(std::env::var("APXINF_DREAMZERO_REFERENCE_RUN").unwrap());
    let config = DreamZeroConfig::from_json_file(&checkpoint.join("config.json")).unwrap();
    drop(config);
    let ctx = Rc::new(
        crate::cuda_dreamzero::backend::Context::new_with_autotune(
            0,
            std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some(),
        )
        .unwrap(),
    );
    let weights =
        crate::cuda_dreamzero::weights::Bf16Weights::from_safetensors(&ctx, &checkpoint)
            .unwrap();
    let model = DreamZeroModel::new(
        ctx,
        weights,
        crate::cuda_dreamzero::config::Policies::default(),
    );
    let runner = DreamZeroModelRunner::new(model);
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("report.json")).unwrap()).unwrap();
    let names: Vec<_> = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "initial",
            "repeat",
            "history",
            "history2",
            "history3",
            "window",
            "reset",
            "task",
            "episode",
            "nine",
            "video-noise",
            "action-noise",
            "public"
        ],
        "full lifecycle fixture must contain every required case"
    );
    let mut numerical_failures = Vec::new();
    let mut episode = "episode-a";
    let mut first_action = None;
    let mut first_video = None;
    let mut cases = report["cases"].as_array().unwrap().clone();
    // Same episode/task and a multi-frame observation, but another caller.
    // Reuse the official explicit-reset fixture to prove session isolation.
    match selection {
        CheckpointCases::Full => {
            cases.push(serde_json::json!({"name":"session", "position_after":3}));
        }
        CheckpointCases::Standalone => cases.retain(|case| {
            ["initial", "video-noise", "public"].contains(&case["name"].as_str().unwrap())
        }),
        // Start from the independent official episode reset, then consume the
        // nine-frame observation. This preserves the reference history while
        // isolating sensitivity to prepared GEMM algorithm selection.
        CheckpointCases::NineFrameHistory => {
            cases.retain(|case| ["episode", "nine"].contains(&case["name"].as_str().unwrap()))
        }
    }
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        if name == "reset" {
            runner.reset_history().unwrap();
        }
        if name == "episode" {
            episode = "episode-b";
        }
        let fixture_name = if name == "session" { "reset" } else { name };
        let data = read(&fixtures.join(fixture_name).join("native.safetensors"));
        let frames = data["images"].shape().dims()[0] / (352 * 640);
        let observation = Observation {
            vision: VisionObservation::Patches(data["images"].clone()),
            state: Some(data["state"].clone()),
            token_ids: data["tokens"]
                .to_f32_vec()
                .unwrap()
                .iter()
                .map(|&x| x as u32)
                .collect(),
            action_mask: None,
        };
        let mask: Vec<_> = data["mask"]
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|&x| x as u8)
            .collect();
        let grid = [[frames as u32, 352, 640]];
        let request = VlaRequest::provided_with_metadata(
            &observation,
            &data["noise"],
            VlaMetadata {
                attention_mask: Some(&mask),
                image_grid_thw: Some(&grid),
                embodiment_id: Some(17),
                history: Some(HistoryContext {
                    session: if name == "session" {
                        "independent-caller"
                    } else {
                        "acceptance"
                    },
                    episode,
                }),
                ..Default::default()
            },
        );
        let before = runner.session.borrow().as_ref().map(|e| e.key.next_frame);
        let start = std::time::Instant::now();
        let plan = runner
            .make_plan(
                &request,
                &validate(&request).unwrap(),
                ExecutionPolicy::RequireGraph,
            )
            .unwrap();
        assert_eq!(
            before,
            runner.session.borrow().as_ref().map(|e| e.key.next_frame),
            "preparation published history"
        );
        assert!(matches!(
            plan.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                ..
            }
        ));
        let eager_action = host(&runner, &plan.work.borrow().actions());
        let eager_video = host(&runner, &plan.work.borrow().video());
        let prepared_seconds = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let action = plan.run(&request).unwrap();
        let values = host(&runner, action.tensor());
        let video = host(&runner, &plan.work.borrow().video());
        let replay_seconds = start.elapsed().as_secs_f64();
        assert_eq!(values, eager_action, "{name} action eager/graph parity");
        assert_eq!(video, eager_video, "{name} video eager/graph parity");
        // Keep the frozen gates, but exercise every history/profile transition
        // before reporting numerical failures from this expensive checkpoint run.
        for (kind, actual, expected, actions) in [
            ("action", &values, &data["action"], true),
            ("video", &video, &data["video"], false),
        ] {
            let checkpoint = format!("{name}.{kind}");
            if !metric(&checkpoint, actual, expected, actions) {
                numerical_failures.push(checkpoint);
            }
        }
        // Initial proves repeated replay; every later case still checks changed
        // inputs and its own eager/graph parity before committing history.
        for _ in 0..if name == "initial" && selection == CheckpointCases::Full {
            2
        } else {
            0
        } {
            for graph in plan.graphs.as_ref().unwrap() {
                graph.replay().unwrap();
            }
            assert_eq!(
                host(&runner, action.tensor()),
                values,
                "repeat graph replay"
            );
        }
        let position = runner.session.borrow().as_ref().unwrap().key.next_frame;
        assert_eq!(position, case["position_after"].as_u64().unwrap() as usize);
        runner.clear_prepared().unwrap();
        assert_eq!(
            position,
            runner.session.borrow().as_ref().unwrap().key.next_frame,
            "clear_prepared reset history"
        );
        if !plan.profile.initial() {
            assert!(
                plan.run(&request).is_err(),
                "old history profile accepted a later observation"
            );
        }
        if name == "initial" {
            first_action = Some(values.clone());
            first_video = Some(video.clone());
        }
        if name == "initial" && selection == CheckpointCases::Full {
            let invalid = Tensor::from_f32(vec![768], &vec![0.; 768]).unwrap();
            let invalid_request =
                VlaRequest::provided_with_metadata(&observation, &invalid, request.metadata);
            assert!(plan.run(&invalid_request).is_err());
            assert_eq!(
                position,
                runner.session.borrow().as_ref().unwrap().key.next_frame
            );
            let seeded = VlaRequest::generated_with_metadata(
                &observation,
                apxinf_core::RngKey::new(71, 0, 0),
                request.metadata,
            );
            let first = host(&runner, plan.run(&seeded).unwrap().tensor());
            let first_future = host(&runner, &plan.work.borrow().video());
            assert_eq!(first, host(&runner, plan.run(&seeded).unwrap().tensor()));
            assert_eq!(first_future, host(&runner, &plan.work.borrow().video()));
            let next = VlaRequest::generated_with_metadata(
                &observation,
                apxinf_core::RngKey::new(71, 0, 1),
                request.metadata,
            );
            assert_ne!(first, host(&runner, plan.run(&next).unwrap().tensor()));
            assert_ne!(first_future, host(&runner, &plan.work.borrow().video()));
            assert_eq!(values, host(&runner, plan.run(&request).unwrap().tensor()));
        }
        if name == "repeat" {
            assert_eq!(first_action.as_ref().unwrap(), &values);
            assert_eq!(first_video.as_ref().unwrap(), &video);
        }
        if name == "video-noise" || name == "action-noise" {
            assert_ne!(
                first_action.as_ref().unwrap(),
                &values,
                "joint noise did not affect actions"
            );
            assert_ne!(
                first_video.as_ref().unwrap(),
                &video,
                "joint noise did not affect video"
            );
        }
        eprintln!(
            "{}",
            serde_json::json!({"case":name,"prepare_seconds":prepared_seconds,"replay_seconds":replay_seconds,"next_frame":position,"graphs":3})
        );
        drop(plan);
        assert_eq!(
            host(&runner, action.tensor()),
            values,
            "returned action storage outlived plan"
        );
        if name == "initial" && selection == CheckpointCases::Full {
            // Finite, representable BF16 noise can still overflow the network.
            // A failed public output must not publish another episode's K/V.
            let extreme = Tensor::from_f32(vec![NOISE], &vec![1e38; NOISE]).unwrap();
            let failed = VlaRequest::provided_with_metadata(
                &observation,
                &extreme,
                VlaMetadata {
                    history: Some(HistoryContext {
                        session: "failed-caller",
                        episode: "failed-episode",
                    }),
                    ..request.metadata
                },
            );
            let previous = runner
                .session
                .borrow()
                .as_ref()
                .unwrap()
                .key
                .identity
                .clone();
            let error = runner.infer_host_f32(&failed).unwrap_err();
            assert!(error.to_string().contains("nonfinite actions"), "{error}");
            assert_eq!(
                runner.session.borrow().as_ref().unwrap().key.identity,
                previous
            );
            assert_eq!(
                runner.session.borrow().as_ref().unwrap().key.next_frame,
                position
            );
            runner.clear_prepared().unwrap();
            eprintln!("nonfinite public output preserves the committed episode");
            for policy in [ExecutionPolicy::Eager, ExecutionPolicy::PreferGraph] {
                // A separate runner's owner can be dropped while the explicit
                // plan retains its model, session and graph resources.
                let owner = DreamZeroModelRunner {
                    model: runner.model.clone(),
                    session: Rc::new(RefCell::new(None)),
                    implicit: RefCell::new(None),
                };
                let alternate = owner.prepare_for(&request, policy).unwrap();
                let expected_mode = if policy == ExecutionPolicy::Eager {
                    ExecutionMode::Eager
                } else {
                    ExecutionMode::Graph
                };
                assert!(
                    matches!(alternate.status(),PreparationStatus::Ready {mode,..} if mode==expected_mode)
                );
                drop(owner);
                let result = alternate.run(&request).unwrap();
                assert_eq!(values, host(&runner, result.tensor()));
                drop(alternate);
                assert_eq!(values, host(&runner, result.tensor()));
                eprintln!("full initial {policy:?} / dropped runner and retained output passed");
            }
        }
    }
    runner.reset_history().unwrap();
    assert!(runner.session.borrow().is_none());
    assert!(
        numerical_failures.is_empty(),
        "numerical gates failed: {numerical_failures:?}"
    );
}

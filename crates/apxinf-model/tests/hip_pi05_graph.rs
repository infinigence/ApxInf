//! π0.5 on HIP, graph-capture acceptance.
//!
//! Capture's quiet failure mode is a replay that reproduces the tensors it
//! recorded instead of the inputs it was just given: the graph holds
//! addresses, and an input that never reached those addresses looks exactly
//! like a correct run. So the acceptance here is bit-identity between a
//! `RequireGraph` plan and an eager plan under the captured inputs **and**
//! under different ones — nothing separates the two paths but the traversal
//! being recorded.
//!
//! ```sh
//! APXINF_PI05_CHECKPOINT=~/models/lerobot-pi05_libero_base \
//!   cargo test -p apxinf-model --features hip --test hip_pi05_graph -- --nocapture
//! ```
//!
//! Every variable must be set or the test skips, so the suite still runs on a
//! machine without the checkpoint. Use absolute paths: relative ones resolve
//! against `crates/apxinf-model`.

#![cfg(feature = "hip")]

use std::path::Path;
use std::sync::Arc;

use apxinf_core::Tensor;
use apxinf_hip::HipContext;
use apxinf_model::hip_pi05::{load_runner, Config};
use apxinf_model::vla::{
    Action, ExecutionMode, ExecutionPolicy, Observation, PreparationStatus, PreparedInference,
    VisionObservation, VlaRequest, VlaRuntime,
};

/// The checkpoint's `config.json` decides view count and horizon; these only
/// fix the parts the request chooses. Ten prompt tokens, like the bench.
const TOKEN_COUNT: usize = 10;

fn checkpoint() -> Option<String> {
    let path = std::env::var("APXINF_PI05_CHECKPOINT").ok()?;
    Path::new(&path).is_dir().then_some(path)
}

/// A patches observation and provided noise, each filled with one value.
fn observation(config: &Config, patch_value: f32, noise_value: f32) -> (Observation, Tensor) {
    let patches = vec![
        config.num_views * config.patches_per_view(),
        3 * config.patch_size * config.patch_size,
    ];
    let patch_count = patches[0] * patches[1];
    let noise_shape = vec![config.action_horizon, config.action_dim];
    let noise_count = noise_shape[0] * noise_shape[1];
    let to_bf16 = |value: f32| half::bf16::from_f32(value);
    (
        Observation {
            vision: VisionObservation::Patches(
                Tensor::from_bf16(patches, &vec![to_bf16(patch_value); patch_count]).unwrap(),
            ),
            token_ids: vec![1; TOKEN_COUNT],
            state: None,
            action_mask: None,
        },
        Tensor::from_bf16(noise_shape, &vec![to_bf16(noise_value); noise_count]).unwrap(),
    )
}

/// Action values as bit patterns: bit-identity is the standard, and `==` on
/// the floats would fold signed zeros and NaNs that distinguish paths. The
/// action lives on the device, so it is downloaded first — `to_f32_vec`
/// alone would fail on a resident GPU tensor.
fn action_bits(ctx: &HipContext, action: &Action) -> Vec<u32> {
    ctx.to_cpu(action.tensor())
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Run one plan on one input case, returning the action bits.
fn run_bits(
    plan: &dyn PreparedInference,
    config: &Config,
    ctx: &HipContext,
    patch_value: f32,
    noise_value: f32,
) -> Vec<u32> {
    let (case, noise) = observation(config, patch_value, noise_value);
    let request = VlaRequest::provided(&case, &noise);
    action_bits(ctx, &plan.run(&request).unwrap())
}

/// Eager and captured plans agree bit for bit, under the captured inputs and
/// under changed ones, and the replays are counted.
#[test]
fn captured_plan_matches_the_eager_plan_under_changed_inputs() {
    let Some(path) = checkpoint() else {
        eprintln!("skipping: APXINF_PI05_CHECKPOINT is not set");
        return;
    };
    let ctx = Arc::new(HipContext::new(0).unwrap());
    let runner = load_runner(Path::new(&path), Arc::clone(&ctx), &Default::default(), None).unwrap();
    let config = runner.model().config().clone();

    let (first, _) = observation(&config, 0.0, 0.0);
    let spec = first.inference_spec();
    let cases = [
        ("the captured inputs", 0.0, 0.0),
        ("changed inputs", 0.375, -0.5),
        ("back to the first", 0.0, 0.0),
    ];

    let eager = runner.prepare_plan(&spec, ExecutionPolicy::Eager).unwrap();
    let eager_bits: Vec<_> = cases
        .iter()
        .map(|(_, patch, noise)| run_bits(&eager, &config, &ctx, *patch, *noise))
        .collect();
    assert_eq!(
        run_bits(&eager, &config, &ctx, cases[0].1, cases[0].2),
        eager_bits[0],
        "the eager plan is not deterministic across its own runs"
    );

    // RequireGraph, not PreferGraph: a capture that fell back would make this
    // eager-versus-eager and pass without testing capture at all.
    let captured = runner.prepare_plan(&spec, ExecutionPolicy::RequireGraph).unwrap();
    assert_eq!(
        captured.status(),
        PreparationStatus::Ready {
            mode: ExecutionMode::Graph,
            fallback_reason: None
        },
        "the plan must capture; a fallback would compare eager with eager"
    );
    let (replays_before, nodes) = captured.graph_usage();
    assert!(nodes > 0, "a π0.5 capture with no kernels is not this model");

    for (index, (label, patch, noise)) in cases.iter().enumerate() {
        let actual = run_bits(&captured, &config, &ctx, *patch, *noise);
        assert_eq!(actual, eager_bits[index], "eager and captured disagree on {label}");
    }
    let (replays, _) = captured.graph_usage();
    assert_eq!(
        replays - replays_before,
        cases.len(),
        "the runs above must have replayed, not run eagerly"
    );

    // The implicit `infer` path asks for a graph too, and reports it.
    let (case, noise) = observation(&config, 0.25, -0.125);
    let request = VlaRequest::provided(&case, &noise);
    let action = runner.infer(&request).unwrap();
    assert_eq!(runner.execution_mode(), "graph");
    assert_eq!(
        action_bits(&ctx, &action),
        run_bits(&eager, &config, &ctx, 0.25, -0.125),
        "the cached plan's replay and an eager forward disagree"
    );
}

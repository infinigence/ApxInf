//! End-to-end acceptance for GR00T N1.7 on HIP, through [`VlaRuntime`].
//!
//! Runs the frozen LIBERO fixture through the registered runner and compares
//! the action chunk against the NVIDIA GPU reference. This is the gate from
//! `doc/gr00t-n1.7.md`: BF16 cosine at least `0.999`, relative L2 at most
//! `0.05`.
//!
//! Unlike the `hip_gr00t_stage_dump` example, this goes through the public
//! runtime, so it also covers request validation, the plan cache and the
//! preparation contract -- the parts a stage dump bypasses.
//!
//! ```text
//! APXINF_GR00T_CHECKPOINT=/models/GR00T-N1.7-LIBERO/libero_10 \
//!   APXINF_GR00T_FIXTURE=devlocal/gr00t-hip-port/fixtures/libero-task0-2view \
//!   APXINF_GR00T_REFERENCE=devlocal/gr00t-hip-port/fixtures/nvidia-reference/final_action.npy \
//!   cargo test -p apxinf-model --features hip --test hip_gr00t_acceptance -- --nocapture
//! ```
//!
//! The reference is one F32 `.npy` extracted from the reference `.npz`; see
//! that directory's `PROVENANCE.md`. Every variable must be set or the test
//! skips, so the suite still runs on a machine without the checkpoint.

#![cfg(feature = "hip")]

use std::path::PathBuf;
use std::sync::Arc;

use apxinf_core::Tensor;
use apxinf_hip::HipContext;
use apxinf_model::hip_gr00t::load_runner;
use apxinf_model::vla::PreparedInference as _;
use apxinf_model::vla::{
    ExecutionPolicy, InferenceSpec, Observation, VisionObservation, VlaMetadata, VlaRequest,
    VlaRuntime,
};
use apxinf_model::LoadOptions;

/// `doc/gr00t-n1.7.md`: BF16 release gate.
const MIN_COSINE: f64 = 0.999;
const MAX_RELATIVE_L2: f64 = 0.05;

struct Fixture {
    pixel_values: Tensor,
    grid: Vec<[u32; 3]>,
    token_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    state: Tensor,
    noise: Tensor,
    embodiment_id: usize,
}

/// `None` when the variable is unset, so the suite skips cleanly on a machine
/// without the checkpoint. A variable that *is* set but names a missing path
/// panics instead: silently skipping there turns a typo -- or a relative path
/// resolved against the crate directory rather than the workspace root -- into
/// a green run that tested nothing.
fn env_dir(name: &str) -> Option<PathBuf> {
    let value = std::env::var_os(name)?;
    let path = PathBuf::from(&value);
    assert!(
        path.exists(),
        "{name} is set to {} which does not exist (relative paths resolve against \
         crates/apxinf-model, not the workspace root)",
        path.display()
    );
    Some(path)
}

fn read_bf16(root: &PathBuf, name: &str, dims: Vec<usize>) -> Tensor {
    let bytes = std::fs::read(root.join(name)).expect("fixture tensor");
    assert_eq!(
        bytes.len(),
        dims.iter().product::<usize>() * 2,
        "{name} size does not match {dims:?}"
    );
    let values: Vec<half::bf16> = bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Tensor::from_bf16(dims, &values).expect("fixture tensor")
}

fn load_fixture(root: &PathBuf) -> Fixture {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).expect("manifest"))
            .expect("manifest json");
    assert_eq!(
        manifest["schema"].as_str(),
        Some("apxinf.gr00t-n1.7.preprocessed-fixture.v1"),
        "unexpected fixture schema"
    );
    let grid_bytes = std::fs::read(root.join("image_grid_thw.bin")).expect("grid");
    let grid: Vec<[u32; 3]> = grid_bytes
        .chunks_exact(12)
        .map(|c| {
            let v = |i: usize| u32::from_le_bytes([c[i], c[i + 1], c[i + 2], c[i + 3]]);
            [v(0), v(4), v(8)]
        })
        .collect();
    let patches: usize = grid
        .iter()
        .map(|&[t, h, w]| t as usize * h as usize * w as usize)
        .sum();
    let token_bytes = std::fs::read(root.join("token_ids.bin")).expect("tokens");
    Fixture {
        pixel_values: read_bf16(root, "pixel_values.bin", vec![patches, 1536]),
        grid,
        token_ids: token_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        attention_mask: std::fs::read(root.join("attention_mask.bin")).expect("mask"),
        state: read_bf16(root, "state.bin", vec![1, 132]),
        noise: read_bf16(root, "noise.bin", vec![40, 132]),
        embodiment_id: manifest["embodiment_id"].as_u64().expect("embodiment_id") as usize,
    }
}

/// Minimal F32 `.npy` reader: enough for a C-order 2-D `<f4` array.
fn read_npy_f32(path: &PathBuf) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("reference npy");
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a .npy file");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("npy header");
    assert!(header.contains("'<f4'"), "reference must be little-endian f32: {header}");
    assert!(
        header.contains("'fortran_order': False"),
        "reference must be C-order: {header}"
    );
    let shape_text = header
        .split("'shape': (")
        .nth(1)
        .and_then(|rest| rest.split(')').next())
        .expect("npy shape");
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    let values: Vec<f32> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(values.len(), shape.iter().product::<usize>(), "npy size");
    (shape, values)
}

fn cosine_and_relative_l2(got: &[f32], want: &[f32]) -> (f64, f64) {
    let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let norm = |values: &[f32]| values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let difference = got
        .iter()
        .zip(want)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let want_norm = norm(want);
    (dot / (norm(got) * want_norm), difference / want_norm)
}

#[test]
fn the_libero_fixture_matches_the_nvidia_reference_within_the_release_gate() {
    let (Some(checkpoint), Some(fixture_dir), Some(reference)) = (
        env_dir("APXINF_GR00T_CHECKPOINT"),
        env_dir("APXINF_GR00T_FIXTURE"),
        env_dir("APXINF_GR00T_REFERENCE"),
    ) else {
        eprintln!(
            "set APXINF_GR00T_CHECKPOINT, APXINF_GR00T_FIXTURE and APXINF_GR00T_REFERENCE; skipping"
        );
        return;
    };

    let fixture = load_fixture(&fixture_dir);
    let ctx = Arc::new(HipContext::new(0).expect("HIP device 0"));
    let runner = load_runner(&checkpoint, ctx, &LoadOptions::default()).expect("load GR00T");

    let contract = runner.contract();
    assert_eq!(contract.action_shape, [40, 132]);
    assert!(!contract.accepts_rgb_u8, "GR00T consumes processor patches");

    let observation = Observation {
        vision: VisionObservation::Patches(fixture.pixel_values),
        token_ids: fixture.token_ids,
        state: Some(fixture.state),
        action_mask: None,
    };
    let metadata = VlaMetadata {
        attention_mask: Some(&fixture.attention_mask),
        image_grid_thw: Some(&fixture.grid),
        embodiment_id: Some(fixture.embodiment_id),
        planning: None,
    };
    let request = VlaRequest::provided_with_metadata(&observation, &fixture.noise, metadata);

    let actions = runner.infer_host_f32(&request).expect("inference");
    let (shape, want) = read_npy_f32(&reference);
    assert_eq!(
        shape.iter().product::<usize>(),
        actions.len(),
        "reference shape {shape:?} does not match the action chunk"
    );
    assert!(
        actions.iter().all(|v| v.is_finite()),
        "the action chunk contains non-finite values"
    );

    let (cosine, relative_l2) = cosine_and_relative_l2(&actions, &want);
    println!("final_action cosine {cosine:.7} relative L2 {relative_l2:.5}");
    assert!(
        cosine >= MIN_COSINE,
        "cosine {cosine:.7} is below the BF16 gate {MIN_COSINE}"
    );
    assert!(
        relative_l2 <= MAX_RELATIVE_L2,
        "relative L2 {relative_l2:.5} exceeds the BF16 gate {MAX_RELATIVE_L2}"
    );

    // The worst single action step, not just the aggregate: one bad step in a
    // chunk of 40 is a real failure the full-tensor cosine can hide.
    let dim = shape[shape.len() - 1];
    let worst = (0..actions.len() / dim)
        .map(|row| {
            let span = row * dim..(row + 1) * dim;
            cosine_and_relative_l2(&actions[span.clone()], &want[span]).0
        })
        .fold(f64::INFINITY, f64::min);
    println!("worst per-step cosine {worst:.7}");
    assert!(
        worst >= MIN_COSINE,
        "the worst action step's cosine {worst:.7} is below the gate {MIN_COSINE}"
    );

    // A second run of the same request must reuse the cached plan and produce
    // the identical chunk: the plan holds the arena and the uploaded inputs,
    // so a stale buffer would show up here.
    let again = runner.infer_host_f32(&request).expect("second inference");
    assert_eq!(actions, again, "a repeated request changed the action chunk");
    assert_eq!(runner.execution_mode(), "graph");
}

#[test]
fn preparation_refuses_a_spec_that_cannot_bind_grooots_shapes() {
    let (Some(checkpoint), Some(fixture_dir)) = (
        env_dir("APXINF_GR00T_CHECKPOINT"),
        env_dir("APXINF_GR00T_FIXTURE"),
    ) else {
        eprintln!("set APXINF_GR00T_CHECKPOINT and APXINF_GR00T_FIXTURE; skipping");
        return;
    };
    let fixture = load_fixture(&fixture_dir);
    let ctx = Arc::new(HipContext::new(0).expect("HIP device 0"));
    let runner = load_runner(&checkpoint, ctx, &LoadOptions::default()).expect("load GR00T");

    // `InferenceSpec` carries no grid and no embodiment, so a plan built from
    // it would bind neither. The contract says such a `prepare` must fail
    // rather than return a plan that proves nothing.
    let spec = InferenceSpec {
        token_count: fixture.token_ids.len(),
        image_layout: None,
    };
    let error = match runner.prepare(&spec) {
        Ok(_) => panic!("prepare accepted a spec that cannot bind GR00T's shapes"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("image_grid_thw"), "{error}");
    assert!(error.contains("embodiment_id"), "{error}");
    assert!(error.contains("prepare_for"), "{error}");

    // `prepare_for` sees a whole request and does bind everything.
    let observation = Observation {
        vision: VisionObservation::Patches(fixture.pixel_values),
        token_ids: fixture.token_ids,
        state: Some(fixture.state),
        action_mask: None,
    };
    let metadata = VlaMetadata {
        attention_mask: Some(&fixture.attention_mask),
        image_grid_thw: Some(&fixture.grid),
        embodiment_id: Some(fixture.embodiment_id),
        planning: None,
    };
    let request = VlaRequest::provided_with_metadata(&observation, &fixture.noise, metadata);
    let prepared = runner
        .prepare_plan(&request, ExecutionPolicy::PreferGraph)
        .expect("prepare_plan");
    prepared.run(&request).expect("prepared run");

    // The arena bound is hand-audited against the forward path, so it can
    // drift as that path changes. An arena one allocation short fails the run
    // outright, so hold the bound accountable to the measured peak here --
    // and flag a bound so loose it is no longer an audit.
    let (peak, capacity) = prepared.arena_usage();
    println!(
        "arena peak {:.0} MiB of {:.0} MiB bound",
        peak as f64 / (1 << 20) as f64,
        capacity as f64 / (1 << 20) as f64
    );
    assert!(
        peak <= capacity,
        "arena peak {peak} exceeded its {capacity}-byte bound"
    );
    assert!(
        capacity <= peak * 2,
        "arena bound {capacity} is more than twice the {peak}-byte peak; the audit has drifted"
    );

    // `prepare_for` must accept the same request through the portable trait.
    let boxed = runner
        .prepare_for(&request, ExecutionPolicy::PreferGraph)
        .expect("prepare_for");
    assert_eq!(boxed.spec().token_count, observation.token_ids.len());

    // Preparation honors the policy: RequireGraph prepares a plan that
    // reports Graph mode (the warmup and capture happened inside `prepare`,
    // not on the first `run`), and Eager prepares one that reports Eager.
    let graphed = runner
        .prepare_plan(&request, ExecutionPolicy::RequireGraph)
        .expect("RequireGraph prepares and captures");
    assert_eq!(
        graphed.status(),
        apxinf_model::vla::PreparationStatus::Ready {
            mode: apxinf_model::vla::ExecutionMode::Graph,
            fallback_reason: None,
        }
    );
    let eager = runner
        .prepare_plan(&request, ExecutionPolicy::Eager)
        .expect("Eager prepares");
    assert_eq!(
        eager.status(),
        apxinf_model::vla::PreparationStatus::Ready {
            mode: apxinf_model::vla::ExecutionMode::Eager,
            fallback_reason: None,
        }
    );
    // An eager plan never captured, so it has served zero replays.
    let (replays, nodes) = eager.graph_usage();
    assert_eq!((replays, nodes), (0, 0));
}

#[test]
fn a_different_prompt_changes_the_action_on_a_shared_plan() {
    // The processor pads every prompt to the same length, so two different
    // instructions produce the same token count, the same placeholder layout
    // and therefore the same plan key -- the plan is reused. The token IDs
    // are the only per-request value in that plan, so if they are not
    // refilled the model keeps answering the first prompt: no error, no
    // degenerate output, just a robot that performs the previous task.
    //
    // The earlier version of this check only asserted that `A, B, A` returns
    // the same answer for both `A`s. That passes even when the prompt is
    // ignored entirely. Asserting that `A` and `B` *differ* is the half that
    // catches it.
    let (Some(checkpoint), Some(fixture_dir)) = (
        env_dir("APXINF_GR00T_CHECKPOINT"),
        env_dir("APXINF_GR00T_FIXTURE"),
    ) else {
        eprintln!("set APXINF_GR00T_CHECKPOINT and APXINF_GR00T_FIXTURE; skipping");
        return;
    };
    let fixture = load_fixture(&fixture_dir);
    let ctx = Arc::new(HipContext::new(0).expect("HIP device 0"));
    let runner = load_runner(&checkpoint, ctx, &LoadOptions::default()).expect("load GR00T");

    // The fixture's tokens, and the same tokens with the instruction span
    // altered. Only non-image, non-padding positions are touched, so the
    // plan key -- count, grid, placeholder layout -- is unchanged.
    let image_token = 151655u32;
    let mut other_tokens = fixture.token_ids.clone();
    let mut altered = 0usize;
    for token in other_tokens.iter_mut() {
        if *token != image_token && *token > 1000 && altered < 4 {
            *token += 1;
            altered += 1;
        }
    }
    assert_eq!(altered, 4, "the fixture had too few instruction tokens to alter");
    assert_ne!(other_tokens, fixture.token_ids);

    let run = |token_ids: Vec<u32>| {
        let observation = Observation {
            vision: VisionObservation::Patches(fixture.pixel_values.clone()),
            token_ids,
            state: Some(fixture.state.clone()),
            action_mask: None,
        };
        let metadata = VlaMetadata {
            attention_mask: Some(&fixture.attention_mask),
            image_grid_thw: Some(&fixture.grid),
            embodiment_id: Some(fixture.embodiment_id),
            planning: None,
        };
        let request =
            VlaRequest::provided_with_metadata(&observation, &fixture.noise, metadata);
        runner.infer_host_f32(&request).expect("inference")
    };

    let first = run(fixture.token_ids.clone());
    let other = run(other_tokens.clone());
    let first_again = run(fixture.token_ids.clone());

    assert_eq!(
        first, first_again,
        "the same prompt gave a different action after another prompt ran; \
         the plan is not restoring state"
    );
    assert_ne!(
        first, other,
        "two different prompts gave the identical action chunk, so the token \
         IDs are not reaching the model -- the plan's token buffer is stale"
    );
    assert_eq!(runner.execution_mode(), "graph");
}

#[test]
fn a_request_missing_grooots_metadata_is_refused() {
    let Some(fixture_dir) = env_dir("APXINF_GR00T_FIXTURE") else {
        eprintln!("set APXINF_GR00T_FIXTURE; skipping");
        return;
    };
    let Some(checkpoint) = env_dir("APXINF_GR00T_CHECKPOINT") else {
        eprintln!("set APXINF_GR00T_CHECKPOINT; skipping");
        return;
    };
    let fixture = load_fixture(&fixture_dir);
    let ctx = Arc::new(HipContext::new(0).expect("HIP device 0"));
    let runner = load_runner(&checkpoint, ctx, &LoadOptions::default()).expect("load GR00T");

    let observation = Observation {
        vision: VisionObservation::Patches(fixture.pixel_values),
        token_ids: fixture.token_ids,
        state: Some(fixture.state),
        action_mask: None,
    };
    // Each of these would otherwise be guessed, and a guessed embodiment
    // silently selects the wrong adapters.
    for (label, metadata) in [
        (
            "image_grid_thw",
            VlaMetadata {
                attention_mask: Some(&fixture.attention_mask),
                image_grid_thw: None,
                embodiment_id: Some(fixture.embodiment_id),
                planning: None,
            },
        ),
        (
            "attention_mask",
            VlaMetadata {
                attention_mask: None,
                image_grid_thw: Some(&fixture.grid),
                embodiment_id: Some(fixture.embodiment_id),
                planning: None,
            },
        ),
        (
            "embodiment_id",
            VlaMetadata {
                attention_mask: Some(&fixture.attention_mask),
                image_grid_thw: Some(&fixture.grid),
                embodiment_id: None,
                planning: None,
            },
        ),
    ] {
        let request = VlaRequest::provided_with_metadata(&observation, &fixture.noise, metadata);
        let error = runner.infer(&request).unwrap_err().to_string();
        assert!(error.contains(label), "expected {label} in: {error}");
    }
}


/// The GR00T forward must be capturable into a HIP graph.
///
/// Not a performance test -- capture is worth about -0.03 +/- 0.11 ms here,
/// which is nothing. It is an executable form of the fixed-profile contract in
/// `doc/model-execution-wiring.md#accelerator-port-acceptance`: device-resident from input upload to output transfer, with
/// host round-trips counted as unfinished work rather than debt.
///
/// Capturing checks exactly that, and checks it rather than trusting a
/// comment. Device allocation is illegal while a capture is open; a captured
/// copy from host memory replays from a pointer that no longer exists; a
/// synchronize inside a capture is an error. A forward that captures does none
/// of those. `doc/model-lifecycle.md` makes the converse point: that eager
/// execution works says nothing about whether capture will.
#[test]
fn the_forward_is_device_resident_enough_to_capture() {
    let (Some(checkpoint), Some(fixture)) = (
        env_dir("APXINF_GR00T_CHECKPOINT"),
        env_dir("APXINF_GR00T_FIXTURE"),
    ) else {
        eprintln!("set APXINF_GR00T_CHECKPOINT and APXINF_GR00T_FIXTURE; skipping");
        return;
    };

    let loaded = load_fixture(&fixture);
    let ctx = Arc::new(HipContext::new(0).expect("HIP device 0"));
    let runner = load_runner(&checkpoint, ctx, &LoadOptions::default()).expect("load GR00T");

    let observation = Observation {
        vision: VisionObservation::Patches(loaded.pixel_values),
        token_ids: loaded.token_ids,
        state: Some(loaded.state),
        action_mask: None,
    };
    let metadata = VlaMetadata {
        attention_mask: Some(&loaded.attention_mask),
        image_grid_thw: Some(&loaded.grid),
        embodiment_id: Some(loaded.embodiment_id),
        planning: None,
    };
    let request = VlaRequest::provided_with_metadata(&observation, &loaded.noise, metadata);

    // The implicit `infer` path prepares under PreferGraph, so the first
    // inference is the warmup inside preparation and every later one replays
    // the captured graph.
    let first = runner.infer_host_f32(&request).expect("first inference");
    let second = runner.infer_host_f32(&request).expect("second inference");
    let third = runner.infer_host_f32(&request).expect("third inference");

    let prepared = runner.prepared_for_report().expect("a plan was cached");
    let (replays, nodes) = prepared.graph_usage();
    assert!(
        nodes > 0,
        "the forward did not capture. Something in it allocates on the device, copies \
         from host memory rebuilt per call, or synchronizes -- all of which a fixed \
         profile must not do. The eager path still returns the right answer, which is \
         why this needs its own assertion."
    );
    assert!(
        replays > 0,
        "the graph captured {nodes} nodes but served none of the three inferences"
    );
    // A replay that drifts is worse than no replay: it returns a plausible
    // wrong answer from addresses recorded earlier.
    assert_eq!(second, first, "the captured run differs from the eager one");
    assert_eq!(third, second, "a later replay differs from the first");
}

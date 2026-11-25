//! Maintained behaviour tests of the runner.
//!
//! `synthetic_prompts_have_the_structure_the_checkpoint_test_relies_on` runs everywhere. The checkpoint test
//! is `#[ignore]`d: it needs a CUDA device and the public `allenai/MolmoAct2-LIBERO` checkpoint,
//! `APXINF_MOLMOACT2_TEST_CHECKPOINT=<checkpoint directory> cargo test -p apxinf-model --features cuda
//! --release --lib cuda_molmoact2::model_runner::tests -- --ignored --nocapture`.
//! Its inputs are built from the checkpoint config alone (a pseudo-random image pair, a prompt of
//! the right image-token layout, a pseudo-random noise tensor), so no exported reference data is needed. It is
//! deliberately one sequential test: it loads the 21 GB of weights once and drops every plan before the next
//! one is built, so only one prepared profile is alive at a time.
//!
//! What it fixes: the three execution paths (raw eager, prepared eager, prepared graph) return bit-identical
//! actions; changed images / prompt / noise take effect and restoring them restores the result; returned
//! actions survive later requests and plan teardown; a different token structure replaces the cached plan;
//! rejected requests leave the plan usable; a failed capture is an error for `RequireGraph` and a recorded
//! fallback for `PreferGraph` and does not touch `Eager`. Numerical agreement with the official model is
//! not tested here (see `doc/molmoact2.md`).
use super::prepare::inject_capture_failure;
use super::runner::ModelRunner;
use crate::cuda_molmoact2::backend::ops::GemmPolicy;
use crate::cuda_molmoact2::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_molmoact2::config::tests::libero_json;
use crate::cuda_molmoact2::config::MolmoAct2Config;
use crate::cuda_molmoact2::geometry::{encoder_attention_mask, plan_prefill_segments, two_view_layout, PoolingMaps, NUM_VIEWS};
use crate::cuda_molmoact2::weights::Bf16Weights;
use crate::vla::*;
use apxinf_core::{DType, Error, Result, RngKey, Tensor};
use std::{path::PathBuf, rc::Rc};

const HORIZON: usize = 10;
const WIDTH: usize = 32;
const PHYSICAL_DIMS: usize = 7;

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}

/// Two NHWC RGB views of pseudo-random bytes.
fn image(cfg: &MolmoAct2Config, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..NUM_VIEWS * cfg.vit.image_size * cfg.vit.image_size * 3).map(|_| lcg(&mut state) as u8).collect()
}

/// Full noise `[1, horizon, width]` in `[-2, 2)`.
fn noise(seed: u64) -> Tensor {
    let mut state = seed;
    let values: Vec<f32> = (0..HORIZON * WIDTH).map(|_| (lcg(&mut state) % 4096) as f32 / 1024.0 - 2.0).collect();
    Tensor::from_f32(vec![1, HORIZON, WIDTH], &values).unwrap()
}

/// `[bos] (start, patches, end, text) x views, text...` with `text_len` plain tokens; when `action_span` is set
/// the last three text tokens are an action span instead (same length, three fewer visible context rows).
fn prompt(cfg: &MolmoAct2Config, text_len: usize, salt: u32, action_span: bool) -> Vec<u32> {
    let tokens = &cfg.image_tokens;
    let patches = PoolingMaps::new(NUM_VIEWS, cfg.patch_side(), cfg.pooling).rows / NUM_VIEWS;
    let mut ids = vec![cfg.eos_token_id];
    for _ in 0..NUM_VIEWS {
        ids.push(tokens.start);
        ids.extend(std::iter::repeat(tokens.patch).take(patches));
        ids.push(tokens.end);
        ids.push(1000); // text between the two image runs keeps them two runs
    }
    ids.extend((0..text_len as u32).map(|i| 2000 + (salt * 131 + i * 7) % 4000));
    if action_span {
        let n = ids.len();
        ids[n - 3] = cfg.action_start_token_id;
        ids[n - 1] = cfg.action_end_token_id;
    }
    ids
}

#[test]
fn synthetic_prompts_have_the_structure_the_checkpoint_test_relies_on() {
    let cfg = MolmoAct2Config::from_json(&libero_json()).unwrap();
    // (token layout, prefill segments, visible context rows)
    let analyse = |ids: &[u32]| {
        let valid = vec![true; ids.len()];
        let (layout, _) = two_view_layout(&cfg, ids, &valid).unwrap();
        let segments = plan_prefill_segments(&layout.token_type, &valid).unwrap();
        let rows = encoder_attention_mask(&cfg, ids, &valid).iter().filter(|&&visible| visible).count();
        (layout, segments, rows)
    };
    let a = analyse(&prompt(&cfg, 24, 0, false));
    assert_eq!(a.0.image_runs.len(), NUM_VIEWS);
    let same_structure = analyse(&prompt(&cfg, 24, 1, false));
    assert_eq!((&a.0, &a.1, a.2), (&same_structure.0, &same_structure.1, same_structure.2));
    let longer = analyse(&prompt(&cfg, 30, 0, false));
    assert_ne!(a.0.token_type.len(), longer.0.token_type.len());
    let span = analyse(&prompt(&cfg, 24, 0, true));
    assert_eq!((&a.0, &a.1), (&span.0, &span.1), "only the visible rows differ");
    assert_eq!(span.2 + 3, a.2);
}

struct Input {
    observation: Observation,
    noise: Tensor,
}

impl Input {
    fn new(cfg: &MolmoAct2Config, image_seed: u64, token_ids: &[u32], noise_seed: u64) -> Self {
        let observation = Observation {
            vision: VisionObservation::RgbU8 { bytes: image(cfg, image_seed), layout: ImageLayout::Nhwc },
            token_ids: token_ids.to_vec(),
            state: None,
            action_mask: None,
        };
        Self { observation, noise: noise(noise_seed) }
    }

    fn request(&self) -> VlaRequest<'_> {
        VlaRequest::provided(&self.observation, &self.noise)
    }
}

fn bytes(action: &Action) -> Vec<u8> {
    let mut out = vec![0u8; HORIZON * WIDTH * 4];
    CudaBuffer::from_tensor(action.tensor()).unwrap().copy_to_host(&mut out).unwrap();
    let values: Vec<f32> = out.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
    assert!(values.iter().all(|v| v.is_finite()), "non-finite action");
    assert!(values.chunks(WIDTH).all(|row| row[PHYSICAL_DIMS..].iter().all(|v| *v == 0.0)), "padded action dimensions must be zero");
    out
}

fn mode(plan: &dyn PreparedInference) -> (ExecutionMode, Option<String>) {
    match plan.status() {
        PreparationStatus::Ready { mode, fallback_reason } => (mode, fallback_reason),
        other => panic!("plan is not ready: {other:?}"),
    }
}

/// Resets the injected capture failure even when an assertion fails.
struct CaptureFailure;

impl CaptureFailure {
    fn inject() -> Self {
        inject_capture_failure(true);
        Self
    }
}

impl Drop for CaptureFailure {
    fn drop(&mut self) {
        inject_capture_failure(false);
    }
}

#[test]
#[ignore = "requires a CUDA device and APXINF_MOLMOACT2_TEST_CHECKPOINT"]
fn execution_paths_agree_and_follow_the_runner_contract() -> Result<()> {
    let dir = PathBuf::from(std::env::var("APXINF_MOLMOACT2_TEST_CHECKPOINT").expect("APXINF_MOLMOACT2_TEST_CHECKPOINT"));
    let cfg = MolmoAct2Config::from_file(&dir.join("config.json"))?;
    let ctx = Rc::new(CudaContext::new_with_autotune(0, false).map_err(Error::Cuda)?);
    let weights = Rc::new(Bf16Weights::load(&dir, &cfg, 0)?);
    let gemm = GemmPolicy { online_tune: false, ..Default::default() };
    let attention = AttentionPolicy { online_tune: false, ..Default::default() };
    // The `Eager` policy: `infer` is prepared eager. Graph plans are built explicitly with `prepare_for`.
    let runner = ModelRunner::with_policy(ctx, weights, cfg.clone(), gemm, attention, ExecutionPolicy::Eager)?;

    let ids_a = prompt(&cfg, 24, 0, false);
    let a = Input::new(&cfg, 1, &ids_a, 1);
    let ids_c = prompt(&cfg, 30, 0, false); // another token structure
    let c = Input::new(&cfg, 1, &ids_c, 1);
    let ids_e = prompt(&cfg, 24, 0, true); // same length, fewer visible context rows
    let e = Input::new(&cfg, 1, &ids_e, 1);

    // The reference path: raw eager, nothing prepared and nothing shared between calls.
    let raw_a = bytes(&runner.run(&a.request())?);

    // ---- public `infer` with the Eager policy is the prepared eager path
    assert_eq!(runner.execution_mode(), "eager");
    assert_eq!(bytes(&runner.infer(&a.request())?), raw_a, "prepared eager equals raw eager");
    let plan = runner.implicit_plan().expect("infer prepares a plan");
    assert!(!plan.is_graph());
    assert_eq!(mode(&*plan), (ExecutionMode::Eager, None));
    assert_eq!(bytes(&runner.infer(&a.request())?), raw_a);
    assert!(Rc::ptr_eq(&plan, &runner.implicit_plan().unwrap()), "the same structure reuses the plan");

    // ---- changed content takes effect; the original request restores the result
    let other_image = Input::new(&cfg, 2, &ids_a, 1);
    let other_prompt = Input::new(&cfg, 1, &prompt(&cfg, 24, 1, false), 1);
    let other_noise = Input::new(&cfg, 1, &ids_a, 2);
    for (label, input) in [("image", &other_image), ("prompt", &other_prompt), ("noise", &other_noise)] {
        assert_ne!(bytes(&runner.infer(&input.request())?), raw_a, "changing the {label} must change the actions");
        assert!(Rc::ptr_eq(&plan, &runner.implicit_plan().unwrap()));
        assert_eq!(bytes(&runner.infer(&a.request())?), raw_a, "restoring the {label} restores the result");
    }
    let all_changed = Input::new(&cfg, 2, &prompt(&cfg, 24, 1, false), 2);
    let raw_changed = bytes(&runner.run(&all_changed.request())?);
    assert_eq!(bytes(&runner.infer(&all_changed.request())?), raw_changed, "in-place rebinding equals a fresh raw run");
    drop(plan); // no handle to a plan may outlive its eviction below

    // ---- an earlier result is independent of later requests and of plan teardown
    let held = runner.infer(&a.request())?;
    assert_eq!(bytes(&held), raw_a);
    runner.infer(&all_changed.request())?;
    assert_eq!(bytes(&held), raw_a, "a later request must not overwrite an earlier result");
    runner.clear_prepared()?;
    assert!(runner.implicit_plan().is_none());
    assert_eq!(bytes(&held), raw_a, "a result outlives its plan");

    // ---- rejected requests leave the plan usable
    assert_eq!(bytes(&runner.infer(&a.request())?), raw_a);
    let plan = runner.implicit_plan().unwrap();
    let mut with_state = Input::new(&cfg, 1, &ids_a, 1);
    with_state.observation.state = Some(Tensor::zeros(vec![8], DType::F32));
    assert!(runner.infer(&with_state.request()).is_err(), "a continuous state is rejected");
    let wrong_noise = Tensor::zeros(vec![HORIZON, WIDTH], DType::F32);
    assert!(runner.infer(&VlaRequest::provided(&a.observation, &wrong_noise)).is_err(), "noise without the batch axis is rejected");
    assert!(Rc::ptr_eq(&plan, &runner.implicit_plan().unwrap()), "a rejection keeps the cached plan");
    assert_eq!(bytes(&runner.infer(&a.request())?), raw_a);
    let plan_a = Rc::downgrade(&plan);
    drop(plan);

    // ---- device-generated noise: deterministic per key, different between keys
    let generated = |seed: u64| -> Result<Vec<u8>> { Ok(bytes(&runner.infer(&VlaRequest::generated(&a.observation, RngKey::new(seed, 0, 0)))?)) };
    let (g1, g1_again, g2) = (generated(1)?, generated(1)?, generated(2)?);
    assert_eq!(g1, g1_again);
    assert_ne!(g1, g2);

    // ---- a different structure replaces (and releases) the plan, a fewer-rows one too; going back works
    runner.infer(&c.request())?;
    assert!(plan_a.upgrade().is_none(), "a longer prompt prepares another plan and releases the old one");
    let plan_c = Rc::downgrade(&runner.implicit_plan().unwrap());
    runner.infer(&e.request())?;
    assert!(plan_c.upgrade().is_none(), "fewer visible context rows prepare another plan");
    assert_eq!(bytes(&runner.infer(&a.request())?), raw_a);
    runner.clear_prepared()?;

    // ---- explicit plans, one at a time
    let eager = runner.prepare_for(&a.request(), ExecutionPolicy::Eager)?;
    assert_eq!(mode(&*eager), (ExecutionMode::Eager, None));
    assert_eq!(bytes(&eager.run(&a.request())?), raw_a);
    assert!(eager.run(&c.request()).is_err(), "another token count does not fit the plan");
    assert!(eager.run(&e.request()).is_err(), "other visible context rows do not fit the plan");
    assert_eq!(bytes(&eager.run(&a.request())?), raw_a, "a rejected request leaves the plan usable");
    drop(eager);

    let graph = runner.prepare_for(&a.request(), ExecutionPolicy::RequireGraph)?;
    assert_eq!(mode(&*graph), (ExecutionMode::Graph, None));
    assert_eq!(bytes(&graph.run(&a.request())?), raw_a, "graph equals raw eager");
    assert_eq!(bytes(&graph.run(&a.request())?), raw_a, "replays repeat");
    assert_eq!(bytes(&graph.run(&all_changed.request())?), raw_changed, "changed inputs reach the graph");
    assert!(graph.run(&e.request()).is_err());
    assert_eq!(bytes(&graph.run(&a.request())?), raw_a);
    drop(graph);

    // a second token structure: graph equals prepared eager
    let (eager_c, graph_c) = {
        let eager = runner.prepare_for(&c.request(), ExecutionPolicy::Eager)?;
        let eager_c = bytes(&eager.run(&c.request())?);
        drop(eager);
        let graph = runner.prepare_for(&c.request(), ExecutionPolicy::RequireGraph)?;
        (eager_c, bytes(&graph.run(&c.request())?))
    };
    assert_eq!(eager_c, graph_c);
    assert_ne!(eager_c, raw_a);

    // ---- capture failure: an error for RequireGraph, a recorded fallback for PreferGraph, nothing for Eager
    {
        let _failing = CaptureFailure::inject();
        assert!(runner.prepare_for(&a.request(), ExecutionPolicy::RequireGraph).is_err());
        let fallback = runner.prepare_for(&a.request(), ExecutionPolicy::PreferGraph)?;
        let (fallback_mode, reason) = mode(&*fallback);
        assert_eq!(fallback_mode, ExecutionMode::Eager);
        assert!(reason.is_some_and(|r| r.contains("injected capture failure")));
        assert_eq!(bytes(&fallback.run(&a.request())?), raw_a, "the fallback plan computes the same actions");
        drop(fallback);
        let eager = runner.prepare_for(&a.request(), ExecutionPolicy::Eager)?;
        assert_eq!(mode(&*eager), (ExecutionMode::Eager, None), "Eager never captures");
    }
    let recovered = runner.prepare_for(&a.request(), ExecutionPolicy::RequireGraph)?;
    assert_eq!(mode(&*recovered), (ExecutionMode::Graph, None));
    assert_eq!(bytes(&recovered.run(&a.request())?), raw_a);
    Ok(())
}

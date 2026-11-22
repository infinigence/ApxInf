//! `VlaRuntime` for LingBot-VLA 2.0.
//!
//! The request carries Qwen3-VL patch rows for three 256x256 views, the prompt ids (valid tokens only),
//! the normalised 55-wide state and the initial noise. A plan is keyed by the prompt length, because the
//! prefix length, its rotary tables and every attention extent depend on it. Plans are captured into one
//! CUDA graph covering vision, language prefix and all ten flow steps; the most recent plan is kept.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use half::bf16;

use crate::cuda_lingbot_vla2::backend::{ops, Context, DeviceBuffer, RuntimeBackend};
use crate::cuda_lingbot_vla2::config::*;
use crate::cuda_lingbot_vla2::math;
use crate::cuda_lingbot_vla2::model::language::KvCaches;
use crate::cuda_lingbot_vla2::model::{self, Inputs, PlanTables, PrefixLayout};
use crate::cuda_lingbot_vla2::weights::model_weights::LingbotWeights;
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus,
    PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};

/// Arena capacities tried in order when capturing; the first that fits is kept.
const ARENA_CAPACITIES: [usize; 8] = [
    128 << 20,
    192 << 20,
    256 << 20,
    384 << 20,
    512 << 20,
    768 << 20,
    1024 << 20,
    1280 << 20,
];

pub struct LingbotVla2ModelRunner {
    backend: Arc<RuntimeBackend>,
    weights: Arc<LingbotWeights>,
    gemm: ops::GemmPolicy,
    attention: ops::AttentionPolicy,
    prepared: RefCell<Option<Plan>>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(format!("LingBot-VLA 2.0: {}", message.into()))
}

/// Fixed-address device inputs of one plan.
struct PlanInputs {
    patches: Tensor,
    token_embeds: Tensor,
    state: Tensor,
    noise: Tensor,
}

/// Everything a replay needs: the graph, its fixed inputs/outputs and the buffers it reads.
struct Plan {
    prompt_tokens: usize,
    layout: PrefixLayout,
    inputs: PlanInputs,
    output: Tensor,
    graph: Option<Box<dyn apxinf_core::Graph>>,
    // Kept alive because the graph reads them.
    _tables: PlanTables,
    caches: KvCaches,
    session: ops::ExecutionSession,
    eager_fallback: Option<String>,
}

impl LingbotVla2ModelRunner {
    pub(crate) fn new(
        backend: Arc<RuntimeBackend>,
        weights: Arc<LingbotWeights>,
        gemm: ops::GemmPolicy,
        attention: ops::AttentionPolicy,
    ) -> Self {
        Self {
            backend,
            weights,
            gemm,
            attention,
            prepared: RefCell::new(None),
        }
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
    }

    fn device_tensor(&self, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
        let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
        DeviceBuffer::alloc_zeros(bytes.max(1), self.ctx().device_id())
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(shape), dtype)
            .map_err(Error::Cuda)
    }

    fn build_plan(&self, prompt_tokens: usize, policy: ExecutionPolicy) -> Result<Plan> {
        if prompt_tokens == 0 || prompt_tokens > MAX_PROMPT_TOKENS {
            return Err(invalid(format!(
                "prompt must have 1..={MAX_PROMPT_TOKENS} tokens, got {prompt_tokens}"
            )));
        }
        let ctx = self.ctx();
        let layout = PrefixLayout::new(prompt_tokens);
        let grids = [[1u32, GRID, GRID]; NUM_VIEWS];
        let positions = math::rope_index(&grids, prompt_tokens, PREFIX_QUERY_TOKENS);
        let flat: Vec<u32> = positions
            .iter()
            .flat_map(|p| [p[0] as u32, p[1] as u32, p[2] as u32])
            .collect();
        let start = math::suffix_position_offset(&positions);
        let suffix: Vec<u32> = (0..SUFFIX_TOKENS as i64)
            .flat_map(|i| [(start + i) as u32; 3])
            .collect();
        let tables = PlanTables {
            prefix_positions: crate::cuda_lingbot_vla2::weights::device::upload_u32_buffer(
                ctx, &flat,
            )?,
            suffix_positions: crate::cuda_lingbot_vla2::weights::device::u32_tensor(
                ctx,
                &suffix,
                vec![SUFFIX_TOKENS, 3],
            )?,
        };
        let inputs = PlanInputs {
            patches: self.device_tensor(
                vec![NUM_VIEWS * PATCHES_PER_VIEW, VIT_PATCH_DIM],
                DType::BF16,
            )?,
            token_embeds: self.device_tensor(vec![prompt_tokens + 2, VLM_HIDDEN], DType::BF16)?,
            state: self.device_tensor(vec![1, PADDED_DIM], DType::BF16)?,
            noise: self.device_tensor(vec![ACTION_HORIZON, PADDED_DIM], DType::BF16)?,
        };
        let caches = KvCaches::new(ctx, layout.total)?;

        let mut last_error = None;
        for capacity in ARENA_CAPACITIES {
            let session = ops::ExecutionSession::with_capacity(capacity, ctx.device_id())?;
            let run = |_: ()| -> Result<Tensor> {
                let bound = Inputs {
                    patches: &inputs.patches,
                    token_embeds: &inputs.token_embeds,
                    state: &inputs.state,
                    noise: &inputs.noise,
                };
                model::infer(
                    ctx,
                    &self.weights,
                    &layout,
                    &tables,
                    &bound,
                    &caches,
                    &self.gemm,
                    &self.attention,
                )
            };
            // Warm up (tuning, native plan creation) outside capture.
            let warm = ops::prepare_with_session(&session, || run(()));
            let output = match warm {
                Ok(t) => t,
                Err(e) if e.to_string().contains("workspace exhausted") => {
                    last_error = Some(e.to_string());
                    continue;
                }
                Err(e) => return Err(invalid(format!("warm-up failed: {e}"))),
            };
            self.backend.synchronize()?;
            drop(output);
            eprintln!(
                "[apxinf] LingBot-VLA 2.0 prefix {} rows: arena {}/{} MiB used after warm-up",
                layout.total,
                session.used() >> 20,
                session.capacity() >> 20
            );
            let (graph, output) = match policy {
                ExecutionPolicy::Eager => (None, None),
                _ => {
                    match self
                        .backend
                        .capture_graph(|| ops::with_session(&session, || run(())))
                    {
                        Ok((g, out)) => (Some(g), Some(out)),
                        Err(e) if policy == ExecutionPolicy::PreferGraph => {
                            eprintln!("[apxinf] LingBot-VLA 2.0 graph capture unavailable, using eager: {e}");
                            (None, None)
                        }
                        Err(e) => return Err(e),
                    }
                }
            };
            let output = match output {
                Some(o) => o,
                None => {
                    // Eager plans re-run the model per request; allocate a stable output to copy into.
                    self.device_tensor(vec![ACTION_HORIZON, PADDED_DIM], DType::BF16)?
                }
            };
            return Ok(Plan {
                prompt_tokens,
                layout,
                inputs,
                output,
                eager_fallback: graph.is_none().then(|| "eager".to_string()),
                graph,
                _tables: tables,
                caches,
                session,
            });
        }
        Err(invalid(format!(
            "graph workspace exhausted at {} bytes: {}",
            ARENA_CAPACITIES[ARENA_CAPACITIES.len() - 1],
            last_error.unwrap_or_default()
        )))
    }

    fn run_plan(&self, plan: &Plan, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        let VisionObservation::Patches(patches) = &observation.vision else {
            return Err(invalid(
                "expects preprocessed Qwen3-VL patch rows, not raw RGB",
            ));
        };
        let expected = [NUM_VIEWS * PATCHES_PER_VIEW, VIT_PATCH_DIM];
        if patches.shape().dims() != expected {
            return Err(invalid(format!(
                "patches must be {expected:?}, got {:?}",
                patches.shape().dims()
            )));
        }
        if observation.token_ids.len() != plan.prompt_tokens {
            return Err(invalid(format!(
                "request has {} prompt tokens but the plan was prepared for {}",
                observation.token_ids.len(),
                plan.prompt_tokens
            )));
        }
        if let Some(grids) = request.metadata.image_grid_thw {
            if grids.len() != NUM_VIEWS || grids.iter().any(|g| *g != [1, GRID, GRID]) {
                return Err(invalid(format!(
                    "image_grid_thw must be {NUM_VIEWS} x [1, {GRID}, {GRID}], got {grids:?}"
                )));
            }
        }
        let state = observation
            .state
            .as_ref()
            .ok_or_else(|| invalid("a state vector is required"))?;
        let state = state.to_f32_vec()?;
        if state.len() != STATE_DIM {
            return Err(invalid(format!(
                "state must have {STATE_DIM} values, got {}",
                state.len()
            )));
        }
        self.backend.synchronize()?;
        // inputs are rewritten in place; the graph reads these addresses
        write_bf16_from_f32(&plan.inputs.patches, &patches.to_f32_vec()?)?;
        let mut ids: Vec<i32> = observation.token_ids.iter().map(|t| *t as i32).collect();
        ids.push(VISION_START_ID as i32);
        ids.push(VISION_END_ID as i32);
        write_bf16(
            &plan.inputs.token_embeds,
            &self.weights.vlm.embed.gather(&ids)?,
        )?;
        let mut padded_state = state;
        padded_state.resize(PADDED_DIM, 0.0);
        write_bf16_from_f32(&plan.inputs.state, &padded_state)?;
        let noise = match request.initial_latent {
            InitialLatent::Provided(noise) => noise.to_f32_vec()?,
            InitialLatent::Generate { rng } => host_normal(
                rng.seed,
                rng.sequence,
                rng.draw,
                ACTION_HORIZON * ACTION_DIM,
            ),
        };
        if noise.len() != ACTION_HORIZON * ACTION_DIM {
            return Err(invalid(format!(
                "noise must be [{ACTION_HORIZON}, {ACTION_DIM}], got {} values",
                noise.len()
            )));
        }
        let mut padded = vec![0f32; ACTION_HORIZON * PADDED_DIM];
        for row in 0..ACTION_HORIZON {
            padded[row * PADDED_DIM..row * PADDED_DIM + ACTION_DIM]
                .copy_from_slice(&noise[row * ACTION_DIM..(row + 1) * ACTION_DIM]);
        }
        write_bf16_from_f32(&plan.inputs.noise, &padded)?;
        match &plan.graph {
            Some(graph) => graph.replay()?,
            None => {
                let ctx = self.ctx();
                let bound = Inputs {
                    patches: &plan.inputs.patches,
                    token_embeds: &plan.inputs.token_embeds,
                    state: &plan.inputs.state,
                    noise: &plan.inputs.noise,
                };
                let out = ops::with_session(&plan.session, || {
                    model::infer(
                        ctx,
                        &self.weights,
                        &plan.layout,
                        &plan._tables,
                        &bound,
                        &plan.caches,
                        &self.gemm,
                        &self.attention,
                    )
                })?;
                ops::copy_tensor(ctx, &out, &plan.output)?;
            }
        }
        self.backend.synchronize()?;
        Ok(Action::new(plan.output.clone()))
    }

    /// `[50, 64]` BF16 device output -> `[50, 55]` f32 host values.
    fn download(&self, action: &Action) -> Result<Vec<f32>> {
        let host = self.backend.to_cpu(action.tensor())?.to_f32_vec()?;
        let mut out = Vec::with_capacity(ACTION_HORIZON * ACTION_DIM);
        for row in 0..ACTION_HORIZON {
            out.extend_from_slice(&host[row * PADDED_DIM..row * PADDED_DIM + ACTION_DIM]);
        }
        Ok(out)
    }

    fn ensure_plan(&self, prompt_tokens: usize) -> Result<()> {
        let stale = self
            .prepared
            .borrow()
            .as_ref()
            .is_none_or(|p| p.prompt_tokens != prompt_tokens);
        if stale {
            // Retire the old graph before allocating the new plan: the 4060 has no room for two.
            *self.prepared.borrow_mut() = None;
            self.backend.synchronize()?;
            let plan = self.build_plan(prompt_tokens, ExecutionPolicy::PreferGraph)?;
            *self.prepared.borrow_mut() = Some(plan);
        }
        Ok(())
    }
}

fn write_bf16_from_f32(tensor: &Tensor, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect();
    DeviceBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_from_host(&bytes)
        .map_err(Error::Cuda)
}

fn write_bf16(tensor: &Tensor, values: &[bf16]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    DeviceBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_from_host(&bytes)
        .map_err(Error::Cuda)
}

/// Deterministic standard-normal noise for `InitialLatent::Generate` (Box-Muller over a splitmix64 stream).
fn host_normal(seed: u64, sequence: u64, draw: u64, count: usize) -> Vec<f32> {
    let mut state = seed ^ sequence.rotate_left(21) ^ draw.rotate_left(42) ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let (u1, u2) = (next().max(1e-300), next());
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        if out.len() < count {
            out.push((r * (2.0 * std::f64::consts::PI * u2).sin()) as f32);
        }
    }
    out
}

impl VlaRuntime for LingbotVla2ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("int8_convrot")
    }

    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [ACTION_HORIZON, ACTION_DIM],
            patch_shape: [NUM_VIEWS * PATCHES_PER_VIEW, VIT_PATCH_DIM],
            max_token_len: MAX_PROMPT_TOKENS,
            num_views: NUM_VIEWS,
            image_size: IMAGE_SIZE,
            patch_size: VIT_PATCH,
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        request.observation.validate()?;
        self.ensure_plan(request.observation.token_ids.len())?;
        let guard = self.prepared.borrow();
        self.run_plan(guard.as_ref().expect("plan prepared above"), request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::PreferGraph)
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        if spec.image_layout.is_some() {
            return Err(invalid(
                "raw RGB input is not supported; send Qwen3-VL patch rows",
            ));
        }
        *self.prepared.borrow_mut() = None;
        self.backend.synchronize()?;
        let plan = self.build_plan(spec.token_count, policy)?;
        Ok(Box::new(Prepared {
            runner: self.clone_handles(),
            plan,
            spec: *spec,
        }))
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        sample.observation.validate()?;
        self.prepare_with_policy(&sample.observation.inference_spec(), policy)
    }

    fn clear_prepared(&self) -> Result<()> {
        self.backend.synchronize()?;
        *self.prepared.borrow_mut() = None;
        Ok(())
    }

    fn execution_mode(&self) -> &'static str {
        match self.prepared.borrow().as_ref() {
            Some(plan) if plan.graph.is_some() => "graph",
            Some(_) => "eager",
            None => "unprepared",
        }
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.download(&action)
    }

    fn infer_host_f32_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Vec<f32>, BTreeMap<String, f64>)> {
        let started = std::time::Instant::now();
        let values = self.infer_host_f32(request)?;
        Ok((
            values,
            BTreeMap::from([(
                "request_ms".to_string(),
                started.elapsed().as_secs_f64() * 1e3,
            )]),
        ))
    }
}

/// An explicitly prepared plan. It owns its graph and shares weights with the runner.
struct Prepared {
    runner: LingbotVla2ModelRunner,
    plan: Plan,
    spec: InferenceSpec,
}

impl LingbotVla2ModelRunner {
    fn clone_handles(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
            weights: Arc::clone(&self.weights),
            gemm: self.gemm.clone(),
            attention: self.attention.clone(),
            prepared: RefCell::new(None),
        }
    }
}

impl PreparedInference for Prepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: if self.plan.graph.is_some() {
                ExecutionMode::Graph
            } else {
                ExecutionMode::Eager
            },
            fallback_reason: self.plan.eager_fallback.clone(),
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(invalid(
                "request does not match the prepared inference spec",
            ));
        }
        self.runner.run_plan(&self.plan, request)
    }
}

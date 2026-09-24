//! Planning request validation, device input binding and mutable execution state.
use super::prepare::{DirectPlan, GdnGraphs};
use crate::qwen_drive::{
    backend::{kernels, transfers, tuning, DeviceBuffer},
    inputs::ExpertConditioning,
    model::{PlanningInput, PlanningState, QwenDriveModel, ReasoningInput, VisionState},
};
use crate::vla::{
    Action, ExecutionPolicy, InferenceSpec, InitialLatent, PreparedInference, VisionObservation,
    VlaContract, VlaRequest, VlaRuntime,
};
use apxinf_core::{DType, Device, Error, Result, SamplingBackend, Shape, Tensor};
use std::cell::RefCell;

struct ExecutionState {
    // Drop graphs before the buffers whose addresses they capture.
    graphs: GdnGraphs,
    vision: std::rc::Rc<VisionState>,
    backbone: Option<PlanningState>,
}
pub struct QwenDriveModelRunner {
    model: std::rc::Rc<QwenDriveModel>,
    prepared: RefCell<Option<DirectPlan>>,
    last_mode: std::cell::Cell<&'static str>,
    state: RefCell<ExecutionState>,
}
impl QwenDriveModelRunner {
    pub(crate) fn new(model: QwenDriveModel) -> Self {
        Self {
            model: std::rc::Rc::new(model),
            prepared: RefCell::new(None),
            last_mode: std::cell::Cell::new("eager"),
            state: RefCell::new(ExecutionState {
                graphs: GdnGraphs::default(),
                vision: Default::default(),
                backbone: None,
            }),
        }
    }
    fn execute(&self, request: &VlaRequest<'_>) -> Result<Tensor> {
        let valid = validate(&self.model, request)?;
        let backend = self.model.backend();
        let c = self.model.config();
        let observation = request.observation;
        let pixels = valid.pixels;
        let grids = valid.grids;
        let cond = valid.cond;
        let steps = valid.steps;
        let reasoning = request.metadata.planning.and_then(|o| o.reasoning.as_ref());
        let shape = [c.num_future_points, c.trajectory_point_dim];
        let noise = match request.initial_latent {
            InitialLatent::Provided(noise) => match noise.device() {
                Device::Cpu => transfers::to_cuda(noise, backend.device_id())?,
                Device::Cuda(id) if id == backend.device_id() => noise.clone(),
                _ => {
                    return Err(Error::Other(
                        "qwen_drive latent is on another device".into(),
                    ))
                }
            }
            .reshape(shape.to_vec())?,
            InitialLatent::Generate { rng } => {
                let buffer = DeviceBuffer::alloc(shape[0] * shape[1] * 4, backend.device_id())
                    .map_err(Error::Cuda)?;
                let noise = buffer
                    .as_tensor(Shape::new(shape.to_vec()), DType::F32)
                    .map_err(Error::Cuda)?;
                backend
                    .create_normal_generator(noise.clone())?
                    .generate(rng)?;
                kernels::elementwise::scale(backend.context(), &noise, c.noise_init_std)?
            }
        };
        let input = PlanningInput {
            token_ids: &observation.token_ids,
            pixels,
            grids,
            conditioning: &cond,
            noise: &noise,
            steps,
            reasoning: reasoning.map(|r| ReasoningInput {
                max_new_tokens: r.max_new_tokens,
                min_new_tokens: r.min_new_tokens,
                terminator_ids: &r.terminator_ids,
                closing_ids: &r.closing_ids,
            }),
        };
        let mut execution = self
            .state
            .try_borrow_mut()
            .map_err(|_| Error::Other("qwen_drive runner is already executing".into()))?;
        // Keep cache addresses stable across requests, but reset all semantic
        // state. Local decode captures retain their existing per-request policy.
        execution.graphs = GdnGraphs::default();
        if let Some(backbone) = execution.backbone.as_mut() {
            self.model.reset_state(backbone)?;
        } else {
            execution.backbone = Some(self.model.new_state(execution.vision.clone())?);
        }
        let ExecutionState {
            graphs, backbone, ..
        } = &mut *execution;
        self.model
            .infer(backbone.as_mut().expect("fresh backbone"), graphs, &input)
    }
}
impl VlaRuntime for QwenDriveModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }
    fn contract(&self) -> VlaContract {
        let c = self.model.config();
        VlaContract {
            action_shape: [c.num_future_points, c.trajectory_point_dim],
            patch_shape: [
                0,
                c.vision.in_channels
                    * c.vision.temporal_patch_size
                    * c.vision.patch_size
                    * c.vision.patch_size,
            ],
            max_token_len: c.text.max_position_embeddings.min(16384),
            num_views: 0,
            image_size: 0,
            patch_size: c.vision.patch_size,
            accepts_rgb_u8: false,
        }
    }
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let use_graph = request
            .metadata
            .planning
            .and_then(|p| p.reasoning.as_ref())
            .is_none()
            && !matches!(
                std::env::var("APXINF_QWEN_GRAPH").as_deref(),
                Ok("0") | Ok("off")
            );
        if !use_graph {
            let result = self.execute(request)?;
            self.last_mode
                .set(if std::env::var_os("APXINF_QWEN_DECODE_GRAPH").is_some() {
                    "eager-with-opt-in-gdn-graphs"
                } else {
                    "eager"
                });
            return Ok(Action::new(result));
        }
        let valid = validate(&self.model, request)?;
        let mut cache = self
            .prepared
            .try_borrow_mut()
            .map_err(|_| Error::Other("qwen_drive runner is already executing".into()))?;
        if cache
            .as_ref()
            .is_none_or(|plan| !plan.matches(request, &valid) || !plan.is_current())
        {
            // Online tuning needs the real request operands, and the direct
            // plan suppresses it through both preparation and replay because a
            // capture cannot afford a tactic search mid-graph. So traverse once
            // eagerly first, as the PI0.5 and GR00T runtimes do, and let every
            // GEMM shape this model issues reach the tuner before graph
            // preparation freezes the plans it selected. Without this an
            // autotune pass over direct planning writes nothing.
            if self.model.backend().context().tuning().mode() == tuning::TuningMode::AutoTune {
                let tuned = self.execute(request)?;
                self.model
                    .backend()
                    .context()
                    .synchronize()
                    .map_err(Error::Cuda)?;
                drop(tuned);
            }
            self.model
                .backend()
                .context()
                .synchronize()
                .map_err(Error::Cuda)?;
            *cache = None;
            *cache = Some(DirectPlan::prepare(
                self.model.clone(),
                request,
                ExecutionPolicy::PreferGraph,
            )?);
        }
        let plan = cache.as_ref().expect("prepared direct plan");
        let result = plan.run(request)?;
        self.last_mode.set(plan.execution_mode());
        Ok(result)
    }
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        transfers::to_cpu(action.tensor())?.to_f32_vec()
    }
    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other("qwen_drive requires prepare_for(sample, policy): InferenceSpec does not describe image grids, image-token positions or planner steps".into()))
    }
    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(DirectPlan::prepare(
            self.model.clone(),
            sample,
            policy,
        )?))
    }
    fn clear_prepared(&self) -> Result<()> {
        let mut cache = self
            .prepared
            .try_borrow_mut()
            .map_err(|_| Error::Other("qwen_drive runner is already executing".into()))?;
        self.model
            .backend()
            .context()
            .synchronize()
            .map_err(Error::Cuda)?;
        *cache = None;
        self.last_mode.set("eager");
        Ok(())
    }
    fn execution_mode(&self) -> &'static str {
        let mode = self.last_mode.get();
        if matches!(mode, "graph" | "eager-fallback")
            && self
                .prepared
                .borrow()
                .as_ref()
                .is_some_and(|plan| !plan.is_current())
        {
            "invalidated"
        } else {
            mode
        }
    }
}

pub(super) struct Validated<'a> {
    pub pixels: &'a Tensor,
    pub grids: &'a [[u32; 3]],
    pub cond: ExpertConditioning,
    pub steps: usize,
}
pub(super) fn validate<'a>(
    model: &QwenDriveModel,
    request: &'a VlaRequest<'a>,
) -> Result<Validated<'a>> {
    let c = model.config();
    let observation = request.observation;
    observation.validate()?;
    if observation.action_mask.is_some() || request.metadata.embodiment_id.is_some() {
        return Err(Error::Other(
            "qwen_drive does not accept action masks or embodiment IDs".into(),
        ));
    }
    if let Some(mask) = request.metadata.attention_mask {
        if mask.len() != observation.token_ids.len() || mask.iter().any(|&x| x != 1) {
            return Err(Error::Other(
                "qwen_drive requires an unpadded prompt (all-one attention mask)".into(),
            ));
        }
    }
    let pixels = match &observation.vision {
        VisionObservation::Patches(p) => p,
        _ => {
            return Err(Error::Other(
                "qwen_drive requires canonical image patches".into(),
            ))
        }
    };
    if !matches!(pixels.device(), Device::Cpu)
        && pixels.device() != Device::Cuda(model.backend().device_id())
    {
        return Err(Error::Other(
            "qwen_drive pixels are on another device".into(),
        ));
    }
    let grids = request
        .metadata
        .image_grid_thw
        .ok_or_else(|| Error::Other("qwen_drive requires image_grid_thw".into()))?;
    let width = c.vision.in_channels
        * c.vision.temporal_patch_size
        * c.vision.patch_size
        * c.vision.patch_size;
    let merge = c.vision.spatial_merge_size as u32;
    let rows = grids.iter().try_fold(0usize, |sum, g| {
        if g.contains(&0) || g[1] % merge != 0 || g[2] % merge != 0 {
            return Err(Error::Other("qwen_drive invalid image grid".into()));
        }
        let n = g
            .iter()
            .try_fold(1usize, |n, &x| n.checked_mul(x as usize))
            .and_then(|n| sum.checked_add(n));
        n.ok_or_else(|| Error::Other("qwen_drive image grid overflow".into()))
    })?;
    if grids.is_empty()
        || pixels.shape().dims() != [rows, width]
        || !matches!(pixels.dtype(), DType::F32 | DType::BF16)
    {
        return Err(Error::Other(
            "qwen_drive patch shape/dtype does not match image grids".into(),
        ));
    }
    if observation
        .token_ids
        .iter()
        .any(|&id| id as usize >= c.text.vocab_size)
    {
        return Err(Error::Other(
            "qwen_drive token ID exceeds vocabulary".into(),
        ));
    }
    let state = observation
        .state
        .as_ref()
        .ok_or_else(|| Error::Other("qwen_drive requires packed planning conditioning".into()))?;
    if state.device() != Device::Cpu
        || state.dtype() != DType::F32
        || state.shape().dims().len() != 1
    {
        return Err(Error::Other(
            "qwen_drive conditioning must be a host f32 vector".into(),
        ));
    }
    let cond = ExpertConditioning::from_packed(c, &state.to_f32_vec()?)?;
    let options = request.metadata.planning;
    let steps = options
        .and_then(|o| o.num_steps)
        .unwrap_or(c.num_inference_steps);
    if steps == 0 {
        return Err(Error::Other("qwen_drive num_steps must be positive".into()));
    }
    let reasoning = options.and_then(|o| o.reasoning.as_ref());
    if let Some(r) = reasoning {
        if r.max_new_tokens == 0
            || r.min_new_tokens > r.max_new_tokens
            || r.terminator_ids.is_empty()
            || r.closing_ids.is_empty()
            || r.terminator_ids
                .iter()
                .chain(&r.closing_ids)
                .any(|&id| id as usize >= c.text.vocab_size)
        {
            return Err(Error::Other(
                "qwen_drive invalid reasoning token bounds or turn delimiters".into(),
            ));
        }
    }
    let reserve = reasoning
        .map_or(Some(0), |r| {
            r.max_new_tokens.checked_add(r.closing_ids.len())
        })
        .ok_or_else(|| Error::Other("qwen_drive reasoning capacity overflow".into()))?;
    if observation
        .token_ids
        .len()
        .checked_add(reserve)
        .is_none_or(|n| n > c.text.max_position_embeddings.min(16384))
    {
        return Err(Error::Other(
            "qwen_drive prompt and reasoning exceed cache capacity".into(),
        ));
    }
    if let InitialLatent::Provided(noise) = request.initial_latent {
        let shape = [c.num_future_points, c.trajectory_point_dim];
        let dims = noise.shape().dims();
        if (dims != shape && dims != [1, shape[0], shape[1]]) || noise.dtype() != DType::F32 {
            return Err(Error::Other(
                "qwen_drive latent must be f32 [horizon,3] or [1,horizon,3]".into(),
            ));
        }
        match noise.device() {
            Device::Cpu => {
                if noise.to_f32_vec()?.iter().any(|x| !x.is_finite()) {
                    return Err(Error::Other(
                        "qwen_drive latent contains non-finite values".into(),
                    ));
                }
            }
            Device::Cuda(id) if id == model.backend().device_id() => {}
            _ => {
                return Err(Error::Other(
                    "qwen_drive latent is on another device".into(),
                ))
            }
        }
    }
    // Validate image-token runs before a malformed request can evict a plan.
    model.validate_layout(&observation.token_ids, grids)?;
    Ok(Validated {
        pixels,
        grids,
        cond,
        steps,
    })
}

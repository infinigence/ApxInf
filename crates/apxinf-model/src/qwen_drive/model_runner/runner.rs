//! Planning request validation, device input binding and mutable execution state.
use super::prepare::DirectPlan;
use crate::qwen_drive::{
    backend::{kernels, transfers, tuning, Context, DeviceBuffer},
    inputs::ExpertConditioning,
    model::{PlanningInput, PlanningState, QwenDriveModel, ReasoningInput, VisionState},
};
use crate::vla::{
    Action, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent, PreparedInference,
    RawRgbResizeFrame, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};
use apxinf_core::{Backend, DType, Device, Error, Result, SamplingBackend, Shape, Tensor};
use std::cell::RefCell;

struct ExecutionState {
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
        let mut rgb_hold = None;
        let pixels = match valid.vision {
            ValidatedVision::Patches(pixels) => (*pixels).clone(),
            ValidatedVision::Rgb(bytes) => {
                let ctx = backend.context();
                let resize = valid
                    .resize_frames()
                    .map(|frames| kernels::pillow_bicubic::PillowBicubicRgbPlan::new(ctx, &frames))
                    .transpose()?;
                let raw = if let Some(plan) = &resize {
                    plan.raw().clone()
                } else {
                    DeviceBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?
                };
                raw.copy_from_host(bytes).map_err(Error::Cuda)?;
                let output_bytes = valid
                    .rows
                    .checked_mul(valid.width)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(|| Error::Other("qwen_drive BF16 patch size overflow".into()))?;
                let output = DeviceBuffer::alloc(output_bytes, ctx.device_id())
                    .map_err(Error::Cuda)?
                    .as_tensor(Shape::new(vec![valid.rows, valid.width]), DType::BF16)
                    .map_err(Error::Cuda)?;
                let lut = rgb_bf16_lut(ctx)?;
                if let Some(plan) = &resize {
                    if let Err(error) = plan.run(ctx) {
                        // A prior axis may already be queued on this stream.
                        ctx.synchronize().map_err(Error::Cuda)?;
                        return Err(error);
                    }
                }
                let rgb = resize.as_ref().map(|plan| plan.final_rgb()).unwrap_or(&raw);
                if let Err(error) = kernels::preprocess::rgb_u8_to_temporal2_merge2_rect_bf16(
                    ctx,
                    rgb,
                    &output,
                    &lut,
                    &valid.rgb_frames(),
                ) {
                    // Earlier frame launches may already be queued on the stream.
                    ctx.synchronize().map_err(Error::Cuda)?;
                    return Err(error);
                }
                rgb_hold = Some((raw, lut, resize));
                output
            }
        };
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
            prompt_len: valid.prompt_len,
            pixels: &pixels,
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
        // state. Reasoning decode uses ordinary eager execution.
        if let Some(backbone) = execution.backbone.as_mut() {
            self.model.reset_state(backbone)?;
        } else {
            execution.backbone = Some(self.model.new_state(execution.vision.clone())?);
        }
        let backbone = &mut execution.backbone;
        let result = self
            .model
            .infer(backbone.as_mut().expect("fresh backbone"), &input);
        // Eager RGB input is temporary; complete its stream consumers before
        // dropping the raw bytes and LUT. The direct graph uses stable buffers.
        if rgb_hold.is_some() {
            backend.synchronize()?;
        }
        result
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
            accepts_rgb_u8: c.vision.in_channels == 3
                && c.vision.patch_size == 16
                && c.vision.temporal_patch_size == 2
                && c.vision.spatial_merge_size == 2,
        }
    }
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let use_graph = request
            .metadata
            .planning
            .and_then(|p| p.reasoning.as_ref())
            .is_none();
        if !use_graph {
            let result = self.execute(request)?;
            self.last_mode.set("eager");
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

#[derive(Clone, Copy)]
pub(super) enum ValidatedVision<'a> {
    Patches(&'a Tensor),
    Rgb(&'a [u8]),
}

pub(super) struct Validated<'a> {
    pub vision: ValidatedVision<'a>,
    pub grids: &'a [[u32; 3]],
    pub rows: usize,
    pub width: usize,
    pub cond: ExpertConditioning,
    pub steps: usize,
    pub raw_resize: Option<&'a [RawRgbResizeFrame]>,
    /// Unmasked prompt tokens. Equal to `token_ids.len()` unless the caller padded
    /// the prompt, in which case `token_ids.len()` is the padded width and this is
    /// the length everything semantic must use.
    pub prompt_len: usize,
}
impl Validated<'_> {
    pub fn rgb_frames(&self) -> Vec<kernels::preprocess::RgbRectFrame> {
        self.grids
            .iter()
            .map(|g| kernels::preprocess::RgbRectFrame {
                grid_h: g[1] as usize,
                grid_w: g[2] as usize,
            })
            .collect()
    }
    pub fn resize_frames(&self) -> Option<Vec<kernels::pillow_bicubic::RgbResizeFrame>> {
        self.raw_resize.map(|frames| {
            frames
                .iter()
                .map(|f| kernels::pillow_bicubic::RgbResizeFrame {
                    source_width: f.source_width,
                    source_height: f.source_height,
                    stage_width: f.stage_width,
                    stage_height: f.stage_height,
                    final_width: f.final_width,
                    final_height: f.final_height,
                })
                .collect()
        })
    }
    pub fn input_dtype(&self) -> DType {
        match self.vision {
            ValidatedVision::Patches(p) => p.dtype(),
            ValidatedVision::Rgb(_) => DType::BF16,
        }
    }
}

pub(super) fn rgb_bf16_lut(ctx: &Context) -> Result<DeviceBuffer> {
    // Same f32 sequence as QwenDrivePolicy._patchify, rounded once to BF16.
    let mut bytes = Vec::with_capacity(512);
    for value in 0..=255 {
        let f = ((value as f32 / 255.0f32) - 0.5f32) / 0.5f32;
        bytes.extend_from_slice(&half::bf16::from_f32(f).to_bits().to_ne_bytes());
    }
    let lut = DeviceBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    lut.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(lut)
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
    // Without a mask every token is real; a mask may pad the prompt out to a
    // fixed length so the shape-specialised kernels stay reachable.
    let mut prompt_len = observation.token_ids.len();
    if let Some(mask) = request.metadata.attention_mask {
        if mask.len() != observation.token_ids.len() {
            return Err(Error::Other(
                "qwen_drive attention mask length does not match the prompt".into(),
            ));
        }
        // Padding is inert only because it is trailing: the backbone is causal and
        // the expert masks a suffix of its joint KV, so an interior zero would be
        // silently attended to rather than rejected. Require the zeros to reach the
        // end, and take the real length from the mask instead of carrying a second,
        // desynchronisable copy of it.
        let real_len = mask.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
        if real_len == 0 {
            return Err(Error::Other(
                "qwen_drive requires at least one unmasked prompt token".into(),
            ));
        }
        if mask[..real_len].iter().any(|&x| x != 1) {
            return Err(Error::Other(
                "qwen_drive requires trailing attention padding; the mask has an interior zero"
                    .into(),
            ));
        }
        prompt_len = real_len;
    }
    if prompt_len != observation.token_ids.len() {
        if request
            .metadata
            .planning
            .and_then(|p| p.reasoning.as_ref())
            .is_some()
        {
            return Err(Error::Other(
                "qwen_drive padded reasoning is unsupported".into(),
            ));
        }
        if observation.token_ids[prompt_len..]
            .iter()
            .any(|&id| id != 0)
        {
            return Err(Error::Other(
                "qwen_drive padding token IDs must be zero".into(),
            ));
        }
    }
    let grids = request
        .metadata
        .image_grid_thw
        .ok_or_else(|| Error::Other("qwen_drive requires image_grid_thw".into()))?;
    let raw_resize = request
        .metadata
        .planning
        .and_then(|options| options.raw_rgb_resize.as_deref());
    let width = c
        .vision
        .in_channels
        .checked_mul(c.vision.temporal_patch_size)
        .and_then(|n| n.checked_mul(c.vision.patch_size))
        .and_then(|n| n.checked_mul(c.vision.patch_size))
        .ok_or_else(|| Error::Other("qwen_drive patch width overflow".into()))?;
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
    if grids.is_empty() {
        return Err(Error::Other("qwen_drive requires image grids".into()));
    }
    rows.checked_mul(width)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Other("qwen_drive pixel buffer size overflow".into()))?;
    let vision = match &observation.vision {
        VisionObservation::Patches(pixels) => {
            if raw_resize.is_some() {
                return Err(Error::Other(
                    "raw RGB resize metadata requires RGB input".into(),
                ));
            }
            if (pixels.device() != Device::Cpu
                && pixels.device() != Device::Cuda(model.backend().device_id()))
                || pixels.shape().dims() != [rows, width]
                || !matches!(pixels.dtype(), DType::F32 | DType::BF16)
            {
                return Err(Error::Other(
                    "qwen_drive patch shape/dtype/device does not match image grids".into(),
                ));
            }
            ValidatedVision::Patches(pixels)
        }
        VisionObservation::RgbU8 { bytes, layout } => {
            if *layout != ImageLayout::Nhwc
                || c.vision.patch_size != 16
                || c.vision.temporal_patch_size != 2
                || c.vision.spatial_merge_size != 2
                || c.vision.in_channels != 3
                || grids.iter().any(|g| {
                    g[0] != 1
                        || g[1] > i32::MAX as u32
                        || g[2] > i32::MAX as u32
                        || (g[1] as u64) * (g[2] as u64) > i32::MAX as u64
                })
            {
                return Err(Error::Other(
                    "qwen_drive RGB path requires NHWC still frames and 16/2/2 patch geometry"
                        .into(),
                ));
            }
            let final_expected = rows
                .checked_mul(c.vision.patch_size)
                .and_then(|n| n.checked_mul(c.vision.patch_size))
                .and_then(|n| n.checked_mul(c.vision.in_channels))
                .ok_or_else(|| Error::Other("qwen_drive RGB input byte count overflow".into()))?;
            let expected = if let Some(frames) = raw_resize {
                if frames.len() != grids.len() || frames.len() > 65535 {
                    return Err(Error::Other(
                        "raw RGB resize frame count does not match grids".into(),
                    ));
                }
                let mut total_final = 0usize;
                let mut total_raw = 0usize;
                for (frame, grid) in frames.iter().zip(grids) {
                    if frame.source_width == 0
                        || frame.source_height == 0
                        || frame.stage_width == 0
                        || frame.stage_height == 0
                        || [
                            frame.source_width,
                            frame.source_height,
                            frame.stage_width,
                            frame.stage_height,
                            frame.final_width,
                            frame.final_height,
                        ]
                        .iter()
                        .any(|&size| size > 8192)
                        || [
                            (frame.source_width, frame.source_height),
                            (frame.stage_width, frame.stage_height),
                            (frame.final_width, frame.final_height),
                            (frame.stage_width, frame.source_height),
                            (frame.final_width, frame.stage_height),
                        ]
                        .iter()
                        .any(|&(w, h)| u64::from(w) * u64::from(h) > i32::MAX as u64)
                        || frame.final_width
                            != grid[2].checked_mul(c.vision.patch_size as u32).unwrap_or(0)
                        || frame.final_height
                            != grid[1].checked_mul(c.vision.patch_size as u32).unwrap_or(0)
                    {
                        return Err(Error::Other(
                            "raw RGB resize geometry does not match final grid".into(),
                        ));
                    }
                    total_final = total_final
                        .checked_add(
                            (frame.final_width as usize)
                                .checked_mul(frame.final_height as usize)
                                .and_then(|n| n.checked_mul(3))
                                .ok_or_else(|| Error::Other("final RGB extent overflow".into()))?,
                        )
                        .ok_or_else(|| Error::Other("final RGB total overflow".into()))?;
                    total_raw = total_raw
                        .checked_add(
                            (frame.source_width as usize)
                                .checked_mul(frame.source_height as usize)
                                .and_then(|n| n.checked_mul(3))
                                .ok_or_else(|| Error::Other("raw RGB extent overflow".into()))?,
                        )
                        .ok_or_else(|| Error::Other("raw RGB total overflow".into()))?;
                }
                if total_final != final_expected
                    || total_raw > 512 * 1024 * 1024
                    || total_final > 256 * 1024 * 1024
                {
                    return Err(Error::Other(
                        "raw RGB final extent does not match patches".into(),
                    ));
                }
                total_raw
            } else {
                final_expected
            };
            if bytes.len() != expected {
                return Err(Error::Other(
                    "qwen_drive RGB byte count does not match image grids".into(),
                ));
            }
            ValidatedVision::Rgb(bytes)
        }
    };
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
        vision,
        grids,
        rows,
        width,
        cond,
        steps,
        raw_resize,
        prompt_len,
    })
}

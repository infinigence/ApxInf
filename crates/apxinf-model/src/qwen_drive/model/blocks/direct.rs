//! Fixed-profile device inputs for direct planning. No capture policy lives here.
use super::bf16::{
    expert, upload_u32, vision, BackboneBf16, BackboneState, PlannerBf16, VisionState,
};
use super::{GdnExecution, GdnRequest};
use crate::qwen_drive::{
    backend::{kernels, nvtx, Context, DeviceBuffer},
    inputs::ExpertConditioning,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

pub(crate) struct DirectInputs {
    pub pixels: Tensor,
    pub rgb: Option<RawRgbInput>,
    pub tokens: DeviceBuffer,
    pub noise: Tensor,
    rows: DeviceBuffer,
    cos: Tensor,
    sin: Tensor,
    vision: vision::PreparedVision,
    visual_output: Tensor,
    expert: expert::PreparedInputs,
    conditioning_shape: ExpertConditioning,
    scene: Vec<(Tensor, Tensor)>,
    token_count: usize,
    prompt_len: usize,
    anchor: i64,
    steps: usize,
}

/// Stable captured input addresses and frame geometry for resized RGB mode.
pub(crate) struct RawRgbInput {
    pub bytes: DeviceBuffer,
    pub lut: DeviceBuffer,
    pub frames: Vec<kernels::preprocess::RgbRectFrame>,
    pub resize: Option<kernels::pillow_bicubic::PillowBicubicRgbPlan>,
}

/// The runner may reuse a workspace between these sequential computations.
/// Cross-phase values are in persistent storage before the callback returns.
pub(crate) trait DirectExecution {
    fn run(&mut self, operation: &mut dyn FnMut() -> Result<Tensor>) -> Result<Tensor>;
}
struct EagerGdn;
impl GdnExecution for EagerGdn {
    fn run(
        &mut self,
        _: &GdnRequest<'_>,
        x: Tensor,
        eager: &mut dyn FnMut(Tensor) -> Result<Tensor>,
    ) -> Result<(Tensor, bool)> {
        eager(x).map(|x| (x, false))
    }
}
fn tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |n, &d| n.checked_mul(d))
        .ok_or_else(|| Error::Other("qwen_drive direct input size overflow".into()))?;
    DeviceBuffer::alloc(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}
fn copy(ctx: &Context, dst: &Tensor, src: &Tensor) -> Result<()> {
    DeviceBuffer::from_tensor(dst)
        .map_err(Error::Cuda)?
        .copy_from_device_async(
            &DeviceBuffer::from_tensor(src).map_err(Error::Cuda)?,
            src.numel() * src.dtype().size_in_bytes(),
            ctx.stream(),
        )
        .map_err(Error::Cuda)
}
impl DirectInputs {
    pub(crate) fn new(
        b: &BackboneBf16,
        p: &PlannerBf16,
        state: &mut BackboneState,
        vision_state: &VisionState,
        token_ids: &[u32],
        prompt_len: usize,
        pixels: &Tensor,
        grids: &[[u32; 3]],
        cond: ExpertConditioning,
        steps: usize,
    ) -> Result<Self> {
        let ctx = b.ctx();
        let positions = b.rope_index(token_ids, grids)?;
        let last = positions
            .get(prompt_len - 1)
            .ok_or_else(|| Error::Other("qwen_drive empty prompt".into()))?;
        let anchor = *last.iter().max().unwrap() as i64;
        let (cos, sin) = b.mrope_tables(&positions)?;
        let mut ordinal = 0;
        let rows: Vec<u32> = token_ids
            .iter()
            .map(|&id| {
                if id == b.config.image_token_id {
                    let row = ordinal;
                    ordinal += 1;
                    row
                } else {
                    u32::MAX
                }
            })
            .collect();
        let merged_rows: usize = grids
            .iter()
            .map(|g| g.iter().map(|&x| x as usize).product::<usize>())
            .sum::<usize>()
            / b.config.vision.spatial_merge_size.pow(2);
        if ordinal as usize != merged_rows {
            return Err(Error::Other(
                "qwen_drive image token / vision row mismatch".into(),
            ));
        }
        let scene = b.scene_caches(state)?;
        let noise = tensor(
            ctx,
            &[b.config.num_future_points, b.config.trajectory_point_dim],
            DType::F32,
        )?;
        let plan = expert::ExpertPlan {
            scene: &scene,
            scene_len: prompt_len,
            anchor,
            cond: &cond,
            noise: &noise,
            num_steps: steps,
        };
        let expert = expert::prepare_inputs(&p.config, ctx, &plan)?;
        let vision = vision::prepare(
            &b.config,
            &b.weights.vision,
            vision_state,
            ctx,
            pixels,
            grids,
        )?;
        Ok(Self {
            pixels: tensor(ctx, pixels.shape().dims(), pixels.dtype())?,
            rgb: None,
            tokens: upload_u32(ctx, token_ids)?,
            noise,
            rows: upload_u32(ctx, &rows)?,
            cos,
            sin,
            vision,
            visual_output: tensor(ctx, &[merged_rows, b.config.text.hidden_size], DType::BF16)?,
            expert,
            conditioning_shape: cond,
            scene,
            token_count: token_ids.len(),
            prompt_len,
            anchor,
            steps,
        })
    }
    pub(crate) fn update_conditioning(
        &self,
        config: &crate::qwen_drive::config::QwenDriveConfig,
        cond: &ExpertConditioning,
    ) -> Result<()> {
        self.expert.update_conditioning(config, cond)
    }
    pub(crate) fn forward(
        &self,
        b: &BackboneBf16,
        p: &PlannerBf16,
        state: &mut BackboneState,
        execution: &mut dyn DirectExecution,
    ) -> Result<Tensor> {
        let ctx = b.ctx();
        // Reset is part of the captured graph, so every replay starts a new request.
        b.reset_state(state)?;
        execution.run(&mut || {
            let _stage = nvtx::range("qwen_drive/vision");
            let pixels = if let Some(rgb) = &self.rgb {
                if let Some(resize) = &rgb.resize {
                    resize.run(ctx)?;
                }
                let rgb_buffer = rgb
                    .resize
                    .as_ref()
                    .map(|resize| resize.final_rgb())
                    .unwrap_or(&rgb.bytes);
                kernels::preprocess::rgb_u8_to_temporal2_merge2_rect_bf16(
                    ctx,
                    rgb_buffer,
                    &self.pixels,
                    &rgb.lut,
                    &rgb.frames,
                )?;
                self.pixels.clone()
            } else {
                b.upload_pixels(&self.pixels)?
            };
            let features =
                vision::forward_device(&b.config, &b.weights.vision, ctx, &pixels, &self.vision)?;
            copy(ctx, &self.visual_output, &features)?;
            Ok(self.visual_output.clone())
        })?;
        execution.run(&mut || {
            let _stage = nvtx::range("qwen_drive/language");
            let embedded = kernels::embedding::lookup(
                ctx,
                &b.weights.embed_tokens,
                &self.tokens,
                self.token_count,
            )?;
            let embedded = kernels::elementwise::replace_rows_bf16(
                ctx,
                &embedded,
                &self.visual_output,
                &self.rows,
            )?;
            b.run_text_device(
                state,
                &mut EagerGdn,
                embedded,
                self.token_count,
                &self.cos,
                &self.sin,
            )?;
            // Language-to-planner handoff is the persistent shared KV prefix.
            Ok(self.visual_output.clone())
        })?;
        execution.run(&mut || {
            let _stage = nvtx::range("qwen_drive/planner");
            let plan = expert::ExpertPlan {
                scene: &self.scene,
                scene_len: self.prompt_len,
                anchor: self.anchor,
                cond: &self.conditioning_shape,
                noise: &self.noise,
                num_steps: self.steps,
            };
            let flow = expert::prepare_device(&p.config, &p.weights, ctx, &plan, &self.expert)?;
            for step in 0..self.steps {
                p.step(&plan, &flow, step)?;
            }
            Ok(p.output(flow))
        })
    }
}

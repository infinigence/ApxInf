//! Request binding and fixed-profile lifetime for the existing VLA interface.
use crate::cuda_cosmos3_edge::backend::ops::GemmPolicy;
use crate::cuda_cosmos3_edge::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_cosmos3_edge::weights::Bf16Weights;
use crate::vla::*;
use apxinf_core::{DType, Device, Error, Result};
use std::{cell::RefCell, rc::Rc};

use super::prepare::Plan;

/// Frames of RGB the runtime encodes. Only the singleton (frame 0) conditions
/// the denoiser; the 16-frame rollout chunk is dropped because the causal VAE
/// never reads its output back, so the public path takes a single image.
pub(super) const PIXEL_FRAMES: usize = 1;
pub(super) const PIXEL_H: usize = 544;
pub(super) const PIXEL_W: usize = 736;
/// Joint initial noise: the CTHW video latent `[48, 5, 34, 46]`, then `[16, 64]` action.
pub(super) const LATENT_SHAPE: [usize; 2] = [1, 48 * 5 * 34 * 46 + 16 * 64];
pub(super) const ACTION_SHAPE: [usize; 2] = [16, 10];
pub(super) const NUM_TASKS: usize = 1;

pub(super) fn invalid(message: &str) -> Error {
    Error::Other(format!("Cosmos3-Edge: {message}"))
}
pub(super) fn validate_spec(spec: &InferenceSpec) -> Result<()> {
    if spec.token_count != 1 || spec.image_layout != Some(ImageLayout::Nhwc) {
        return Err(invalid(
            "DROID direct requires one cached prompt ID and NHWC RGB",
        ));
    }
    Ok(())
}
pub(super) fn validate_request(request: &VlaRequest<'_>) -> Result<()> {
    validate_spec(&request.observation.inference_spec())?;
    let o = request.observation;
    match &o.vision {
        VisionObservation::RgbU8 {
            bytes,
            layout: ImageLayout::Nhwc,
        } if bytes.len() == PIXEL_FRAMES * PIXEL_H * PIXEL_W * 3 => {}
        _ => return Err(invalid("expected a single 544x736 NHWC RGB frame")),
    }
    if (o.token_ids[0] as usize) >= NUM_TASKS {
        return Err(invalid("cached-prompt ID is outside the supported table"));
    }
    if o.state.is_some() {
        return Err(invalid(
            "the DROID direct profile does not take proprioception",
        ));
    }
    if o.action_mask.is_some()
        || request.metadata.attention_mask.is_some()
        || request
            .metadata
            .image_grid_thw
            .is_some_and(|grid| grid != [[1, PIXEL_H as u32, PIXEL_W as u32]])
        || request.metadata.embodiment_id.is_some()
        || request.metadata.planning.is_some()
    {
        return Err(invalid(
            "direct profile does not take masks, embodiment overrides or planning options",
        ));
    }
    if let InitialLatent::Provided(noise) = request.initial_latent {
        if noise.device() != Device::Cpu
            || noise.dtype() != DType::F32
            || noise.shape().dims() != LATENT_SHAPE.as_slice()
            || noise.as_f32()?.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "supplied noise must be finite host FP32 [1, video + action]",
            ));
        }
    }
    Ok(())
}

pub(in crate::cuda_cosmos3_edge) struct ModelRunner {
    pub(super) ctx: Rc<CudaContext>,
    pub(super) weights: Rc<Bf16Weights>,
    pub(super) gemm: GemmPolicy,
    pub(super) attention: AttentionPolicy,
    /// UniPC denoising step count, driven by the checkpoint config (4 for the
    /// DROID real-time profile, 30 for the LIBERO accuracy profile).
    pub(super) num_inference_steps: usize,
    execution_policy: ExecutionPolicy,
    implicit: RefCell<Option<Plan>>,
}
impl ModelRunner {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
        num_inference_steps: usize,
        execution_policy: ExecutionPolicy,
    ) -> Self {
        Self {
            ctx,
            weights,
            gemm,
            attention,
            num_inference_steps,
            execution_policy,
            implicit: RefCell::new(None),
        }
    }
}

impl VlaRuntime for ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: ACTION_SHAPE,
            patch_shape: [PIXEL_FRAMES * PIXEL_H * PIXEL_W, 3],
            max_token_len: 1,
            num_views: 1,
            image_size: PIXEL_H,
            patch_size: 1,
            accepts_rgb_u8: true,
        }
    }
    fn initial_latent_shape(&self) -> Vec<usize> {
        LATENT_SHAPE.to_vec()
    }
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        validate_request(request)?;
        if self.implicit.borrow().is_none() {
            let plan = self.plan(
                &request.observation.inference_spec(),
                self.execution_policy,
                Some(request),
            )?;
            *self.implicit.borrow_mut() = Some(plan);
        }
        let plan = self.implicit.borrow().as_ref().unwrap().clone();
        plan.run(request)
    }
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::PreferGraph)
    }
    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.plan(spec, policy, None)?))
    }
    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.plan(
            &sample.observation.inference_spec(),
            policy,
            Some(sample),
        )?))
    }
    fn clear_prepared(&self) -> Result<()> {
        *self.implicit.borrow_mut() = None;
        Ok(())
    }
    fn execution_mode(&self) -> &'static str {
        match self.implicit.borrow().as_ref().map(|p| &p.status) {
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                ..
            }) => "graph",
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                ..
            }) => "eager",
            _ => "unprepared",
        }
    }
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        let mut bytes = vec![0; ACTION_SHAPE[0] * ACTION_SHAPE[1] * 4];
        CudaBuffer::from_tensor(action.tensor())
            .map_err(Error::Cuda)?
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
}

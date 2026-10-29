//! Request binding and fixed-profile lifetime for the existing VLA interface.
use crate::cuda_cosmos_policy::backend::ops::GemmPolicy;
use crate::cuda_cosmos_policy::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_cosmos_policy::weights::Bf16Weights;
use crate::vla::*;
use apxinf_core::{DType, Device, Error, Result};
use std::{cell::RefCell, rc::Rc};

use super::prepare::Plan;
pub(super) fn invalid(message: &str) -> Error {
    Error::Other(format!("Cosmos Policy: {message}"))
}
pub(super) fn validate_spec(spec: &InferenceSpec) -> Result<()> {
    if spec.token_count != 1 || spec.image_layout != Some(ImageLayout::Nhwc) {
        return Err(invalid(
            "LIBERO direct requires one cached-task ID and NHWC RGB",
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
        } if bytes.len() == 2 * 224 * 224 * 3 => {}
        _ => return Err(invalid("expected two 224x224 RGB images")),
    }
    if o.token_ids[0] >= 40 {
        return Err(invalid("cached-task ID is outside the supported table"));
    }
    let state = o
        .state
        .as_ref()
        .ok_or_else(|| invalid("normalized proprioception is required"))?;
    if state.device() != Device::Cpu
        || state.dtype() != DType::F32
        || state.shape().dims() != [9]
        || state.as_f32()?.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("proprioception must be finite host FP32 [9]"));
    }
    if o.action_mask.is_some()
        || request.metadata.attention_mask.is_some()
        || request.metadata.image_grid_thw.is_some()
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
            || noise.shape().dims() != [1, 16, 9, 28, 28]
            || noise.as_f32()?.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "supplied noise must be finite host FP32 [1,16,9,28,28]",
            ));
        }
    }
    Ok(())
}
pub(in crate::cuda_cosmos_policy) struct ModelRunner {
    pub(super) ctx: Rc<CudaContext>,
    pub(super) weights: Rc<Bf16Weights>,
    pub(super) gemm: GemmPolicy,
    pub(super) attention: AttentionPolicy,
    implicit: RefCell<Option<Plan>>,
}
impl ModelRunner {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Self {
        Self {
            ctx,
            weights,
            gemm,
            attention,
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
            action_shape: [16, 7],
            patch_shape: [224 * 224 * 2, 3],
            max_token_len: 1,
            num_views: 2,
            image_size: 224,
            patch_size: 1,
            accepts_rgb_u8: true,
        }
    }
    fn initial_latent_shape(&self) -> Vec<usize> {
        vec![1, 16, 9, 28, 28]
    }
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        validate_request(request)?;
        if self.implicit.borrow().is_none() {
            let plan = self.plan(
                &request.observation.inference_spec(),
                ExecutionPolicy::RequireGraph,
                Some(request),
            )?;
            *self.implicit.borrow_mut() = Some(plan);
        }
        let plan = self.implicit.borrow().as_ref().unwrap().clone();
        plan.run(request)
    }
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::RequireGraph)
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
        let mut bytes = vec![0; 16 * 7 * 4];
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

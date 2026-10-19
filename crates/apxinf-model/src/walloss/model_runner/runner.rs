//! WallOSS model runner: implements `VlaRuntime`, owns preparation policy,
//! the bounded prepared-plan cache, and per-call input binding.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Backend, Error, Result, Tensor};

use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus,
    PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};

use super::super::backend::RuntimeBackend;
use super::super::geometry::multimodal_position_ids;
use super::super::model::{ModelVariant, WallossSchedule};
use super::super::WallossConfig;
use super::prepare::{rgb_bytes, PreparedShape, WallossPreparedInference};

/// Default request shape when a request declares no grids: the published
/// LIBERO profile, two 252x252 views (patch grid 18x18). It is also the shape
/// of the public fixed-size RGB contract and of explicit `prepare`.
const DEFAULT_GRIDS: [[usize; 3]; 2] = [[1, 18, 18], [1, 18, 18]];

/// Bounded implicit cache: one prepared plan per runner, as in PI0.5. A
/// request of another shape replaces it (re-preparation and re-capture);
/// callers that alternate shapes own plans explicitly via `prepare`.
const MAX_IMPLICIT_PLANS: usize = 1;

pub(super) enum WallossHostVision {
    Patches(Tensor),
    RgbU8(Vec<u8>),
}

pub(super) struct WallossHostInputs {
    pub(super) vision: WallossHostVision,
    pub(super) prefix_ids: Vec<u32>,
    pub(super) vision_row_map: Vec<u32>,
    pub(super) prefix_position_ids: Vec<u32>,
    pub(super) action_position_ids: Vec<u32>,
    pub(super) initial_state: Option<Tensor>,
    pub(super) action_mask: Tensor,
}

type PlanKey = (InferenceSpec, PreparedShape);

pub struct WallossModelRunner {
    backend: Arc<RuntimeBackend>,
    config: Arc<WallossConfig>,
    schedule: Arc<WallossSchedule>,
    model: Rc<ModelVariant>,
    prepared: RefCell<Vec<(PlanKey, Rc<WallossPreparedInference>)>>,
}

impl WallossModelRunner {
    pub(in crate::walloss) fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<WallossConfig>,
        schedule: Arc<WallossSchedule>,
        model: ModelVariant,
    ) -> Self {
        Self {
            backend,
            config,
            schedule,
            model: Rc::new(model),
            prepared: RefCell::new(Vec::new()),
        }
    }

    fn build(
        &self,
        spec: InferenceSpec,
        shape: PreparedShape,
        policy: ExecutionPolicy,
        first_request: Option<&VlaRequest<'_>>,
    ) -> Result<WallossPreparedInference> {
        WallossPreparedInference::build(
            &self.backend,
            &self.config,
            &self.schedule,
            &self.model,
            spec,
            shape,
            policy,
            first_request,
        )
    }

    /// The cached plan for the request's spec and grids, or a new one that
    /// replaces it (re-preparation and re-capture).
    fn plan_for(&self, request: &VlaRequest<'_>) -> Result<Rc<WallossPreparedInference>> {
        let key = (
            request.observation.inference_spec(),
            request_shape(&self.config, request)?,
        );
        let cached = self
            .prepared
            .borrow()
            .iter()
            .find(|(cached, _)| *cached == key)
            .map(|(_, plan)| Rc::clone(plan));
        if let Some(plan) = cached {
            return Ok(plan);
        }
        // Reject malformed host inputs while the healthy plan still exists.
        // Only a valid replacement may release its workspace; keeping both
        // workspaces alive while building would double the device budget.
        let geometry = self.model.model().geometry(&key.1.grids)?;
        if key.0.image_layout.is_some()
            && (geometry.rgb_views.is_none() || self.model.model().image_processor.is_none())
        {
            return Err(Error::Other(
                "walloss RGB input requires configured preprocessing and uniform still views"
                    .into(),
            ));
        }
        prepare_host_inputs(
            &key.0,
            &key.1,
            &self.config,
            geometry.patch_order.len(),
            request,
        )?;
        {
            // Release the old plan before building its replacement so two
            // graph workspaces never coexist.
            let mut cache = self.prepared.borrow_mut();
            while cache.len() >= MAX_IMPLICIT_PLANS {
                drop(cache.remove(0));
            }
        }
        self.backend.synchronize()?;
        let plan = Rc::new(self.build(
            key.0,
            key.1.clone(),
            ExecutionPolicy::PreferGraph,
            Some(request),
        )?);
        self.prepared.borrow_mut().push((key, Rc::clone(&plan)));
        Ok(plan)
    }

    fn default_contract_profile(&self) -> (usize, usize) {
        let [_, h, w] = DEFAULT_GRIDS[0];
        debug_assert_eq!(h, w);
        (DEFAULT_GRIDS.len(), h * self.config.vision.patch_size)
    }
}

/// The grids a request declares through `VlaMetadata::image_grid_thw`, or the
/// default profile.
fn request_shape(config: &WallossConfig, request: &VlaRequest<'_>) -> Result<PreparedShape> {
    let Some(grids) = request.metadata.image_grid_thw else {
        return Ok(PreparedShape {
            grids: DEFAULT_GRIDS.to_vec(),
        });
    };
    if grids.is_empty() {
        return Err(Error::Other(
            "walloss image_grid_thw cannot be empty".into(),
        ));
    }
    let merge = config.vision.spatial_merge_size;
    let grids = grids
        .iter()
        .map(|&[t, h, w]| {
            let [t, h, w] = [t as usize, h as usize, w as usize];
            if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(Error::Other(format!(
                    "walloss vision grid [{t}, {h}, {w}] must be positive and divisible by merge size {merge}"
                )));
            }
            Ok([t, h, w])
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PreparedShape { grids })
}

/// Validate one request against a prepared plan and build its host inputs.
pub(super) fn prepare_host_inputs(
    spec: &InferenceSpec,
    shape: &PreparedShape,
    config: &WallossConfig,
    patch_rows: usize,
    request: &VlaRequest<'_>,
) -> Result<WallossHostInputs> {
    let observation = request.observation;
    observation.validate()?;
    spec.validate()?;
    if spec.token_count <= config.action.action_horizon {
        return Err(Error::Other(
            "walloss token sequence must contain a language prefix and action suffix".into(),
        ));
    }
    if *spec != observation.inference_spec() {
        return Err(Error::Other(format!(
            "prepared walloss spec {spec:?} does not match observation {:?}",
            observation.inference_spec()
        )));
    }
    let requested = request_shape(config, request)?;
    if requested != *shape {
        return Err(Error::Other(format!(
            "walloss plan was prepared for grids {:?}, request declares {:?}",
            shape.grids, requested.grids
        )));
    }
    let token_ids = &observation.token_ids;
    if token_ids.len() > config.text.max_position_embeddings {
        return Err(Error::Other(format!(
            "walloss sequence of {} tokens exceeds max_position_embeddings {}",
            token_ids.len(),
            config.text.max_position_embeddings
        )));
    }
    let merge = config.vision.spatial_merge_size;
    let image_tokens: usize = shape
        .grids
        .iter()
        .map(|&[t, h, w]| t * (h / merge) * (w / merge))
        .sum();
    let action_tokens = config.action.action_horizon;
    let prefix_tokens = token_ids.len() - action_tokens;
    let prefix_ids = token_ids[..prefix_tokens].to_vec();
    let mut vision_row_map = vec![u32::MAX; prefix_tokens];
    let mut vision_row = 0u32;
    for (row, &token) in prefix_ids.iter().enumerate() {
        if token == config.image_token_id {
            vision_row_map[row] = vision_row;
            vision_row += 1;
        }
    }
    if vision_row as usize != image_tokens {
        return Err(Error::Other(format!(
            "walloss prompt has {vision_row} image tokens, grids {:?} expect {image_tokens}",
            shape.grids
        )));
    }
    // Also checks that each image-token run matches its grid.
    let position_ids =
        multimodal_position_ids(token_ids, &shape.grids, config.image_token_id, merge)?;
    let vision = match &observation.vision {
        VisionObservation::Patches(value) => WallossHostVision::Patches(normalize_host_bf16(
            value,
            vec![
                patch_rows,
                3 * config.vision.temporal_patch_size
                    * config.vision.patch_size
                    * config.vision.patch_size,
            ],
            "patches",
        )?),
        VisionObservation::RgbU8 { bytes, .. } => {
            let expected = rgb_bytes(config, shape).unwrap_or(usize::MAX);
            if bytes.len() != expected {
                return Err(Error::Other(format!(
                    "walloss expected {expected} RGB bytes for grids {:?}, got {}",
                    shape.grids,
                    bytes.len()
                )));
            }
            WallossHostVision::RgbU8(bytes.clone())
        }
    };
    let action_shape = vec![action_tokens, config.action.action_dim];
    let initial_state = match request.initial_latent {
        InitialLatent::Provided(value) => Some(normalize_host_bf16(
            value,
            action_shape.clone(),
            "initial latent",
        )?),
        InitialLatent::Generate { .. } => None,
    };
    let action_mask = match observation.action_mask.as_ref() {
        Some(value) => normalize_host_bf16(value, action_shape, "action mask")?,
        None => Tensor::from_bf16(
            action_shape,
            &vec![half::bf16::ONE; action_tokens * config.action.action_dim],
        )?,
    };
    Ok(WallossHostInputs {
        vision,
        prefix_ids,
        vision_row_map,
        prefix_position_ids: position_ids[..prefix_tokens * 3].to_vec(),
        action_position_ids: position_ids[prefix_tokens * 3..].to_vec(),
        initial_state,
        action_mask,
    })
}

fn normalize_host_bf16(value: &Tensor, shape: Vec<usize>, name: &str) -> Result<Tensor> {
    if value.shape().dims() != shape {
        return Err(Error::Other(format!(
            "walloss {name} shape {:?}, expected {shape:?}",
            value.shape().dims()
        )));
    }
    let values = value
        .to_f32_vec()?
        .into_iter()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    Tensor::from_bf16(shape, &values)
}

impl VlaRuntime for WallossModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some(self.model.name())
    }

    /// The fixed-size fields describe the default profile; requests may
    /// declare other grids through `VlaMetadata::image_grid_thw`.
    fn contract(&self) -> VlaContract {
        let (num_views, image_size) = self.default_contract_profile();
        let model = self.model.model();
        VlaContract {
            action_shape: [
                self.config.action.action_horizon,
                self.config.action.action_dim,
            ],
            patch_shape: [
                DEFAULT_GRIDS.iter().map(|&[t, h, w]| t * h * w).sum(),
                model.patch_width(),
            ],
            max_token_len: self.config.text.max_position_embeddings,
            num_views,
            image_size,
            patch_size: self.config.vision.patch_size,
            accepts_rgb_u8: model.image_processor.is_some(),
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.plan_for(request)?.run(request)
    }

    fn infer_host_f32_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Vec<f32>, BTreeMap<String, f64>)> {
        let (action, profile) = self.plan_for(request)?.run_profiled(request)?;
        Ok((self.backend.to_cpu(action.tensor())?.to_f32_vec()?, profile))
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::PreferGraph)
    }

    /// Explicit plans serve the default profile; requests with other grids
    /// use the implicit path.
    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.build(
            *spec,
            PreparedShape {
                grids: DEFAULT_GRIDS.to_vec(),
            },
            policy,
            None,
        )?))
    }

    fn execution_mode(&self) -> &'static str {
        match self.prepared.borrow().last().map(|(_, plan)| plan.status()) {
            None => "unprepared",
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                ..
            }) => "graph",
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                ..
            }) => "eager",
            Some(_) => "runtime-managed",
        }
    }

    fn clear_prepared(&self) -> Result<()> {
        self.backend.synchronize()?;
        self.prepared.borrow_mut().clear();
        Ok(())
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let output = self.infer(request)?;
        self.backend.to_cpu(output.tensor())?.to_f32_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::LoadOptions;
    use crate::vla::{ImageLayout, Observation};
    use apxinf_core::DType;

    #[test]
    #[ignore = "requires CUDA and APXINF_WALLOSS_TEST_MODEL checkpoint"]
    fn long_prefix_preparation_has_sufficient_workspace() {
        let path = std::env::var("APXINF_WALLOSS_TEST_MODEL").expect("checkpoint is required");
        let backend = Arc::new(RuntimeBackend::new(0).unwrap());
        let runner = super::super::super::load::load_model_runner(
            std::path::Path::new(&path),
            backend,
            &LoadOptions::default(),
        )
        .unwrap();
        let plan = runner
            .prepare_with_policy(
                &InferenceSpec {
                    token_count: 32768 + runner.config.action.action_horizon,
                    image_layout: Some(ImageLayout::Nhwc),
                },
                ExecutionPolicy::Eager,
            )
            .unwrap();
        assert_eq!(
            plan.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                fallback_reason: None
            }
        );
    }

    #[test]
    #[ignore = "requires CUDA and APXINF_WALLOSS_TEST_MODEL checkpoint"]
    fn rejected_requests_preserve_cached_plan_and_actions() {
        let path = std::env::var("APXINF_WALLOSS_TEST_MODEL").expect("checkpoint is required");
        let backend = Arc::new(RuntimeBackend::new(0).unwrap());
        let runner = super::super::super::load::load_model_runner(
            std::path::Path::new(&path),
            backend.clone(),
            &LoadOptions::default(),
        )
        .unwrap();
        let contract = runner.contract();
        let mut token_ids = vec![runner.config.image_token_id; 81];
        token_ids.push(1);
        token_ids.extend(vec![runner.config.image_token_id; 81]);
        token_ids.extend(vec![1; runner.config.action.action_horizon]);
        let good = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: vec![
                    127;
                    contract.num_views * contract.image_size * contract.image_size * 3
                ],
                layout: ImageLayout::Nhwc,
            },
            token_ids,
            state: None,
            action_mask: None,
        };
        let latent = Tensor::zeros(contract.action_shape.to_vec(), DType::BF16);
        let request = VlaRequest::provided(&good, &latent);
        let plan = runner.plan_for(&request).unwrap();
        if std::env::var_os("APXINF_WALLOSS_NO_GRAPH").is_some() {
            assert!(matches!(plan.status(), PreparationStatus::Ready {
                mode: ExecutionMode::Eager, fallback_reason: Some(reason),
            } if reason.contains("APXINF_WALLOSS_NO_GRAPH")));
            assert!(runner
                .prepare_with_policy(&good.inference_spec(), ExecutionPolicy::RequireGraph)
                .is_err());
        } else {
            assert_eq!(
                plan.status(),
                PreparationStatus::Ready {
                    mode: ExecutionMode::Graph,
                    fallback_reason: None,
                }
            );
        }
        let before = backend
            .to_cpu(plan.run(&request).unwrap().tensor())
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let mut bad = good.clone();
        bad.token_ids.truncate(runner.config.action.action_horizon);
        assert!(runner
            .plan_for(&VlaRequest::provided(&bad, &latent))
            .is_err());
        assert!(Rc::ptr_eq(&runner.prepared.borrow()[0].1, &plan));
        bad = good.clone();
        bad.token_ids.push(1); // New key, but invalid RGB buffer.
        if let VisionObservation::RgbU8 { bytes, .. } = &mut bad.vision {
            bytes.pop();
        }
        assert!(runner
            .plan_for(&VlaRequest::provided(&bad, &latent))
            .is_err());
        assert!(Rc::ptr_eq(&runner.prepared.borrow()[0].1, &plan));
        bad = good.clone();
        bad.token_ids.push(1);
        let wrong_latent = Tensor::zeros((1, 1), DType::BF16);
        assert!(runner
            .plan_for(&VlaRequest::provided(&bad, &wrong_latent))
            .is_err());
        assert!(Rc::ptr_eq(&runner.prepared.borrow()[0].1, &plan));
        bad.action_mask = Some(Tensor::zeros((1, 1), DType::BF16));
        assert!(runner
            .plan_for(&VlaRequest::provided(&bad, &latent))
            .is_err());
        assert!(Rc::ptr_eq(&runner.prepared.borrow()[0].1, &plan));
        for token_count in [
            0,
            runner.config.action.action_horizon,
            runner.config.text.max_position_embeddings + 1,
        ] {
            let invalid = InferenceSpec {
                token_count,
                image_layout: Some(ImageLayout::Nhwc),
            };
            assert!(runner
                .prepare_with_policy(&invalid, ExecutionPolicy::Eager)
                .is_err());
        }
        assert!(Rc::ptr_eq(&runner.plan_for(&request).unwrap(), &plan));
        let after = backend
            .to_cpu(plan.run(&request).unwrap().tensor())
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert!(before.iter().all(|x| x.is_finite()));
        assert_eq!(before, after);
    }
}

//! WallOSS prepared inference: stable buffers, captured graph ownership, and
//! the explicit preparation contract.
//!
//! Preparation allocates every graph-visible input at a stable address, runs
//! one eager pass on placeholder inputs to resolve GEMM plans and size the
//! workspace, and captures the CUDA graph unless the policy is `Eager`.
//! `run` neither captures nor tunes: it binds per-call inputs in place and
//! replays (or runs eagerly).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Error, Graph, NormalGenerator, Result, SamplingBackend, Tensor};

use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus,
    PreparedInference, VlaRequest,
};

use super::super::backend::{ops, transfers, CudaEventTimer, DeviceBuffer, RuntimeBackend};
use super::super::geometry::{upload_u32, DeviceVisionGeometry};
use super::super::model::{
    ModelVariant, WallossDeviceInputs, WallossDeviceVision, WallossSchedule,
};
use super::super::WallossConfig;
use super::runner::{prepare_host_inputs, WallossHostInputs, WallossHostVision};

/// Vision grids a prepared plan is built for, `[t, h, w]` patches per image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PreparedShape {
    pub grids: Vec<[usize; 3]>,
}

struct CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
}

/// Owning prepared WallOSS inference plan. Graph actions alias reusable
/// device output: copy them before the next run if a stable result is
/// needed. Runs are serialized on this plan.
pub struct WallossPreparedInference {
    spec: InferenceSpec,
    shape: PreparedShape,
    backend: Arc<RuntimeBackend>,
    config: Arc<WallossConfig>,
    schedule: Arc<WallossSchedule>,
    model: Rc<ModelVariant>,
    geometry: DeviceVisionGeometry,
    inputs: WallossDeviceInputs,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
    workspace: ops::ExecutionSession,
    graph: Option<CapturedGraph>,
    fallback_reason: Option<String>,
}

/// Choose the execution mode for `policy`; `PreferGraph` falls back to eager
/// with the capture error as the recorded reason.
pub(super) fn select_graph<G>(
    policy: ExecutionPolicy,
    capture: impl FnOnce() -> Result<G>,
) -> Result<(Option<G>, Option<String>)> {
    match policy {
        ExecutionPolicy::Eager => Ok((None, None)),
        ExecutionPolicy::RequireGraph => capture().map(|graph| (Some(graph), None)),
        ExecutionPolicy::PreferGraph => match capture() {
            Ok(graph) => Ok((Some(graph), None)),
            Err(error) => Ok((None, Some(error.to_string()))),
        },
    }
}

/// An environment opt-out may relax a preference, never a requirement.
fn effective_policy(
    policy: ExecutionPolicy,
    disabled: bool,
) -> Result<(ExecutionPolicy, Option<String>)> {
    if !disabled || policy == ExecutionPolicy::Eager {
        return Ok((policy, None));
    }
    let reason = "APXINF_WALLOSS_NO_GRAPH disables CUDA graph capture";
    if policy == ExecutionPolicy::RequireGraph {
        return Err(Error::Other(format!(
            "RequireGraph conflicts with {reason}"
        )));
    }
    Ok((ExecutionPolicy::Eager, Some(reason.into())))
}

impl WallossPreparedInference {
    pub(super) fn build(
        backend: &Arc<RuntimeBackend>,
        config: &Arc<WallossConfig>,
        schedule: &Arc<WallossSchedule>,
        model: &Rc<ModelVariant>,
        spec: InferenceSpec,
        shape: PreparedShape,
        policy: ExecutionPolicy,
        first_request: Option<&VlaRequest<'_>>,
    ) -> Result<Self> {
        let (policy, disabled_reason) = effective_policy(
            policy,
            std::env::var_os("APXINF_WALLOSS_NO_GRAPH").is_some(),
        )?;
        spec.validate()?;
        let action_tokens = config.action.action_horizon;
        if spec.token_count <= action_tokens {
            return Err(Error::Other(
                "walloss token sequence must contain a language prefix and action suffix".into(),
            ));
        }
        if spec.token_count > config.text.max_position_embeddings {
            return Err(Error::Other(format!(
                "walloss sequence of {} tokens exceeds max_position_embeddings {}",
                spec.token_count, config.text.max_position_embeddings,
            )));
        }
        let host_geometry = model.model().geometry(&shape.grids)?;
        if spec.image_layout.is_some() && host_geometry.rgb_views.is_none() {
            return Err(Error::Other(format!(
                "walloss RGB input requires still views of one size, got grids {:?}; send patches instead",
                shape.grids
            )));
        }
        let geometry = host_geometry.upload(backend.context())?;
        let device = backend.context().device_id();
        let patch_rows = host_geometry.patch_order.len();
        let prefix_tokens = spec.token_count - action_tokens;
        let vision = match spec.image_layout {
            None => WallossDeviceVision::Patches,
            Some(layout) => {
                let bytes = rgb_bytes(config, &shape)
                    .ok_or_else(|| Error::Other("walloss RGB byte count overflows".into()))?;
                WallossDeviceVision::RgbU8 {
                    raw_images: DeviceBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?,
                    layout,
                }
            }
        };
        let action_shape = (action_tokens, config.action.action_dim);
        let inputs = WallossDeviceInputs {
            patches: backend.to_device(&Tensor::zeros(
                (patch_rows, model.model().patch_width()),
                DType::BF16,
            ))?,
            vision,
            prefix_ids: upload_u32(device, &vec![0; prefix_tokens])?,
            vision_row_map: upload_u32(device, &vec![0; prefix_tokens])?,
            prefix_position_ids: upload_u32(device, &vec![0; prefix_tokens * 3])?,
            action_position_ids: upload_u32(device, &vec![0; action_tokens * 3])?,
            initial_state: backend.to_device(&Tensor::zeros(action_shape, DType::BF16))?,
            action_mask: backend.to_device(&Tensor::from_bf16(
                vec![action_shape.0, action_shape.1],
                &vec![half::bf16::ONE; action_shape.0 * action_shape.1],
            )?)?,
            prefix_tokens,
        };
        let normal_generator = backend.create_normal_generator(inputs.initial_state.clone())?;
        let requirements = model
            .model()
            .workspace_requirements(patch_rows, prefix_tokens)?;
        let workspace = ops::ExecutionSession::with_capacity(requirements.bytes, device)?;
        let mut plan = Self {
            spec,
            shape,
            backend: Arc::clone(backend),
            config: Arc::clone(config),
            schedule: Arc::clone(schedule),
            model: Rc::clone(model),
            geometry,
            inputs,
            normal_generator: RefCell::new(normal_generator),
            workspace,
            graph: None,
            fallback_reason: None,
        };
        plan.warm_up(first_request)?;
        (plan.graph, plan.fallback_reason) = select_graph(policy, || plan.capture())?;
        plan.fallback_reason = plan.fallback_reason.or(disabled_reason);
        if let Some(reason) = &plan.fallback_reason {
            eprintln!("[apxinf] WallOSS graph capture unavailable, using eager: {reason}");
        }
        Ok(plan)
    }

    /// One eager pass before capture resolves every GEMM plan. With the
    /// request that created the plan, it runs on real inputs so missing GEMM
    /// keys can be autotuned; explicit preparation runs on placeholders.
    fn warm_up(&self, request: Option<&VlaRequest<'_>>) -> Result<()> {
        let forward = || {
            self.model
                .model()
                .infer(&self.schedule, &self.inputs, &self.geometry)
        };
        let output = match request {
            Some(request) => {
                let host = prepare_host_inputs(
                    &self.spec,
                    &self.shape,
                    &self.config,
                    self.inputs.patches.shape().dims()[0],
                    request,
                )?;
                self.bind(&host, request)?;
                ops::prepare_with_session(&self.workspace, forward)?
            }
            None => ops::prepare_with_session(&self.workspace, forward)?,
        };
        self.backend.synchronize()?;
        drop(output);
        Ok(())
    }

    fn capture(&self) -> Result<CapturedGraph> {
        self.backend.begin_capture()?;
        let output = match ops::with_session(&self.workspace, || {
            self.model
                .model()
                .infer(&self.schedule, &self.inputs, &self.geometry)
        }) {
            Ok(output) => output,
            Err(error) => {
                let _ = self.backend.end_capture();
                return Err(error);
            }
        };
        let graph = self.backend.end_capture()?;
        Ok(CapturedGraph { graph, output })
    }

    /// Diagnostic run: stage GPU times from one eager pass (vision includes RGB
    /// preprocessing; eager denoise includes launch gaps), plus the graph
    /// replay time when a graph is captured. The actions come from the graph.
    pub(super) fn run_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Action, BTreeMap<String, f64>)> {
        let host = prepare_host_inputs(
            &self.spec,
            &self.shape,
            &self.config,
            self.inputs.patches.shape().dims()[0],
            request,
        )?;
        self.bind(&host, request)?;
        let (eager, mut profile) = ops::prepare_with_session(&self.workspace, || {
            self.model
                .model()
                .infer_profiled(&self.schedule, &self.inputs, &self.geometry)
        })?;
        let mib = |bytes: usize| bytes as f64 / f64::from(1u32 << 20);
        profile.insert(
            "workspace_peak_live_mib".to_string(),
            mib(self.workspace.used()),
        );
        profile.insert(
            "workspace_capacity_mib".to_string(),
            mib(self.workspace.capacity()),
        );
        let Some(captured) = &self.graph else {
            return Ok((Action::new(eager), profile));
        };
        drop(eager);
        let timer = CudaEventTimer::new()?;
        timer.start(self.backend.context())?;
        captured.graph.replay()?;
        timer.stop(self.backend.context())?;
        profile.insert("graph_ms".to_string(), timer.elapsed_ms()?);
        Ok((Action::new(captured.output.clone()), profile))
    }

    /// Copy one request's inputs into the stable device buffers.
    fn bind(&self, host: &WallossHostInputs, request: &VlaRequest<'_>) -> Result<()> {
        let inputs = &self.inputs;
        self.backend.synchronize()?;
        match (&host.vision, &inputs.vision) {
            (WallossHostVision::Patches(patches), WallossDeviceVision::Patches) => {
                transfers::copy_cpu_to_cuda(patches, &inputs.patches)?;
            }
            (WallossHostVision::RgbU8(bytes), WallossDeviceVision::RgbU8 { raw_images, .. }) => {
                raw_images.copy_from_host(bytes).map_err(Error::Cuda)?;
            }
            _ => unreachable!("the prepared spec fixes the vision input mode"),
        }
        transfers::copy_cpu_to_cuda(&host.action_mask, &inputs.action_mask)?;
        copy_u32(&inputs.prefix_ids, &host.prefix_ids)?;
        copy_u32(&inputs.vision_row_map, &host.vision_row_map)?;
        copy_u32(&inputs.prefix_position_ids, &host.prefix_position_ids)?;
        copy_u32(&inputs.action_position_ids, &host.action_position_ids)?;
        match (&request.initial_latent, &host.initial_state) {
            (InitialLatent::Provided(_), Some(state)) => {
                transfers::copy_cpu_to_cuda(state, &inputs.initial_state)
            }
            (InitialLatent::Generate { rng }, _) => self
                .normal_generator
                .borrow_mut()
                .generate(*rng)
                .map(|_| ()),
            (InitialLatent::Provided(_), None) => {
                unreachable!("host inputs carry every provided latent")
            }
        }
    }
}

impl PreparedInference for WallossPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: if self.graph.is_some() {
                ExecutionMode::Graph
            } else {
                ExecutionMode::Eager
            },
            fallback_reason: self.fallback_reason.clone(),
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let host = prepare_host_inputs(
            &self.spec,
            &self.shape,
            &self.config,
            self.inputs.patches.shape().dims()[0],
            request,
        )?;
        self.bind(&host, request)?;
        let output = match &self.graph {
            Some(captured) => {
                captured.graph.replay()?;
                captured.output.clone()
            }
            None => ops::prepare_with_session(&self.workspace, || {
                self.model
                    .model()
                    .infer(&self.schedule, &self.inputs, &self.geometry)
            })?,
        };
        Ok(Action::new(output))
    }
}

/// Packed `u8` byte count of the RGB views for `shape`, or `None` on overflow.
pub(super) fn rgb_bytes(config: &WallossConfig, shape: &PreparedShape) -> Option<usize> {
    let patch = config.vision.patch_size;
    shape.grids.iter().try_fold(0usize, |total, &[t, h, w]| {
        t.checked_mul(h)?
            .checked_mul(w)?
            .checked_mul(patch * patch * 3)
            .and_then(|bytes| total.checked_add(bytes))
    })
}

fn copy_u32(buffer: &DeviceBuffer, values: &[u32]) -> Result<()> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    if bytes.len() != buffer.len() {
        return Err(Error::Other(format!(
            "walloss prepared u32 input has {} bytes, expected {}",
            bytes.len(),
            buffer.len()
        )));
    }
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_opt_out_preserves_explicit_requirements() {
        assert!(effective_policy(ExecutionPolicy::RequireGraph, true)
            .unwrap_err()
            .to_string()
            .contains("RequireGraph"));
        let (policy, reason) = effective_policy(ExecutionPolicy::PreferGraph, true).unwrap();
        assert_eq!(policy, ExecutionPolicy::Eager);
        assert!(reason.unwrap().contains("APXINF_WALLOSS_NO_GRAPH"));
        assert_eq!(
            effective_policy(ExecutionPolicy::Eager, true).unwrap(),
            (ExecutionPolicy::Eager, None)
        );
        for policy in [
            ExecutionPolicy::Eager,
            ExecutionPolicy::PreferGraph,
            ExecutionPolicy::RequireGraph,
        ] {
            assert_eq!(effective_policy(policy, false).unwrap(), (policy, None));
        }
    }

    #[test]
    fn graph_policy_selection_and_fallback() {
        let failure = || -> Result<u8> { Err(Error::Other("capture failed".into())) };
        assert!(select_graph(ExecutionPolicy::RequireGraph, failure).is_err());
        let (graph, reason) = select_graph(ExecutionPolicy::PreferGraph, failure).unwrap();
        assert!(graph.is_none());
        assert!(reason.unwrap().contains("capture failed"));
        let (graph, reason) =
            select_graph::<u8>(ExecutionPolicy::Eager, || panic!("eager must not capture"))
                .unwrap();
        assert!(graph.is_none() && reason.is_none());
        assert_eq!(
            select_graph(ExecutionPolicy::RequireGraph, || Ok(7)).unwrap(),
            (Some(7), None)
        );
    }
}

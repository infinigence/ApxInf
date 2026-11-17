#[cfg(feature = "musa")]
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::{Rc, Weak};

use apxinf_core::{DType, Device, Error, Result, Tensor};

#[cfg(feature = "musa")]
use crate::vla::ExecutionMode;
use crate::vla::{
    Action, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus, PreparedInference,
    VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};

use super::super::backend::kernels;
#[cfg(feature = "musa")]
use super::super::geometry::Gr00tBackboneTokenGroups;
#[cfg(feature = "musa")]
use super::super::model::executor::{Gr00tCapturedGraph, Gr00tGraphKey, Gr00tPreparedInputs};
use super::super::model::executor::{Gr00tExecutor, Gr00tObservation, Gr00tPrecisionExecution};
use super::super::Gr00tConfig;

pub(crate) struct Gr00tModelRunner<E: Gr00tPrecisionExecution> {
    engine: Rc<RefCell<Gr00tExecutor<E>>>,
    /// The implicit fixed-profile MUSA plan used by the convenient `infer`
    /// entry point. CUDA retains its existing executor-managed graph cache.
    #[cfg(feature = "musa")]
    prepared: RefCell<Option<(Gr00tGraphKey, Rc<Gr00tPreparedInference<E>>)>>,
    #[cfg(feature = "musa")]
    execution_policy: Cell<ExecutionPolicy>,
    /// Tracks the sole resource-owning plan, including externally held handles.
    active_strategy: Rc<Gr00tActiveStrategy>,
}

pub(super) struct Gr00tPreparedInference<E: Gr00tPrecisionExecution> {
    spec: InferenceSpec,
    engine: Rc<RefCell<Gr00tExecutor<E>>>,
    strategy: Rc<RefCell<Gr00tPreparedStrategy>>,
    active_strategy: Rc<Gr00tActiveStrategy>,
}

#[derive(Default)]
struct Gr00tActiveStrategy {
    strategy: RefCell<Weak<RefCell<Gr00tPreparedStrategy>>>,
}

impl Gr00tActiveStrategy {
    fn register(&self, strategy: &Rc<RefCell<Gr00tPreparedStrategy>>) {
        *self.strategy.borrow_mut() = Rc::downgrade(strategy);
    }

    /// Retire the active plan and return its arena for reuse.
    ///
    /// Synchronization must succeed before the graph and its arena are
    /// destroyed. The retired payload is dropped only after the plan's
    /// `RefCell` borrow ends, so graph destruction never runs under it.
    #[cfg(feature = "musa")]
    fn retire(
        &self,
        synchronize: impl FnOnce() -> Result<()>,
    ) -> Result<Option<kernels::GraphWorkspace>> {
        let active = self.strategy.borrow().upgrade();
        let Some(active) = active else {
            *self.strategy.borrow_mut() = Weak::new();
            return Ok(None);
        };
        synchronize()?;
        let retired = {
            let mut active = active.try_borrow_mut().map_err(|_| {
                Error::Other(
                    "GR00T prepared plan is in use; retry after the current run".into(),
                )
            })?;
            std::mem::replace(&mut *active, Gr00tPreparedStrategy::Invalidated)
        };
        *self.strategy.borrow_mut() = Weak::new();
        Ok(retired.take_workspace())
    }

    fn invalidate(&self, synchronize: impl FnOnce() -> Result<()>) -> Result<()> {
        #[cfg(feature = "musa")]
        {
            self.retire(synchronize).map(|_| ())
        }
        #[cfg(not(feature = "musa"))]
        {
            let active = self.strategy.borrow().upgrade();
            if let Some(active) = active {
                synchronize()?;
                *active.borrow_mut() = Gr00tPreparedStrategy::Invalidated;
            }
            *self.strategy.borrow_mut() = Weak::new();
            Ok(())
        }
    }
}

enum Gr00tPreparedStrategy {
    Invalidated,
    /// Compatibility for the legacy shape-only `prepare` entry point, whose
    /// public spec does not carry enough GR00T geometry to allocate a plan.
    RuntimeManaged,
    #[cfg(feature = "musa")]
    Graph {
        key: Gr00tGraphKey,
        graph: Gr00tCapturedGraph,
    },
    #[cfg(feature = "musa")]
    NoGraph {
        key: Gr00tGraphKey,
        token_groups: Gr00tBackboneTokenGroups,
        inputs: Gr00tPreparedInputs,
        workspace: kernels::GraphWorkspace,
        fallback_reason: Option<String>,
    },
}

#[cfg(feature = "musa")]
impl Gr00tPreparedStrategy {
    /// Consume the plan and return its arena, if it owns one.
    ///
    /// A captured graph is destroyed first, so the returned arena carries no
    /// addresses recorded by the retired plan.
    fn take_workspace(self) -> Option<kernels::GraphWorkspace> {
        match self {
            Gr00tPreparedStrategy::Invalidated | Gr00tPreparedStrategy::RuntimeManaged => None,
            Gr00tPreparedStrategy::Graph { graph, .. } => Some(graph.into_workspace()),
            Gr00tPreparedStrategy::NoGraph { workspace, .. } => Some(workspace),
        }
    }
}

impl<E: Gr00tPrecisionExecution> Gr00tModelRunner<E> {
    pub(crate) fn new(engine: Gr00tExecutor<E>) -> Self {
        Self {
            engine: Rc::new(RefCell::new(engine)),
            #[cfg(feature = "musa")]
            prepared: RefCell::new(None),
            #[cfg(feature = "musa")]
            execution_policy: Cell::new(ExecutionPolicy::PreferGraph),
            active_strategy: Rc::new(Gr00tActiveStrategy::default()),
        }
    }

    pub(crate) fn with_execution_policy(self, policy: ExecutionPolicy) -> Self {
        self.execution_policy.set(policy);
        self
    }

    fn build_input(request: &VlaRequest<'_>, config: &Gr00tConfig) -> Result<Gr00tObservation> {
        let pixel_values = match &request.observation.vision {
            VisionObservation::Patches(tensor) => to_bf16(tensor, "pixel_values")?,
            VisionObservation::RgbU8 { .. } => {
                return Err(Error::Other(
                    "GR00T expects pixel_values from the checkpoint's official processor, not raw RGB"
                        .into(),
                ));
            }
        };
        if pixel_values.shape().dims().len() != 2 {
            return Err(Error::Other(format!(
                "GR00T pixel_values must be rank 2, got {:?}",
                pixel_values.shape().dims()
            )));
        }

        let image_grid_thw = request
            .metadata
            .image_grid_thw
            .ok_or_else(|| Error::Other("GR00T requires VlaMetadata.image_grid_thw".into()))?;
        #[cfg(feature = "musa")]
        if image_grid_thw.len() != 2 {
            return Err(Error::Other(format!(
                "GR00T MUSA minimal BF16 slice requires exactly two camera grids, got {}",
                image_grid_thw.len()
            )));
        }
        let attention_mask = request
            .metadata
            .attention_mask
            .ok_or_else(|| Error::Other("GR00T requires VlaMetadata.attention_mask".into()))?;
        let embodiment_id = request
            .metadata
            .embodiment_id
            .ok_or_else(|| Error::Other("GR00T requires VlaMetadata.embodiment_id".into()))?;
        let state = request.observation.state.as_ref().ok_or_else(|| {
            Error::Other("GR00T requires a normalized, padded state tensor".into())
        })?;
        let state = normalize_batched_bf16(
            state,
            config.state_history_length,
            config.max_state_dim,
            "state",
        )?;
        let latent = match request.initial_latent {
            InitialLatent::Provided(latent) => latent,
            InitialLatent::Generate { .. } => {
                return Err(Error::Other(
                    "GR00T requires processor-owned initial noise; pass an explicit latent".into(),
                ));
            }
        };
        let noise = normalize_batched_bf16(
            latent,
            config.action_horizon,
            config.max_action_dim,
            "initial latent",
        )?;

        Ok(Gr00tObservation {
            pixel_values,
            image_grid_thw: image_grid_thw.to_vec(),
            token_ids: request.observation.token_ids.clone(),
            attention_mask: attention_mask.to_vec(),
            state,
            embodiment_id,
            noise,
        })
    }

    fn run_runtime_managed(&self, request: &VlaRequest<'_>) -> Result<Action> {
        run_engine(&self.engine, &self.active_strategy, request)
    }

    #[cfg(feature = "musa")]
    fn build_prepared_no_graph(
        &self,
        input: &Gr00tObservation,
        fallback_reason: Option<String>,
        reused: Option<kernels::GraphWorkspace>,
    ) -> Result<Gr00tPreparedInference<E>> {
        let engine = self.engine.borrow();
        let key = engine.prepared_key(input)?;
        let token_groups = engine.prepared_token_groups(input)?;
        let inputs = engine.allocate_prepared_inputs(input)?;
        let workspace = match reused {
            Some(workspace) => workspace,
            None => engine.allocate_prepared_workspace()?,
        };
        drop(engine);
        Ok(Gr00tPreparedInference {
            spec: InferenceSpec {
                token_count: input.token_ids.len(),
                image_layout: None,
            },
            engine: Rc::clone(&self.engine),
            active_strategy: Rc::clone(&self.active_strategy),
            strategy: Rc::new(RefCell::new(Gr00tPreparedStrategy::NoGraph {
                key,
                token_groups,
                inputs,
                workspace,
                fallback_reason,
            })),
        })
    }

    /// Release executable resources even when a caller retains the old handle.
    /// Synchronization must succeed before destroying a graph or its arena.
    fn invalidate_prepared(&self) -> Result<()> {
        self.active_strategy
            .invalidate(|| self.engine.borrow().backend().synchronize())
    }

    #[cfg(feature = "musa")]
    fn build_prepared(
        &self,
        input: &Gr00tObservation,
        policy: ExecutionPolicy,
    ) -> Result<Gr00tPreparedInference<E>> {
        // Validate before retiring a usable plan. Once construction starts,
        // failure leaves no active plan; a later request can retry safely.
        self.engine.borrow().prepared_key(input)?;
        let reused = self
            .active_strategy
            .retire(|| self.engine.borrow().backend().synchronize())?;
        let plan = self.build_prepared_impl(input, policy, reused)?;
        self.active_strategy.register(&plan.strategy);
        Ok(plan)
    }

    #[cfg(feature = "musa")]
    fn build_prepared_impl(
        &self,
        input: &Gr00tObservation,
        policy: ExecutionPolicy,
        reused: Option<kernels::GraphWorkspace>,
    ) -> Result<Gr00tPreparedInference<E>> {
        if policy == ExecutionPolicy::Eager {
            return self.build_prepared_no_graph(input, None, reused);
        }

        let captured = self
            .engine
            .borrow_mut()
            .capture_prepared_graph(input, reused);
        match captured {
            Ok((key, graph)) => Ok(Gr00tPreparedInference {
                spec: InferenceSpec {
                    token_count: input.token_ids.len(),
                    image_layout: None,
                },
                engine: Rc::clone(&self.engine),
                active_strategy: Rc::clone(&self.active_strategy),
                strategy: Rc::new(RefCell::new(Gr00tPreparedStrategy::Graph { key, graph })),
            }),
            Err(error) if policy == ExecutionPolicy::PreferGraph => {
                let reason = error.to_string();
                eprintln!(
                    "[apxinf] GR00T MUSA graph capture unavailable, using prepared no_graph: {reason}"
                );
                self.build_prepared_no_graph(input, Some(reason), None)
            }
            Err(error) => Err(Error::Other(format!(
                "GR00T MUSA graph capture is required but failed: {error}"
            ))),
        }
    }

    #[cfg(feature = "musa")]
    fn run_prepared_musa(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let config = self.engine.borrow().config().clone();
        let input = Self::build_input(request, &config)?;
        let prepared = {
            let mut cache = self.prepared.borrow_mut();
            let matches = match cache.as_ref() {
                Some((cached_key, plan))
                    if !matches!(&*plan.strategy.borrow(), Gr00tPreparedStrategy::Invalidated) =>
                {
                    self.engine
                        .borrow()
                        .prepared_input_matches(cached_key, &input)?
                }
                _ => false,
            };
            if !matches {
                let plan = Rc::new(self.build_prepared(&input, self.execution_policy.get())?);
                let key = plan
                    .profile_key()
                    .expect("MUSA implicit preparation always builds a fixed-profile plan");
                *cache = Some((key, plan));
            }
            Rc::clone(&cache.as_ref().expect("prepared plan was just built").1)
        };
        // Cache selection above already performed the complete value/profile
        // validation. Do not rebuild the owned key and validate it a second
        // and third time inside the executor.
        prepared.run_validated_input(&input, &config)
    }
}

impl<E: Gr00tPrecisionExecution> Gr00tPreparedInference<E> {
    #[cfg(feature = "musa")]
    fn profile_key(&self) -> Option<Gr00tGraphKey> {
        match &*self.strategy.borrow() {
            Gr00tPreparedStrategy::RuntimeManaged | Gr00tPreparedStrategy::Invalidated => None,
            Gr00tPreparedStrategy::Graph { key, .. }
            | Gr00tPreparedStrategy::NoGraph { key, .. } => Some(key.clone()),
        }
    }

    fn run_input(&self, input: &Gr00tObservation, config: &Gr00tConfig) -> Result<Action> {
        let output = match &*self.strategy.borrow() {
            Gr00tPreparedStrategy::Invalidated => {
                return Err(Error::Other(
                    "GR00T prepared plan was invalidated; prepare a new plan".into(),
                ))
            }
            Gr00tPreparedStrategy::RuntimeManaged => {
                infer_runtime_managed(&self.engine, &self.active_strategy, input)?
            }
            #[cfg(feature = "musa")]
            Gr00tPreparedStrategy::Graph { key, .. }
            | Gr00tPreparedStrategy::NoGraph { key, .. } => {
                if !self.engine.borrow().prepared_input_matches(key, input)? {
                    return Err(Error::Other(format!(
                        "GR00T prepared profile mismatch: request does not match {key:?}"
                    )));
                }
                return self.run_validated_input(input, config);
            }
        };
        self.finish_action(output, config)
    }

    #[cfg(feature = "musa")]
    fn run_validated_input(
        &self,
        input: &Gr00tObservation,
        config: &Gr00tConfig,
    ) -> Result<Action> {
        let output = match &*self.strategy.borrow() {
            Gr00tPreparedStrategy::Invalidated => {
                return Err(Error::Other(
                    "GR00T prepared plan was invalidated; prepare a new plan".into(),
                ))
            }
            Gr00tPreparedStrategy::RuntimeManaged => {
                infer_runtime_managed(&self.engine, &self.active_strategy, input)?
            }
            Gr00tPreparedStrategy::Graph { graph, .. } => {
                self.engine.borrow_mut().run_prepared_graph(graph, input)?
            }
            Gr00tPreparedStrategy::NoGraph {
                token_groups,
                inputs,
                workspace,
                ..
            } => self.engine.borrow_mut().infer_prepared_no_graph(
                input,
                workspace,
                token_groups,
                inputs,
            )?,
        };
        self.finish_action(output, config)
    }

    fn finish_action(&self, output: Tensor, config: &Gr00tConfig) -> Result<Action> {
        Ok(Action::new(output.reshape(vec![
            config.action_horizon,
            config.max_action_dim,
        ])?))
    }
}

impl<E: Gr00tPrecisionExecution> PreparedInference for Gr00tPreparedInference<E> {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        match &*self.strategy.borrow() {
            Gr00tPreparedStrategy::Invalidated => PreparationStatus::Invalidated,
            Gr00tPreparedStrategy::RuntimeManaged => PreparationStatus::RuntimeManaged,
            #[cfg(feature = "musa")]
            Gr00tPreparedStrategy::Graph { .. } => PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                fallback_reason: None,
            },
            #[cfg(feature = "musa")]
            Gr00tPreparedStrategy::NoGraph {
                fallback_reason, ..
            } => PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                fallback_reason: fallback_reason.clone(),
            },
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(format!(
                "GR00T prepared inference expects {:?}, got {:?}",
                self.spec,
                request.observation.inference_spec()
            )));
        }
        let config = self.engine.borrow().config().clone();
        let input = Gr00tModelRunner::<E>::build_input(request, &config)?;
        self.run_input(&input, &config)
    }
}

impl<E: Gr00tPrecisionExecution> VlaRuntime for Gr00tModelRunner<E> {
    fn contract(&self) -> VlaContract {
        let engine = self.engine.borrow();
        let config = engine.config();
        VlaContract {
            action_shape: [config.action_horizon, config.max_action_dim],
            // Processor output has a request-dependent row count. A zero row
            // count advertises that only the feature width is fixed.
            patch_shape: [0, engine.pixel_width()],
            max_token_len: engine.max_token_len(),
            num_views: 0,
            image_size: config.image_target_size.map_or(0, |shape| shape[0]),
            patch_size: engine.patch_size(),
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        #[cfg(feature = "musa")]
        {
            if musa_prepared_arena_enabled() {
                return self.run_prepared_musa(request);
            }
        }
        self.run_runtime_managed(request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        if spec.image_layout.is_some() {
            return Err(Error::Other(
                "GR00T prepared inference accepts processor-produced patches only".into(),
            ));
        }
        Ok(Box::new(Gr00tPreparedInference {
            spec: *spec,
            engine: Rc::clone(&self.engine),
            active_strategy: Rc::clone(&self.active_strategy),
            strategy: Rc::new(RefCell::new(Gr00tPreparedStrategy::RuntimeManaged)),
        }))
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        #[cfg(feature = "musa")]
        {
            let config = self.engine.borrow().config().clone();
            let input = Self::build_input(sample, &config)?;
            Ok(Box::new(self.build_prepared(&input, policy)?))
        }
    }

    fn clear_prepared(&self) -> Result<()> {
        #[cfg(feature = "musa")]
        {
            self.engine.borrow().backend().synchronize()?;
            drop(self.prepared.borrow_mut().take());
            Ok(())
        }
    }

    fn execution_mode(&self) -> &'static str {
        #[cfg(feature = "musa")]
        if let Some((_, plan)) = self.prepared.borrow().as_ref() {
            return match &*plan.strategy.borrow() {
                Gr00tPreparedStrategy::Invalidated => "no_graph",
                Gr00tPreparedStrategy::Graph { .. } => "musa-graph",
                Gr00tPreparedStrategy::NoGraph { .. } => "no_graph",
                Gr00tPreparedStrategy::RuntimeManaged => "runtime-managed",
            };
        }
        if self.engine.borrow().has_captured_graph() {
            "cuda-graph"
        } else {
            "no_graph"
        }
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.infer(request)?.tensor().to_f32_vec()
    }

    fn calibration_amax(&self, request: &VlaRequest<'_>) -> Result<BTreeMap<String, f32>> {
        let config = self.engine.borrow().config().clone();
        let input = Self::build_input(request, &config)?;
        self.invalidate_prepared()?;
        self.engine.borrow_mut().calibration_amax(&input)
    }

    fn calibration_plan(&self) -> Result<Vec<String>> {
        Ok(self.engine.borrow().calibration_plan())
    }
}

#[cfg(feature = "musa")]
fn musa_prepared_arena_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("APXINF_GR00T_PREPARED_ARENA")
            .map(|value| value != "0")
            .unwrap_or(true)
    })
}

fn run_engine<E: Gr00tPrecisionExecution>(
    engine: &Rc<RefCell<Gr00tExecutor<E>>>,
    active_strategy: &Gr00tActiveStrategy,
    request: &VlaRequest<'_>,
) -> Result<Action> {
    let config = engine.borrow().config().clone();
    let input = Gr00tModelRunner::<E>::build_input(request, &config)?;
    let output = infer_runtime_managed(engine, active_strategy, &input)?
        .reshape(vec![config.action_horizon, config.max_action_dim])?;
    Ok(Action::new(output))
}

fn infer_runtime_managed<E: Gr00tPrecisionExecution>(
    engine: &Rc<RefCell<Gr00tExecutor<E>>>,
    active_strategy: &Gr00tActiveStrategy,
    input: &Gr00tObservation,
) -> Result<Tensor> {
    active_strategy.invalidate(|| engine.borrow().backend().synchronize())?;
    engine.borrow_mut().infer(input)
}

#[cfg(test)]
mod lifecycle_tests {
    use super::{Gr00tActiveStrategy, Gr00tPreparedStrategy};
    use apxinf_core::Error;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn shared_registry_invalidates_retained_handle_after_synchronization() {
        let registry = Rc::new(Gr00tActiveStrategy::default());
        let runtime_managed_registry = Rc::clone(&registry);
        let strategy = Rc::new(RefCell::new(Gr00tPreparedStrategy::RuntimeManaged));
        registry.register(&strategy);
        drop(registry);

        runtime_managed_registry
            .invalidate(|| {
                assert!(matches!(
                    *strategy.borrow(),
                    Gr00tPreparedStrategy::RuntimeManaged
                ));
                Ok(())
            })
            .unwrap();

        assert!(matches!(
            *strategy.borrow(),
            Gr00tPreparedStrategy::Invalidated
        ));
        assert!(runtime_managed_registry
            .strategy
            .borrow()
            .upgrade()
            .is_none());
        runtime_managed_registry
            .invalidate(|| panic!("already retired plans need no synchronization"))
            .unwrap();
    }

    #[test]
    fn failed_synchronization_preserves_plan_and_registration() {
        let registry = Gr00tActiveStrategy::default();
        let strategy = Rc::new(RefCell::new(Gr00tPreparedStrategy::RuntimeManaged));
        registry.register(&strategy);

        assert!(registry
            .invalidate(|| Err(Error::Other("synchronization failed".into())))
            .is_err());
        assert!(matches!(
            *strategy.borrow(),
            Gr00tPreparedStrategy::RuntimeManaged
        ));
        assert!(Rc::ptr_eq(
            &registry.strategy.borrow().upgrade().unwrap(),
            &strategy
        ));

        registry.invalidate(|| Ok(())).unwrap();
        assert!(matches!(
            *strategy.borrow(),
            Gr00tPreparedStrategy::Invalidated
        ));
    }

    #[test]
    fn replacement_is_tracked_without_reviving_retired_plan() {
        let registry = Gr00tActiveStrategy::default();
        let old_strategy = Rc::new(RefCell::new(Gr00tPreparedStrategy::RuntimeManaged));
        registry.register(&old_strategy);
        registry.invalidate(|| Ok(())).unwrap();

        let replacement = Rc::new(RefCell::new(Gr00tPreparedStrategy::RuntimeManaged));
        registry.register(&replacement);
        assert!(matches!(
            *old_strategy.borrow(),
            Gr00tPreparedStrategy::Invalidated
        ));
        assert!(matches!(
            *replacement.borrow(),
            Gr00tPreparedStrategy::RuntimeManaged
        ));
        registry.invalidate(|| Ok(())).unwrap();
        assert!(matches!(
            *replacement.borrow(),
            Gr00tPreparedStrategy::Invalidated
        ));
    }

    #[test]
    fn registry_does_not_keep_dropped_handles_alive() {
        let registry = Gr00tActiveStrategy::default();
        let strategy = Rc::new(RefCell::new(Gr00tPreparedStrategy::RuntimeManaged));
        registry.register(&strategy);
        drop(strategy);

        registry
            .invalidate(|| panic!("dropped plans need no synchronization"))
            .unwrap();
    }

    #[cfg(feature = "musa")]
    #[test]
    fn plans_without_an_arena_have_no_workspace_to_reuse() {
        assert!(Gr00tPreparedStrategy::Invalidated
            .take_workspace()
            .is_none());
        assert!(Gr00tPreparedStrategy::RuntimeManaged
            .take_workspace()
            .is_none());
    }
}

fn to_bf16(tensor: &Tensor, name: &str) -> Result<Tensor> {
    match tensor.dtype() {
        DType::BF16 => Ok(tensor.clone()),
        DType::F32 if tensor.device() == Device::Cpu => {
            let values = tensor
                .to_f32_vec()?
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            Tensor::from_bf16(tensor.shape().dims().to_vec(), &values)
        }
        dtype => Err(Error::Other(format!(
            "GR00T {name} must be CPU f32 or bf16, got {dtype} on {}",
            tensor.device()
        ))),
    }
}

fn normalize_batched_bf16(tensor: &Tensor, rows: usize, cols: usize, name: &str) -> Result<Tensor> {
    let tensor = to_bf16(tensor, name)?;
    match tensor.shape().dims() {
        dims if dims == [1, rows, cols] => Ok(tensor),
        dims if dims == [rows, cols] => tensor.reshape(vec![1, rows, cols]),
        dims => Err(Error::Other(format!(
            "GR00T {name} must have shape [{rows}, {cols}] or [1, {rows}, {cols}], got {dims:?}"
        ))),
    }
}

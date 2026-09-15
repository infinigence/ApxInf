use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, InferenceSpec, InitialLatent, PreparedInference, VisionObservation, VlaContract,
    VlaRequest, VlaRuntime,
};

use super::backend::{downcast_arc, RuntimeBackend};
use super::executor::{Gr00tExecutor, Gr00tObservation, Gr00tPrecisionExecution};
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

const BACKBONE_ASSET: &str = "backbone";

pub(super) struct Gr00tVlaRuntime<E: Gr00tPrecisionExecution> {
    engine: Rc<RefCell<Gr00tExecutor<E>>>,
}

pub(super) struct Gr00tPreparedInference<E: Gr00tPrecisionExecution> {
    spec: InferenceSpec,
    engine: Rc<RefCell<Gr00tExecutor<E>>>,
}

impl<E: Gr00tPrecisionExecution> Gr00tVlaRuntime<E> {
    pub(super) fn new(engine: Gr00tExecutor<E>) -> Self {
        Self {
            engine: Rc::new(RefCell::new(engine)),
        }
    }

    fn build_input(
        request: &VlaRequest<'_>,
        config: &super::Gr00tConfig,
    ) -> Result<Gr00tObservation> {
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

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        run_engine(&self.engine, request)
    }
}

impl<E: Gr00tPrecisionExecution> PreparedInference for Gr00tPreparedInference<E> {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(format!(
                "GR00T prepared inference expects {:?}, got {:?}",
                self.spec,
                request.observation.inference_spec()
            )));
        }
        run_engine(&self.engine, request)
    }
}

impl<E: Gr00tPrecisionExecution> VlaRuntime for Gr00tVlaRuntime<E> {
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
        self.run(request)
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
        }))
    }

    fn execution_mode(&self) -> &'static str {
        if self.engine.borrow().has_captured_graph() {
            "cuda-graph"
        } else {
            "eager"
        }
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.infer(request)?.tensor().to_f32_vec()
    }

    fn calibration_amax(&self, request: &VlaRequest<'_>) -> Result<BTreeMap<String, f32>> {
        let config = self.engine.borrow().config().clone();
        let input = Self::build_input(request, &config)?;
        self.engine.borrow_mut().calibration_amax(&input)
    }

    fn calibration_plan(&self) -> Result<Vec<String>> {
        Ok(self.engine.borrow().calibration_plan())
    }
}

fn run_engine<E: Gr00tPrecisionExecution>(
    engine: &Rc<RefCell<Gr00tExecutor<E>>>,
    request: &VlaRequest<'_>,
) -> Result<Action> {
    let config = engine.borrow().config().clone();
    let input = Gr00tVlaRuntime::<E>::build_input(request, &config)?;
    let output = engine
        .borrow_mut()
        .infer(&input)?
        .reshape(vec![config.action_horizon, config.max_action_dim])?;
    Ok(Action::new(output))
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

pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.synthetic.is_some()
        || options.config.is_some()
        || options.uniform_fp8_scale.is_some()
    {
        return Err(Error::Other(
            "GR00T does not support PI0.5 synthetic/config load options".into(),
        ));
    }
    if let Some(name) = options
        .assets
        .keys()
        .find(|name| name.as_str() != BACKBONE_ASSET)
    {
        return Err(Error::Other(format!(
            "GR00T does not recognize the named asset {name:?}; expected only {BACKBONE_ASSET:?}"
        )));
    }
    let backbone_path = options.assets.get(BACKBONE_ASSET).cloned().ok_or_else(|| {
        Error::Other(
            "GR00T loading requires LoadOptions.assets[\"backbone\"] pointing to Cosmos-Reason2-2B"
                .into(),
        )
    })?;
    let backend: Arc<RuntimeBackend> = downcast_arc(backend)
        .ok_or_else(|| Error::Other("GR00T is only registered for CUDA".into()))?;
    let precision = resolve_precision(options.precision, backend.context().caps().sm)?;
    let config = Gr00tConfig::from_json_file(&path.join("config.json"))?;
    let (backbone_config, weights) = Gr00tWeights::from_safetensors(&config, &backbone_path, path)?;
    let runtime: Box<dyn VlaRuntime> = match precision {
        ModelPrecision::Bf16 => Box::new(super::bf16_runtime::build(
            config,
            backbone_config,
            weights,
            backend,
        )?),
        ModelPrecision::Fp8 => {
            let calibration_path = options.calibration_path.as_deref().ok_or_else(|| {
                Error::Other(
                    "GR00T FP8 requires LoadOptions.calibration_path pointing to a validated calibration JSON"
                        .into(),
                )
            })?;
            Box::new(super::fp8_runtime::build(
                path,
                &backbone_path,
                calibration_path,
                config,
                backbone_config,
                weights,
                backend,
            )?)
        }
        ModelPrecision::W8A8 => Box::new(super::int8_runtime::build(
            config,
            backbone_config,
            weights,
            backend,
        )?),
        ModelPrecision::Auto => unreachable!("automatic precision was resolved"),
    };
    Ok(LoadedModel::Vla(runtime))
}

fn resolve_precision(requested: ModelPrecision, sm: u32) -> Result<ModelPrecision> {
    let precision = match requested {
        ModelPrecision::Auto => ModelPrecision::Bf16,
        explicit => explicit,
    };
    match (sm, precision) {
        (100.., ModelPrecision::Bf16 | ModelPrecision::Fp8) => Ok(precision),
        (80..100, ModelPrecision::Bf16 | ModelPrecision::W8A8) => Ok(precision),
        (100.., ModelPrecision::W8A8) => Err(Error::Other(format!(
            "GR00T W8A8 is an Orin-class SM80 path and is not supported on sm_{sm}; use bf16 or fp8"
        ))),
        (80..100, ModelPrecision::Fp8) => Err(Error::Other(format!(
            "GR00T FP8 requires Thor-class sm_100 or newer, got sm_{sm}; use bf16 or int8"
        ))),
        (_, _) => Err(Error::Other(format!(
            "GR00T supports Thor-class sm_100+ and Orin-class sm_80..sm_99 GPUs, got sm_{sm}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelPrecision;

    #[test]
    fn normalizes_unbatched_processor_tensors_without_changing_values() {
        let input = Tensor::from_f32(vec![2, 3], &[0.0, 1.0, -2.0, 3.5, 4.0, -5.0]).unwrap();
        let normalized = normalize_batched_bf16(&input, 2, 3, "fixture").unwrap();

        assert_eq!(normalized.dtype(), DType::BF16);
        assert_eq!(normalized.shape().dims(), [1, 2, 3]);
        assert_eq!(
            normalized.to_f32_vec().unwrap(),
            vec![0.0, 1.0, -2.0, 3.5, 4.0, -5.0]
        );
    }

    #[test]
    fn rejects_processor_tensor_with_wrong_shape() {
        let input = Tensor::from_f32(vec![1, 3], &[0.0, 1.0, 2.0]).unwrap();
        let error = normalize_batched_bf16(&input, 2, 3, "fixture").unwrap_err();

        assert!(error.to_string().contains("[2, 3] or [1, 2, 3]"));
    }

    #[test]
    fn unsupported_precision_families_are_not_silently_remapped() {
        assert_eq!(
            super::resolve_precision(ModelPrecision::Auto, 110).unwrap(),
            ModelPrecision::Bf16
        );
        assert!(super::resolve_precision(ModelPrecision::W8A8, 110).is_err());
        assert!(super::resolve_precision(ModelPrecision::Fp8, 87).is_err());
    }
}

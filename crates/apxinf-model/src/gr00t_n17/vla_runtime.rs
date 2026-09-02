//! Owning CUDA runtime and whole-model CUDA Graph for GR00T N1.7 LIBERO.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{
    Backend, DType, Device, Error, Graph, NormalGenerator, Result, SamplingBackend, Tensor,
};
use apxinf_cuda::buffer::CudaBuffer;
use apxinf_cuda::kernels::preprocess::{self, ImageLayout as KernelImageLayout};
use apxinf_cuda::kernels::{self, GraphWorkspace};
use apxinf_cuda::transfers;
use apxinf_cuda::CudaBackend;

use crate::accelerator::cuda::downcast_arc;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, InferenceSpec, InitialLatent, PreparedInference, VisionObservation, VlaRequest,
    VlaRuntime,
};

use super::{
    GrootN17ActionExecutor, GrootN17ActionWeights, GrootN17BackboneExecutor,
    GrootN17BackboneWeights, GrootN17Config,
};

const GRAPH_WORKSPACE_BYTES: usize = 3 * 1024 * 1024 * 1024;

pub struct GrootN17VlaRuntime {
    backend: Arc<CudaBackend>,
    backbone_weights: Arc<GrootN17BackboneWeights>,
    action: Arc<GrootN17ActionExecutor>,
    prepared: RefCell<Option<Rc<GrootPrepared>>>,
}

struct GrootPrepared {
    spec: InferenceSpec,
    backbone: GrootN17BackboneExecutor,
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    state: Tensor,
    noise: Tensor,
    normal: RefCell<Box<dyn NormalGenerator>>,
    _raw_images: Option<CudaBuffer>,
    _text_rows: apxinf_cuda::kernels::elementwise::RowIndices,
    _image_rows: apxinf_cuda::kernels::elementwise::RowIndices,
    _workspace: GraphWorkspace,
    _action: Arc<GrootN17ActionExecutor>,
}

impl GrootN17VlaRuntime {
    fn build_prepared(&self, spec: &InferenceSpec) -> Result<GrootPrepared> {
        let view_count = match spec.token_count {
            76 => 1,
            142 => 2,
            _ => return Err(Error::Other(format!(
                "GR00T N1.7 requires the 76-token one-view or 142-token two-view patch profile, got {spec:?}"))),
        };
        let mut template = vec![0u32; spec.token_count];
        for row in 4..68 {
            template[row] = 151655;
        }
        if view_count == 2 {
            for row in 70..134 {
                template[row] = 151655;
            }
        }
        let backbone = GrootN17BackboneExecutor::new(
            Arc::clone(&self.backbone_weights),
            &template,
            self.backend.device_id(),
        )?;
        let patches = self
            .backend
            .to_device(&Tensor::zeros((view_count * 256, 1536), DType::BF16))?;
        let raw_images = spec
            .image_layout
            .map(|_| CudaBuffer::alloc(view_count * 256 * 256 * 3, self.backend.device_id()))
            .transpose()
            .map_err(Error::Cuda)?;
        let state = self
            .backend
            .to_device(&Tensor::zeros((1, 132), DType::BF16))?;
        let noise = self
            .backend
            .to_device(&Tensor::zeros((40, 132), DType::BF16))?;
        let normal = self.backend.create_normal_generator(noise.clone())?;
        let workspace = GraphWorkspace::new(GRAPH_WORKSPACE_BYTES, self.backend.device_id())?;
        let image_rows = image_rows(self.backend.device_id(), &template)?;
        let text_rows = text_rows(self.backend.device_id(), &template)?;
        let execute = || -> Result<Tensor> {
            if let (Some(images), Some(layout)) = (&raw_images, spec.image_layout) {
                preprocess::groot_rgb_u8_to_patches_bf16(
                    self.backend.context(),
                    images,
                    &patches,
                    view_count,
                    kernel_image_layout(layout),
                )?;
            }
            let features = backbone
                .forward(self.backend.context(), &patches)
                .map_err(|error| Error::Other(format!("GR00T backbone execution: {error}")))?;
            let refined = self
                .action
                .refine_backbone(self.backend.context(), &features)
                .map_err(|error| Error::Other(format!("GR00T VL refinement: {error}")))?;
            let output = self
                .action
                .infer(
                    self.backend.context(),
                    &refined,
                    &state,
                    &noise,
                    &text_rows,
                    &image_rows,
                    &*self.backend,
                )
                .map_err(|error| Error::Other(format!("GR00T action execution: {error}")))?;
            Ok(output)
        };
        let eager = kernels::prepare_with_workspace(&workspace, || execute())?;
        self.backend.synchronize()?;
        drop(eager);
        self.backend.begin_capture()?;
        let output = match kernels::with_workspace(&workspace, || execute()) {
            Ok(value) => value,
            Err(error) => {
                let _ = self.backend.end_capture();
                return Err(error);
            }
        };
        let graph = self.backend.end_capture()?;
        Ok(GrootPrepared {
            spec: *spec,
            backbone,
            graph,
            output,
            patches,
            state,
            noise,
            normal: RefCell::new(normal),
            _raw_images: raw_images,
            _workspace: workspace,
            _text_rows: text_rows,
            _image_rows: image_rows,
            _action: Arc::clone(&self.action),
        })
    }
}

impl PreparedInference for GrootPrepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "GR00T prepared shape does not match request".into(),
            ));
        }
        match &request.observation.vision {
            VisionObservation::Patches(value) => {
                if self._raw_images.is_some() {
                    return Err(Error::Other(
                        "GR00T prepared RGB graph received patches".into(),
                    ));
                }
                let patch_rows = if self.spec.token_count == 76 {
                    256
                } else {
                    512
                };
                let patches = normalize_cpu(value, [patch_rows, 1536], "patches")?;
                transfers::copy_cpu_to_cuda(&patches, &self.patches)?;
            }
            VisionObservation::RgbU8 { bytes, .. } => {
                let images = self._raw_images.as_ref().ok_or_else(|| {
                    Error::Other("GR00T prepared patch graph received RGB".into())
                })?;
                if bytes.len() != images.len() {
                    return Err(Error::Other(format!(
                        "GR00T RGB input expected {} bytes, got {}",
                        images.len(),
                        bytes.len()
                    )));
                }
                images.copy_from_host(bytes).map_err(Error::Cuda)?;
            }
        }
        let state = normalize_cpu(
            request
                .state
                .ok_or_else(|| Error::Other("GR00T inference requires normalized state".into()))?,
            [1, 132],
            "state",
        )?;
        transfers::copy_cpu_to_cuda(&state, &self.state)?;
        self.backbone
            .update_token_ids(&request.observation.token_ids)?;
        match request.initial_latent {
            InitialLatent::Provided(value) => {
                let noise = normalize_cpu(value, [40, 132], "initial latent")?;
                transfers::copy_cpu_to_cuda(&noise, &self.noise)?;
            }
            InitialLatent::Generate { rng } => {
                self.normal.borrow_mut().generate(rng)?;
            }
        }
        self.graph.replay()?;
        Ok(Action::new(self.output.clone()))
    }
}

impl VlaRuntime for GrootN17VlaRuntime {
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let spec = request.observation.inference_spec();
        let prepared = self
            .prepared
            .borrow()
            .as_ref()
            .filter(|value| value.spec == spec)
            .map(Rc::clone);
        let prepared = if let Some(value) = prepared {
            value
        } else {
            self.prepared.borrow_mut().take();
            let value = Rc::new(self.build_prepared(&spec)?);
            *self.prepared.borrow_mut() = Some(Rc::clone(&value));
            value
        };
        prepared.run(request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.build_prepared(spec)?))
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.backend.to_cpu(action.tensor())?.to_f32_vec()
    }
}

pub(crate) fn load_registered(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Other("GR00T requires CUDA".into()));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) {
        return Err(Error::Other(
            "GR00T N1.7 currently requires BF16 precision".into(),
        ));
    }
    let backend =
        downcast_arc(backend).ok_or_else(|| Error::Other("GR00T requires CudaBackend".into()))?;
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let config = Arc::new(GrootN17Config::from_json_file(&root.join("config.json"))?);
    let (mut tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load GR00T checkpoint: {error}")))?;
    let backbone_weights =
        Arc::new(GrootN17BackboneWeights::from_map(&mut tensors)?.to_device(&*backend)?);
    let action_weights =
        Arc::new(GrootN17ActionWeights::from_map(&config, &mut tensors)?.to_device(&*backend)?);
    let action = Arc::new(GrootN17ActionExecutor::new(
        Arc::clone(&config),
        action_weights,
        &*backend,
        &backend,
    )?);
    Ok(LoadedModel::Vla(Box::new(GrootN17VlaRuntime {
        backend,
        backbone_weights,
        action,
        prepared: RefCell::new(None),
    })))
}

fn normalize_cpu<const N: usize>(
    tensor: &Tensor,
    shape: [usize; N],
    label: &str,
) -> Result<Tensor> {
    if tensor.device() != Device::Cpu || tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "GR00T {label} must be CPU {:?}, got {:?} on {}",
            shape,
            tensor.shape().dims(),
            tensor.device()
        )));
    }
    if tensor.dtype() == DType::BF16 {
        return Ok(tensor.clone());
    }
    Tensor::from_bf16(
        shape.to_vec(),
        &tensor
            .to_f32_vec()?
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>(),
    )
}

fn image_rows(
    device: usize,
    tokens: &[u32],
) -> Result<apxinf_cuda::kernels::elementwise::RowIndices> {
    let rows = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token == 151655).then_some(index as u32))
        .collect::<Vec<_>>();
    apxinf_cuda::kernels::elementwise::RowIndices::new(device, &rows)
}
fn text_rows(
    device: usize,
    tokens: &[u32],
) -> Result<apxinf_cuda::kernels::elementwise::RowIndices> {
    let rows = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, &token)| (token != 151655).then_some(index as u32))
        .collect::<Vec<_>>();
    apxinf_cuda::kernels::elementwise::RowIndices::new(device, &rows)
}

fn kernel_image_layout(layout: crate::vla::ImageLayout) -> KernelImageLayout {
    match layout {
        crate::vla::ImageLayout::Nhwc => KernelImageLayout::Nhwc,
        crate::vla::ImageLayout::Nchw => KernelImageLayout::Nchw,
    }
}

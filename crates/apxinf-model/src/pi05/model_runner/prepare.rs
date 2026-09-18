//! Shared PI0.5 preparation and captured resource ownership.
use crate::pi05::backend::{
    kernels, transfers, Context, DeviceBuffer as CudaBuffer, RuntimeBackend,
};
use crate::pi05::model::Pi05Model;
use crate::pi05::model::{ModelOperation, ModelVariant, PrepareBlocks, WorkspaceRequirements};
use crate::pi05::{Pi05Config, Pi05ImageLayout};
use apxinf_core::{Backend, Error, Graph, Result, Tensor};
use std::sync::Arc;

fn allocate_workspace(
    requirements: &WorkspaceRequirements,
    device: usize,
) -> Result<kernels::GraphWorkspace> {
    match requirements.fp8_scratch {
        Some((a, w)) => kernels::GraphWorkspace::new_fp8(requirements.bytes, a, w, device),
        None => kernels::GraphWorkspace::new(requirements.bytes, device),
    }
}
pub struct CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<CudaBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    token_ids: CudaBuffer,
    token_count: usize,
    backend: Arc<RuntimeBackend>,
    // Retain every fixed weight referenced by the captured computation.
    _fixed: Box<dyn std::any::Any>,
    workspace: kernels::GraphWorkspace,
}

impl CapturedGraph {
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    pub fn replay_and_synchronize(&self) -> Result<()> {
        self.graph.replay()?;
        self.backend.synchronize()
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn raw_image_layout(&self) -> Option<Pi05ImageLayout> {
        self.raw_image_layout
    }

    fn update_tokens(&self, token_ids: &[u32]) -> Result<()> {
        let bytes = token_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        self.token_ids.copy_from_host(&bytes).map_err(Error::Cuda)
    }

    pub fn update_inputs(&self, patches: &Tensor, token_ids: &[u32], noise: &Tensor) -> Result<()> {
        self.update_inputs_without_noise(patches, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    pub fn update_inputs_without_noise(&self, patches: &Tensor, token_ids: &[u32]) -> Result<()> {
        if self.raw_images.is_some() {
            return Err(Error::Other(
                "π0.5 graph uses raw RGB input; call update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        self.backend.synchronize()?;
        transfers::copy_cpu_to_cuda(patches, &self.patches)?;
        self.update_tokens(token_ids)
    }

    pub fn update_raw_image_inputs(
        &self,
        images: &[u8],
        token_ids: &[u32],
        noise: &Tensor,
    ) -> Result<()> {
        self.update_raw_image_inputs_without_noise(images, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    pub fn update_raw_image_inputs_without_noise(
        &self,
        images: &[u8],
        token_ids: &[u32],
    ) -> Result<()> {
        let raw_images = self.raw_images.as_ref().ok_or_else(|| {
            Error::Other("π0.5 graph uses patch input; call update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "π0.5 graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        self.backend.synchronize()?;
        raw_images.copy_from_host(images).map_err(Error::Cuda)?;
        self.update_tokens(token_ids)
    }

    pub fn workspace_bytes(&self) -> usize {
        self.workspace.capacity()
    }

    pub fn workspace_used_bytes(&self) -> usize {
        self.workspace.used()
    }
}

struct CaptureBuilder<'a, B: PrepareBlocks> {
    model: &'a Arc<Pi05Model<B>>,
    backend: &'a Arc<RuntimeBackend>,
    config: &'a Pi05Config,
}
impl<B: PrepareBlocks> CaptureBuilder<'_, B> {
    fn ctx(&self) -> &Context {
        self.backend.context()
    }
    #[allow(clippy::too_many_arguments)]
    fn infer_captured_inputs(
        &self,
        patches: &Tensor,
        raw_images: Option<&CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        modulation: &[B::StepModulation],
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => {
                self.model
                    .infer_with_modulation(patches, token_ids, token_count, noise, modulation)
            }
            (Some(images), Some(layout)) => {
                self.model.preprocess(images, patches, layout)?;
                self.model.infer_with_native_modulation(
                    patches,
                    token_ids,
                    token_count,
                    noise,
                    modulation,
                )
            }
            _ => Err(Error::Other(
                "π0.5 raw image capture state is inconsistent".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_infer_impl(
        &self,
        patches: Tensor,
        raw_images: Option<CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<CapturedGraph> {
        let backend = &self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "π0.5 raw image capture state is inconsistent".into(),
            ));
        }
        let modulation = self.model.prepare_all_modulation(time_embeddings)?;
        backend.synchronize()?;
        let workspace = allocate_workspace(
            &self.model.workspace_requirements(token_count)?,
            self.ctx().device_id(),
        )?;
        let mut stable = false;
        for _ in 0..4 {
            let generation = self.ctx().tuning().generation();
            let eager_output = kernels::prepare_with_workspace(&workspace, || {
                self.infer_captured_inputs(
                    &patches,
                    raw_images.as_ref(),
                    raw_image_layout,
                    token_ids,
                    token_count,
                    noise,
                    &modulation,
                )
            })?;
            backend.synchronize()?;
            drop(eager_output);
            if self.ctx().tuning().generation() == generation {
                stable = true;
                break;
            }
        }
        if !stable {
            return Err(Error::Other(
                "GEMM tactic store did not stabilize before PI0.5 graph capture".into(),
            ));
        }

        let (graph, output) = backend.capture_graph(|| {
            kernels::with_workspace(&workspace, || {
                self.infer_captured_inputs(
                    &patches,
                    raw_images.as_ref(),
                    raw_image_layout,
                    token_ids,
                    token_count,
                    noise,
                    &modulation,
                )
            })
        })?;
        Ok(CapturedGraph {
            graph,
            output,
            patches,
            raw_images,
            raw_image_layout,
            noise: noise.clone(),
            token_ids: token_ids.clone(),
            token_count,
            backend: Arc::clone(&self.backend),
            _fixed: Box::new((Arc::clone(&self.model), modulation)),
            workspace,
        })
    }

    pub fn capture_infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<CapturedGraph> {
        self.capture_infer_impl(
            patches.clone(),
            None,
            None,
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }

    pub fn capture_infer_rgb_u8(
        &self,
        layout: Pi05ImageLayout,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<CapturedGraph> {
        let backend = &self.backend;
        let raw_image_bytes = self
            .config
            .num_views
            .checked_mul(3)
            .and_then(|n| n.checked_mul(self.config.image_size))
            .and_then(|n| n.checked_mul(self.config.image_size))
            .ok_or_else(|| Error::Other("PI0.5 raw image size overflow".into()))?;
        let raw_images = CudaBuffer::alloc_zeros(raw_image_bytes, self.ctx().device_id())
            .map_err(Error::Cuda)?;
        let patch_rows = self.config.num_views * self.config.patches_per_view();
        let patch_width = 3 * self.config.patch_size * self.config.patch_size;
        let patches = backend.to_device(&Tensor::zeros(
            vec![patch_rows, patch_width],
            self.model.raw_patch_dtype(),
        ))?;
        self.capture_infer_impl(
            patches,
            Some(raw_images),
            Some(layout),
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }
}

pub(super) enum CaptureInput<'a> {
    Patches(&'a Tensor),
    Rgb(Pi05ImageLayout),
}
pub(super) fn capture<B: PrepareBlocks>(
    model: &Arc<Pi05Model<B>>,
    input: CaptureInput<'_>,
    tokens: &CudaBuffer,
    count: usize,
    noise: &Tensor,
    embeddings: &[Tensor],
) -> Result<CapturedGraph> {
    let builder = CaptureBuilder {
        model,
        backend: model.backend(),
        config: model.config(),
    };
    match input {
        CaptureInput::Rgb(layout) => {
            builder.capture_infer_rgb_u8(layout, tokens, count, noise, embeddings)
        }
        CaptureInput::Patches(patches) => {
            builder.capture_infer(patches, tokens, count, noise, embeddings)
        }
    }
}
/// Prepare a low-level graph with caller-owned stable patch/token/noise tensors.
/// Their storage is retained; sizes and addresses cannot change between replays.
pub fn capture_patches<B: PrepareBlocks>(
    model: &Arc<Pi05Model<B>>,
    patches: &Tensor,
    tokens: &CudaBuffer,
    count: usize,
    noise: &Tensor,
    embeddings: &[Tensor],
) -> Result<CapturedGraph> {
    capture(
        model,
        CaptureInput::Patches(patches),
        tokens,
        count,
        noise,
        embeddings,
    )
}
/// Prepare a graph that owns RGB input storage and device preprocessing.
pub fn capture_rgb<B: PrepareBlocks>(
    model: &Arc<Pi05Model<B>>,
    layout: Pi05ImageLayout,
    tokens: &CudaBuffer,
    count: usize,
    noise: &Tensor,
    embeddings: &[Tensor],
) -> Result<CapturedGraph> {
    capture(
        model,
        CaptureInput::Rgb(layout),
        tokens,
        count,
        noise,
        embeddings,
    )
}

pub(super) fn capture_loaded(
    model: &ModelVariant,
    spec: &crate::vla::InferenceSpec,
    patches: &Tensor,
    tokens: &CudaBuffer,
    noise: &Tensor,
) -> Result<CapturedGraph> {
    struct CaptureOperation<'a> {
        input: CaptureInput<'a>,
        tokens: &'a CudaBuffer,
        count: usize,
        noise: &'a Tensor,
    }
    impl ModelOperation for CaptureOperation<'_> {
        type Output = CapturedGraph;
        fn run<B: PrepareBlocks>(
            self,
            model: &Arc<Pi05Model<B>>,
            embeddings: &[Tensor],
        ) -> Result<Self::Output> {
            capture(
                model,
                self.input,
                self.tokens,
                self.count,
                self.noise,
                embeddings,
            )
        }
    }
    let input = match spec.image_layout {
        Some(layout) => CaptureInput::Rgb(super::runner::kernel_image_layout(layout)),
        None => CaptureInput::Patches(patches),
    };
    model.with_model(CaptureOperation {
        input,
        tokens,
        count: spec.token_count,
        noise,
    })
}

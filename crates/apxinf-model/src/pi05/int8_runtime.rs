//! Fixed-shape W8A8 INT8 π0.5 inference runtime.

use std::sync::Arc;

use super::backend::{kernels, transfers, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use apxinf_core::{Backend, DType, Error, Graph, Result, Tensor};
use half::bf16;
use kernels::preprocess;

use super::{sinusoidal_time_embedding, Pi05Config, Pi05ImageLayout, StaticInt8Pi05Weights};

pub use super::blocks::Int8PrefixKvCache;
use super::blocks::{Int8StepStyles, W8A8Blocks};
use super::network::Pi05Int8Network;

pub struct Pi05Int8CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<CudaBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    _styles: Vec<Int8StepStyles>,
    token_ids: CudaBuffer,
    token_count: usize,
    backend: Arc<RuntimeBackend>,
    _network: Arc<Pi05Int8Network>,
    workspace: kernels::GraphWorkspace,
}

impl Pi05Int8CapturedGraph {
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
                "π0.5 INT8 graph uses raw RGB input; call update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 INT8 graph expects {} token IDs, got {}",
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
            Error::Other("π0.5 INT8 graph uses patch input; call update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "π0.5 INT8 graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 INT8 graph expects {} token IDs, got {}",
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

#[derive(Clone)]
pub struct Pi05Int8CudaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    network: Arc<Pi05Int8Network>,
}

impl Pi05Int8CudaRuntime {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Pi05Config>,
        weights: Arc<StaticInt8Pi05Weights>,
    ) -> Result<Self> {
        let network = Arc::new(Pi05Int8Network::from_blocks(W8A8Blocks::new(
            Arc::clone(&backend),
            Arc::clone(&config),
            weights,
        )?));
        Ok(Self {
            backend,
            config,
            network,
        })
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
    }

    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        self.network.encode_vision(patches)
    }

    pub fn embed_prefix(
        &self,
        vision_tokens: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
    ) -> Result<Tensor> {
        self.network
            .embed_prefix(vision_tokens, token_ids, token_count)
    }

    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Int8PrefixKvCache> {
        self.network.prefix_forward(prefix)
    }

    fn prepare_all_styles(&self, time_embeddings: &[Tensor]) -> Result<Vec<Int8StepStyles>> {
        self.network.prepare_all_styles(time_embeddings)
    }

    pub fn denoise_step(
        &self,
        state: &Tensor,
        time_embedding: &Tensor,
        prefix: &Int8PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        self.network.denoise_step(state, time_embedding, prefix, dt)
    }

    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        time_embeddings: &[Tensor],
        prefix: &Int8PrefixKvCache,
    ) -> Result<Tensor> {
        self.network
            .denoise_all_steps(noise, time_embeddings, prefix)
    }

    fn infer_with_styles(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Int8StepStyles],
    ) -> Result<Tensor> {
        self.network
            .infer_with_styles(patches, token_ids, token_count, noise, styles)
    }

    pub fn infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        self.network
            .infer(patches, token_ids, token_count, noise, time_embeddings)
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
        styles: &[Int8StepStyles],
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => self.infer_with_styles(patches, token_ids, token_count, noise, styles),
            (Some(images), Some(layout)) => {
                preprocess::rgb_u8_to_patches_bf16(
                    self.ctx(),
                    images,
                    patches,
                    self.config.num_views,
                    self.config.image_size,
                    self.config.patch_size,
                    layout,
                )?;
                self.infer_with_styles(patches, token_ids, token_count, noise, styles)
            }
            _ => Err(Error::Other(
                "π0.5 INT8 raw image capture state is inconsistent".into(),
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
    ) -> Result<Pi05Int8CapturedGraph> {
        let backend = &self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "π0.5 INT8 raw image capture state is inconsistent".into(),
            ));
        }
        let styles = self.prepare_all_styles(time_embeddings)?;
        backend.synchronize()?;
        let workspace = kernels::GraphWorkspace::new(
            self.config.cuda_graph_workspace_bytes_int8(token_count)?,
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
                    &styles,
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
                "GEMM tactic store did not stabilize before PI0.5 INT8 graph capture".into(),
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
                    &styles,
                )
            })
        })?;
        Ok(Pi05Int8CapturedGraph {
            graph,
            output,
            patches,
            raw_images,
            raw_image_layout,
            noise: noise.clone(),
            _styles: styles,
            token_ids: token_ids.clone(),
            token_count,
            backend: Arc::clone(&self.backend),
            _network: Arc::clone(&self.network),
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
    ) -> Result<Pi05Int8CapturedGraph> {
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
    ) -> Result<Pi05Int8CapturedGraph> {
        let backend = &self.backend;
        let raw_image_bytes =
            self.config.num_views * 3 * self.config.image_size * self.config.image_size;
        let raw_images = CudaBuffer::alloc_zeros(raw_image_bytes, self.ctx().device_id())
            .map_err(Error::Cuda)?;
        let patch_rows = self.config.num_views * self.config.patches_per_view();
        let patch_width = 3 * self.config.patch_size * self.config.patch_size;
        let patches =
            backend.to_device(&Tensor::zeros(vec![patch_rows, patch_width], DType::BF16))?;
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

pub fn upload_time_embeddings_int8(
    config: &Pi05Config,
    backend: &dyn Backend,
) -> Result<Vec<Tensor>> {
    (0..config.num_flow_steps)
        .map(|step| {
            let time = config.flow_start_time * (1.0 - step as f32 / config.num_flow_steps as f32);
            let values = sinusoidal_time_embedding(
                time,
                config.action_expert.width,
                config.time_min_period,
                config.time_max_period,
            )
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
            backend.to_device(&Tensor::from_bf16(
                vec![1, config.action_expert.width],
                &values,
            )?)
        })
        .collect()
}

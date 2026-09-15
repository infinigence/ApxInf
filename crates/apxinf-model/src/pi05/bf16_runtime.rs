//! PI0.5 BF16 execution resources and capture lifecycle.
//!
//! Network mathematics lives in `network`; this module owns input updates,
//! stable workspaces and captured resource lifetime. Existing public runtime
//! methods remain compatibility entry points during the staged migration.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::backend::{kernels, transfers, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use apxinf_core::{Backend, DType, Error, Graph, Result, Tensor};
use apxinf_cuda::CudaArchFamily;
use half::bf16 as HalfBf16;
use kernels::preprocess;

use super::{sinusoidal_time_embedding, Pi05Config, Pi05ImageLayout, StaticBf16Pi05Weights};

pub use super::blocks::Bf16PrefixKvCache;
use super::blocks::{Bf16Blocks, Bf16StepStyles};
use super::network::Pi05Bf16Network;

pub struct Pi05Bf16CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<CudaBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    _styles: Vec<Bf16StepStyles>,
    token_ids: CudaBuffer,
    token_count: usize,
    backend: Arc<RuntimeBackend>,
    // Retain every fixed weight referenced by the captured computation.
    _network: Arc<Pi05Bf16Network>,
    workspace: kernels::GraphWorkspace,
}

impl Pi05Bf16CapturedGraph {
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
                "π0.5 BF16 graph uses raw RGB input; call update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 BF16 graph expects {} token IDs, got {}",
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
            Error::Other("π0.5 BF16 graph uses patch input; call update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "π0.5 BF16 graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 BF16 graph expects {} token IDs, got {}",
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
pub struct Pi05Bf16CudaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    network: Arc<Pi05Bf16Network>,
}

impl Pi05Bf16CudaRuntime {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Pi05Config>,
        weights: Arc<StaticBf16Pi05Weights>,
    ) -> Result<Self> {
        let network = Arc::new(Pi05Bf16Network::from_blocks(Bf16Blocks::new(
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

    fn graph_workspace_bytes(&self, token_count: usize) -> Result<usize> {
        let mut bytes = self.config.cuda_graph_workspace_bytes_bf16(token_count)?;
        if self.ctx().caps().arch_family == CudaArchFamily::Sm80 {
            bytes = bytes
                .checked_add(self.splitkv_workspace_bytes(token_count)?)
                .ok_or_else(|| Error::Other("pi05 BF16 split-KV workspace overflow".into()))?;
        }
        Ok(bytes)
    }

    fn splitkv_workspace_bytes(&self, token_count: usize) -> Result<usize> {
        let patches = self.config.num_views * self.config.patches_per_view();
        let prefix = patches
            .checked_add(token_count)
            .ok_or_else(|| Error::Other("pi05 split-KV prefix length overflow".into()))?;
        let horizon = self.config.action_horizon;
        let action = self.config.action_expert;
        if action.num_heads <= action.num_kv_heads || action.head_dim != 256 || horizon > 64 {
            return Ok(0);
        }
        let key_tokens = prefix
            .checked_add(horizon)
            .ok_or_else(|| Error::Other("pi05 split-KV key length overflow".into()))?;
        let max_splits = key_tokens.div_ceil(64).min(128);
        let lse = max_splits
            .checked_mul(horizon)
            .and_then(|value| value.checked_mul(action.num_heads))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("pi05 split-KV LSE workspace overflow".into()))?;
        let output = max_splits
            .checked_mul(horizon)
            .and_then(|value| value.checked_mul(action.num_heads))
            .and_then(|value| value.checked_mul(action.head_dim))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("pi05 split-KV output workspace overflow".into()))?;
        lse.checked_add(output)
            .and_then(|value| value.checked_mul(self.config.action_expert.depth))
            .and_then(|value| value.checked_mul(self.config.num_flow_steps))
            .ok_or_else(|| Error::Other("pi05 split-KV workspace overflow".into()))
    }

    // Compatibility entry points for existing low-level callers. All eager and
    // capture traversals use the computation owned by Pi05Bf16Network.
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

    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Bf16PrefixKvCache> {
        self.network.prefix_forward(prefix)
    }

    pub fn denoise_step(
        &self,
        state: &Tensor,
        time_embedding: &Tensor,
        prefix: &Bf16PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        self.network.denoise_step(state, time_embedding, prefix, dt)
    }

    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        time_embeddings: &[Tensor],
        prefix: &Bf16PrefixKvCache,
    ) -> Result<Tensor> {
        self.network
            .denoise_all_steps(noise, time_embeddings, prefix)
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

    pub fn calibrate(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<BTreeMap<String, f32>> {
        self.network
            .calibrate(patches, token_ids, token_count, noise, time_embeddings)
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
        styles: &[Bf16StepStyles],
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => {
                self.network
                    .infer_with_styles(patches, token_ids, token_count, noise, styles)
            }
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
                self.network
                    .infer_with_styles(patches, token_ids, token_count, noise, styles)
            }
            _ => Err(Error::Other(
                "π0.5 BF16 raw image capture state is inconsistent".into(),
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
    ) -> Result<Pi05Bf16CapturedGraph> {
        let backend = &self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "π0.5 BF16 raw image capture state is inconsistent".into(),
            ));
        }
        let styles = self.network.prepare_all_styles(time_embeddings)?;
        backend.synchronize()?;
        let workspace = kernels::GraphWorkspace::new(
            self.graph_workspace_bytes(token_count)?,
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
                "GEMM tactic store did not stabilize before PI0.5 BF16 graph capture".into(),
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
        Ok(Pi05Bf16CapturedGraph {
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
    ) -> Result<Pi05Bf16CapturedGraph> {
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
    ) -> Result<Pi05Bf16CapturedGraph> {
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

pub fn upload_time_embeddings_bf16(
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
            .map(HalfBf16::from_f32)
            .collect::<Vec<_>>();
            backend.to_device(&Tensor::from_bf16(
                vec![1, config.action_expert.width],
                &values,
            )?)
        })
        .collect()
}

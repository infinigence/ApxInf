//! WallOSS forward computation: RGB/patch canonical tensors to actions.
//!
//! This module owns the shared forward order (vision tower -> language prefix
//! KV -> fixed-step action denoise), modality connections, and the flow
//! schedule. `ModelVariant` is the private dispatch among constructed
//! precision-specific weight representations; it does not own graph or cache
//! policy, and nothing here references `VlaRequest`, `InferenceSpec`, or
//! runner types. `model_runner/` owns execution policy, workspaces, capture,
//! and input binding.

pub(super) mod blocks;

use std::collections::BTreeMap;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};

use super::backend::{kernels, transfers, CudaEventTimer, DeviceBuffer, RuntimeBackend};
use super::geometry::{DeviceVisionGeometry, RgbViews, VisionGeometry};
use super::schedule::{sinusoidal_time_embedding, solver_times};
use super::weights::WallossActionWeights;
use super::{WallossConfig, WallossImageProcessorConfig};
use blocks::{action_stack, language_prefix, solver_update, vision_tower, PrefixCache};

/// Vision input accepted by the model after the runner bound host input.
pub(super) enum WallossDeviceVision {
    Patches,
    RgbU8 {
        raw_images: DeviceBuffer,
        layout: crate::vla::ImageLayout,
    },
}

/// Device-resident inputs for one fixed-shape WallOSS forward pass. Every
/// buffer lives at a stable address so a captured graph replays in place.
pub(super) struct WallossDeviceInputs {
    pub patches: Tensor,
    pub vision: WallossDeviceVision,
    pub prefix_ids: DeviceBuffer,
    pub vision_row_map: DeviceBuffer,
    pub prefix_position_ids: DeviceBuffer,
    pub action_position_ids: DeviceBuffer,
    /// Initial action latent; generated noise and caller-provided latents are
    /// both written here.
    pub initial_state: Tensor,
    pub action_mask: Tensor,
    pub prefix_tokens: usize,
}

/// Precision-specific device weights assembled by `weights/` at load time.
pub(super) enum WallossDeviceWeights {
    Bf16(super::weights::WallossWeights),
    DynamicFp8(super::weights::WallossDynamicFp8Weights),
}

/// Graph workspace the model needs for one prepared shape.
pub struct WorkspaceRequirements {
    pub bytes: usize,
}

/// Loaded WallOSS computation. `WallossModel` owns forward order and modality
/// connections; `ModelVariant` dispatches among precision-specific weights.
pub(super) struct WallossModel {
    pub backend: Arc<RuntimeBackend>,
    pub config: Arc<WallossConfig>,
    pub image_processor: Option<Arc<WallossImageProcessorConfig>>,
    pub weights: WallossDeviceWeights,
}

/// Fixed time-step assets derived from the checkpoint schedule. These belong
/// to the model because they are checkpoint-fixed forward inputs, not request
/// state.
pub(super) struct WallossSchedule {
    step_deltas: Vec<f32>,
    time_embeddings: Vec<Tensor>,
}

impl WallossSchedule {
    pub(super) fn new(
        config: &WallossConfig,
        backend: &RuntimeBackend,
    ) -> Result<Self> {
        let steps = config.action.solver_steps;
        let times = solver_times(steps, config.action.scheduler_s, 1.0)?;
        let action_tokens = config.action.action_horizon;
        let mut step_deltas = Vec::with_capacity(steps);
        let mut time_embeddings = Vec::with_capacity(steps);
        for step in 0..steps {
            step_deltas.push(times[step + 1] - times[step]);
            let embedding = sinusoidal_time_embedding(times[step], config.action.hidden_size)?;
            let repeated = embedding
                .iter()
                .copied()
                .cycle()
                .take(action_tokens * embedding.len())
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            time_embeddings.push(transfers::to_cuda(
                &Tensor::from_bf16(vec![action_tokens, config.action.hidden_size], &repeated)?,
                backend.device_id(),
            )?);
        }
        Ok(Self {
            step_deltas,
            time_embeddings,
        })
    }
}

pub(super) enum ModelVariant {
    Bf16(WallossModel),
    DynamicFp8(WallossModel),
}

impl ModelVariant {
    pub(super) fn new_bf16(
        backend: Arc<RuntimeBackend>,
        config: Arc<WallossConfig>,
        image_processor: Option<Arc<WallossImageProcessorConfig>>,
        weights: super::weights::WallossWeights,
    ) -> Self {
        Self::Bf16(WallossModel {
            backend,
            config,
            image_processor,
            weights: WallossDeviceWeights::Bf16(weights),
        })
    }

    pub(super) fn new_dynamic_fp8(
        backend: Arc<RuntimeBackend>,
        config: Arc<WallossConfig>,
        image_processor: Option<Arc<WallossImageProcessorConfig>>,
        weights: super::weights::WallossDynamicFp8Weights,
    ) -> Self {
        Self::DynamicFp8(WallossModel {
            backend,
            config,
            image_processor,
            weights: WallossDeviceWeights::DynamicFp8(weights),
        })
    }

    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::Bf16(_) => "bf16",
            Self::DynamicFp8(_) => "fp8_dynamic",
        }
    }

    pub(super) fn model(&self) -> &WallossModel {
        match self {
            Self::Bf16(model) | Self::DynamicFp8(model) => model,
        }
    }
}

impl WallossModel {
    /// Vision geometry for one declared grid list. Geometry is model-owned
    /// because it encodes checkpoint vision topology; the runner uploads and
    /// retains the device copy per prepared shape.
    pub(super) fn geometry(&self, grids: &[[usize; 3]]) -> Result<VisionGeometry> {
        VisionGeometry::new(&self.config.vision, grids)
    }

    /// Raw patch row width: `3 * temporal_patch_size * patch_size^2`.
    pub(super) fn patch_width(&self) -> usize {
        3 * self.config.vision.temporal_patch_size
            * self.config.vision.patch_size
            * self.config.vision.patch_size
    }

    /// Graph workspace for one prepared shape: the high-water mark of every
    /// intermediate allocated during one forward pass, which grows with the
    /// raw patch rows (vision tower) and the prefix length (language prefix
    /// and the per-step action attention over it).
    pub(super) fn workspace_requirements(
        &self,
        patch_rows: usize,
        prefix_tokens: usize,
    ) -> Result<WorkspaceRequirements> {
        Ok(WorkspaceRequirements {
            bytes: super::workspace::workspace_bytes(&self.config, patch_rows, prefix_tokens)?,
        })
    }

    /// Canonical RGB canonicalization: raw `u8` bytes at the stable device
    /// buffer into normalized temporal-merged BF16 patches.
    pub(super) fn preprocess_rgb(
        &self,
        raw_images: &DeviceBuffer,
        patches: &Tensor,
        layout: crate::vla::ImageLayout,
        views: RgbViews,
    ) -> Result<()> {
        let image_processor = self.image_processor.as_ref().ok_or_else(|| {
            Error::Other("walloss RGB input requires checkpoint preprocessor_config.json".into())
        })?;
        kernels::preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16(
            self.backend.context(),
            raw_images,
            patches,
            views.views,
            views.grid_h * self.config.vision.patch_size,
            views.grid_w * self.config.vision.patch_size,
            self.config.vision.patch_size,
            self.config.vision.temporal_patch_size,
            self.config.vision.spatial_merge_size,
            kernel_image_layout(layout),
            image_processor.rescale_factor,
            image_processor.image_mean,
            image_processor.image_std,
        )
    }

    /// Shared forward order: vision tower -> language prefix KV -> fixed-step
    /// action denoise. `geometry` must match the prepared shape.
    pub(super) fn infer(
        &self,
        schedule: &WallossSchedule,
        inputs: &WallossDeviceInputs,
        geometry: &DeviceVisionGeometry,
    ) -> Result<Tensor> {
        self.forward(schedule, inputs, geometry, None)
    }

    /// Eager forward pass with CUDA-event timing of the vision (including RGB
    /// preprocessing), language-prefix and denoise stages.
    pub(super) fn infer_profiled(
        &self,
        schedule: &WallossSchedule,
        inputs: &WallossDeviceInputs,
        geometry: &DeviceVisionGeometry,
    ) -> Result<(Tensor, BTreeMap<String, f64>)> {
        let timers = [
            CudaEventTimer::new()?,
            CudaEventTimer::new()?,
            CudaEventTimer::new()?,
        ];
        let output = self.forward(schedule, inputs, geometry, Some(&timers))?;
        let mut profile = BTreeMap::new();
        for (name, timer) in ["vision_ms", "prefix_ms", "denoise_ms"].iter().zip(&timers) {
            profile.insert((*name).to_string(), timer.elapsed_ms()?);
        }
        Ok((output, profile))
    }

    fn forward(
        &self,
        schedule: &WallossSchedule,
        inputs: &WallossDeviceInputs,
        geometry: &DeviceVisionGeometry,
        timers: Option<&[CudaEventTimer; 3]>,
    ) -> Result<Tensor> {
        let context = self.backend.context();
        let mark = |stage: usize, start: bool| -> Result<()> {
            match timers {
                Some(timers) if start => timers[stage].start(context),
                Some(timers) => timers[stage].stop(context),
                None => Ok(()),
            }
        };
        mark(0, true)?;
        if let WallossDeviceVision::RgbU8 { raw_images, layout } = &inputs.vision {
            let views = geometry.rgb_views.ok_or_else(|| {
                Error::Other("walloss RGB input requires still views of one size".into())
            })?;
            self.preprocess_rgb(raw_images, &inputs.patches, *layout, views)
                .map_err(|e| Error::Other(format!("walloss preprocess_rgb: {e}")))?;
        }
        let prefix = match &self.weights {
            WallossDeviceWeights::Bf16(weights) => self.encode_prefix(
                inputs,
                geometry,
                &weights.vision,
                &weights.language_layers,
                &weights.token_embedding,
                &mark,
            ),
            WallossDeviceWeights::DynamicFp8(weights) => self.encode_prefix(
                inputs,
                geometry,
                &weights.vision,
                &weights.language_layers,
                &weights.token_embedding,
                &mark,
            ),
        }
        .map_err(|e| Error::Other(format!("walloss encode_prefix: {e}")))?;
        mark(2, true)?;
        let output = match &self.weights {
            WallossDeviceWeights::Bf16(weights) => self.denoise(
                schedule,
                inputs,
                &weights.action_layers,
                &weights.action,
                &weights.action_norm,
                &prefix,
            ),
            WallossDeviceWeights::DynamicFp8(weights) => self.denoise(
                schedule,
                inputs,
                &weights.action_layers,
                &weights.action,
                &weights.action_norm,
                &prefix,
            ),
        }
        .map_err(|e| Error::Other(format!("walloss denoise: {e}")))?;
        mark(2, false)?;
        Ok(output)
    }

    fn encode_prefix<V: blocks::VisionTowerWeights, L: blocks::TransformerWeights>(
        &self,
        inputs: &WallossDeviceInputs,
        geometry: &DeviceVisionGeometry,
        vision_weights: &V,
        language_weights: &[L],
        token_embedding: &Tensor,
        mark: &dyn Fn(usize, bool) -> Result<()>,
    ) -> Result<PrefixCache> {
        let vision = vision_tower(
            self.backend.context(),
            &self.config.vision,
            vision_weights,
            geometry,
            &inputs.patches,
        )?;
        mark(0, false)?;
        mark(1, true)?;
        let prefix = language_prefix(
            self.backend.context(),
            &self.config.text,
            language_weights,
            token_embedding,
            &inputs.prefix_ids,
            &vision,
            &inputs.vision_row_map,
            &inputs.prefix_position_ids,
            inputs.prefix_tokens,
            inputs.prefix_tokens + self.config.action.action_horizon,
        )?;
        mark(1, false)?;
        Ok(prefix)
    }

    fn denoise<A: blocks::TransformerWeights>(
        &self,
        schedule: &WallossSchedule,
        inputs: &WallossDeviceInputs,
        transformer_weights: &[A],
        action_weights: &WallossActionWeights,
        action_norm: &Tensor,
        prefix: &PrefixCache,
    ) -> Result<Tensor> {
        let mut state = inputs.initial_state.clone();
        for (step, time_embedding) in schedule.time_embeddings.iter().enumerate() {
            let velocity = action_stack(
                self.backend.context(),
                &self.config.text,
                transformer_weights,
                action_weights,
                action_norm,
                prefix,
                &state,
                &inputs.action_mask,
                time_embedding,
                &inputs.action_position_ids,
            )
            .map_err(|e| Error::Other(format!("walloss action_stack step {step}: {e}")))?;
            state = solver_update(
                self.backend.context(),
                &state,
                &velocity,
                schedule.step_deltas[step],
            )
            .map_err(|e| Error::Other(format!("walloss euler step {step}: {e}")))?;
        }
        Ok(state)
    }
}

fn kernel_image_layout(layout: crate::vla::ImageLayout) -> kernels::preprocess::ImageLayout {
    match layout {
        crate::vla::ImageLayout::Nhwc => kernels::preprocess::ImageLayout::Nhwc,
        crate::vla::ImageLayout::Nchw => kernels::preprocess::ImageLayout::Nchw,
    }
}

#[cfg(test)]
mod tests {
    use apxinf_core::{Backend, DType, Tensor};

    use super::kernels;
    use crate::walloss::backend::{DeviceBuffer, RuntimeBackend};

    const VIEWS: usize = 2;
    const HEIGHT: usize = 8;
    const WIDTH: usize = 12;
    const PATCH: usize = 2;
    const TEMPORAL: usize = 2;
    const MERGE: usize = 2;

    fn preprocess(backend: &RuntimeBackend, input: &DeviceBuffer, output: &Tensor) {
        kernels::preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16(
            backend.context(),
            input,
            output,
            VIEWS,
            HEIGHT,
            WIDTH,
            PATCH,
            TEMPORAL,
            MERGE,
            kernels::preprocess::ImageLayout::Nhwc,
            1.0 / 255.0,
            [0.481_454_66, 0.457_827_5, 0.408_210_72],
            [0.268_629_55, 0.261_302_6, 0.275_777_1],
        )
        .unwrap();
    }

    /// The graph-captured RGB preprocessing reads the stable input buffer on
    /// every replay, so a request's new bytes reach the model.
    #[test]
    fn captured_rgb_preprocess_observes_updated_input_bytes() {
        let backend = RuntimeBackend::new(0).unwrap();
        let bytes = VIEWS * HEIGHT * WIDTH * 3;
        let rows = VIEWS * (HEIGHT / PATCH) * (WIDTH / PATCH);
        let width = 3 * TEMPORAL * PATCH * PATCH;
        let input = DeviceBuffer::alloc(bytes, backend.device_id()).unwrap();
        let output = backend
            .to_device(&Tensor::zeros((rows, width), DType::BF16))
            .unwrap();
        let first = (0..bytes).map(|i| (i * 17 % 256) as u8).collect::<Vec<_>>();
        let changed = (0..bytes).map(|i| (255 - i * 29 % 256) as u8).collect::<Vec<_>>();
        let eager = |data: &[u8]| {
            input.copy_from_host(data).unwrap();
            preprocess(&backend, &input, &output);
            backend.synchronize().unwrap();
            backend.to_cpu(&output).unwrap()
        };
        let expected_changed = eager(&changed);
        let expected_first = eager(&first);

        backend.begin_capture().unwrap();
        preprocess(&backend, &input, &output);
        let graph = backend.end_capture().unwrap();
        let replay = |data: &[u8]| {
            input.copy_from_host(data).unwrap();
            graph.replay().unwrap();
            backend.synchronize().unwrap();
            backend.to_cpu(&output).unwrap()
        };
        assert_eq!(replay(&first).as_bf16().unwrap(), expected_first.as_bf16().unwrap());
        let actual = replay(&changed);
        assert_eq!(actual.as_bf16().unwrap(), expected_changed.as_bf16().unwrap());
        assert_ne!(actual.as_bf16().unwrap(), expected_first.as_bf16().unwrap());
    }
}

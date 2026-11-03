//! PI0.5 Ascend dataflow: vision → prefix/KV → fixed-step action generation.
//!
//! Mirrors `pi05::model` but uses `ascend_pi05::backend` for device ops.

use super::backend::DeviceBuffer;
use apxinf_core::{Error, Result, Tensor};
use std::collections::BTreeMap;
use blocks::Blocks;

mod blocks;
mod calibration;
mod model;

pub(in crate::ascend_pi05) use blocks::L3Policies;
pub use calibration::Pi05CalibrationObserver;
pub use model::{build_bf16_model, upload_time_embeddings_bf16};
pub(in crate::ascend_pi05) use model::{build_bf16_model_with_policies, ModelVariant};

use crate::ascend_pi05::backend::{Context, DeviceBuffer as _DeviceBuffer};
use crate::ascend_pi05::AscendPi05ImageLayout;
use apxinf_core::DType;
use std::sync::Arc;

/// Resource requirements supplied by a model implementation.
pub struct WorkspaceRequirements {
    pub bytes: usize,
}

pub trait PrepareBlocks: Blocks + 'static {
    fn backend(&self) -> &Arc<Context>;
    fn workspace_requirements(&self, tokens: usize) -> Result<WorkspaceRequirements>;
    fn raw_patch_dtype(&self) -> DType;
    fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: AscendPi05ImageLayout,
    ) -> Result<()>;
}

impl<B: PrepareBlocks> Pi05Model<B> {
    pub(in crate::ascend_pi05) fn backend(&self) -> &Arc<Context> {
        self.blocks.backend()
    }
    pub(in crate::ascend_pi05) fn config(&self) -> &crate::ascend_pi05::Pi05Config {
        self.blocks.config()
    }
    pub(in crate::ascend_pi05) fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> Result<WorkspaceRequirements> {
        self.blocks.workspace_requirements(tokens)
    }
}

pub use blocks::bf16::{
    action_layer_bf16, language_layer_bf16, vision_layer_bf16, vision_patch_embed_bf16,
    Bf16ActionLayerOutput, Bf16LanguageLayerOutput,
};
pub use blocks::Bf16PrefixKvCache;
pub type Bf16Model = std::sync::Arc<Pi05Model<blocks::Bf16Blocks>>;

/// Per-layer dump output (still on device; caller copies to host).
pub struct DumpOutput {
    pub action: Tensor,
    pub vision_output: Tensor,
    pub vision_layer_outputs: Vec<Tensor>,
    pub prefix_embedding: Tensor,
    pub language_layer_outputs: Vec<Tensor>,
    /// (state_after_step, action_layer_outputs) per denoise step.
    pub denoise_step_outputs: Vec<(Tensor, Vec<Tensor>)>,
    /// Internal captures from the FIRST transformer block (action_layer[0])
    /// of each denoise step. Length == num_flow_steps. Each entry is Some
    /// when the dump variant ran, None otherwise.
    /// Per-step map: layer_index -> internals. Controlled by env var
    ///  (comma-separated layer indices,
    /// e.g. "0,1"). Defaults to "0" (first block only) to preserve old
    /// behaviour. Empty string disables all internal captures.
    pub denoise_step_first_block_internals: Vec<Option<blocks::bf16::Bf16ActionLayerInternals>>,
    pub num_language_layers: usize,
    pub num_vision_layers: usize,
    pub num_flow_steps: usize,
}


impl<B: PrepareBlocks> Pi05Model<B> {
    pub(in crate::ascend_pi05) fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: AscendPi05ImageLayout,
    ) -> Result<()> {
        self.blocks.preprocess(images, patches, layout)
    }
    pub(in crate::ascend_pi05) fn raw_patch_dtype(&self) -> DType {
        self.blocks.raw_patch_dtype()
    }
}

pub struct Pi05Model<B: Blocks> {
    blocks: B,
}

impl<B: Blocks> Pi05Model<B> {
    pub fn from_blocks(blocks: B) -> Self {
        Self { blocks }
    }

    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        self.blocks.vision(patches, false)
    }

    pub fn embed_prefix(&self, vision: &Tensor, ids: &DeviceBuffer, count: usize) -> Result<Tensor> {
        self.blocks.embed_prefix(vision, ids, count)
    }
    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<B::Prefix> {
        self.blocks.prefix(prefix)
    }
    pub fn prepare_all_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<B::StepModulation>> {
        self.blocks.prepare_modulation(embeddings)
    }
    pub fn denoise_step(
        &self,
        state: &Tensor,
        embedding: &Tensor,
        prefix: &B::Prefix,
        dt: f32,
    ) -> Result<Tensor> {
        self.blocks.step(state, embedding, prefix, dt)
    }

    pub fn denoise_all_steps_with_modulation(
        &self,
        noise: &Tensor,
        modulation: &[B::StepModulation],
        prefix: &B::Prefix,
    ) -> Result<Tensor> {
        let config = self.blocks.config();
        if modulation.len() != config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} precomputed modulation sets, got {}",
                config.num_flow_steps,
                modulation.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -config.flow_start_time / config.num_flow_steps as f32;
        for modulation in modulation {
            state = self
                .blocks
                .step_with_modulation(&state, modulation, prefix, dt)?;
        }
        Ok(state)
    }
    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        embeddings: &[Tensor],
        prefix: &B::Prefix,
    ) -> Result<Tensor> {
        if let Some(modulation) = self.blocks.eager_modulation(embeddings)? {
            return self.denoise_all_steps_with_modulation(noise, &modulation, prefix);
        }
        self.denoise_embeddings(noise, embeddings, prefix)
    }

    fn denoise_embeddings(
        &self,
        noise: &Tensor,
        embeddings: &[Tensor],
        prefix: &B::Prefix,
    ) -> Result<Tensor> {
        let config = self.blocks.config();
        if embeddings.len() != config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} timestep embeddings, got {}",
                config.num_flow_steps,
                embeddings.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -config.flow_start_time / config.num_flow_steps as f32;
        for embedding in embeddings {
            state = self.blocks.step(&state, embedding, prefix, dt)?;
        }
        Ok(state)
    }

    fn infer_impl(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
        native: bool,
        position_offset: Option<usize>,
    ) -> Result<Tensor> {
        let modulation = self.blocks.eager_modulation(embeddings)?;
        let vision = self.blocks.vision(patches, native)?;
        let prefix = self.blocks.embed_prefix(&vision, ids, count)?;
        let prefix = self.blocks.prefix_with_offset(&prefix, position_offset)?;
        match modulation {
            Some(modulation) => self.denoise_all_steps_with_modulation(noise, &modulation, &prefix),
            None => self.denoise_embeddings(noise, embeddings, &prefix),
        }
    }
    pub fn infer(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
        position_offset: Option<usize>,
    ) -> Result<Tensor> {
        self.infer_impl(patches, ids, count, noise, embeddings, false, position_offset)
    }
    pub fn infer_native(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
        position_offset: Option<usize>,
    ) -> Result<Tensor> {
        self.infer_impl(patches, ids, count, noise, embeddings, true, position_offset)
    }
    fn infer_modulation_impl(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        modulation: &[B::StepModulation],
        native: bool,
        position_offset: Option<usize>,
    ) -> Result<Tensor> {
        let vision = self.blocks.vision(patches, native)?;
        let prefix = self.blocks.embed_prefix(&vision, ids, count)?;
        let prefix = self.blocks.prefix_with_offset(&prefix, position_offset)?;
        self.denoise_all_steps_with_modulation(noise, modulation, &prefix)
    }
    pub fn infer_with_modulation(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        modulation: &[B::StepModulation],
        position_offset: Option<usize>,
    ) -> Result<Tensor> {
        self.infer_modulation_impl(patches, ids, count, noise, modulation, false, position_offset)
    }
    pub fn infer_with_native_modulation(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        modulation: &[B::StepModulation],
        position_offset: Option<usize>,
    ) -> Result<Tensor> {
        self.infer_modulation_impl(patches, ids, count, noise, modulation, true, position_offset)
    }
}

/// BF16-specialized inference with per-layer activation dump.
///
/// Mirrors `infer_impl` but captures intermediate tensors at each stage.
/// Only `Bf16Blocks` provides the `*_with_dump` helpers.
impl Pi05Model<blocks::Bf16Blocks> {
    pub fn infer_with_dump(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
        position_offset: Option<usize>,
    ) -> Result<DumpOutput> {
        let config = self.blocks.config();
        let modulation = self.blocks.eager_modulation(embeddings)?;

        // Stage 1: Vision encoder
        let (vision_output, vision_layer_outputs) =
            self.blocks.encode_vision_with_dump(patches)?;

        // Stage 2: Embed prefix (vision + token embedding)
        let prefix_embedding = self.blocks.embed_prefix(&vision_output, ids, count)?;

        // Stage 3: Language model prefix pass (KV cache)
        let (prefix_cache, language_layer_outputs) =
            self.blocks.prefix_forward_with_dump(&prefix_embedding, position_offset)?;

        // Stage 4: Flow-matching denoise loop
        let mut state = noise.clone();
        let dt = -config.flow_start_time / config.num_flow_steps as f32;
        let mut denoise_step_outputs = Vec::with_capacity(config.num_flow_steps);
        let mut denoise_step_first_block_internals: Vec<Option<blocks::bf16::Bf16ActionLayerInternals>> =
            Vec::with_capacity(config.num_flow_steps);

        match modulation {
            Some(modulation) => {
                for modulation in &modulation {
                    let (next_state, action_layer_outputs, internals) = self
                        .blocks
                        .denoise_step_with_modulation_and_dump(&state, modulation, &prefix_cache, dt)?;
                    state = next_state;
                    denoise_step_outputs.push((state.clone(), action_layer_outputs));
                    denoise_step_first_block_internals.push(internals);
                }
            }
            None => {
                for embedding in embeddings {
                    state = self.blocks.step(&state, embedding, &prefix_cache, dt)?;
                    denoise_step_outputs.push((state.clone(), Vec::new()));
                }
            }
        }

        Ok(DumpOutput {
            action: state,
            vision_output,
            vision_layer_outputs,
            prefix_embedding,
            language_layer_outputs,
            denoise_step_outputs,
            denoise_step_first_block_internals,
            num_language_layers: config.language.depth,
            num_vision_layers: config.vision_depth,
            num_flow_steps: config.num_flow_steps,
        })
    }
}

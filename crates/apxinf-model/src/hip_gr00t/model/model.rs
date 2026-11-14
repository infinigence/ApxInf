//! Precision variants of the GR00T model, and the static dispatch over them.
//!
//! Construction and variant dispatch live here, behind the family seam; the
//! runner knows neither [`Blocks`] nor a concrete variant type.

use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_hip::native::HipArena;
use apxinf_hip::HipContext;

use super::super::config::{Gr00tConfig, Qwen3VLConfig};
use super::blocks::{Bf16Blocks, Blocks, PreparedBlocks, VisionOutput, WorkspaceRequirements};
use super::HipGr00tModel;
use crate::hip_gr00t::Embodiment;

/// One loaded GR00T model, precision-dispatched. The runner holds this and
/// nothing below it.
pub enum ModelVariant {
    Bf16 { model: HipGr00tModel<Bf16Blocks> },
}

impl ModelVariant {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bf16 { .. } => "bf16",
        }
    }

    pub fn context(&self) -> &Arc<HipContext> {
        match self {
            Self::Bf16 { model } => model.blocks.context(),
        }
    }

    pub fn config(&self) -> &Gr00tConfig {
        match self {
            Self::Bf16 { model } => model.blocks.config(),
        }
    }

    pub fn backbone_config(&self) -> &Qwen3VLConfig {
        match self {
            Self::Bf16 { model } => model.blocks.backbone_config(),
        }
    }

    /// The requested embodiment's adapters, uploading them on first use.
    ///
    /// One deployment uses one embodiment, so this is normally a single
    /// upload. A request that switches embodiment uploads the new set and
    /// drops the old one's allocations, so alternating costs bandwidth but
    /// does not grow memory.
    pub fn embodiment(&self, id: usize) -> Result<Embodiment> {
        match self {
            Self::Bf16 { model } => model.embodiment(id),
        }
    }

    pub fn prepare_shape(
        &self,
        patches: usize,
        grid_thw: &[[u32; 3]],
        token_ids: &[u32],
        attention_mask: &[u8],
    ) -> Result<PreparedBlocks> {
        match self {
            Self::Bf16 { model } => {
                model
                    .blocks
                    .prepare_shape(patches, grid_thw, token_ids, attention_mask)
            }
        }
    }

    pub fn workspace_requirements(
        &self,
        patches: usize,
        tokens: usize,
    ) -> Result<WorkspaceRequirements> {
        match self {
            Self::Bf16 { model } => model.blocks.workspace_requirements(patches, tokens),
        }
    }

    pub fn infer(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        blocks: &PreparedBlocks,
        embodiment: &Embodiment,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16 { model } => {
                model
                    .blocks
                    .infer(arena, pixel_values, state, noise, blocks, embodiment)
            }
        }
    }

    /// The vision tower alone: the reference's `step2_vision_output` stage.
    pub fn vision_stage(
        &self,
        arena: &HipArena,
        pixel_values: &Tensor,
        blocks: &PreparedBlocks,
    ) -> Result<VisionOutput> {
        match self {
            Self::Bf16 { model } => model
                .blocks
                .vision_stage(arena, pixel_values, &blocks.vision),
        }
    }

    /// The Qwen3-VL text backbone alone: `backbone_features_raw`.
    pub fn backbone_stage(
        &self,
        arena: &HipArena,
        vision_output: &VisionOutput,
        blocks: &PreparedBlocks,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16 { model } => model
                .blocks
                .backbone_stage(arena, vision_output, &blocks.backbone),
        }
    }

    /// The VLLN adapter and VL self-attention blocks alone:
    /// `backbone_features_post_adapter`.
    pub fn adapter_stage(
        &self,
        arena: &HipArena,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16 { model } => model.blocks.adapter_stage(arena, hidden),
        }
    }

    /// The DiT action head alone: split, flow loop, decode.
    #[allow(clippy::too_many_arguments)]
    pub fn action_stage(
        &self,
        arena: &HipArena,
        adapted: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        blocks: &PreparedBlocks,
        embodiment: &Embodiment,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16 { model } => {
                model
                    .blocks
                    .action_stage(arena, adapted, state, noise, &blocks.rows, embodiment)
            }
        }
    }
}

/// Internal static-dispatch seam. The caller supplies an operation; the model
/// knows neither capture policy nor runner. No per-layer virtual calls.
pub(in crate::hip_gr00t) trait ModelOperation {
    type Output;
    fn run<B: Blocks>(self, model: &HipGr00tModel<B>) -> Result<Self::Output>;
}

impl ModelVariant {
    pub(in crate::hip_gr00t) fn with_model<O: ModelOperation>(&self, operation: O) -> Result<O::Output> {
        match self {
            Self::Bf16 { model } => operation.run(model),
        }
    }
}

/// Reject a model whose geometry the port cannot serve, before any upload.
pub fn validate_servable(config: &Gr00tConfig) -> Result<()> {
    if !config.use_alternate_vl_dit {
        return Err(Error::Contract(
            "GR00T on HIP implements AlternateVLDiT only; this checkpoint disables it",
        ));
    }
    if config.num_inference_timesteps == 0 {
        return Err(Error::Contract(
            "GR00T needs at least one inference timestep",
        ));
    }
    Ok(())
}

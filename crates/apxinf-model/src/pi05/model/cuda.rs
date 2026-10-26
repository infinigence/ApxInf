//! One PI0.5 dataflow: vision -> prefix/KV -> fixed-step action generation.
//! Precision, fused topology and physical weight representations live in Blocks.
//! ModelRunner and prepare own execution policy, capture, workspaces and input binding.

use super::blocks::{self, Blocks};
use apxinf_core::{Result, Tensor};

pub use super::calibration::Pi05CalibrationObserver;
pub use super::model::{
    build_bf16_model, build_fp8_static_model, build_int8_dynamic_model,
    upload_time_embeddings_bf16, upload_time_embeddings_fp8_static,
    upload_time_embeddings_int8_dynamic,
};
pub(in crate::pi05) use super::model::{
    build_bf16_model_with_policies, build_fp8_static_model_with_policies,
    build_int8_dynamic_model_with_policies,
};
pub(in crate::pi05) use super::model::{ModelOperation, ModelVariant};
pub(in crate::pi05) use blocks::L3Policies;
#[cfg(test)]
pub(in crate::pi05) use blocks::L3PolicySnapshot;

use crate::pi05::backend::{Context, DeviceBuffer};
use crate::pi05::Pi05ImageLayout;
use apxinf_core::DType;
use std::sync::Arc;
/// Resource requirements supplied by a model implementation. Execution owns
/// allocation, warmup, capture and lifetime; this description allocates nothing.
pub struct WorkspaceRequirements {
    pub bytes: usize,
}
pub trait PrepareBlocks: Blocks<TokenIds = DeviceBuffer> + 'static {
    fn backend(&self) -> &Arc<Context>;
    fn workspace_requirements(&self, tokens: usize) -> Result<WorkspaceRequirements>;
    fn raw_patch_dtype(&self) -> DType;
    fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: Pi05ImageLayout,
    ) -> Result<()>;
}

impl<B: PrepareBlocks> super::Pi05Model<B> {
    pub(in crate::pi05) fn backend(&self) -> &Arc<Context> {
        self.blocks.backend()
    }
    pub(in crate::pi05) fn config(&self) -> &crate::pi05::Pi05Config {
        self.blocks.config()
    }
    pub(in crate::pi05) fn workspace_requirements(
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
pub use blocks::fp8_static::{
    action_layer_fp8_static, language_layer_fp8_static, vision_layer_fp8_static,
    vision_patch_embed_fp8_static, vision_patch_embed_fp8_static_native,
    Fp8StaticActionLayerOutput, Fp8StaticLanguageLayerOutput,
};
pub use blocks::int8_dynamic::{
    action_layer_int8_dynamic, language_layer_int8_dynamic, vision_layer_int8_dynamic,
    vision_patch_embed_int8_dynamic, Int8DynamicActionLayerOutput, Int8DynamicLanguageLayerOutput,
};
pub use blocks::{Bf16PrefixKvCache, Fp8StaticPrefixKvCache, Int8DynamicPrefixKvCache};
pub type Bf16Model = std::sync::Arc<super::Pi05Model<blocks::Bf16Blocks>>;
pub type Fp8StaticModel = std::sync::Arc<super::Pi05Model<blocks::Fp8StaticBlocks>>;
pub type Int8DynamicModel = std::sync::Arc<super::Pi05Model<blocks::Int8DynamicBlocks>>;

impl<B: PrepareBlocks> super::Pi05Model<B> {
    pub(in crate::pi05) fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: Pi05ImageLayout,
    ) -> Result<()> {
        self.blocks.preprocess(images, patches, layout)
    }
    pub(in crate::pi05) fn raw_patch_dtype(&self) -> DType {
        self.blocks.raw_patch_dtype()
    }
}

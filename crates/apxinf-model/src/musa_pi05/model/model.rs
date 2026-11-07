//! Construction and typed dispatch of loaded PI0.5 computation. No execution policy or capture ownership.
use super::blocks::{Bf16Blocks, Int8DynamicBlocks, L3Policies};
use super::Pi05Model;
use super::{Bf16Model, Int8DynamicModel};
use crate::musa_pi05::backend::{self, Context};
use crate::musa_pi05::weights::{Bf16Weights, Int8DynamicWeights};
use crate::musa_pi05::{sinusoidal_time_embedding, Pi05Config};
use apxinf_core::{Result, Tensor};
use std::sync::Arc;
pub fn build_bf16_model(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
    embeddings: &[Tensor],
) -> Result<Arc<Pi05Model<Bf16Blocks>>> {
    build_bf16_model_with_policies(backend, config, weights, L3Policies::default(), embeddings)
}
pub(in crate::musa_pi05) fn build_bf16_model_with_policies(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
    policies: L3Policies,
    embeddings: &[Tensor],
) -> Result<Arc<Pi05Model<Bf16Blocks>>> {
    Ok(Arc::new(Pi05Model::new(
        Bf16Blocks::new(backend, config, weights, policies)?,
        embeddings,
    )?))
}
pub fn build_int8_dynamic_model(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Int8DynamicWeights>,
    embeddings: &[Tensor],
) -> Result<Arc<Pi05Model<Int8DynamicBlocks>>> {
    build_int8_dynamic_model_with_policies(
        backend,
        config,
        weights,
        L3Policies::default(),
        embeddings,
    )
}
pub(in crate::musa_pi05) fn build_int8_dynamic_model_with_policies(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Int8DynamicWeights>,
    policies: L3Policies,
    embeddings: &[Tensor],
) -> Result<Arc<Pi05Model<Int8DynamicBlocks>>> {
    Ok(Arc::new(Pi05Model::new(
        Int8DynamicBlocks::new(backend, config, weights, policies)?,
        embeddings,
    )?))
}
pub fn upload_time_embeddings_bf16(
    config: &Pi05Config,
    context: &Context,
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
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
            backend::to_device(context, &Tensor::from_bf16(
                vec![1, config.action_expert.width],
                &values,
            )?)
        })
        .collect()
}
use crate::musa_pi05::backend::DeviceBuffer;
/// The time conditioning is precision-independent: both routes embed the same
/// schedule and feed it to projections of the same width.
pub use upload_time_embeddings_bf16 as upload_time_embeddings_int8_dynamic;
#[derive(Clone)]
pub(in crate::musa_pi05) enum ModelVariant {
    Bf16 {
        model: Bf16Model,
    },
    Int8Dynamic {
        model: Int8DynamicModel,
        /// Which `int8_dynamic` spelling this is, for reporting
        /// ([`crate::musa_pi05::ModelVariantChoice::as_str`]). The two scopes
        /// share the model type, every dispatch arm and both caches; only
        /// weight preparation and this label differ.
        name: &'static str,
    },
}

impl ModelVariant {
    pub(in crate::musa_pi05) fn name(&self) -> &'static str {
        match self {
            Self::Bf16 { .. } => "bf16",
            Self::Int8Dynamic { name, .. } => name,
        }
    }

    pub(in crate::musa_pi05) fn workspace_requirements(
        &self,
        token_count: usize,
    ) -> Result<super::WorkspaceRequirements> {
        match self {
            Self::Bf16 { model, .. } => model.workspace_requirements(token_count),
            Self::Int8Dynamic { model, .. } => model.workspace_requirements(token_count),
        }
    }

    pub(in crate::musa_pi05) fn infer(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16 { model } => model.infer(patches, token_ids, token_count, noise),
            Self::Int8Dynamic { model, .. } => model.infer(patches, token_ids, token_count, noise),
        }
    }

    pub(in crate::musa_pi05) fn with_model<O: ModelOperation>(&self, operation: O) -> Result<O::Output> {
        match self {
            Self::Bf16 { model } => operation.run(model),
            Self::Int8Dynamic { model, .. } => operation.run(model),
        }
    }
}

impl ModelVariant {
    pub(in crate::musa_pi05) fn preprocess_rgb(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: crate::musa_pi05::Pi05ImageLayout,
    ) -> Result<()> {
        use super::PrepareBlocks;
        match self {
            Self::Bf16 { model, .. } => model.blocks.preprocess(images, patches, layout),
            Self::Int8Dynamic { model, .. } => model.blocks.preprocess(images, patches, layout),
        }
    }
}

/// Internal static-dispatch seam. The caller supplies an operation; Model
/// knows neither capture policy nor ModelRunner. No per-layer virtual calls.
pub(in crate::musa_pi05) trait ModelOperation {
    type Output;
    fn run<B: super::PrepareBlocks>(self, model: &Arc<Pi05Model<B>>) -> Result<Self::Output>;
}

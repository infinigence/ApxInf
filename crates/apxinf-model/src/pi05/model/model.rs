//! Construction and typed dispatch of loaded PI0.5 computation. No execution policy or capture ownership.
use super::blocks::{Bf16Blocks, Fp8StaticBlocks, Int8DynamicBlocks, L3Policies};
use super::Pi05Model;
use super::{Bf16Model, Fp8StaticModel, Int8DynamicModel};
use crate::pi05::backend::{self, Context};
use crate::pi05::weights::{
    Bf16Weights, Fp8StaticActivationScales, Fp8StaticWeights, Int8DynamicWeights,
};
use crate::pi05::{sinusoidal_time_embedding, Pi05Config};
use apxinf_core::{Result, Tensor};
use std::sync::Arc;
pub fn build_bf16_model(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
) -> Result<Arc<Pi05Model<Bf16Blocks>>> {
    build_bf16_model_with_policies(backend, config, weights, L3Policies::default())
}
pub(in crate::pi05) fn build_bf16_model_with_policies(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
    policies: L3Policies,
) -> Result<Arc<Pi05Model<Bf16Blocks>>> {
    Ok(Arc::new(Pi05Model::from_blocks(Bf16Blocks::new(
        backend, config, weights, policies,
    )?)))
}
pub fn build_fp8_static_model(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Fp8StaticWeights>,
    scales: Arc<Fp8StaticActivationScales>,
) -> Result<Arc<Pi05Model<Fp8StaticBlocks>>> {
    build_fp8_static_model_with_policies(backend, config, weights, scales, L3Policies::default())
}
pub(in crate::pi05) fn build_fp8_static_model_with_policies(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Fp8StaticWeights>,
    scales: Arc<Fp8StaticActivationScales>,
    policies: L3Policies,
) -> Result<Arc<Pi05Model<Fp8StaticBlocks>>> {
    Ok(Arc::new(Pi05Model::from_blocks(Fp8StaticBlocks::new(
        backend, config, weights, scales, policies,
    )?)))
}
pub fn build_int8_dynamic_model(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Int8DynamicWeights>,
) -> Result<Arc<Pi05Model<Int8DynamicBlocks>>> {
    build_int8_dynamic_model_with_policies(backend, config, weights, L3Policies::default())
}
pub(in crate::pi05) fn build_int8_dynamic_model_with_policies(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Int8DynamicWeights>,
    policies: L3Policies,
) -> Result<Arc<Pi05Model<Int8DynamicBlocks>>> {
    Ok(Arc::new(Pi05Model::from_blocks(Int8DynamicBlocks::new(
        backend, config, weights, policies,
    )?)))
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
pub fn upload_time_embeddings_fp8_static(
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
            .map(half::f16::from_f32)
            .collect::<Vec<_>>();
            let tensor = Tensor::from_f16(vec![1, config.action_expert.width], &values)?;
            backend::to_device(context, &tensor)
        })
        .collect()
}

pub use upload_time_embeddings_bf16 as upload_time_embeddings_int8_dynamic;

use crate::pi05::backend::DeviceBuffer;
use apxinf_core::{DType, Error};
#[derive(Clone)]
pub(in crate::pi05) enum ModelVariant {
    Fp8Static {
        model: Fp8StaticModel,
        time_embeddings: Arc<Vec<Tensor>>,
    },
    Bf16 {
        model: Bf16Model,
        time_embeddings: Arc<Vec<Tensor>>,
    },
    Int8Dynamic {
        model: Int8DynamicModel,
        time_embeddings: Arc<Vec<Tensor>>,
    },
}

impl ModelVariant {
    pub(in crate::pi05) fn name(&self) -> &'static str {
        match self {
            Self::Bf16 { .. } => "bf16",
            Self::Fp8Static { .. } => "fp8_static",
            Self::Int8Dynamic { .. } => "int8_dynamic",
        }
    }

    #[cfg(test)]
    pub(in crate::pi05) fn l3_policy_snapshot(&self) -> super::L3PolicySnapshot {
        match self {
            Self::Bf16 { model, .. } => model.blocks.policies.snapshot(),
            Self::Fp8Static { model, .. } => model.blocks.policies.snapshot(),
            Self::Int8Dynamic { model, .. } => model.blocks.policies.snapshot(),
        }
    }

    pub(in crate::pi05) fn input_dtype(&self) -> DType {
        match self {
            Self::Fp8Static { .. } => DType::F16,
            Self::Bf16 { .. } | Self::Int8Dynamic { .. } => DType::BF16,
        }
    }

    pub(in crate::pi05) fn captured_patch_dtype(&self, raw_rgb: bool) -> DType {
        match (self, raw_rgb) {
            (Self::Fp8Static { .. }, true) => DType::F8E4M3,
            _ => self.input_dtype(),
        }
    }

    pub(in crate::pi05) fn workspace_requirements(
        &self,
        token_count: usize,
    ) -> Result<super::WorkspaceRequirements> {
        match self {
            Self::Bf16 { model, .. } => model.workspace_requirements(token_count),
            Self::Fp8Static { model, .. } => model.workspace_requirements(token_count),
            Self::Int8Dynamic { model, .. } => model.workspace_requirements(token_count),
        }
    }

    pub(in crate::pi05) fn infer(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
        prequantized_fp8_patches: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Fp8Static {
                model,
                time_embeddings,
                ..
            } if prequantized_fp8_patches => {
                model.infer_native(patches, token_ids, token_count, noise, time_embeddings)
            }
            Self::Fp8Static {
                model,
                time_embeddings,
                ..
            } => model.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::Bf16 {
                model,
                time_embeddings,
            } => model.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::Int8Dynamic {
                model,
                time_embeddings,
            } => model.infer(patches, token_ids, token_count, noise, time_embeddings),
        }
    }

    pub(in crate::pi05) fn with_model<O: ModelOperation>(&self, operation: O) -> Result<O::Output> {
        match self {
            Self::Bf16 {
                model,
                time_embeddings,
            } => operation.run(model, time_embeddings),
            Self::Fp8Static {
                model,
                time_embeddings,
            } => operation.run(model, time_embeddings),
            Self::Int8Dynamic {
                model,
                time_embeddings,
            } => operation.run(model, time_embeddings),
        }
    }
}

impl ModelVariant {
    pub(in crate::pi05) fn preprocess_rgb(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: crate::pi05::Pi05ImageLayout,
    ) -> Result<()> {
        use super::PrepareBlocks;
        match self {
            Self::Bf16 { model, .. } => model.blocks.preprocess(images, patches, layout),
            Self::Fp8Static { model, .. } => model.blocks.preprocess(images, patches, layout),
            Self::Int8Dynamic { model, .. } => model.blocks.preprocess(images, patches, layout),
        }
    }
    pub(in crate::pi05) fn calibrate(
        &self,
        patches: &Tensor,
        tokens: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
    ) -> Result<std::collections::BTreeMap<String, f32>> {
        match self {
            Self::Bf16 {
                model,
                time_embeddings,
            } => model.calibrate(patches, tokens, count, noise, time_embeddings),
            _ => Err(Error::Other(
                "PI0.5 calibration requires model_variant=bf16".into(),
            )),
        }
    }
}

/// Internal static-dispatch seam. The caller supplies an operation; Model
/// knows neither capture policy nor ModelRunner. No per-layer virtual calls.
pub(in crate::pi05) trait ModelOperation {
    type Output;
    fn run<B: super::PrepareBlocks>(
        self,
        model: &Arc<Pi05Model<B>>,
        embeddings: &[Tensor],
    ) -> Result<Self::Output>;
}

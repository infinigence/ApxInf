//! Construction and typed dispatch of loaded PI0.5 computation (Ascend, BF16).
use super::blocks::{Bf16Blocks, L3Policies};
use super::Pi05Model;
use super::Bf16Model;
use crate::ascend_pi05::backend::{self, Context};
use crate::ascend_pi05::weights::Bf16Weights;
use crate::ascend_pi05::{sinusoidal_time_embedding, Pi05Config};
use apxinf_core::{Result, Tensor};
use std::sync::Arc;

pub fn build_bf16_model(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
) -> Result<Arc<Pi05Model<Bf16Blocks>>> {
    build_bf16_model_with_policies(backend, config, weights, L3Policies::default())
}

pub(in crate::ascend_pi05) fn build_bf16_model_with_policies(
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
    policies: L3Policies,
) -> Result<Arc<Pi05Model<Bf16Blocks>>> {
    Ok(Arc::new(Pi05Model::from_blocks(Bf16Blocks::new(
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

use crate::ascend_pi05::backend::DeviceBuffer;
use apxinf_core::{DType, Error};

#[derive(Clone)]
pub(in crate::ascend_pi05) enum ModelVariant {
    Bf16 {
        model: Bf16Model,
        time_embeddings: Arc<Vec<Tensor>>,
    },
}

impl ModelVariant {
    pub(in crate::ascend_pi05) fn name(&self) -> &'static str {
        match self {
            Self::Bf16 { .. } => "bf16",
        }
    }

    pub(in crate::ascend_pi05) fn input_dtype(&self) -> DType {
        match self {
            Self::Bf16 { .. } => DType::BF16,
        }
    }

    pub(in crate::ascend_pi05) fn captured_patch_dtype(&self, _raw_rgb: bool) -> DType {
        self.input_dtype()
    }

    pub(in crate::ascend_pi05) fn workspace_requirements(
        &self,
        token_count: usize,
    ) -> Result<super::WorkspaceRequirements> {
        match self {
            Self::Bf16 { model, .. } => model.workspace_requirements(token_count),
        }
    }

    pub(in crate::ascend_pi05) fn infer(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
        _prequantized_fp8_patches: bool,
        position_offset: Option<usize>,
    ) -> Result<Tensor> {
        match self {
            Self::Bf16 {
                model,
                time_embeddings,
            } => model.infer(patches, token_ids, token_count, noise, time_embeddings, position_offset),
        }
    }
    pub(in crate::ascend_pi05) fn infer_with_dump(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
        position_offset: Option<usize>,
    ) -> Result<super::DumpOutput> {
        match self {
            Self::Bf16 {
                model,
                time_embeddings,
            } => model.infer_with_dump(patches, token_ids, token_count, noise, time_embeddings, position_offset),
        }
    }

    pub(in crate::ascend_pi05) fn with_model<O: ModelOperation>(
        &self,
        operation: O,
    ) -> Result<O::Output> {
        match self {
            Self::Bf16 {
                model,
                time_embeddings,
            } => operation.run(model, time_embeddings),
        }
    }
}

impl ModelVariant {
    pub(in crate::ascend_pi05) fn preprocess_rgb(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: crate::ascend_pi05::AscendPi05ImageLayout,
    ) -> Result<()> {
        use super::PrepareBlocks;
        match self {
            Self::Bf16 { model, .. } => model.blocks.preprocess(images, patches, layout),
        }
    }
    pub(in crate::ascend_pi05) fn calibrate(
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

/// Internal static-dispatch seam.
pub(in crate::ascend_pi05) trait ModelOperation {
    type Output;
    fn run<B: super::PrepareBlocks>(
        self,
        model: &Arc<Pi05Model<B>>,
        embeddings: &[Tensor],
    ) -> Result<Self::Output>;
}

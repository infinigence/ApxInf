//! Construction and typed dispatch of loaded PI0.5 computation. No execution policy or capture ownership.
use super::blocks::{Bf16Blocks, Fp8StaticBlocks, Int8DynamicBlocks};
use super::Pi05Network;
use super::{Bf16Network, Fp8StaticNetwork, Int8DynamicNetwork};
use crate::pi05::backend::RuntimeBackend;
use crate::pi05::weights::{
    Bf16Weights, Fp8StaticActivationScales, Fp8StaticWeights, Int8DynamicWeights,
};
use crate::pi05::{sinusoidal_time_embedding, Pi05Config};
use apxinf_core::{Backend, Result, Tensor};
use std::sync::Arc;
pub fn build_bf16_network(
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<Bf16Weights>,
) -> Result<Arc<Pi05Network<Bf16Blocks>>> {
    Ok(Arc::new(Pi05Network::from_blocks(Bf16Blocks::new(
        backend, config, weights,
    )?)))
}
pub fn build_fp8_static_network(
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<Fp8StaticWeights>,
    scales: Arc<Fp8StaticActivationScales>,
) -> Result<Arc<Pi05Network<Fp8StaticBlocks>>> {
    Ok(Arc::new(Pi05Network::from_blocks(Fp8StaticBlocks::new(
        backend, config, weights, scales,
    )?)))
}
pub fn build_int8_dynamic_network(
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<Int8DynamicWeights>,
) -> Result<Arc<Pi05Network<Int8DynamicBlocks>>> {
    Ok(Arc::new(Pi05Network::from_blocks(Int8DynamicBlocks::new(
        backend, config, weights,
    )?)))
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
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>();
            backend.to_device(&Tensor::from_bf16(
                vec![1, config.action_expert.width],
                &values,
            )?)
        })
        .collect()
}
pub fn upload_time_embeddings_fp8_static(
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
            .map(half::f16::from_f32)
            .collect::<Vec<_>>();
            let tensor = Tensor::from_f16(vec![1, config.action_expert.width], &values)?;
            backend.to_device(&tensor)
        })
        .collect()
}

pub use upload_time_embeddings_bf16 as upload_time_embeddings_int8_dynamic;

use crate::pi05::backend::DeviceBuffer;
use apxinf_core::{DType, Error};
#[derive(Clone)]
pub(in crate::pi05) enum LoadedCompute {
    Fp8Static {
        network: Fp8StaticNetwork,
        time_embeddings: Arc<Vec<Tensor>>,
    },
    Bf16 {
        network: Bf16Network,
        time_embeddings: Arc<Vec<Tensor>>,
    },
    Int8Dynamic {
        network: Int8DynamicNetwork,
        time_embeddings: Arc<Vec<Tensor>>,
    },
}

impl LoadedCompute {
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
                network,
                time_embeddings,
                ..
            } if prequantized_fp8_patches => {
                network.infer_native(patches, token_ids, token_count, noise, time_embeddings)
            }
            Self::Fp8Static {
                network,
                time_embeddings,
                ..
            } => network.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::Bf16 {
                network,
                time_embeddings,
            } => network.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::Int8Dynamic {
                network,
                time_embeddings,
            } => network.infer(patches, token_ids, token_count, noise, time_embeddings),
        }
    }

    pub(in crate::pi05) fn with_network<O: NetworkOperation>(
        &self,
        operation: O,
    ) -> Result<O::Output> {
        match self {
            Self::Bf16 {
                network,
                time_embeddings,
            } => operation.run(network, time_embeddings),
            Self::Fp8Static {
                network,
                time_embeddings,
            } => operation.run(network, time_embeddings),
            Self::Int8Dynamic {
                network,
                time_embeddings,
            } => operation.run(network, time_embeddings),
        }
    }
}

impl LoadedCompute {
    pub(in crate::pi05) fn preprocess_rgb(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: crate::pi05::Pi05ImageLayout,
    ) -> Result<()> {
        use super::PrepareBlocks;
        match self {
            Self::Bf16 { network, .. } => network.blocks.preprocess(images, patches, layout),
            Self::Fp8Static { network, .. } => network.blocks.preprocess(images, patches, layout),
            Self::Int8Dynamic { network, .. } => network.blocks.preprocess(images, patches, layout),
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
                network,
                time_embeddings,
            } => network.calibrate(patches, tokens, count, noise, time_embeddings),
            _ => Err(Error::Other(
                "PI0.5 calibration requires compute_variant=bf16".into(),
            )),
        }
    }
}

/// Internal static-dispatch seam. The caller supplies an operation; Network
/// knows neither capture policy nor Session. No per-layer virtual calls.
pub(in crate::pi05) trait NetworkOperation {
    type Output;
    fn run<B: super::PrepareBlocks>(
        self,
        network: &Arc<Pi05Network<B>>,
        embeddings: &[Tensor],
    ) -> Result<Self::Output>;
}

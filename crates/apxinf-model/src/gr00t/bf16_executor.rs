//! Native-BF16 GR00T layer composition.

use std::collections::BTreeMap;
use std::sync::Arc;

use apxinf_core::Result;

use super::action_weights::Gr00tLinearWeights;
use super::backbone::{Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights};
use super::backend::RuntimeBackend;
use super::bf16_weights::Gr00tBf16LinearWeights;
use super::executor::{Gr00tExecutor, Gr00tPrecisionExecution};
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

pub(super) struct Gr00tBf16Execution;

impl Gr00tPrecisionExecution for Gr00tBf16Execution {
    type Dense = Gr00tBf16LinearWeights;
    type FeedForward = Gr00tBf16LinearWeights;
    type FusedQkv = Gr00tBf16LinearWeights;
    type Backbone = Gr00tBf16LinearWeights;

    const NAME: &'static str = "bf16";
    const SUPPORTS_CALIBRATION: bool = true;

    fn transfer_dense(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::Dense> {
        Gr00tBf16LinearWeights::from_host(weights, name, backend)
    }

    fn transfer_feed_forward(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::FeedForward> {
        Gr00tBf16LinearWeights::from_host(weights, name, backend)
    }

    fn transfer_fused_qkv(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::FusedQkv>> {
        Ok(Some(Gr00tBf16LinearWeights::from_host(
            weights, name, backend,
        )?))
    }

    fn transfer_backbone_linears(
        &self,
        _text: &Qwen3VLTextWeights,
        _vision: &Qwen3VLVisionWeights,
        _backend: &RuntimeBackend,
    ) -> Result<BTreeMap<String, Self::Backbone>> {
        Ok(BTreeMap::new())
    }
}

pub(super) type Gr00tBf16Executor = Gr00tExecutor<Gr00tBf16Execution>;

pub(super) fn build(
    config: Gr00tConfig,
    backbone: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tBf16Executor> {
    Gr00tExecutor::from_backend(config, backbone, weights, Gr00tBf16Execution, backend)
}

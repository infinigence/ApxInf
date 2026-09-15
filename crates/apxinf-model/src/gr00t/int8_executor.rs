//! W8A8 INT8 GR00T layer composition for Orin-class GPUs.

use std::collections::BTreeMap;
use std::sync::Arc;

use apxinf_core::Result;

use super::action_weights::Gr00tLinearWeights;
use super::backbone::{Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights};
use super::backend::RuntimeBackend;
use super::bf16_weights::Gr00tBf16LinearWeights;
use super::executor::{Gr00tExecutor, Gr00tPrecisionExecution};
use super::int8_weights::Gr00tInt8LinearWeights;
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

pub(super) struct Gr00tInt8Execution;

impl Gr00tPrecisionExecution for Gr00tInt8Execution {
    // GR00T's validated Orin plan keeps attention, adapters, encoders, and
    // decoders in BF16.  Only FFN matrices and eligible fused self-QKV
    // projections use W8A8.
    type Dense = Gr00tBf16LinearWeights;
    type FeedForward = Gr00tInt8LinearWeights;
    type FusedQkv = Gr00tInt8LinearWeights;
    type Backbone = Gr00tInt8LinearWeights;

    const NAME: &'static str = "int8";
    const SUPPORTS_CALIBRATION: bool = false;

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
        _name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::FeedForward> {
        Gr00tInt8LinearWeights::linear(weights, backend)
    }

    fn transfer_fused_qkv(
        &self,
        weights: Gr00tLinearWeights,
        _name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::FusedQkv>> {
        if weights.weight.shape().dims() == [1536, 4608] {
            Ok(Some(Gr00tInt8LinearWeights::linear(weights, backend)?))
        } else {
            Ok(None)
        }
    }

    fn transfer_backbone_linears(
        &self,
        text: &Qwen3VLTextWeights,
        _vision: &Qwen3VLVisionWeights,
        backend: &RuntimeBackend,
    ) -> Result<BTreeMap<String, Self::Backbone>> {
        let mut output = BTreeMap::new();
        for (index, layer) in text.layers.iter().enumerate() {
            let prefix = format!("backbone.text.layers.{index}");
            for (projection, weight) in [
                ("gate", &layer.w_gate),
                ("up", &layer.w_up),
                ("down", &layer.w_down),
            ] {
                output.insert(
                    format!("{prefix}.{projection}"),
                    Gr00tInt8LinearWeights::matrix(weight, backend)?,
                );
            }
        }
        Ok(output)
    }
}

pub(super) type Gr00tInt8Executor = Gr00tExecutor<Gr00tInt8Execution>;

pub(super) fn build(
    config: Gr00tConfig,
    backbone: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tInt8Executor> {
    Gr00tExecutor::from_backend(config, backbone, weights, Gr00tInt8Execution, backend)
}

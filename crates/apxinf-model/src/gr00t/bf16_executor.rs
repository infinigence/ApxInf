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

    const USE_BIAS_RESIDUAL_ADAPTIVE_LAYER_NORM: bool = true;
    const USE_BIAS_RESIDUAL_LAYER_NORM: bool = true;
    const NAME: &'static str = "bf16";
    const SUPPORTS_CALIBRATION: bool = true;
    const USE_HDIM96_BM64_ATTENTION: bool = true;
    const USE_FUSED_QK_RMS_MROPE: bool = true;
    const OPTIMIZED_SMS: &'static [u32] = &[87, 110];
    const USE_CACHED_VISION_RESIDUAL_LAYER_NORM: bool = true;
    const USE_VISION_POINTWISE_PACK: bool = true;
    const USE_BATCHED_VISION_ATTENTION: bool = true;
    const VISION_POINTWISE_LEGACY_ENV: Option<&'static str> =
        Some("APXINF_GR00T_BF16_LEGACY_VISION_POINTWISE_PACK");
    const QK_RMS_MROPE_LEGACY_ENV: Option<&'static str> =
        Some("APXINF_GR00T_BF16_LEGACY_QK_RMS_MROPE");
    const USE_DIRECT_BF16_VISION_SEGMENT_OUTPUT: bool = true;
    const USE_PRECOMPUTED_BF16_VISION_ROPE: bool = true;
    const USE_FUSED_BF16_VISION_MLP_RESIDUAL_NORM: bool = true;
    const USE_PACKED8_BF16_BIAS_ACTIVATION: bool = true;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_explicitly_opts_into_hdim96_bm64_attention() {
        assert!(Gr00tBf16Execution::USE_HDIM96_BM64_ATTENTION);
    }

    #[test]
    fn bf16_explicitly_opts_into_qk_rms_mrope_fusion() {
        assert!(Gr00tBf16Execution::USE_FUSED_QK_RMS_MROPE);
    }

    #[test]
    fn bf16_explicitly_opts_into_direct_vision_segment_output() {
        assert!(Gr00tBf16Execution::USE_DIRECT_BF16_VISION_SEGMENT_OUTPUT);
    }

    #[test]
    fn bf16_explicitly_opts_into_precomputed_vision_rope() {
        assert!(Gr00tBf16Execution::USE_PRECOMPUTED_BF16_VISION_ROPE);
    }

    #[test]
    fn bf16_explicitly_opts_into_vision_mlp_residual_norm_fusion() {
        assert!(Gr00tBf16Execution::USE_FUSED_BF16_VISION_MLP_RESIDUAL_NORM);
    }

    #[test]
    fn bf16_explicitly_opts_into_packed8_bias_activation() {
        assert!(Gr00tBf16Execution::USE_PACKED8_BF16_BIAS_ACTIVATION);
    }

    #[test]
    fn bf16_explicitly_opts_into_bias_residual_adaptive_layer_norm() {
        assert!(Gr00tBf16Execution::USE_BIAS_RESIDUAL_ADAPTIVE_LAYER_NORM);
    }

    #[test]
    fn bf16_explicitly_opts_into_bias_residual_layer_norm() {
        assert!(Gr00tBf16Execution::USE_BIAS_RESIDUAL_LAYER_NORM);
    }
}

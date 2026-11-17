use super::action_weights::Gr00tLinearWeights;
use super::backbone::{Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights};
use super::backend::RuntimeBackend;
use super::executor::Gr00tPrecisionExecution;
use super::precision::Gr00tBf16Execution;
use super::weights::fp8::Gr00tFp8LinearWeights;
use super::weights::fp8_calibration::Gr00tFp8Calibration;
use super::Gr00tConfig;
use apxinf_core::{Result, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

pub(crate) struct Gr00tFp8Execution {
    calibration: Gr00tFp8Calibration,
}

impl Gr00tFp8Execution {
    pub(crate) fn from_file(
        path: &Path,
        checkpoint: &Path,
        backbone: &Path,
        config: &Gr00tConfig,
        backbone_config: &Qwen3VLConfig,
    ) -> Result<Self> {
        let consumers = super::calibration::fp8_consumers(config, backbone_config);
        Ok(Self {
            calibration: Gr00tFp8Calibration::from_json_file(
                path, checkpoint, backbone, &consumers,
            )?,
        })
    }

    fn linear(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Gr00tFp8LinearWeights> {
        let scale = if Gr00tFp8LinearWeights::candidate(&weights.weight, name) {
            self.calibration.scale(name)?
        } else {
            1.0
        };
        Gr00tFp8LinearWeights::from_host(weights, name, scale, backend)
    }
}

impl Gr00tPrecisionExecution for Gr00tFp8Execution {
    type Dense = Gr00tFp8LinearWeights;
    type FeedForward = Gr00tFp8LinearWeights;
    type FusedQkv = Gr00tFp8LinearWeights;
    type Backbone = Gr00tFp8LinearWeights;
    const NAME: &'static str = "fp8";
    const SUPPORTS_CALIBRATION: bool = false;
    const USE_FUSED_QK_RMS_MROPE: bool = Gr00tBf16Execution::USE_FUSED_QK_RMS_MROPE;
    const QK_RMS_MROPE_LEGACY_ENV: Option<&'static str> =
        Gr00tBf16Execution::QK_RMS_MROPE_LEGACY_ENV;
    const USE_PRECOMPUTED_BF16_VISION_ROPE: bool =
        Gr00tBf16Execution::USE_PRECOMPUTED_BF16_VISION_ROPE;
    const USE_FUSED_BF16_VISION_MLP_RESIDUAL_NORM: bool =
        Gr00tBf16Execution::USE_FUSED_BF16_VISION_MLP_RESIDUAL_NORM;
    const USE_PACKED8_BF16_BIAS_ACTIVATION: bool =
        Gr00tBf16Execution::USE_PACKED8_BF16_BIAS_ACTIVATION;
    const USE_BIAS_RESIDUAL_ADAPTIVE_LAYER_NORM: bool =
        Gr00tBf16Execution::USE_BIAS_RESIDUAL_ADAPTIVE_LAYER_NORM;
    const USE_BIAS_RESIDUAL_LAYER_NORM: bool = Gr00tBf16Execution::USE_BIAS_RESIDUAL_LAYER_NORM;
    const USE_CACHED_VISION_RESIDUAL_LAYER_NORM: bool =
        Gr00tBf16Execution::USE_CACHED_VISION_RESIDUAL_LAYER_NORM;
    const USE_BATCHED_VISION_ATTENTION: bool = Gr00tBf16Execution::USE_BATCHED_VISION_ATTENTION;

    fn transfer_dense(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::Dense> {
        self.linear(weights, name, backend)
    }
    fn transfer_feed_forward(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::FeedForward> {
        self.linear(weights, name, backend)
    }
    fn transfer_fused_qkv(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::FusedQkv>> {
        Ok(Some(self.linear(weights, name, backend)?))
    }
    fn transfer_backbone_linears(
        &self,
        text: &Qwen3VLTextWeights,
        vision: &Qwen3VLVisionWeights,
        backend: &RuntimeBackend,
    ) -> Result<BTreeMap<String, Self::Backbone>> {
        let mut output = BTreeMap::new();
        let mut insert = |name: String, weight: &Tensor| -> Result<()> {
            if Gr00tFp8LinearWeights::candidate(weight, &name) {
                let linear = Gr00tFp8LinearWeights::from_device_matrix(
                    weight,
                    &name,
                    self.calibration.scale(&name)?,
                    backend,
                )?;
                output.insert(name, linear);
            }
            Ok(())
        };
        for (index, layer) in text.layers.iter().enumerate() {
            for (projection, weight) in [
                ("query", &layer.wq),
                ("key", &layer.wk),
                ("value", &layer.wv),
                ("output", &layer.wo),
                ("gate", &layer.w_gate),
                ("up", &layer.w_up),
                ("down", &layer.w_down),
            ] {
                insert(format!("backbone.text.layers.{index}.{projection}"), weight)?;
            }
        }
        insert(
            "backbone.vision.patch_embed".into(),
            &vision.patch_embed_weight,
        )?;
        for (index, block) in vision.blocks.iter().enumerate() {
            for (projection, weight) in [
                ("qkv", &block.qkv_w),
                ("output", &block.proj_w),
                ("fc1", &block.fc1_w),
                ("fc2", &block.fc2_w),
            ] {
                insert(
                    format!("backbone.vision.blocks.{index}.{projection}"),
                    weight,
                )?;
            }
        }
        insert("backbone.vision.merger.fc1".into(), &vision.merger.fc1_w)?;
        insert("backbone.vision.merger.fc2".into(), &vision.merger.fc2_w)?;
        for (index, merger) in vision.deepstack_mergers.iter().enumerate() {
            insert(
                format!("backbone.vision.deepstack.{index}.fc1"),
                &merger.fc1_w,
            )?;
            insert(
                format!("backbone.vision.deepstack.{index}.fc2"),
                &merger.fc2_w,
            )?;
        }
        Ok(output)
    }
}

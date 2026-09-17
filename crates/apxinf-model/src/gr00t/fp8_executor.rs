//! Static-FP8 GR00T layer composition for Thor-class GPUs.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};

use super::action_weights::Gr00tLinearWeights;
use super::backbone::{Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights};
use super::backend::RuntimeBackend;
use super::executor::{Gr00tExecutor, Gr00tPrecisionExecution};
use super::fp8_weights::{Gr00tFp8Calibration, Gr00tFp8LinearWeights};
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

pub(super) struct Gr00tFp8Execution {
    calibration: Gr00tFp8Calibration,
}

impl Gr00tFp8Execution {
    pub(super) fn from_file(
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
        Gr00tFp8LinearWeights::from_host(weights, self.calibration.scale(name)?, backend)
    }

    fn matrix(
        &self,
        weight: &Tensor,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Gr00tFp8LinearWeights> {
        Gr00tFp8LinearWeights::matrix(weight, self.calibration.scale(name)?, backend)
    }
}

impl Gr00tPrecisionExecution for Gr00tFp8Execution {
    type Dense = Gr00tFp8LinearWeights;
    type FeedForward = Gr00tFp8LinearWeights;
    type FusedQkv = Gr00tFp8LinearWeights;
    type Backbone = Gr00tFp8LinearWeights;

    const NAME: &'static str = "fp8";
    const SUPPORTS_CALIBRATION: bool = false;

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
        _weights: Gr00tLinearWeights,
        _name: &str,
        _backend: &RuntimeBackend,
    ) -> Result<Option<Self::FusedQkv>> {
        // The static-FP8 execution plan assigns independent Q/K/V activation
        // scales, so concatenating them would erase calibrated boundaries.
        Ok(None)
    }

    fn transfer_backbone_linears(
        &self,
        text: &Qwen3VLTextWeights,
        vision: &Qwen3VLVisionWeights,
        backend: &RuntimeBackend,
    ) -> Result<BTreeMap<String, Self::Backbone>> {
        let mut output = BTreeMap::new();
        for (index, layer) in text.layers.iter().enumerate() {
            let projections = [
                ("query", &layer.wq),
                ("key", &layer.wk),
                ("value", &layer.wv),
                ("output", &layer.wo),
                ("gate", &layer.w_gate),
                ("up", &layer.w_up),
                ("down", &layer.w_down),
            ];
            for (projection, weight) in projections {
                let name = format!("backbone.text.layers.{index}.{projection}");
                output.insert(name.clone(), self.matrix(weight, &name, backend)?);
            }
        }
        let mut insert = |name: String, weight: &Tensor| -> Result<()> {
            output.insert(name.clone(), self.matrix(weight, &name, backend)?);
            Ok(())
        };
        insert(
            "backbone.vision.patch_embed".into(),
            &vision.patch_embed_weight,
        )?;
        for (index, block) in vision.blocks.iter().enumerate() {
            insert(format!("backbone.vision.blocks.{index}.qkv"), &block.qkv_w)?;
            insert(
                format!("backbone.vision.blocks.{index}.output"),
                &block.proj_w,
            )?;
            insert(format!("backbone.vision.blocks.{index}.fc1"), &block.fc1_w)?;
            insert(format!("backbone.vision.blocks.{index}.fc2"), &block.fc2_w)?;
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
        if output.is_empty() {
            return Err(Error::Other("GR00T FP8 backbone plan is empty".into()));
        }
        Ok(output)
    }
}

pub(super) type Gr00tFp8Executor = Gr00tExecutor<Gr00tFp8Execution>;

pub(super) fn build(
    checkpoint_path: &Path,
    backbone_path: &Path,
    calibration_path: &Path,
    config: Gr00tConfig,
    backbone: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tFp8Executor> {
    let execution = Gr00tFp8Execution::from_file(
        calibration_path,
        checkpoint_path,
        backbone_path,
        &config,
        &backbone,
    )?;
    Gr00tExecutor::from_backend(config, backbone, weights, execution, backend)
}

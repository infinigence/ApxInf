//! Static-FP8 GR00T runtime for Thor-class GPUs.

use std::path::Path;
use std::sync::Arc;

use apxinf_core::Result;

use super::backbone::Qwen3VLConfig;
use super::backend::RuntimeBackend;
use super::fp8_executor::{self, Gr00tFp8Executor};
use super::vla_runtime::Gr00tVlaRuntime;
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

pub(super) type Gr00tFp8Runtime = Gr00tVlaRuntime<super::fp8_executor::Gr00tFp8Execution>;

pub(super) fn build(
    checkpoint_path: &Path,
    backbone_path: &Path,
    calibration_path: &Path,
    config: Gr00tConfig,
    backbone: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tFp8Runtime> {
    let executor: Gr00tFp8Executor = fp8_executor::build(
        checkpoint_path,
        backbone_path,
        calibration_path,
        config,
        backbone,
        weights,
        backend,
    )?;
    Ok(Gr00tVlaRuntime::new(executor))
}

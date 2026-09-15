//! W8A8 INT8 GR00T runtime for Orin-class GPUs.

use std::sync::Arc;

use apxinf_core::Result;

use super::backbone::Qwen3VLConfig;
use super::backend::RuntimeBackend;
use super::int8_executor::{self, Gr00tInt8Executor};
use super::vla_runtime::Gr00tVlaRuntime;
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

pub(super) type Gr00tInt8Runtime = Gr00tVlaRuntime<super::int8_executor::Gr00tInt8Execution>;

pub(super) fn build(
    config: Gr00tConfig,
    backbone: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tInt8Runtime> {
    let executor: Gr00tInt8Executor = int8_executor::build(config, backbone, weights, backend)?;
    Ok(Gr00tVlaRuntime::new(executor))
}

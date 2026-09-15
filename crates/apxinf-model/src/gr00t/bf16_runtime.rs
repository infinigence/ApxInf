//! Native-BF16 GR00T runtime.

use std::sync::Arc;

use apxinf_core::Result;

use super::backbone::Qwen3VLConfig;
use super::backend::RuntimeBackend;
use super::bf16_executor::{self, Gr00tBf16Executor};
use super::vla_runtime::Gr00tVlaRuntime;
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

pub(super) type Gr00tBf16Runtime = Gr00tVlaRuntime<super::bf16_executor::Gr00tBf16Execution>;

pub(super) fn build(
    config: Gr00tConfig,
    backbone: Qwen3VLConfig,
    weights: Gr00tWeights,
    backend: Arc<RuntimeBackend>,
) -> Result<Gr00tBf16Runtime> {
    let executor: Gr00tBf16Executor = bf16_executor::build(config, backbone, weights, backend)?;
    Ok(Gr00tVlaRuntime::new(executor))
}

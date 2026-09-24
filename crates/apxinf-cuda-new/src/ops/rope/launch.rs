use apxinf_core::Result;

use super::contracts::Normalized;
use crate::ffi::abi::{rope as abi, status};
use crate::CudaContext;

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    let _storage = &normalized.storage;
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    unsafe {
        status::check(abi::apxinf_rope_launch(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

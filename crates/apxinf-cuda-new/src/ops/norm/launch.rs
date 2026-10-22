use apxinf_core::Result;

use super::contracts::Normalized;
use crate::ffi::abi::{norm as abi, status};
use crate::CudaContext;

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    let storage = std::rc::Rc::new(normalized.storage);
    crate::workspace::retain_resource(&storage);
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    unsafe {
        status::check(abi::apxinf_norm_launch(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

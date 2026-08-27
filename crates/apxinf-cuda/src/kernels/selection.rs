//! Token-selection operator contracts.

use apxinf_core::{DType, Error, Result};

use super::contracts::{check_cuda, checked_bytes, require_address, require_buffers};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::ffi;

pub const ARGMAX_PARTIAL_BYTES: usize = 128 * 8;

/// Select the lowest-index maximum from one BF16 logit row and publish its
/// token ID to caller-owned device-visible storage.
pub fn argmax_bf16_into(
    ctx: &CudaContext,
    logits: &CudaBuffer,
    partials: &CudaBuffer,
    output: CudaDeviceAddress,
    count: usize,
) -> Result<()> {
    if count > u32::MAX as usize {
        return Err(Error::Other("BF16 argmax element count exceeds u32".into()));
    }
    let bytes = checked_bytes(DType::BF16, &[count], "BF16 argmax")?;
    require_buffers(
        ctx,
        "BF16 argmax",
        &[
            ("logits", logits, bytes),
            ("partials", partials, ARGMAX_PARTIAL_BYTES),
        ],
    )?;
    require_address(ctx, "BF16 argmax", "output", output, 4)?;
    check_cuda(unsafe {
        ffi::apxinf_argmax_bf16(
            logits.ptr(),
            count as u32,
            partials.ptr(),
            output.ptr(),
            ctx.stream().handle(),
        )
    })
}

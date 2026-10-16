use apxinf_core::Result;

use super::contracts::{normalize, QuantizationArgs};
use super::launch;
use crate::CudaContext;

/// Execute a validated quantization or representation-conversion semantic.
pub fn quantization(ctx: &CudaContext, args: QuantizationArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

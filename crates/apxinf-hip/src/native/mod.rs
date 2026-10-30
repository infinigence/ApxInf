//! Operators for models written directly against HIP, starting with pi05.
//!
//! This is a different layer from the crate's `Backend` impl. Models that
//! compose through `dyn Backend` get allocating, shape-generic operators;
//! a native model gets these instead:
//!
//! - **Outputs are passed in.** Every operator writes into tensors the caller
//!   allocated, typically from a [`HipArena`], so activation addresses are
//!   fixed per inference and a captured HIP graph can replay them.
//! - **No hidden synchronization.** No operator waits on the device or copies
//!   from the host; inputs reach the device through [`HipDeviceBytes`] or
//!   [`upload`] before the work is issued.
//! - **BF16 storage, F32 math.** Each kernel follows the rounding points of the
//!   cuda-new kernel pi05 runs on CUDA, so the two backends are comparable
//!   stage by stage.
//!
//! The operator set is pi05's. It grows with the models that use it, not ahead
//! of them.

mod arena;
mod attention;
mod elementwise;
mod gemm;
mod norm;

use std::ffi::c_void;
use std::sync::Arc;

use apxinf_core::{DType, Error, NormalGenerator, Result, Tensor};

pub use arena::HipArena;
pub use attention::{attention, AttentionGeometry};
pub use elementwise::{
    add_position, bias_activation, copy_rows, embedding, euler_update, geglu, rgb_to_patches,
    split_qkv_bias, split_qkv_rope, Activation, HipDeviceBytes, ImageLayout, QkvRope,
};
pub use gemm::gemm;
pub use norm::{
    ada_gate_residual_rms_norm, ada_rms_norm, bias_residual, bias_residual_layer_norm,
    bias_residual_rms_norm, layer_norm, rms_norm,
};

use crate::runtime::HipContext;

/// Copy a host tensor into an existing device tensor of the same dtype and
/// element count, keeping the destination's address. This is how a prepared
/// plan refills its fixed input buffers. Waits for the copy, so the host memory
/// may be reused immediately.
pub fn upload(ctx: &HipContext, src: &Tensor, dst: &Tensor) -> Result<()> {
    if src.dtype() != dst.dtype() {
        return Err(Error::DTypeMismatch {
            expected: dst.dtype(),
            got: src.dtype(),
        });
    }
    require_numel(src, dst.numel(), || dst.shape().to_string())?;
    let bytes = src.storage().as_cpu().ok_or(Error::Contract(
        "apxinf-hip upload: source must be a host tensor",
    ))?;
    let len = dst.size_in_bytes();
    if bytes.len() < len {
        return Err(Error::DataLengthMismatch {
            expected: len,
            got: bytes.len(),
        });
    }
    let ptr = ctx.ptr(dst)?;
    ctx.bind()?;
    ctx.upload(ptr, &bytes[..len])
}

/// A standard-normal generator that refills `output` (on this device) in
/// place, with the host generator's exact RNG stream.
pub fn normal_generator(ctx: &Arc<HipContext>, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
    Ok(Box::new(crate::sampling::HipNormalGenerator::new(
        Arc::clone(ctx),
        output,
    )?))
}

fn require_bf16(tensor: &Tensor) -> Result<()> {
    if tensor.dtype() == DType::BF16 {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            got: tensor.dtype(),
            allowed: "bf16",
        })
    }
}

/// `[..., cols]` as (rows, cols), the leading dimensions folded into rows.
fn rows_cols(tensor: &Tensor) -> Result<(usize, usize)> {
    let cols = *tensor.shape().dims().last().ok_or(Error::Contract(
        "apxinf-hip native: operand has no dimensions",
    ))?;
    if cols == 0 {
        return Err(Error::Contract(
            "apxinf-hip native: operand has zero columns",
        ));
    }
    Ok((tensor.numel() / cols, cols))
}

fn same_shape(expected: &Tensor, got: &Tensor) -> Result<()> {
    if expected.shape() == got.shape() {
        Ok(())
    } else {
        Err(Error::ShapeMismatch {
            expected: expected.shape().to_string(),
            got: got.shape().to_string(),
        })
    }
}

fn require_numel(tensor: &Tensor, numel: usize, expected: impl FnOnce() -> String) -> Result<()> {
    if tensor.numel() == numel {
        Ok(())
    } else {
        Err(Error::ShapeMismatch {
            expected: expected(),
            got: tensor.shape().to_string(),
        })
    }
}

/// Device address of a BF16 tensor on this context's device.
fn bf16_ptr(ctx: &HipContext, tensor: &Tensor) -> Result<*mut c_void> {
    require_bf16(tensor)?;
    ctx.ptr(tensor)
}

/// Address of an optional BF16 operand of exactly `numel` elements, or null.
fn optional_ptr(ctx: &HipContext, tensor: Option<&Tensor>, numel: usize) -> Result<*const c_void> {
    match tensor {
        None => Ok(std::ptr::null()),
        Some(tensor) => {
            require_numel(tensor, numel, || format!("[{numel}]"))?;
            Ok(bf16_ptr(ctx, tensor)?.cast_const())
        }
    }
}

/// Kernel extents are `int`; refuse anything larger instead of truncating.
fn int(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Contract("apxinf-hip native: extent exceeds i32"))
}

fn long(value: usize) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::Contract("apxinf-hip native: extent exceeds i64"))
}

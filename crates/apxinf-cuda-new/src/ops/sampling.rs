//! Device-side token selection operators.

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::{elementwise as elementwise_abi, sampling as sampling_abi, status};
use crate::{CudaBuffer, CudaContext};

mod suppress;

pub use suppress::suppress_logits;

/// Device argmax. `remap`, when present, maps the winning local column to the
/// token id written to `index`; without it, `index` receives the column index.
pub struct ArgmaxArgs<'a> {
    pub logits: &'a Tensor,
    pub index: &'a Tensor,
    pub remap: Option<&'a Tensor>,
}

impl<'a> ArgmaxArgs<'a> {
    pub fn new(logits: &'a Tensor, index: &'a Tensor) -> Self {
        Self {
            logits,
            index,
            remap: None,
        }
    }

    pub fn with_remap(mut self, remap: &'a Tensor) -> Self {
        self.remap = Some(remap);
        self
    }
}

fn tensor_buffer(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
    name: &str,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != dtype
        || tensor.shape().numel() != elements
    {
        return Err(Error::Other(format!(
            "argmax {name} must contain {elements} {dtype} values on CUDA{}",
            ctx.device_id()
        )));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let required = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other(format!("argmax {name} size overflow")))?;
    if buffer.len() < required {
        return Err(Error::Other(format!(
            "argmax {name} storage has {} bytes, need {required}",
            buffer.len()
        )));
    }
    Ok(buffer)
}

/// Index of the largest logit, reduced on device. Input dtype is inferred from
/// `logits`; the current kernel supports BF16 and rejects other dtypes.
pub fn argmax(ctx: &CudaContext, args: ArgmaxArgs<'_>) -> Result<()> {
    let count = args.logits.shape().numel();
    if count == 0 || count > i32::MAX as usize {
        return Err(Error::Other(
            "argmax logit count must be in 1..=i32::MAX".into(),
        ));
    }
    let logits = tensor_buffer(ctx, args.logits, DType::BF16, count, "logits")?;
    let index = tensor_buffer(ctx, args.index, DType::I32, 1, "index")?;
    if let Some(remap) = args.remap {
        let remap = tensor_buffer(ctx, remap, DType::I32, count, "remap")?;
        unsafe {
            status::check(elementwise_abi::apxinf_elementwise_argmax_remap_bf16(
                logits.ptr(),
                count as u32,
                remap.ptr(),
                index.ptr(),
                ctx.stream().handle(),
            ))
        }
    } else {
        unsafe {
            status::check(sampling_abi::apxinf_sampling_argmax(
                logits.ptr(),
                index.ptr(),
                count as i64,
                ctx.stream().handle(),
            ))
        }
    }
}

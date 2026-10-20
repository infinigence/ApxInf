use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::{rope as abi, status};
use crate::{CudaBuffer, CudaContext};

/// Immutable double-precision `(cos, sin)` coefficients for adjacent-pair
/// rotation. Construction performs the host-to-device upload before capture.
/// Coefficients are shared by all heads; callers may compose arbitrary axes
/// into this `[tokens, head_dim / 2]` table without changing the operator.
pub struct PairwiseRopeTable {
    storage: CudaBuffer,
    tokens: usize,
    head_dim: usize,
}

impl PairwiseRopeTable {
    pub fn new(
        ctx: &CudaContext,
        tokens: usize,
        head_dim: usize,
        coefficients: &[(f64, f64)],
    ) -> Result<Self> {
        if tokens == 0
            || tokens > i32::MAX as usize
            || head_dim == 0
            || head_dim % 2 != 0
            || head_dim > 2048
        {
            return Err(Error::Other(
                "Pairwise RoPE requires positive tokens and even head_dim <= 2048".into(),
            ));
        }
        let count = tokens
            .checked_mul(head_dim / 2)
            .ok_or_else(|| Error::Other("Pairwise RoPE size overflow".into()))?;
        if coefficients.len() != count
            || coefficients
                .iter()
                .any(|&(c, s)| !c.is_finite() || !s.is_finite())
        {
            return Err(Error::Other(
                "Pairwise RoPE coefficient shape or value is invalid".into(),
            ));
        }
        count
            .checked_mul(16)
            .ok_or_else(|| Error::Other("Pairwise RoPE table size overflow".into()))?;
        let bytes: Vec<u8> = coefficients
            .iter()
            .flat_map(|&(c, s)| c.to_ne_bytes().into_iter().chain(s.to_ne_bytes()))
            .collect();
        let storage = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
        storage.copy_from_host(&bytes).map_err(Error::Cuda)?;
        Ok(Self {
            storage,
            tokens,
            head_dim,
        })
    }
}

/// Adjacent-pair rotation of contiguous `[tokens, heads * head_dim]` input.
/// BF16 input is multiplied by FP64 coefficients in FP64, then converted to
/// either BF16 or F32. No FP32 coefficient rounding or fused multiply-add is
/// introduced. Exact input/output aliasing is allowed for BF16 output only.
pub struct PairwiseRopeArgs<'a> {
    pub input: &'a Tensor,
    pub table: &'a PairwiseRopeTable,
    pub output: &'a mut Tensor,
    pub heads: usize,
}

pub fn pairwise_rope(ctx: &CudaContext, args: PairwiseRopeArgs<'_>) -> Result<()> {
    let invalid = || Error::Other("Invalid pairwise RoPE shape, dtype, device, or alias".into());
    let table = args.table;
    let width = args.heads.checked_mul(table.head_dim).ok_or_else(invalid)?;
    let count = table.tokens.checked_mul(width).ok_or_else(invalid)?;
    if args.heads == 0
        || args.heads > i32::MAX as usize
        || table.tokens > i32::MAX as usize
        || args.input.shape().dims() != [table.tokens, width]
        || args.output.shape().dims() != [table.tokens, width]
        || args.input.dtype() != DType::BF16
        || !matches!(args.output.dtype(), DType::BF16 | DType::F32)
        || args.input.device() != Device::Cuda(ctx.device_id())
        || args.output.device() != Device::Cuda(ctx.device_id())
        || table.storage.device() != ctx.device_id()
    {
        return Err(invalid());
    }
    let input = CudaBuffer::from_tensor(args.input).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(args.output).map_err(Error::Cuda)?;
    let input_bytes = count.checked_mul(2).ok_or_else(invalid)?;
    let output_bytes = count
        .checked_mul(args.output.dtype().size_in_bytes())
        .ok_or_else(invalid)?;
    let i = input.ptr() as usize;
    let o = output.ptr() as usize;
    let i_end = i.checked_add(input_bytes).ok_or_else(invalid)?;
    let o_end = o.checked_add(output_bytes).ok_or_else(invalid)?;
    if input.len() < input_bytes
        || output.len() < output_bytes
        || i % 2 != 0
        || o % args.output.dtype().size_in_bytes() != 0
        || (i < o_end && o < i_end && !(i == o && args.output.dtype() == DType::BF16))
    {
        return Err(invalid());
    }
    crate::workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize)?;
    unsafe {
        status::check(abi::apxinf_pairwise_rope_launch(
            ctx.runtime(),
            1,
            input.ptr() as _,
            table.storage.ptr() as _,
            output.ptr() as _,
            table.tokens as i64,
            args.heads as i64,
            table.head_dim as i64,
            if args.output.dtype() == DType::F32 {
                0
            } else {
                2
            },
            ctx.stream().handle() as _,
        ))?;
    }
    crate::workspace::retain_resource(&std::rc::Rc::new((input, output, table.storage.clone())));
    Ok(())
}

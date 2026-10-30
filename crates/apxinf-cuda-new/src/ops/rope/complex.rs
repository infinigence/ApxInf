use super::{
    contracts::{tensor_storage, Normalized},
    launch,
};
use crate::{ffi::abi::rope as abi, CudaBuffer, CudaContext};
use apxinf_core::{DType, Error, Result, Tensor};

/// Immutable interleaved complex coefficients [tokens, head_dim/2, real/imag].
/// FP64 storage and arithmetic preserve reference complex rotations without
/// extending the public Tensor dtype set. Construct outside graph capture.
pub struct ComplexRotaryTable {
    storage: CudaBuffer,
    tokens: usize,
    head_dim: usize,
}
impl ComplexRotaryTable {
    pub fn new(ctx: &CudaContext, tokens: usize, head_dim: usize, values: &[f64]) -> Result<Self> {
        if tokens == 0
            || tokens > i32::MAX as usize
            || head_dim == 0
            || head_dim > 2048
            || head_dim % 2 != 0
            || tokens.checked_mul(head_dim) != Some(values.len())
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Other(
                "invalid complex rotary table shape or coefficients".into(),
            ));
        }
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let storage = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
        storage.copy_from_host(&bytes).map_err(Error::Cuda)?;
        Ok(Self {
            storage,
            tokens,
            head_dim,
        })
    }
}

/// Rotate adjacent scalar pairs in [tokens, heads, head_dim]. FP64 complex
/// products are cast through FP32 to F16/BF16, matching Torch's scalar cast.
/// Exact in-place
/// operation is supported; partial overlap is rejected.
pub fn complex_rotary(
    ctx: &CudaContext,
    table: &ComplexRotaryTable,
    input: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let dims = input.shape().dims();
    if dims.len() != 3
        || dims[0] != table.tokens
        || dims[2] != table.head_dim
        || dims[1] == 0
        || dims[1] > 65535
        || output.shape() != input.shape()
        || table.storage.device() != ctx.device_id()
    {
        return Err(Error::Other(
            "complex rotary input/output/table shape or device mismatch".into(),
        ));
    }
    let dtype = match input.dtype() {
        DType::F16 => 1,
        DType::BF16 => 2,
        _ => return Err(Error::Other("complex rotary requires F16/BF16".into())),
    };
    let n = dims
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| Error::Other("complex rotary size overflow".into()))?;
    let src = tensor_storage(ctx, input, input.dtype(), n)?;
    let dst = tensor_storage(ctx, output, input.dtype(), n)?;
    let bytes = n
        .checked_mul(2)
        .ok_or_else(|| Error::Other("complex rotary byte overflow".into()))?;
    let (a, b) = (src.ptr() as usize, dst.ptr() as usize);
    if a != b && a.abs_diff(b) < bytes {
        return Err(Error::Other(
            "complex rotary partially overlapping input/output".into(),
        ));
    }
    let align = |p: usize| (1usize << p.trailing_zeros().min(8)) as u32;
    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: 8,
        layout: abi::LAYOUT_SINGLE_TENSOR,
        cache_write: abi::CACHE_WRITE_NONE,
        rotation_source: abi::ROTATION_PRECOMPUTED_TABLE,
        rotary_convention: abi::CONVENTION_NONE,
        dtype,
        has_bias: 1,
        q_heads: dims[1] as u32,
        kv_heads: dims[1] as u32,
        head_dim: dims[2] as u32,
        rotary_dim: 0,
        section_t: 0,
        section_h: 0,
        section_w: 0,
        qkv_alignment: align(a),
        bias_alignment: align(table.storage.ptr() as usize),
        q_alignment: align(b),
        kv_alignment: 256,
        rotation_alignment: align(table.storage.ptr() as usize),
        tokens: table.tokens as i64,
        cache_capacity: 0,
    };
    let bindings = abi::Bindings {
        qkv: src.ptr(),
        bias: table.storage.ptr(),
        q: dst.ptr(),
        k: std::ptr::null_mut(),
        v: std::ptr::null_mut(),
        key_input: std::ptr::null(),
        value_input: std::ptr::null(),
        position: std::ptr::null(),
        rotation_table: table.storage.ptr(),
        stream: ctx.stream().handle(),
        theta: 0.,
        position_offset: 0,
        kv_output_offset: 0,
    };
    launch::execute(
        ctx,
        Normalized {
            spec,
            bindings,
            storage: vec![src, dst, table.storage.clone()],
        },
    )
}

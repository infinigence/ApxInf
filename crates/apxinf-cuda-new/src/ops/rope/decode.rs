use apxinf_core::{DType, Device, Error, Result, Tensor};

use super::contracts::Normalized;
use super::launch;
use crate::ffi::abi::rope as abi;
use crate::{CudaBuffer, CudaContext, CudaDeviceAddress};

/// Single-token graph-safe decode transform.
///
/// Q/K/V inputs are `[1, 1, heads, head_dim]`. Q is rotated into
/// `query_out`; K is rotated directly into `key_cache`; V is appended to
/// `value_cache`. Both caches use the canonical token-major layout
/// `[1, cache_capacity, kv_heads, head_dim]`.
///
/// `position` points to one device-visible `u32`. Its address is stable and
/// is captured as a stable binding address, but its value is read by the
/// kernel on every launch/graph replay.
pub struct DecodeRopeArgs<'a> {
    pub query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub query_out: &'a mut Tensor,
    pub key_cache: &'a mut Tensor,
    pub value_cache: &'a mut Tensor,
    pub position: CudaDeviceAddress,
    pub theta: f32,
}

impl<'a> DecodeRopeArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        query: &'a Tensor,
        key: &'a Tensor,
        value: &'a Tensor,
        query_out: &'a mut Tensor,
        key_cache: &'a mut Tensor,
        value_cache: &'a mut Tensor,
        position: CudaDeviceAddress,
        theta: f32,
    ) -> Self {
        Self {
            query,
            key,
            value,
            query_out,
            key_cache,
            value_cache,
            position,
            theta,
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn alignment(pointer: *const std::ffi::c_void) -> u32 {
    let address = pointer as usize;
    (1usize << address.trailing_zeros().min(8)) as u32
}

fn storage(ctx: &CudaContext, tensor: &Tensor, shape: &[usize]) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != DType::BF16
        || tensor.shape().dims() != shape
    {
        return Err(invalid(
            "dynamic decode RoPE tensor device/dtype/shape mismatch",
        ));
    }
    let expected = shape
        .iter()
        .try_fold(DType::BF16.size_in_bytes(), |bytes, dim| {
            bytes.checked_mul(*dim)
        })
        .ok_or_else(|| invalid("dynamic decode RoPE size overflow"))?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < expected {
        return Err(invalid("dynamic decode RoPE tensor storage is too small"));
    }
    Ok(buffer)
}

pub(crate) fn normalize(ctx: &CudaContext, args: DecodeRopeArgs<'_>) -> Result<Normalized> {
    let q_shape = args.query.shape().dims();
    let k_shape = args.key.shape().dims();
    let v_shape = args.value.shape().dims();
    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(invalid("dynamic decode RoPE requires rank-4 Q/K/V"));
    }
    let (batch, tokens, q_heads, head_dim) =
        (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (k_batch, k_tokens, kv_heads, k_head_dim) =
        (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    let cache_shape = args.key_cache.shape().dims();
    if batch != 1
        || tokens != 1
        || k_batch != 1
        || k_tokens != 1
        || v_shape != k_shape
        || q_heads == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || head_dim == 0
        || head_dim > 2048
        || head_dim % 2 != 0
        || k_head_dim != head_dim
        || args.query_out.shape().dims() != q_shape
        || cache_shape.len() != 4
        || cache_shape[0] != 1
        || cache_shape[1] == 0
        || cache_shape[2] != kv_heads
        || cache_shape[3] != head_dim
        || args.value_cache.shape().dims() != cache_shape
    {
        return Err(invalid("invalid dynamic decode RoPE shape contract"));
    }
    if !args.theta.is_finite() || args.theta <= 0.0 {
        return Err(invalid(
            "dynamic decode RoPE theta must be finite and positive",
        ));
    }
    if args.position.device() != ctx.device_id() || args.position.len() < 4 {
        return Err(invalid(
            "dynamic decode RoPE position address device/size mismatch",
        ));
    }
    let cache_capacity = cache_shape[1];
    for value in [q_heads, kv_heads, head_dim, cache_capacity] {
        if value > i32::MAX as usize {
            return Err(invalid("dynamic decode RoPE dimension exceeds native limits"));
        }
    }

    let q = storage(ctx, args.query, q_shape)?;
    let key_input = storage(ctx, args.key, k_shape)?;
    let value_input = storage(ctx, args.value, v_shape)?;
    let q_out = storage(ctx, args.query_out, q_shape)?;
    let key_cache = storage(ctx, args.key_cache, cache_shape)?;
    let value_cache = storage(ctx, args.value_cache, cache_shape)?;
    let position = args.position.ptr().cast_const().cast::<u32>();

    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: 2,
        dtype: 2,
        has_bias: 0,
        q_heads: q_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        qkv_alignment: alignment(q.ptr()),
        bias_alignment: 256,
        q_alignment: alignment(q_out.ptr()),
        kv_alignment: alignment(key_cache.ptr()),
        position_alignment: alignment(position.cast()),
        tokens: 1,
        cache_capacity: cache_capacity as i64,
    };
    let bindings = abi::Bindings {
        qkv: q.ptr(),
        bias: std::ptr::null(),
        q: q_out.ptr(),
        k: key_cache.ptr(),
        v: value_cache.ptr(),
        key_input: key_input.ptr(),
        value_input: value_input.ptr(),
        position,
        stream: ctx.stream().handle(),
        theta: args.theta,
        position_offset: 0,
        kv_output_offset: 0,
    };
    Ok(Normalized {
        spec,
        bindings,
        storage: vec![q, key_input, value_input, q_out, key_cache, value_cache],
    })
}

/// Apply dynamic single-token RoPE and append K/V into token-major caches.
pub fn decode_rope(ctx: &CudaContext, args: DecodeRopeArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::rope as abi;
use crate::{CudaBuffer, CudaContext};

/// Which packed-QKV split the bindings describe. See `rope_types.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeSemantic {
    /// Grouped-query split with rotary embedding on Q and K.
    SplitQkvRope,
    /// Vision MHA split with an optional bias and no rotation.
    SplitQkvBias,
}

impl RopeSemantic {
    fn code(self) -> u32 {
        match self {
            Self::SplitQkvRope => 0,
            Self::SplitQkvBias => 1,
        }
    }

    fn applies_rope(self) -> bool {
        matches!(self, Self::SplitQkvRope)
    }
}

/// Splits a packed QKV projection into separate Q, K and V buffers.
///
/// `k` and `v` may point either at fresh per-call buffers (`kv_output_offset`
/// 0) or into a KV cache, in which case `kv_output_offset` is the row at which
/// this call's tokens are appended and `kv_rows` is the cache capacity.
pub struct RopeArgs<'a> {
    pub semantic: RopeSemantic,
    pub qkv: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub q: &'a mut Tensor,
    pub k: &'a mut Tensor,
    pub v: &'a mut Tensor,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub position_offset: usize,
    pub kv_output_offset: usize,
}

pub(crate) struct Normalized {
    pub spec: abi::Spec,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("RoPE currently supports F16 and BF16")),
    }
}

fn alignment_of(pointer: usize) -> u32 {
    if pointer == 0 {
        return 256;
    }
    (1usize << pointer.trailing_zeros().min(8)) as u32
}

/// Validates device and dtype and returns the buffer. Unlike the other
/// families the element count is only a lower bound, because K and V may be
/// slices of a larger cache.
fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    minimum_elements: usize,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != dtype {
        return Err(invalid("RoPE tensor device/dtype mismatch"));
    }
    if tensor.shape().numel() < minimum_elements {
        return Err(invalid("RoPE tensor is too small for the requested split"));
    }
    let expected = minimum_elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("RoPE size overflow"))?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < expected || (buffer.ptr() as usize) % dtype.size_in_bytes() != 0 {
        return Err(invalid("RoPE tensor storage is invalid"));
    }
    Ok(buffer)
}

pub(crate) fn normalize(ctx: &CudaContext, args: RopeArgs<'_>) -> Result<Normalized> {
    let semantic = args.semantic;
    let dims = args.qkv.shape().dims();
    if dims.len() != 2 {
        return Err(invalid("RoPE qkv must be rank 2 [tokens, fused_width]"));
    }
    let (tokens, fused_width) = (dims[0], dims[1]);
    if tokens == 0 {
        return Err(invalid("RoPE qkv must be non-empty"));
    }
    let tokens_abi = i32::try_from(tokens)
        .map_err(|_| invalid("RoPE token count exceeds the CUDA kernel range"))?;
    if args.head_dim == 0 || args.head_dim > 2048 || args.head_dim % 2 != 0 {
        return Err(invalid(
            "RoPE head_dim must be even, non-zero, and at most 2048",
        ));
    }
    if args.q_heads == 0 || args.kv_heads == 0 || args.q_heads % args.kv_heads != 0 {
        return Err(invalid("RoPE q_heads must be a multiple of kv_heads"));
    }
    let projection_heads = args
        .kv_heads
        .checked_mul(2)
        .and_then(|kv| args.q_heads.checked_add(kv))
        .ok_or_else(|| invalid("RoPE head count overflow"))?;
    if projection_heads > 65_535 {
        return Err(invalid("RoPE launch has too many Q/K/V heads"));
    }
    if !semantic.applies_rope() && args.q_heads != args.kv_heads {
        return Err(invalid("unrotated QKV split requires q_heads == kv_heads"));
    }
    if semantic.applies_rope() && !(args.theta.is_finite() && args.theta > 0.0) {
        return Err(invalid("RoPE theta must be finite and positive"));
    }
    if !semantic.applies_rope() && (args.position_offset != 0 || args.kv_output_offset != 0) {
        return Err(invalid("unrotated QKV split does not take offsets"));
    }

    let q_width = args
        .q_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| invalid("RoPE Q width overflow"))?;
    let kv_width = args
        .kv_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| invalid("RoPE KV width overflow"))?;
    let expected_fused = kv_width
        .checked_mul(2)
        .and_then(|kv| q_width.checked_add(kv))
        .ok_or_else(|| invalid("RoPE fused width overflow"))?;
    if fused_width != expected_fused {
        return Err(invalid(format!(
            "RoPE qkv width {fused_width} does not match {expected_fused}"
        )));
    }

    let dtype = args.qkv.dtype();
    let mut storage = Vec::new();

    let qkv_elements = tokens
        .checked_mul(fused_width)
        .ok_or_else(|| invalid("RoPE QKV size overflow"))?;
    let qkv_buffer = tensor_storage(ctx, args.qkv, dtype, qkv_elements)?;
    let qkv = qkv_buffer.ptr() as *const std::ffi::c_void;
    let qkv_alignment = alignment_of(qkv_buffer.ptr() as usize);
    storage.push(qkv_buffer);

    let (bias, bias_alignment) = match args.bias {
        Some(tensor) => {
            let buffer = tensor_storage(ctx, tensor, dtype, fused_width)?;
            let pointer = buffer.ptr() as *const std::ffi::c_void;
            let alignment = alignment_of(buffer.ptr() as usize);
            storage.push(buffer);
            (pointer, alignment)
        }
        None => (std::ptr::null(), 256),
    };

    let q_elements = tokens
        .checked_mul(q_width)
        .ok_or_else(|| invalid("RoPE Q size overflow"))?;
    let q_buffer = tensor_storage(ctx, args.q, dtype, q_elements)?;
    let q = q_buffer.ptr() as *mut std::ffi::c_void;
    let q_alignment = alignment_of(q_buffer.ptr() as usize);
    storage.push(q_buffer);

    // K and V must hold every row this call writes, which for a cache means
    // the append offset plus this call's tokens.
    let kv_rows_needed = args
        .kv_output_offset
        .checked_add(tokens)
        .ok_or_else(|| invalid("RoPE size overflow"))?;
    let kv_elements = kv_rows_needed
        .checked_mul(kv_width)
        .ok_or_else(|| invalid("RoPE size overflow"))?;

    let k_buffer = tensor_storage(ctx, args.k, dtype, kv_elements)?;
    let k = k_buffer.ptr() as *mut std::ffi::c_void;
    let kv_alignment = alignment_of(k_buffer.ptr() as usize);
    storage.push(k_buffer);

    let v_buffer = tensor_storage(ctx, args.v, dtype, kv_elements)?;
    let v = v_buffer.ptr() as *mut std::ffi::c_void;
    storage.push(v_buffer);

    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: semantic.code(),
        dtype: dtype_code(dtype)?,
        has_bias: u32::from(!bias.is_null()),
        q_heads: u32::try_from(args.q_heads)
            .map_err(|_| invalid("RoPE Q head count exceeds the ABI range"))?,
        kv_heads: u32::try_from(args.kv_heads)
            .map_err(|_| invalid("RoPE KV head count exceeds the ABI range"))?,
        head_dim: u32::try_from(args.head_dim)
            .map_err(|_| invalid("RoPE head dimension exceeds the ABI range"))?,
        qkv_alignment,
        bias_alignment,
        q_alignment,
        kv_alignment,
        // No position pointer is bound for split/prefill semantics. The ABI
        // uses the maximum guaranteed alignment as the canonical null value,
        // matching the other optional bindings.
        position_alignment: 256,
        tokens: i64::from(tokens_abi),
        cache_capacity: 0,
    };

    let bindings = abi::Bindings {
        qkv,
        bias,
        q,
        k,
        v,
        key_input: std::ptr::null(),
        value_input: std::ptr::null(),
        position: std::ptr::null(),
        stream: ctx.stream().handle() as abi::CudaStream,
        theta: args.theta,
        position_offset: i32::try_from(args.position_offset)
            .map_err(|_| invalid("RoPE position offset exceeds the ABI range"))?,
        kv_output_offset: i32::try_from(args.kv_output_offset)
            .map_err(|_| invalid("RoPE KV output offset exceeds the ABI range"))?,
    };

    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

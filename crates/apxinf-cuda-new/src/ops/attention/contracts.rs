use std::ops::Range;

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::attention as abi;
use crate::{CudaBuffer, CudaContext};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Semantic {
    Dense = abi::SEMANTIC_DENSE,
    KvCache = abi::SEMANTIC_KV_CACHE,
    Segmented = abi::SEMANTIC_SEGMENTED,
}

impl Semantic {
    pub(crate) const fn abi_value(self) -> u32 {
        self as u32
    }

    #[cfg(test)]
    pub(crate) const ALL: &'static [Self] = &[Self::Dense, Self::KvCache, Self::Segmented];

    #[cfg(test)]
    pub(crate) const fn doc_id(self) -> &'static str {
        match self {
            Self::Dense => "attention",
            Self::KvCache => "kv_cache_attention",
            Self::Segmented => "segmented_attention",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionMask {
    None,
    Causal,
}

#[derive(Clone, Debug)]
pub struct AttentionPolicy {
    pub workspace_limit: usize,
    pub online_tune: bool,
    pub allow_fallback: bool,
    pub graph_safe: bool,
    pub deterministic: bool,
    pub cache_dir: Option<String>,
}

impl Default for AttentionPolicy {
    fn default() -> Self {
        Self {
            workspace_limit: 256 * 1024 * 1024,
            online_tune: true,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
            cache_dir: None,
        }
    }
}

/// Canonical dense attention contract. Tensors are contiguous row-major:
/// Q `[batch, query_tokens, query_heads, head_dim]`, K/V
/// `[batch, key_tokens, kv_heads, head_dim]`, and output has Q's shape.
pub struct AttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub out: &'a mut Tensor,
    pub mask: AttentionMask,
    pub scale: f32,
    /// Static dequantization scale for an E4M3 output. The stored value is
    /// `round_to_e4m3(attention / output_scale)`. Must remain `1.0` for
    /// ordinary F16/BF16 output.
    pub output_scale: f32,
    pub policy: AttentionPolicy,
}

impl<'a> AttentionArgs<'a> {
    pub fn new(query: &'a Tensor, key: &'a Tensor, value: &'a Tensor, out: &'a mut Tensor) -> Self {
        let head_dim = query.shape().dims().get(3).copied().unwrap_or(1);
        Self {
            query,
            key,
            value,
            out,
            mask: AttentionMask::None,
            scale: 1.0 / (head_dim as f32).sqrt(),
            output_scale: 1.0,
            policy: AttentionPolicy::default(),
        }
    }

    pub fn causal(mut self) -> Self {
        self.mask = AttentionMask::Causal;
        self
    }
}

pub(crate) struct Normalized {
    pub spec: abi::Spec,
    pub policy: AttentionPolicy,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

pub(crate) fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Attention currently supports F16 and BF16")),
    }
}

fn output_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::F8E4M3 => Ok(3),
        _ => Err(invalid(
            "Attention output currently supports F16, BF16, and E4M3",
        )),
    }
}

pub(crate) fn required_bytes(dtype: DType, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid("Attention size overflow"))
}

pub(crate) fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    shape: &[usize],
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != dtype
        || tensor.shape().dims() != shape
    {
        return Err(invalid("Attention tensor device/dtype/shape mismatch"));
    }
    let expected = required_bytes(dtype, shape)?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < expected || (buffer.ptr() as usize) % dtype.size_in_bytes() != 0 {
        return Err(invalid("Attention tensor storage is invalid"));
    }
    Ok(buffer)
}

pub(crate) fn range(buffer: &CudaBuffer, bytes: usize) -> Result<Range<usize>> {
    let start = buffer.ptr() as usize;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| invalid("Attention address range overflow"))?;
    Ok(start..end)
}

pub(crate) fn alignment(pointer: *const std::ffi::c_void) -> u32 {
    let address = pointer as usize;
    (1usize << address.trailing_zeros().min(8)) as u32
}

pub(crate) fn normalize(ctx: &CudaContext, args: AttentionArgs<'_>) -> Result<Normalized> {
    let q_shape = args.query.shape().dims();
    let k_shape = args.key.shape().dims();
    let v_shape = args.value.shape().dims();
    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(invalid("Attention requires rank-4 Q/K/V tensors"));
    }
    let (batch, query_tokens, query_heads, head_dim) =
        (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (key_batch, key_tokens, kv_heads, key_head_dim) =
        (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    if [
        batch,
        query_tokens,
        query_heads,
        head_dim,
        key_tokens,
        kv_heads,
    ]
    .contains(&0)
        || batch != key_batch
        || v_shape != k_shape
        || head_dim != key_head_dim
        || query_heads % kv_heads != 0
        || args.out.shape().dims() != q_shape
        || args.query.dtype() != args.key.dtype()
        || args.query.dtype() != args.value.dtype()
        || (args.out.dtype() != args.query.dtype()
            && !(args.query.dtype() == DType::F16 && args.out.dtype() == DType::F8E4M3))
        || (args.mask == AttentionMask::Causal && key_tokens < query_tokens)
    {
        return Err(invalid("invalid Attention shape or dtype contract"));
    }
    if !args.scale.is_finite() || args.scale <= 0.0 {
        return Err(invalid("Attention scale must be finite and positive"));
    }
    if !args.output_scale.is_finite()
        || args.output_scale <= 0.0
        || (args.out.dtype() != DType::F8E4M3 && args.output_scale != 1.0)
    {
        return Err(invalid(
            "Attention output_scale must be finite and positive, and is only meaningful for E4M3 output",
        ));
    }
    if [
        batch,
        query_tokens,
        key_tokens,
        query_heads,
        kv_heads,
        head_dim,
    ]
    .iter()
    .any(|value| *value > i32::MAX as usize)
    {
        return Err(invalid("Attention dimension exceeds native limits"));
    }

    let dtype = args.query.dtype();
    let q = tensor_storage(ctx, args.query, dtype, q_shape)?;
    let k = tensor_storage(ctx, args.key, dtype, k_shape)?;
    let v = tensor_storage(ctx, args.value, dtype, v_shape)?;
    let output_dtype = args.out.dtype();
    let out = tensor_storage(ctx, args.out, output_dtype, q_shape)?;
    let out_range = range(&out, required_bytes(output_dtype, q_shape)?)?;
    for (name, input, shape) in [
        ("query", &q, q_shape),
        ("key", &k, k_shape),
        ("value", &v, v_shape),
    ] {
        let input_range = range(input, required_bytes(dtype, shape)?)?;
        if out_range.start < input_range.end && input_range.start < out_range.end {
            return Err(invalid(format!(
                "Attention output overlaps read-only {name} storage"
            )));
        }
    }
    let default_scale = 1.0 / (head_dim as f32).sqrt();
    let bindings = abi::Bindings {
        query: q.ptr(),
        key: k.ptr(),
        value: v.ptr(),
        offsets: std::ptr::null(),
        output: out.ptr(),
        stream: ctx.stream().handle(),
        scale: args.scale,
        output_scale: args.output_scale,
    };
    Ok(Normalized {
        spec: abi::Spec {
            version: abi::SPEC_VERSION,
            semantic: Semantic::Dense.abi_value(),
            dtype: dtype_code(dtype)?,
            output_dtype: output_dtype_code(output_dtype)?,
            mask: args.mask as u32,
            q_alignment: alignment(bindings.query),
            k_alignment: alignment(bindings.key),
            v_alignment: alignment(bindings.value),
            output_alignment: alignment(bindings.output.cast_const()),
            offsets_alignment: 0,
            batch: batch as i64,
            query_tokens: query_tokens as i64,
            key_tokens: key_tokens as i64,
            key_capacity: key_tokens as i64,
            query_heads: query_heads as i64,
            kv_heads: kv_heads as i64,
            head_dim: head_dim as i64,
            query_start: if args.mask == AttentionMask::Causal {
                (key_tokens - query_tokens) as i64
            } else {
                0
            },
            segments: 0,
            max_segment_tokens: 0,
            offsets_hash: 0,
            scale_is_default: u32::from(args.scale == default_scale),
        },
        policy: args.policy,
        bindings,
        storage: vec![q, k, v, out],
    })
}

use apxinf_core::{DType, Result, Tensor};

use super::contracts::{
    alignment, dtype_code, invalid, range, required_bytes, tensor_storage, Normalized,
};
use super::{execution, AttentionMask, AttentionPolicy};
use crate::ffi::abi::attention as abi;
use crate::{CudaBuffer, CudaContext};

/// Packed variable-length self-attention.
///
/// Q/K/V/output are `[total_tokens, heads, head_dim]`. Device-resident U32
/// offsets partition them into independent segments. `host_offsets` must
/// exactly describe the device buffer; it is the immutable metadata snapshot
/// used to validate the contract and key native preparation/autotuning.
pub struct SegmentedAttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub out: &'a mut Tensor,
    pub offsets: &'a CudaBuffer,
    pub host_offsets: &'a [u32],
    pub scale: f32,
    pub policy: AttentionPolicy,
}

impl<'a> SegmentedAttentionArgs<'a> {
    pub fn new(
        query: &'a Tensor,
        key: &'a Tensor,
        value: &'a Tensor,
        out: &'a mut Tensor,
        offsets: &'a CudaBuffer,
        host_offsets: &'a [u32],
    ) -> Self {
        let head_dim = query.shape().dims().get(2).copied().unwrap_or(1);
        Self {
            query,
            key,
            value,
            out,
            offsets,
            host_offsets,
            scale: 1.0 / (head_dim as f32).sqrt(),
            policy: AttentionPolicy::default(),
        }
    }
}

fn offsets_hash(offsets: &[u32]) -> u64 {
    offsets.iter().fold(0xcbf29ce484222325, |hash, value| {
        value.to_le_bytes().iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
    })
}

pub(crate) fn normalize(ctx: &CudaContext, args: SegmentedAttentionArgs<'_>) -> Result<Normalized> {
    let shape = args.query.shape().dims();
    if shape.len() != 3
        || shape.contains(&0)
        || args.key.shape().dims() != shape
        || args.value.shape().dims() != shape
        || args.out.shape().dims() != shape
        || args.key.dtype() != args.query.dtype()
        || args.value.dtype() != args.query.dtype()
        || args.out.dtype() != args.query.dtype()
        || !matches!(args.query.dtype(), DType::F16 | DType::BF16)
    {
        return Err(invalid(
            "segmented Attention requires matching non-empty [tokens,heads,head_dim] tensors",
        ));
    }
    if shape[0] > i32::MAX as usize
        || args.host_offsets.len() < 2
        || args.host_offsets[0] != 0
        || args.host_offsets.last().copied() != Some(shape[0] as u32)
        || args.host_offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid(
            "segmented Attention offsets must be monotonic and span all tokens",
        ));
    }
    let offsets_bytes = args
        .host_offsets
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| invalid("segmented Attention offsets size overflow"))?;
    if args.offsets.device() != ctx.device_id() || args.offsets.len() < offsets_bytes {
        return Err(invalid(
            "segmented Attention requires matching device U32 offsets",
        ));
    }
    if !args.scale.is_finite() || args.scale <= 0.0 {
        return Err(invalid(
            "segmented Attention scale must be finite and positive",
        ));
    }
    let segments = args.host_offsets.len() - 1;
    let max_segment_tokens = args
        .host_offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as usize)
        .max()
        .unwrap_or(0);
    if [shape[0], shape[1], shape[2], segments, max_segment_tokens]
        .iter()
        .any(|value| *value > i32::MAX as usize)
    {
        return Err(invalid(
            "segmented Attention dimension exceeds native limits",
        ));
    }

    let dtype = args.query.dtype();
    let q = tensor_storage(ctx, args.query, dtype, shape)?;
    let k = tensor_storage(ctx, args.key, dtype, shape)?;
    let v = tensor_storage(ctx, args.value, dtype, shape)?;
    let out = tensor_storage(ctx, args.out, dtype, shape)?;
    let output_range = range(&out, required_bytes(dtype, shape)?)?;
    for (name, input) in [("query", &q), ("key", &k), ("value", &v)] {
        let input_range = range(input, required_bytes(dtype, shape)?)?;
        if output_range.start < input_range.end && input_range.start < output_range.end {
            return Err(invalid(format!(
                "segmented Attention output overlaps read-only {name} storage"
            )));
        }
    }
    let offsets = args
        .offsets
        .view(0, offsets_bytes)
        .map_err(apxinf_core::Error::Cuda)?;
    let offsets_range = range(&offsets, offsets_bytes)?;
    if output_range.start < offsets_range.end && offsets_range.start < output_range.end {
        return Err(invalid(
            "segmented Attention output overlaps read-only offsets storage",
        ));
    }
    let default_scale = 1.0 / (shape[2] as f32).sqrt();
    let bindings = abi::Bindings {
        query: q.ptr(),
        key: k.ptr(),
        value: v.ptr(),
        offsets: offsets.ptr(),
        output: out.ptr(),
        stream: ctx.stream().handle(),
        scale: args.scale,
        output_scale: 1.0,
    };
    Ok(Normalized {
        spec: abi::Spec {
            version: abi::SPEC_VERSION,
            semantic: abi::SEMANTIC_SEGMENTED,
            dtype: dtype_code(dtype)?,
            output_dtype: dtype_code(dtype)?,
            mask: AttentionMask::None as u32,
            q_alignment: alignment(bindings.query),
            k_alignment: alignment(bindings.key),
            v_alignment: alignment(bindings.value),
            output_alignment: alignment(bindings.output.cast_const()),
            offsets_alignment: alignment(bindings.offsets),
            batch: 1,
            query_tokens: shape[0] as i64,
            key_tokens: shape[0] as i64,
            key_capacity: shape[0] as i64,
            query_heads: shape[1] as i64,
            kv_heads: shape[1] as i64,
            head_dim: shape[2] as i64,
            query_start: 0,
            segments: segments as i64,
            max_segment_tokens: max_segment_tokens as i64,
            offsets_hash: offsets_hash(args.host_offsets),
            scale_is_default: u32::from(args.scale == default_scale),
        },
        policy: args.policy,
        bindings,
        storage: vec![q, k, v, offsets, out],
    })
}

pub fn segmented_attention(ctx: &CudaContext, args: SegmentedAttentionArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}

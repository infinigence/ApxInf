use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{tensor_storage, Normalized};
use super::launch;
use crate::ffi::abi::rope as abi;
use crate::CudaContext;

const D3_MROPE_SECTIONS: [usize; 3] = [11, 11, 10];

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn alignment(pointer: *const std::ffi::c_void) -> u32 {
    let address = pointer as usize;
    (1usize << address.trailing_zeros().min(8)) as u32
}

/// Packed-QKV split with two-axis height/width rotary positions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AxialRopeNumerics {
    #[default]
    Flexible,
    /// Round biased Q/K inputs to BF16 before applying the rotation.
    RoundedInput,
}

pub struct AxialRopeArgs<'a> {
    pub qkv: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub positions: &'a Tensor,
    pub query: &'a mut Tensor,
    pub key: &'a mut Tensor,
    pub value: &'a mut Tensor,
    pub theta: f32,
    pub numerics: AxialRopeNumerics,
}

/// Packed GQA split with striped temporal/height/width RoPE and token-major
/// K/V cache writes. Outputs are caller-owned so the same contract supports
/// both prefill and appending a suffix to an existing cache.
pub struct MropeCacheArgs<'a> {
    pub qkv: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub positions: &'a Tensor,
    pub query: &'a mut Tensor,
    pub key_cache: &'a mut Tensor,
    pub value_cache: &'a mut Tensor,
    pub theta: f32,
    pub sections: [usize; 3],
    pub cache_offset: usize,
}

/// Apply the D3/Qwen interleaved temporal/height/width partial RoPE in place.
///
/// The current kernel is intentionally restricted to the d3-mini section
/// layout `[11, 11, 10]`; unsupported layouts fail before launch rather than
/// being interpreted as a nearby layout.
pub struct MultimodalRopeArgs<'a> {
    pub data: &'a mut Tensor,
    pub positions: &'a Tensor,
    pub rotary_dim: usize,
    pub sections: [usize; 3],
    pub theta: f32,
}

fn normalize_axial(ctx: &CudaContext, args: AxialRopeArgs<'_>) -> Result<Normalized> {
    let qkv_dims = args.qkv.shape().dims();
    let output_dims = args.query.shape().dims();
    if qkv_dims.len() != 2 || output_dims.len() != 3 {
        return Err(invalid(
            "axial RoPE expects rank-2 packed input and rank-3 outputs",
        ));
    }
    let (tokens, heads, head_dim) = (output_dims[0], output_dims[1], output_dims[2]);
    let projection = heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("axial RoPE projection width overflow"))?;
    if tokens == 0
        || heads == 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 4 != 0
        || qkv_dims != [tokens, 3 * projection]
        || args.key.shape().dims() != output_dims
        || args.value.shape().dims() != output_dims
        || args.positions.shape().dims() != [tokens, 2]
        || !(args.theta.is_finite() && args.theta > 0.0)
    {
        return Err(invalid("axial RoPE shape or scalar mismatch"));
    }
    for value in [tokens, heads, head_dim] {
        i32::try_from(value).map_err(|_| invalid("axial RoPE dimension exceeds native limits"))?;
    }

    let qkv = tensor_storage(ctx, args.qkv, DType::BF16, tokens * 3 * projection)?;
    let bias = args
        .bias
        .map(|tensor| tensor_storage(ctx, tensor, DType::BF16, 3 * projection))
        .transpose()?;
    let positions = tensor_storage(ctx, args.positions, DType::I32, tokens * 2)?;
    let query = tensor_storage(ctx, args.query, DType::BF16, tokens * projection)?;
    let key = tensor_storage(ctx, args.key, DType::BF16, tokens * projection)?;
    let value = tensor_storage(ctx, args.value, DType::BF16, tokens * projection)?;

    let bias_ptr = bias
        .as_ref()
        .map(|buffer| buffer.ptr().cast_const())
        .unwrap_or(std::ptr::null());
    let (semantic, convention) = match args.numerics {
        AxialRopeNumerics::Flexible => (3, abi::CONVENTION_HALF_SPLIT_AXIAL),
        AxialRopeNumerics::RoundedInput => (5, abi::CONVENTION_HALF_SPLIT_AXIAL_ROUNDED_INPUT),
    };
    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic,
        dtype: 2,
        has_bias: u32::from(!bias_ptr.is_null()),
        layout: abi::LAYOUT_PACKED_QKV,
        cache_write: abi::CACHE_WRITE_TOKEN_MAJOR,
        rotation_source: abi::ROTATION_AXIAL_IDS,
        rotary_convention: convention,
        q_heads: heads as u32,
        kv_heads: heads as u32,
        head_dim: head_dim as u32,
        rotary_dim: head_dim as u32,
        section_t: 0,
        section_h: (head_dim / 4) as u32,
        section_w: (head_dim / 4) as u32,
        qkv_alignment: alignment(qkv.ptr()),
        bias_alignment: bias.as_ref().map_or(256, |buffer| alignment(buffer.ptr())),
        q_alignment: alignment(query.ptr()),
        kv_alignment: alignment(key.ptr()),
        rotation_alignment: alignment(positions.ptr()),
        tokens: tokens as i64,
        cache_capacity: 0,
    };
    let bindings = abi::Bindings {
        qkv: qkv.ptr(),
        bias: bias_ptr,
        q: query.ptr(),
        k: key.ptr(),
        v: value.ptr(),
        key_input: std::ptr::null(),
        value_input: std::ptr::null(),
        position: positions.ptr().cast::<u32>(),
        rotation_table: std::ptr::null(),
        stream: ctx.stream().handle(),
        theta: args.theta,
        position_offset: 0,
        kv_output_offset: 0,
    };
    let mut storage = vec![qkv, positions, query, key, value];
    if let Some(bias) = bias {
        storage.push(bias);
    }
    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

fn normalize_mrope_cache(ctx: &CudaContext, args: MropeCacheArgs<'_>) -> Result<Normalized> {
    let qkv_dims = args.qkv.shape().dims();
    let query_dims = args.query.shape().dims();
    let cache_dims = args.key_cache.shape().dims();
    if qkv_dims.len() != 2 || query_dims.len() != 3 || cache_dims.len() != 3 {
        return Err(invalid(
            "mRoPE cache split expects rank-2 QKV and rank-3 outputs",
        ));
    }
    let (tokens, q_heads, head_dim) = (query_dims[0], query_dims[1], query_dims[2]);
    let (capacity, kv_heads, cache_head_dim) = (cache_dims[0], cache_dims[1], cache_dims[2]);
    let q_width = q_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("mRoPE query width overflow"))?;
    let kv_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("mRoPE cache width overflow"))?;
    let fused_width = q_width
        .checked_add(
            kv_width
                .checked_mul(2)
                .ok_or_else(|| invalid("mRoPE fused width overflow"))?,
        )
        .ok_or_else(|| invalid("mRoPE fused width overflow"))?;
    if tokens == 0
        || q_heads == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || head_dim == 0
        || head_dim > 256
        || head_dim % 2 != 0
        || cache_head_dim != head_dim
        || qkv_dims != [tokens, fused_width]
        || args.value_cache.shape().dims() != cache_dims
        || args.positions.shape().dims() != [tokens, 3]
        || args.cache_offset.checked_add(tokens).is_none()
        || args.cache_offset + tokens > capacity
        || args.sections.iter().sum::<usize>() != head_dim / 2
        || !(args.theta.is_finite() && args.theta > 0.0)
    {
        return Err(invalid("mRoPE cache split shape or scalar mismatch"));
    }
    for value in [
        tokens,
        q_heads,
        kv_heads,
        head_dim,
        capacity,
        args.cache_offset,
    ] {
        i32::try_from(value).map_err(|_| invalid("mRoPE cache dimension exceeds native limits"))?;
    }

    let qkv = tensor_storage(ctx, args.qkv, DType::BF16, tokens * fused_width)?;
    let bias = args
        .bias
        .map(|tensor| tensor_storage(ctx, tensor, DType::BF16, fused_width))
        .transpose()?;
    let positions = tensor_storage(ctx, args.positions, DType::I32, tokens * 3)?;
    let query = tensor_storage(ctx, args.query, DType::BF16, tokens * q_width)?;
    let key = tensor_storage(ctx, args.key_cache, DType::BF16, capacity * kv_width)?;
    let value = tensor_storage(ctx, args.value_cache, DType::BF16, capacity * kv_width)?;
    let bias_ptr = bias
        .as_ref()
        .map(|buffer| buffer.ptr().cast_const())
        .unwrap_or(std::ptr::null());
    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: abi::SEMANTIC_SPLIT_QKV_MROPE_CACHE,
        dtype: 2,
        has_bias: u32::from(!bias_ptr.is_null()),
        layout: abi::LAYOUT_PACKED_QKV,
        cache_write: abi::CACHE_WRITE_TOKEN_MAJOR,
        rotation_source: abi::ROTATION_MULTIMODAL_IDS,
        rotary_convention: abi::CONVENTION_HALF_SPLIT_INTERLEAVED_AXES,
        q_heads: q_heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        rotary_dim: head_dim as u32,
        section_t: args.sections[0] as u32,
        section_h: args.sections[1] as u32,
        section_w: args.sections[2] as u32,
        qkv_alignment: alignment(qkv.ptr()),
        bias_alignment: bias.as_ref().map_or(256, |buffer| alignment(buffer.ptr())),
        q_alignment: alignment(query.ptr()),
        kv_alignment: alignment(key.ptr()),
        rotation_alignment: alignment(positions.ptr()),
        tokens: tokens as i64,
        cache_capacity: capacity as i64,
    };
    let bindings = abi::Bindings {
        qkv: qkv.ptr(),
        bias: bias_ptr,
        q: query.ptr(),
        k: key.ptr(),
        v: value.ptr(),
        key_input: std::ptr::null(),
        value_input: std::ptr::null(),
        position: positions.ptr().cast::<u32>(),
        rotation_table: std::ptr::null(),
        stream: ctx.stream().handle(),
        theta: args.theta,
        position_offset: 0,
        kv_output_offset: args.cache_offset as i32,
    };
    let mut storage = vec![qkv, positions, query, key, value];
    if let Some(bias) = bias {
        storage.push(bias);
    }
    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

fn normalize_multimodal(ctx: &CudaContext, args: MultimodalRopeArgs<'_>) -> Result<Normalized> {
    let dims = args.data.shape().dims();
    if dims.len() != 3 {
        return Err(invalid("multimodal RoPE expects [tokens, heads, head_dim]"));
    }
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    if tokens == 0
        || heads == 0
        || heads > 65_535
        || head_dim % 2 != 0
        || args.rotary_dim == 0
        || args.rotary_dim > head_dim
        || args.rotary_dim % 2 != 0
        || args.sections != D3_MROPE_SECTIONS
        || args.sections.iter().sum::<usize>() != args.rotary_dim / 2
        || args.positions.shape().dims() != [tokens, 3]
        || !(args.theta.is_finite() && args.theta > 0.0)
    {
        return Err(invalid(
            "multimodal RoPE requires the validated d3-mini [11, 11, 10] layout",
        ));
    }
    for value in [tokens, heads, head_dim, args.rotary_dim] {
        i32::try_from(value)
            .map_err(|_| invalid("multimodal RoPE dimension exceeds native limits"))?;
    }
    let data = tensor_storage(ctx, args.data, DType::BF16, tokens * heads * head_dim)?;
    let positions = tensor_storage(ctx, args.positions, DType::I32, tokens * 3)?;
    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: 4,
        dtype: 2,
        has_bias: 0,
        layout: abi::LAYOUT_IN_PLACE,
        cache_write: abi::CACHE_WRITE_NONE,
        rotation_source: abi::ROTATION_MULTIMODAL_IDS,
        rotary_convention: abi::CONVENTION_HALF_SPLIT_INTERLEAVED_AXES,
        q_heads: heads as u32,
        kv_heads: heads as u32,
        head_dim: head_dim as u32,
        rotary_dim: args.rotary_dim as u32,
        section_t: args.sections[0] as u32,
        section_h: args.sections[1] as u32,
        section_w: args.sections[2] as u32,
        qkv_alignment: 256,
        bias_alignment: 256,
        q_alignment: alignment(data.ptr()),
        kv_alignment: 256,
        rotation_alignment: alignment(positions.ptr()),
        tokens: tokens as i64,
        cache_capacity: 0,
    };
    let bindings = abi::Bindings {
        qkv: std::ptr::null(),
        bias: std::ptr::null(),
        q: data.ptr(),
        k: std::ptr::null_mut(),
        v: std::ptr::null_mut(),
        key_input: std::ptr::null(),
        value_input: std::ptr::null(),
        position: positions.ptr().cast::<u32>(),
        rotation_table: std::ptr::null(),
        stream: ctx.stream().handle(),
        theta: args.theta,
        position_offset: 0,
        kv_output_offset: 0,
    };
    Ok(Normalized {
        spec,
        bindings,
        storage: vec![data, positions],
    })
}

pub fn split_qkv_axial_rope(ctx: &CudaContext, args: AxialRopeArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize_axial(ctx, args)?)
}

pub fn split_qkv_mrope_cache(ctx: &CudaContext, args: MropeCacheArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize_mrope_cache(ctx, args)?)
}

pub fn multimodal_rope(ctx: &CudaContext, args: MultimodalRopeArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize_multimodal(ctx, args)?)
}

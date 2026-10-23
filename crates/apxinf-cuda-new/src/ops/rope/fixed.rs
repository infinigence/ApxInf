//! Packed-QKV and caller-owned RoPE operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

pub use crate::ops::attention::QkvTensors;
use crate::{ops, CudaContext};

fn split_impl(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    cache: Option<(&Tensor, &Tensor, usize)>,
) -> Result<QkvTensors> {
    let dims = qkv.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("packed QKV must be rank 2".into()));
    }
    let tokens = dims[0];
    let q_shape = Shape::new(vec![tokens, q_heads, head_dim]);
    let mut q = ctx.allocate_output(q_shape, DType::BF16)?;
    let (mut k, mut v, kv_output_offset) = match cache {
        // Rotate K and copy V directly into caller-owned cache rows.
        Some((k_cache, v_cache, offset)) => (k_cache.clone(), v_cache.clone(), offset),
        None => {
            let kv_shape = Shape::new(vec![tokens, kv_heads, head_dim]);
            (
                ctx.allocate_output(kv_shape.clone(), DType::BF16)?,
                ctx.allocate_output(kv_shape, DType::BF16)?,
                0,
            )
        }
    };
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

/// `split_qkv_apply_bf16`: split a packed GQA projection and rotate Q/K into
/// fresh per-call buffers.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    split_impl(
        ctx,
        qkv,
        bias,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        None,
    )
}

/// `apply_q_write_kv_bf16`: split and rotate, appending K/V into caller-owned
/// caches at `output_offset`, returning only the rotated Q.
#[allow(clippy::too_many_arguments)]
pub fn apply_q_write_kv_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    output_offset: usize,
) -> Result<Tensor> {
    Ok(split_impl(
        ctx,
        qkv,
        bias,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        Some((k_cache, v_cache, output_offset)),
    )?
    .q)
}

// ── gr00t RoPE surface ───────────────────────────────────────────────────
//
// These wrap the legacy rope kernels, ported verbatim into the gr00t adapter.
// They reuse cuda-new's `contracts` helpers where the legacy crate used its
// own, so validation is equivalent.

use crate::ops::validation::{
    checked_bytes, gpu_ptr, make_gpu_tensor, matrix_shape, require_address, require_buffers,
    require_finite,
};
use crate::CudaBuffer;

/// `apply_mrope`: rotary-embed a flat `[seq, n_heads * head_dim]` (or 1-D)
/// tensor with the three-section multimodal layout.
pub fn apply_mrope(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    pos_ids: &CudaBuffer,
) -> Result<Tensor> {
    use crate::ffi::abi::{transformer_bf16 as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    let dims = input.shape().dims();
    let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
    if input.dtype() != DType::BF16 {
        return Err(Error::Other("rope_mrope: only BF16 supported".into()));
    }
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_rope_mrope_bf16(
            input_buffer.ptr(),
            output_buffer.ptr(),
            u32::try_from(head_dim).map_err(|_| Error::Other("head dim exceeds u32".into()))?,
            u32::try_from(n_heads).map_err(|_| Error::Other("head count exceeds u32".into()))?,
            u32::try_from(seq_len).map_err(|_| Error::Other("sequence exceeds u32".into()))?,
            theta,
            pos_ids.ptr(),
            u32::try_from(sections[1]).map_err(|_| Error::Other("sec_h exceeds u32".into()))?,
            u32::try_from(sections[2]).map_err(|_| Error::Other("sec_w exceeds u32".into()))?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// `prepare_vision_2d_rotation_table`: build the reusable `[seq, head_dim/2]`
/// `float2` rotation table consumed by the precomputed vision operators.
pub fn prepare_vision_2d_rotation_table(
    ctx: &CudaContext,
    seq_len: usize,
    head_dim: usize,
    theta: f32,
    pos_ids: &CudaBuffer,
) -> Result<CudaBuffer> {
    use crate::ffi::abi::{transformer_bf16 as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    if seq_len == 0 || head_dim == 0 || head_dim % 2 != 0 || theta <= 0.0 {
        return Err(Error::Other(
            "vision rotation table requires a non-empty sequence, even head dimension, and positive theta"
                .into(),
        ));
    }
    require_finite("vision rotation table", &[theta])?;
    let position_bytes = seq_len
        .checked_mul(2)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| Error::Other("vision rotation position size overflow".into()))?;
    let table_bytes = seq_len
        .checked_mul(head_dim / 2)
        .and_then(|count| count.checked_mul(2 * std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::Other("vision rotation table size overflow".into()))?;
    require_address(
        ctx,
        "vision rotation table",
        "positions",
        pos_ids.address(),
        position_bytes,
    )?;
    let table = CudaBuffer::alloc(table_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_build_vision_rotation_table_f32(
            pos_ids.ptr(),
            table.ptr(),
            u32::try_from(head_dim).map_err(|_| Error::Other("head dim exceeds u32".into()))?,
            u32::try_from(seq_len).map_err(|_| Error::Other("sequence exceeds u32".into()))?,
            theta,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(table)
}

/// `apply_vision_2d_pair`: rotate a `[seq, n_heads, head_dim]` Q/K pair with
/// the two-axis vision RoPE.
pub fn apply_vision_2d_pair(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_ids: &CudaBuffer,
) -> Result<(Tensor, Tensor)> {
    use crate::ffi::abi::{transformer_bf16 as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 {
        return Err(Error::Other("rope_vision_2d_pair: only BF16 supported".into()));
    }
    require_finite("vision 2D-RoPE pair", &[theta])?;
    if head_dim == 0 || head_dim % 2 != 0 || n_heads == 0 || theta <= 0.0 {
        return Err(Error::Other(
            "vision 2D-RoPE pair requires non-zero heads, an even head dimension and positive theta"
                .into(),
        ));
    }
    let q_dims = q.shape().dims();
    if q_dims.len() != 3 || q_dims[1] != n_heads || q_dims[2] != head_dim {
        return Err(Error::Other(format!(
            "vision 2D-RoPE pair expected Q [seq,{n_heads},{head_dim}], got {q_dims:?}"
        )));
    }
    if k.shape().dims() != q_dims {
        return Err(Error::Other("vision 2D-RoPE pair requires equal Q/K shapes".into()));
    }
    let seq_len = q_dims[0];
    let expected_bytes = checked_bytes(DType::BF16, &[seq_len, n_heads, head_dim], "vision 2D-RoPE pair")?;
    let position_bytes = seq_len
        .checked_mul(2)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| Error::Other("vision 2D-RoPE position size overflow".into()))?;
    let q_buffer = CudaBuffer::from_tensor(q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(k).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "vision 2D-RoPE pair",
        &[("q", &q_buffer, expected_bytes), ("k", &k_buffer, expected_bytes)],
    )?;
    require_address(ctx, "vision 2D-RoPE pair", "positions", pos_ids.address(), position_bytes)?;
    let shape = Shape::new(vec![seq_len, n_heads, head_dim]);
    let q_out = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let k_out = ctx.allocate_output(shape, DType::BF16)?;
    let q_out_buffer = CudaBuffer::from_tensor(&q_out).map_err(Error::Cuda)?;
    let k_out_buffer = CudaBuffer::from_tensor(&k_out).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_rope_vision_2d_pair_bf16(
            q_buffer.ptr(),
            k_buffer.ptr(),
            q_out_buffer.ptr(),
            k_out_buffer.ptr(),
            u32::try_from(head_dim).map_err(|_| Error::Other("head dim exceeds u32".into()))?,
            u32::try_from(n_heads).map_err(|_| Error::Other("head count exceeds u32".into()))?,
            u32::try_from(seq_len).map_err(|_| Error::Other("sequence exceeds u32".into()))?,
            theta,
            pos_ids.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok((q_out, k_out))
}

/// `rms_norm_apply_mrope_qk_with_block_threads`: per-head RMSNorm on flat
/// `[seq*heads, head_dim]` Q/K followed by mRoPE, with an explicit block
/// width (128 or 256) chosen by the caller's launch policy.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_apply_mrope_qk_with_block_threads(
    ctx: &CudaContext,
    query: &Tensor,
    query_weight: &Tensor,
    key: &Tensor,
    key_weight: &Tensor,
    seq_len: usize,
    query_heads: usize,
    key_heads: usize,
    head_dim: usize,
    eps: f32,
    theta: f32,
    sections: [usize; 3],
    pos_ids: &CudaBuffer,
    block_threads: u32,
) -> Result<(Tensor, Tensor)> {
    use crate::ffi::abi::{transformer_bf16 as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    if query.dtype() != DType::BF16
        || key.dtype() != DType::BF16
        || query_weight.dtype() != DType::BF16
        || key_weight.dtype() != DType::BF16
        || query.shape().dims() != [seq_len * query_heads, head_dim]
        || key.shape().dims() != [seq_len * key_heads, head_dim]
        || query_weight.shape().dims() != [head_dim]
        || key_weight.shape().dims() != [head_dim]
        || seq_len == 0
        || query_heads == 0
        || key_heads == 0
        || head_dim == 0
        || head_dim > 512
        || head_dim % 2 != 0
        || eps <= 0.0
        || theta <= 0.0
    {
        return Err(Error::Other(
            "Q/K RMSNorm mRoPE shape, dtype, or scalar contract mismatch".into(),
        ));
    }
    let query_bytes = checked_bytes(DType::BF16, &[seq_len, query_heads, head_dim], "Q RMSNorm mRoPE")?;
    let key_bytes = checked_bytes(DType::BF16, &[seq_len, key_heads, head_dim], "K RMSNorm mRoPE")?;
    let weight_bytes = checked_bytes(DType::BF16, &[head_dim], "Q/K RMSNorm weights")?;
    let position_bytes = seq_len
        .checked_mul(3)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| Error::Other("Q RMSNorm mRoPE position size overflow".into()))?;
    let q_buffer = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
    let qw_buffer = CudaBuffer::from_tensor(query_weight).map_err(Error::Cuda)?;
    let kw_buffer = CudaBuffer::from_tensor(key_weight).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "Q/K RMSNorm mRoPE",
        &[
            ("query", &q_buffer, query_bytes),
            ("key", &k_buffer, key_bytes),
            ("query weight", &qw_buffer, weight_bytes),
            ("key weight", &kw_buffer, weight_bytes),
        ],
    )?;
    require_address(ctx, "Q/K RMSNorm mRoPE", "positions", pos_ids.address(), position_bytes)?;
    let q_shape = Shape::new(vec![seq_len, query_heads, head_dim]);
    let k_shape = Shape::new(vec![seq_len, key_heads, head_dim]);
    let q_out = ctx.allocate_output(q_shape, DType::BF16)?;
    let k_out = ctx.allocate_output(k_shape, DType::BF16)?;
    let q_out_buffer = CudaBuffer::from_tensor(&q_out).map_err(Error::Cuda)?;
    let k_out_buffer = CudaBuffer::from_tensor(&k_out).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_qk_rms_norm_mrope_bf16_with_threads(
            q_buffer.ptr(),
            qw_buffer.ptr(),
            q_out_buffer.ptr(),
            k_buffer.ptr(),
            kw_buffer.ptr(),
            k_out_buffer.ptr(),
            u32::try_from(head_dim).map_err(|_| Error::Other("head dim exceeds u32".into()))?,
            u32::try_from(query_heads).map_err(|_| Error::Other("query heads exceed u32".into()))?,
            u32::try_from(key_heads).map_err(|_| Error::Other("key heads exceed u32".into()))?,
            u32::try_from(seq_len).map_err(|_| Error::Other("sequence exceeds u32".into()))?,
            eps,
            theta,
            pos_ids.ptr(),
            u32::try_from(sections[1]).map_err(|_| Error::Other("sec_h exceeds u32".into()))?,
            u32::try_from(sections[2]).map_err(|_| Error::Other("sec_w exceeds u32".into()))?,
            block_threads,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok((q_out, k_out))
}

/// `split_qkv_bias_apply_vision_2d`: split a packed vision QKV, add the bias,
/// and rotate Q/K with the dynamic two-axis vision RoPE.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_bias_apply_vision_2d(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_ids: &CudaBuffer,
) -> Result<QkvTensors> {
    vision_split_impl(ctx, qkv, bias, n_heads, head_dim, pos_ids, theta)
}

/// `split_qkv_bias_apply_vision_2d_precomputed`: precomputed-table variant.
pub fn split_qkv_bias_apply_vision_2d_precomputed(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rotation_table: &CudaBuffer,
) -> Result<QkvTensors> {
    vision_split_precomputed_impl(ctx, qkv, bias, n_heads, head_dim, rotation_table, false)
}

/// `split_qkv_bias_apply_vision_2d_precomputed_vec2`: GR00T exact-shape
/// opt-in using two RoPE pairs per thread.
pub fn split_qkv_bias_apply_vision_2d_precomputed_vec2(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rotation_table: &CudaBuffer,
) -> Result<QkvTensors> {
    vision_split_precomputed_impl(ctx, qkv, bias, n_heads, head_dim, rotation_table, true)
}

#[allow(clippy::too_many_arguments)]
fn vision_split_impl(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    head_dim: usize,
    pos_ids: &CudaBuffer,
    theta: f32,
) -> Result<QkvTensors> {
    let (qkv_buffer, bias_buffer, q, k, v) =
        vision_split_common(ctx, qkv, bias, n_heads, head_dim)?;
    use crate::ffi::abi::{transformer_bf16 as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    require_finite("vision fused QKV 2D-RoPE", &[theta])?;
    let (seq_len, _, _) = (qkv.shape().dims()[0], n_heads, head_dim);
    let q_buffer = CudaBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(&k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(&v).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_transformer_bf16_qkv_split_bias_vision_rope_bf16(
            qkv_buffer.ptr(),
            bias_buffer.ptr(),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            to_u32(head_dim, "head dim")?,
            to_u32(n_heads, "head count")?,
            to_u32(seq_len, "sequence")?,
            theta,
            pos_ids.ptr(),
            ctx.stream().handle(),
        ))?;
    }
    Ok(QkvTensors { q, k, v })
}

#[allow(clippy::too_many_arguments)]
fn vision_split_precomputed_impl(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rotation_table: &CudaBuffer,
    vec2: bool,
) -> Result<QkvTensors> {
    let (seq_len, _, _) = (qkv.shape().dims()[0], n_heads, head_dim);
    let (qkv_buffer, bias_buffer, q, k, v) =
        vision_split_common(ctx, qkv, bias, n_heads, head_dim)?;
    use crate::ffi::abi::{transformer_bf16 as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    let rotation_bytes = seq_len
        .checked_mul(head_dim / 2)
        .and_then(|count| count.checked_mul(2 * std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::Other("precomputed vision rotation size overflow".into()))?;
    require_buffers(
        ctx,
        "precomputed vision fused QKV 2D-RoPE",
        &[("rotation table", rotation_table, rotation_bytes)],
    )?;
    let q_buffer = CudaBuffer::from_tensor(&q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(&k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(&v).map_err(Error::Cuda)?;
    unsafe {
        let code = if vec2 {
            abi::apxinf_transformer_bf16_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
                qkv_buffer.ptr(), bias_buffer.ptr(), q_buffer.ptr(), k_buffer.ptr(),
                v_buffer.ptr(), to_u32(head_dim, "head dim")?, to_u32(n_heads, "head count")?,
                to_u32(seq_len, "sequence")?, rotation_table.ptr(), ctx.stream().handle(),
            )
        } else {
            abi::apxinf_transformer_bf16_qkv_split_bias_vision_rope_precomputed_bf16(
                qkv_buffer.ptr(), bias_buffer.ptr(), q_buffer.ptr(), k_buffer.ptr(),
                v_buffer.ptr(), to_u32(head_dim, "head dim")?, to_u32(n_heads, "head count")?,
                to_u32(seq_len, "sequence")?, rotation_table.ptr(), ctx.stream().handle(),
            )
        };
        raw::check_cuda(code).map_err(Error::Cuda)?;
    }
    Ok(QkvTensors { q, k, v })
}

fn vision_split_common(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, Tensor, Tensor, Tensor)> {
    let (seq_len, width) = matrix_shape(qkv, "vision fused QKV 2D-RoPE")?;
    let projection_width = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("vision fused QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16
        || width != 3 * projection_width
        || bias.dtype() != DType::BF16
        || bias.shape().dims() != [width]
        || head_dim == 0
        || head_dim % 2 != 0
        || n_heads == 0
    {
        return Err(Error::Other("vision fused QKV 2D-RoPE shape or dtype mismatch".into()));
    }
    let qkv_bytes = checked_bytes(DType::BF16, &[seq_len, width], "vision fused QKV")?;
    let bias_bytes = checked_bytes(DType::BF16, &[width], "vision fused QKV bias")?;
    let qkv_buffer = CudaBuffer::from_tensor(qkv).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "vision fused QKV 2D-RoPE",
        &[("qkv", &qkv_buffer, qkv_bytes), ("bias", &bias_buffer, bias_bytes)],
    )?;
    let shape = Shape::new(vec![seq_len, n_heads, head_dim]);
    let q = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let k = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let v = ctx.allocate_output(shape, DType::BF16)?;
    Ok((qkv_buffer, bias_buffer, q, k, v))
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds u32")))
}

// ── smolvla RoPE surface ────────────────────────────────────────────────────

/// `apply_batched`: standalone half-split RoPE over `[seq, n_heads, head_dim]`.
/// Legacy signature kept (it dispatched on dtype); cuda-new carries the BF16
/// operator, which is what SmolVLA's BF16 cross-attention path uses.
pub fn apply_batched(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    pos_offset: u32,
) -> Result<Tensor> {
    match input.dtype() {
        DType::BF16 => {
            ops::rope_apply_batched(ctx, input, n_heads, head_dim, rope_theta, pos_offset)
        }
        dtype => Err(Error::Other(format!(
            "CUDA rope_batched does not support {dtype}"
        ))),
    }
}

/// `apply_half_split_f16`: the half-split RoPE convention used by SmolVLA
/// FP16 cross-attention. Ported verbatim from the legacy wrapper over the
/// `apxinf_transformer_f16_rope_half_split_f16` launcher.
pub fn apply_half_split_f16(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    pos_offset: u32,
) -> Result<Tensor> {
    use crate::ffi::abi::transformer_f16 as abi;
    let dims = input.shape().dims();
    let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
    let out_buf = crate::workspace::output_buffer(ctx, input.size_in_bytes())?;

    require_finite("RoPE", &[rope_theta])?;
    if input.dtype() != DType::F16
        || head_dim % 2 != 0
        || rope_theta <= 0.0
        || dims != [seq_len, n_heads, head_dim]
    {
        return Err(Error::Other(
            "half-split FP16 RoPE received invalid shape or dtype".into(),
        ));
    }

    unsafe {
        crate::ops::validation::check_cuda(abi::apxinf_transformer_f16_rope_half_split_f16(
            gpu_ptr(input)?,
            out_buf.ptr(),
            head_dim as u32,
            n_heads as u32,
            seq_len as u32,
            rope_theta,
            pos_offset,
            ctx.stream().handle(),
        ))?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        ctx.device_id(),
        out_buf,
    ))
}

/// `split_qkv_apply_f16`: split a packed FP16 GQA projection and rotate Q/K
/// into fresh per-call buffers. The cuda-new RoPE operator carries the same
/// dtype-templated split kernel, so this routes through it with F16 outputs.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_f16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    let dims = qkv.shape().dims();
    if dims.len() != 2 || qkv.dtype() != DType::F16 {
        return Err(Error::Other(
            "packed FP16 QKV must be a rank-2 FP16 tensor".into(),
        ));
    }
    let tokens = dims[0];
    let mut q = ctx.allocate_output(Shape::new(vec![tokens, q_heads, head_dim]), DType::F16)?;
    let kv_shape = Shape::new(vec![tokens, kv_heads, head_dim]);
    let mut k = ctx.allocate_output(kv_shape.clone(), DType::F16)?;
    let mut v = ctx.allocate_output(kv_shape, DType::F16)?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───

/// Apply RoPE into caller-owned storage using a device-resident position.
#[allow(clippy::too_many_arguments)]
pub fn apply_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    head_dim: usize,
    heads: usize,
    theta: f32,
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use crate::ops::validation::{check_cuda, checked_bytes, require_address, require_buffers, require_finite};
    use crate::ffi::abi::llm_decode as abi;
    require_finite("RoPE", &[theta])?;
    if head_dim % 2 != 0 || theta <= 0.0 {
        return Err(Error::Other(
            "RoPE requires an even head dimension and positive theta".into(),
        ));
    }
    let bytes = checked_bytes(dtype, &[heads, head_dim], "RoPE")?;
    require_buffers(
        ctx,
        "RoPE",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "RoPE", "position", position, 4)?;
    let status = unsafe {
        match dtype {
            DType::F32 => abi::apxinf_llmdec_rope_decode_f32(
                input.ptr(),
                output.ptr(),
                head_dim as u32,
                heads as u32,
                theta,
                position.ptr(),
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_llmdec_rope_decode_bf16(
                input.ptr(),
                output.ptr(),
                head_dim as u32,
                heads as u32,
                theta,
                position.ptr(),
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode RoPE does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Apply multimodal RoPE into caller-owned BF16 storage. `positions` is a
/// `[3]` u32 device buffer of per-token (t, h, w) axes.
#[allow(clippy::too_many_arguments)]
pub fn apply_mrope_bf16_into(
    ctx: &CudaContext,
    input: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    head_dim: usize,
    heads: usize,
    theta: f32,
    positions: crate::CudaDeviceAddress,
    section_h: usize,
    section_w: usize,
) -> Result<()> {
    use crate::ops::validation::{check_cuda, checked_bytes, require_address, require_buffers, require_finite};
    use crate::ffi::abi::llm_decode as abi;
    require_finite("MRoPE", &[theta])?;
    if head_dim % 2 != 0 || section_h + section_w > head_dim / 2 || theta <= 0.0 {
        return Err(Error::Other(
            "MRoPE received invalid head/section dimensions or theta".into(),
        ));
    }
    let bytes = checked_bytes(DType::BF16, &[heads, head_dim], "MRoPE")?;
    require_buffers(
        ctx,
        "MRoPE",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    require_address(ctx, "MRoPE", "positions", positions, 12)?;
    check_cuda(unsafe {
        abi::apxinf_llmdec_rope_mrope_decode_bf16(
            input.ptr(),
            output.ptr(),
            head_dim as u32,
            heads as u32,
            theta,
            positions.ptr(),
            section_h as u32,
            section_w as u32,
            ctx.stream().handle(),
        )
    })
}

/// Apply BF16 RoPE to K and write it directly into a head-major
/// `[kv_heads, max_seq_len, head_dim]` KV cache at the device-resident
/// position.
#[allow(clippy::too_many_arguments)]
pub fn apply_k_write_cache_bf16(
    ctx: &CudaContext,
    input: &crate::CudaBuffer,
    cache: &crate::CudaBuffer,
    head_dim: usize,
    kv_heads: usize,
    max_seq_len: usize,
    theta: f32,
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use crate::ops::validation::{check_cuda, checked_bytes, require_address, require_buffers, require_finite};
    use crate::ffi::abi::llm_decode as abi;
    require_finite("RoPE K write", &[theta])?;
    require_buffers(
        ctx,
        "RoPE K write",
        &[
            (
                "input",
                input,
                checked_bytes(DType::BF16, &[kv_heads, head_dim], "RoPE K write")?,
            ),
            (
                "cache",
                cache,
                checked_bytes(
                    DType::BF16,
                    &[kv_heads, max_seq_len, head_dim],
                    "RoPE K write",
                )?,
            ),
        ],
    )?;
    require_address(ctx, "RoPE K write", "position", position, 4)?;
    check_cuda(unsafe {
        abi::apxinf_llmdec_rope_k_write_bf16(
            input.ptr(),
            cache.ptr(),
            head_dim as u32,
            kv_heads as u32,
            max_seq_len as u32,
            theta,
            position.ptr(),
            ctx.stream().handle(),
        )
    })
}

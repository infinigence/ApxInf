//! Packed-QKV and caller-owned RoPE operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{alignment_of, Normalized};
use super::launch;
use crate::CudaContext;

// ── gr00t RoPE surface ───────────────────────────────────────────────────
//
// These wrap the legacy rope kernels, ported verbatim into the gr00t adapter.
// They reuse cuda-new's `contracts` helpers where the legacy crate used its
// own, so validation is equivalent.

use crate::ops::validation::{
    checked_bytes, make_gpu_tensor, matrix_shape, require_address, require_buffers, require_finite,
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
    if input.dtype() != DType::BF16 {
        return Err(Error::Other("rope_mrope: only BF16 supported".into()));
    }
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    super::rope::apply_tensor(
        ctx,
        input,
        &output,
        n_heads,
        head_dim,
        theta,
        super::rope::ApplyRotation::Multimodal {
            positions: pos_ids,
            sections,
        },
    )?;
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
    use crate::ffi::abi::rope as abi;
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
        raw::check_cuda(abi::apxinf_rope_build_vision_rotation_table_f32(
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
    use crate::ffi::abi::rope as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 {
        return Err(Error::Other(
            "rope_vision_2d_pair: only BF16 supported".into(),
        ));
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
        return Err(Error::Other(
            "vision 2D-RoPE pair requires equal Q/K shapes".into(),
        ));
    }
    let seq_len = q_dims[0];
    let expected_bytes = checked_bytes(
        DType::BF16,
        &[seq_len, n_heads, head_dim],
        "vision 2D-RoPE pair",
    )?;
    let position_bytes = seq_len
        .checked_mul(2)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| Error::Other("vision 2D-RoPE position size overflow".into()))?;
    let q_buffer = CudaBuffer::from_tensor(q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(k).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "vision 2D-RoPE pair",
        &[
            ("q", &q_buffer, expected_bytes),
            ("k", &k_buffer, expected_bytes),
        ],
    )?;
    require_address(
        ctx,
        "vision 2D-RoPE pair",
        "positions",
        pos_ids.address(),
        position_bytes,
    )?;
    let shape = Shape::new(vec![seq_len, n_heads, head_dim]);
    let q_out = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let k_out = ctx.allocate_output(shape, DType::BF16)?;
    let q_out_buffer = CudaBuffer::from_tensor(&q_out).map_err(Error::Cuda)?;
    let k_out_buffer = CudaBuffer::from_tensor(&k_out).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_rope_vision_2d_pair_bf16(
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

/// `rms_norm_apply_mrope_qk`: per-head RMSNorm on flat
/// `[seq*heads, head_dim]` Q/K followed by mRoPE. Native launch policy owns
/// the block width so models only describe the mathematical operation.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_apply_mrope_qk(
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
) -> Result<(Tensor, Tensor)> {
    use crate::ffi::abi::rope as abi;
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
    let query_bytes = checked_bytes(
        DType::BF16,
        &[seq_len, query_heads, head_dim],
        "Q RMSNorm mRoPE",
    )?;
    let key_bytes = checked_bytes(
        DType::BF16,
        &[seq_len, key_heads, head_dim],
        "K RMSNorm mRoPE",
    )?;
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
    require_address(
        ctx,
        "Q/K RMSNorm mRoPE",
        "positions",
        pos_ids.address(),
        position_bytes,
    )?;
    let q_shape = Shape::new(vec![seq_len, query_heads, head_dim]);
    let k_shape = Shape::new(vec![seq_len, key_heads, head_dim]);
    let q_out = ctx.allocate_output(q_shape, DType::BF16)?;
    let k_out = ctx.allocate_output(k_shape, DType::BF16)?;
    let q_out_buffer = CudaBuffer::from_tensor(&q_out).map_err(Error::Cuda)?;
    let k_out_buffer = CudaBuffer::from_tensor(&k_out).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_rope_qk_rms_norm_mrope_bf16(
            ctx.runtime(),
            q_buffer.ptr(),
            qw_buffer.ptr(),
            q_out_buffer.ptr(),
            k_buffer.ptr(),
            kw_buffer.ptr(),
            k_out_buffer.ptr(),
            u32::try_from(head_dim).map_err(|_| Error::Other("head dim exceeds u32".into()))?,
            u32::try_from(query_heads)
                .map_err(|_| Error::Other("query heads exceed u32".into()))?,
            u32::try_from(key_heads).map_err(|_| Error::Other("key heads exceed u32".into()))?,
            u32::try_from(seq_len).map_err(|_| Error::Other("sequence exceeds u32".into()))?,
            eps,
            theta,
            pos_ids.ptr(),
            u32::try_from(sections[1]).map_err(|_| Error::Other("sec_h exceeds u32".into()))?,
            u32::try_from(sections[2]).map_err(|_| Error::Other("sec_w exceeds u32".into()))?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok((q_out, k_out))
}

/// Position source for the vision packed-QKV RoPE semantic.
pub enum VisionRotation<'a> {
    Positions { ids: &'a CudaBuffer, theta: f32 },
    Table(&'a CudaBuffer),
}

/// Split packed vision QKV, add bias, and apply the selected two-axis RoPE.
/// The position source is semantic input; scalar/vectorized table launchers
/// remain private implementation candidates.
pub struct VisionQkvRopeArgs<'a> {
    pub qkv: &'a Tensor,
    pub bias: &'a Tensor,
    pub q: &'a mut Tensor,
    pub k: &'a mut Tensor,
    pub v: &'a mut Tensor,
    pub heads: usize,
    pub head_dim: usize,
    pub rotation: VisionRotation<'a>,
}

pub fn split_qkv_vision(ctx: &CudaContext, args: VisionQkvRopeArgs<'_>) -> Result<()> {
    let (qkv_buffer, bias_buffer, q_buffer, k_buffer, v_buffer) = vision_split_common(
        ctx,
        args.qkv,
        args.bias,
        args.q,
        args.k,
        args.v,
        args.heads,
        args.head_dim,
    )?;
    use crate::ffi::abi::rope as abi;
    let seq_len = args.qkv.shape().dims()[0];
    let (rotation_source, theta, position, rotation_table, rotation) = match args.rotation {
        VisionRotation::Positions { ids, theta } => {
            require_finite("vision fused QKV 2D-RoPE", &[theta])?;
            if theta <= 0.0 {
                return Err(Error::Other(
                    "vision fused QKV 2D-RoPE theta must be positive".into(),
                ));
            }
            let position_bytes = seq_len
                .checked_mul(2)
                .and_then(|count| count.checked_mul(std::mem::size_of::<u32>()))
                .ok_or_else(|| Error::Other("vision position size overflow".into()))?;
            require_buffers(
                ctx,
                "vision fused QKV 2D-RoPE",
                &[("positions", ids, position_bytes)],
            )?;
            (
                abi::ROTATION_AXIAL_IDS,
                theta,
                ids.ptr().cast_const().cast::<u32>(),
                std::ptr::null(),
                ids.clone(),
            )
        }
        VisionRotation::Table(table) => {
            let rotation_bytes = seq_len
                .checked_mul(args.head_dim / 2)
                .and_then(|count| count.checked_mul(2 * std::mem::size_of::<f32>()))
                .ok_or_else(|| Error::Other("precomputed vision rotation size overflow".into()))?;
            require_buffers(
                ctx,
                "precomputed vision fused QKV 2D-RoPE",
                &[("rotation table", table, rotation_bytes)],
            )?;
            (
                abi::ROTATION_PRECOMPUTED_TABLE,
                0.0,
                std::ptr::null(),
                table.ptr().cast_const(),
                table.clone(),
            )
        }
    };
    let normalized = Normalized {
        spec: abi::Spec {
            version: abi::SPEC_VERSION,
            semantic: 5,
            dtype: 2,
            has_bias: 1,
            layout: abi::LAYOUT_PACKED_QKV,
            cache_write: abi::CACHE_WRITE_TOKEN_MAJOR,
            rotation_source,
            rotary_convention: abi::CONVENTION_HALF_SPLIT_AXIAL,
            q_heads: to_u32(args.heads, "head count")?,
            kv_heads: to_u32(args.heads, "head count")?,
            head_dim: to_u32(args.head_dim, "head dim")?,
            rotary_dim: to_u32(args.head_dim, "rotary dim")?,
            section_t: 0,
            section_h: to_u32(args.head_dim / 4, "height section")?,
            section_w: to_u32(args.head_dim / 4, "width section")?,
            qkv_alignment: alignment_of(qkv_buffer.ptr() as usize),
            bias_alignment: alignment_of(bias_buffer.ptr() as usize),
            q_alignment: alignment_of(q_buffer.ptr() as usize),
            kv_alignment: alignment_of(k_buffer.ptr() as usize),
            rotation_alignment: alignment_of(rotation.ptr() as usize),
            tokens: i64::try_from(seq_len)
                .map_err(|_| Error::Other("sequence exceeds i64".into()))?,
            cache_capacity: 0,
        },
        bindings: abi::Bindings {
            qkv: qkv_buffer.ptr(),
            bias: bias_buffer.ptr(),
            q: q_buffer.ptr(),
            k: k_buffer.ptr(),
            v: v_buffer.ptr(),
            key_input: std::ptr::null(),
            value_input: std::ptr::null(),
            position,
            rotation_table,
            stream: ctx.stream().handle(),
            theta,
            position_offset: 0,
            kv_output_offset: 0,
        },
        storage: vec![
            qkv_buffer,
            bias_buffer,
            q_buffer,
            k_buffer,
            v_buffer,
            rotation,
        ],
    };
    launch::execute(ctx, normalized)?;
    Ok(())
}

fn vision_split_common(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    head_dim: usize,
) -> Result<(CudaBuffer, CudaBuffer, CudaBuffer, CudaBuffer, CudaBuffer)> {
    let (seq_len, width) = matrix_shape(qkv, "vision fused QKV 2D-RoPE")?;
    let projection_width = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("vision fused QKV width overflow".into()))?;
    if qkv.dtype() != DType::BF16
        || width != 3 * projection_width
        || bias.dtype() != DType::BF16
        || bias.shape().dims() != [width]
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || q.shape().dims() != [seq_len, n_heads, head_dim]
        || k.shape().dims() != [seq_len, n_heads, head_dim]
        || v.shape().dims() != [seq_len, n_heads, head_dim]
        || head_dim == 0
        || head_dim % 2 != 0
        || n_heads == 0
    {
        return Err(Error::Other(
            "vision fused QKV 2D-RoPE shape or dtype mismatch".into(),
        ));
    }
    let qkv_bytes = checked_bytes(DType::BF16, &[seq_len, width], "vision fused QKV")?;
    let bias_bytes = checked_bytes(DType::BF16, &[width], "vision fused QKV bias")?;
    let output_bytes = checked_bytes(
        DType::BF16,
        &[seq_len, n_heads, head_dim],
        "vision fused QKV output",
    )?;
    let qkv_buffer = CudaBuffer::from_tensor(qkv).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let q_buffer = CudaBuffer::from_tensor(q).map_err(Error::Cuda)?;
    let k_buffer = CudaBuffer::from_tensor(k).map_err(Error::Cuda)?;
    let v_buffer = CudaBuffer::from_tensor(v).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "vision fused QKV 2D-RoPE",
        &[
            ("qkv", &qkv_buffer, qkv_bytes),
            ("bias", &bias_buffer, bias_bytes),
            ("q", &q_buffer, output_bytes),
            ("k", &k_buffer, output_bytes),
            ("v", &v_buffer, output_bytes),
        ],
    )?;
    Ok((qkv_buffer, bias_buffer, q_buffer, k_buffer, v_buffer))
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds u32")))
}

// ── smolvla RoPE surface ────────────────────────────────────────────────────

/// The half-split RoPE convention used by SmolVLA
/// FP16 cross-attention. The unified RoPE family dispatch preserves the
/// legacy half-split kernel and parameter order as its F16 candidate.
pub fn apply_half_split(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    pos_offset: u32,
) -> Result<Tensor> {
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

    let output = make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        ctx.device_id(),
        out_buf,
    );
    super::rope::apply_tensor(
        ctx,
        input,
        &output,
        n_heads,
        head_dim,
        rope_theta,
        super::rope::ApplyRotation::PositionOffset(pos_offset as usize),
    )?;
    Ok(output)
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───

/// Apply RoPE into caller-owned storage using a device-resident position.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    head_dim: usize,
    heads: usize,
    theta: f32,
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use crate::ffi::abi::rope as abi;
    use crate::ops::validation::{
        check_cuda, checked_bytes, require_address, require_buffers, require_finite,
    };
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
            DType::F32 => abi::apxinf_rope_fixed_decode_decode_f32(
                input.ptr(),
                output.ptr(),
                head_dim as u32,
                heads as u32,
                theta,
                position.ptr(),
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_rope_fixed_decode_decode_bf16(
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
pub(super) fn apply_mrope_into(
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
    use crate::ffi::abi::rope as abi;
    use crate::ops::validation::{
        check_cuda, checked_bytes, require_address, require_buffers, require_finite,
    };
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
        abi::apxinf_rope_fixed_decode_mrope_decode_bf16(
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

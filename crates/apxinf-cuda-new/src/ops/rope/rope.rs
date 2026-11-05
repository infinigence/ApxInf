use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{alignment_of, normalize, tensor_storage, Normalized, RopeArgs};
use super::launch;
use crate::ffi::abi::rope as abi;
use crate::{CudaBuffer, CudaContext};

/// Splits a packed QKV projection, optionally applying rotary embedding.
pub fn rope(ctx: &CudaContext, args: RopeArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

pub(super) enum ApplyRotation<'a> {
    PositionOffset(usize),
    Multimodal {
        positions: &'a CudaBuffer,
        sections: [usize; 3],
    },
    Axial {
        positions: &'a CudaBuffer,
    },
}

pub(super) fn apply_tensor(
    ctx: &CudaContext,
    input: &Tensor,
    output: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    rotation: ApplyRotation<'_>,
) -> Result<()> {
    if n_heads == 0
        || head_dim == 0
        || head_dim > 2048
        || head_dim % 2 != 0
        || !(theta.is_finite() && theta > 0.0)
        || input.dtype() != output.dtype()
        || !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        return Err(Error::Other(
            "standalone RoPE shape, dtype, or theta mismatch".into(),
        ));
    }
    let row_width = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("standalone RoPE width overflow".into()))?;
    let elements = input.shape().numel();
    if elements == 0 || elements % row_width != 0 || output.shape().numel() != elements {
        return Err(Error::Other(
            "standalone RoPE input/output element count mismatch".into(),
        ));
    }
    let tokens = elements / row_width;
    let input_buffer = tensor_storage(ctx, input, input.dtype(), elements)?;
    let output_buffer = tensor_storage(ctx, output, output.dtype(), elements)?;

    let (rotation_source, convention, rotary_dim, sections, position, position_offset) =
        match rotation {
            ApplyRotation::PositionOffset(offset) => (
                abi::ROTATION_POSITION_OFFSET,
                abi::CONVENTION_HALF_SPLIT,
                head_dim,
                [0, 0, 0],
                None,
                offset,
            ),
            ApplyRotation::Multimodal {
                positions,
                sections,
            } => {
                let section_hw = sections[1]
                    .checked_add(sections[2])
                    .ok_or_else(|| Error::Other("multimodal RoPE section overflow".into()))?;
                let position_bytes = tokens
                    .checked_mul(3)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
                    .ok_or_else(|| Error::Other("multimodal RoPE position size overflow".into()))?;
                if input.dtype() != DType::BF16
                    || section_hw > head_dim / 2
                    || positions.device() != ctx.device_id()
                    || positions.len() < position_bytes
                {
                    return Err(Error::Other(
                        "multimodal RoPE position/section contract mismatch".into(),
                    ));
                }
                (
                    abi::ROTATION_MULTIMODAL_IDS,
                    abi::CONVENTION_HALF_SPLIT_INTERLEAVED_AXES,
                    head_dim,
                    sections,
                    Some(positions),
                    0,
                )
            }
            ApplyRotation::Axial { positions } => {
                let position_bytes = tokens
                    .checked_mul(2)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<u32>()))
                    .ok_or_else(|| Error::Other("axial RoPE position size overflow".into()))?;
                if input.dtype() != DType::BF16
                    || positions.device() != ctx.device_id()
                    || positions.len() < position_bytes
                {
                    return Err(Error::Other("axial RoPE position contract mismatch".into()));
                }
                let half = head_dim / 2;
                (
                    abi::ROTATION_AXIAL_IDS,
                    abi::CONVENTION_HALF_SPLIT_AXIAL,
                    head_dim,
                    [0, half / 2, half - half / 2],
                    Some(positions),
                    0,
                )
            }
        };
    let position_ptr = position
        .map(|buffer| buffer.ptr() as *const u32)
        .unwrap_or(std::ptr::null());
    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: abi::SEMANTIC_APPLY,
        dtype: match input.dtype() {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::BF16 => 2,
            _ => unreachable!("validated standalone RoPE dtype"),
        },
        has_bias: 0,
        layout: abi::LAYOUT_SINGLE_TENSOR,
        cache_write: abi::CACHE_WRITE_NONE,
        rotation_source,
        rotary_convention: convention,
        q_heads: u32::try_from(n_heads)
            .map_err(|_| Error::Other("RoPE head count exceeds u32".into()))?,
        kv_heads: u32::try_from(n_heads)
            .map_err(|_| Error::Other("RoPE head count exceeds u32".into()))?,
        head_dim: u32::try_from(head_dim)
            .map_err(|_| Error::Other("RoPE head dimension exceeds u32".into()))?,
        rotary_dim: u32::try_from(rotary_dim)
            .map_err(|_| Error::Other("RoPE rotary dimension exceeds u32".into()))?,
        section_t: u32::try_from(sections[0])
            .map_err(|_| Error::Other("RoPE section exceeds u32".into()))?,
        section_h: u32::try_from(sections[1])
            .map_err(|_| Error::Other("RoPE section exceeds u32".into()))?,
        section_w: u32::try_from(sections[2])
            .map_err(|_| Error::Other("RoPE section exceeds u32".into()))?,
        qkv_alignment: alignment_of(input_buffer.ptr() as usize),
        bias_alignment: 256,
        q_alignment: alignment_of(output_buffer.ptr() as usize),
        kv_alignment: 256,
        rotation_alignment: position.map_or(256, |buffer| alignment_of(buffer.ptr() as usize)),
        tokens: i64::try_from(tokens)
            .map_err(|_| Error::Other("RoPE token count exceeds i64".into()))?,
        cache_capacity: 0,
    };
    let bindings = abi::Bindings {
        qkv: input_buffer.ptr(),
        bias: std::ptr::null(),
        q: output_buffer.ptr(),
        k: std::ptr::null_mut(),
        v: std::ptr::null_mut(),
        key_input: std::ptr::null(),
        value_input: std::ptr::null(),
        position: position_ptr,
        rotation_table: std::ptr::null(),
        stream: ctx.stream().handle(),
        theta,
        position_offset: i32::try_from(position_offset)
            .map_err(|_| Error::Other("RoPE offset exceeds i32".into()))?,
        kv_output_offset: 0,
    };
    let mut storage = vec![input_buffer, output_buffer];
    if let Some(position) = position {
        storage.push(position.clone());
    }
    launch::execute(
        ctx,
        Normalized {
            spec,
            bindings,
            storage,
        },
    )
}

fn upload_u32(ctx: &CudaContext, values: &[u32], what: &str) -> Result<CudaBuffer> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer
        .copy_from_host(&bytes)
        .map_err(|error| Error::Other(format!("{what}: upload position ids failed: {error}")))?;
    Ok(buffer)
}

/// Standalone half-split RoPE over a contiguous `[seq, n_heads, head_dim]`
/// BF16 tensor. This is the portable `Backend::rope` semantic, not a packed
/// QKV split; Q and K each call it with their own head count.
pub fn apply_batched(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_offset: u32,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 3 || dims[1] != n_heads || dims[2] != head_dim {
        return Err(Error::Other(format!(
            "batched RoPE expects [{}, {n_heads}, {head_dim}], got {dims:?}",
            dims.first().copied().unwrap_or(0)
        )));
    }
    let seq_len = dims[0];
    if head_dim == 0 || head_dim % 2 != 0 || n_heads == 0 || seq_len == 0 {
        return Err(Error::Other(
            "batched RoPE requires even non-zero dims".into(),
        ));
    }
    let output = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    apply_tensor(
        ctx,
        input,
        &output,
        n_heads,
        head_dim,
        theta,
        ApplyRotation::PositionOffset(pos_offset as usize),
    )?;
    Ok(output)
}

/// Multimodal 3D RoPE (Qwen3-VL). `pos_ids` is `(t, h, w)` per token;
/// `sections` is `[T, H, W]` splitting the `head_dim/2` frequency pairs.
pub fn apply_mrope(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    pos_ids: &[u32],
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 3 || dims[1] != n_heads || dims[2] != head_dim {
        return Err(Error::Other(
            "mRoPE requires a [seq, heads, dim] tensor".into(),
        ));
    }
    let seq_len = dims[0];
    if pos_ids.len() != seq_len * 3 {
        return Err(Error::Other(format!(
            "mRoPE pos_ids len {} != seq_len {} * 3",
            pos_ids.len(),
            seq_len
        )));
    }
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let ids = upload_u32(ctx, pos_ids, "mRoPE")?;
    apply_tensor(
        ctx,
        input,
        &output,
        n_heads,
        head_dim,
        theta,
        ApplyRotation::Multimodal {
            positions: &ids,
            sections,
        },
    )?;
    Ok(output)
}

/// Vision-tower 2D RoPE (Qwen3-VL ViT). `pos_ids` is `(h, w)` per token;
/// the first half of the frequency pairs uses h, the second half uses w.
pub fn apply_vision_2d(
    ctx: &CudaContext,
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_ids: &[u32],
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 3 || dims[1] != n_heads || dims[2] != head_dim {
        return Err(Error::Other(
            "vision 2D RoPE requires a [seq, heads, dim] tensor".into(),
        ));
    }
    let seq_len = dims[0];
    if pos_ids.len() != seq_len * 2 {
        return Err(Error::Other(format!(
            "vision 2D RoPE pos_ids len {} != seq_len {} * 2",
            pos_ids.len(),
            seq_len
        )));
    }
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let ids = upload_u32(ctx, pos_ids, "vision 2D RoPE")?;
    apply_tensor(
        ctx,
        input,
        &output,
        n_heads,
        head_dim,
        theta,
        ApplyRotation::Axial { positions: &ids },
    )?;
    Ok(output)
}

use apxinf_core::{DType, Device, Error, Result, Tensor};

use super::contracts::{normalize, RopeArgs};
use super::launch;
use crate::ffi::abi::{rope as abi, status};
use crate::{CudaBuffer, CudaContext};

/// Splits a packed QKV projection, optionally applying rotary embedding.
pub fn rope(ctx: &CudaContext, args: RopeArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

fn bf16_rank3_storage(ctx: &CudaContext, tensor: &Tensor, what: &str) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != DType::BF16 {
        return Err(Error::Other(format!("{what} requires a BF16 CUDA tensor")));
    }
    if tensor.shape().dims().len() != 3 {
        return Err(Error::Other(format!("{what} requires a [seq, heads, dim] tensor")));
    }
    CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)
}

fn upload_u32(ctx: &CudaContext, values: &[u32], what: &str) -> Result<CudaBuffer> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(|error| {
        Error::Other(format!("{what}: upload position ids failed: {error}"))
    })?;
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
        return Err(Error::Other("batched RoPE requires even non-zero dims".into()));
    }
    let output = ctx.allocate_output(input.shape().clone(), DType::BF16)?;
    let input_buffer = bf16_rank3_storage(ctx, input, "batched RoPE")?;
    let output_buffer = bf16_rank3_storage(ctx, &output, "batched RoPE")?;
    let n_heads = i32::try_from(n_heads).map_err(|_| Error::Other("RoPE heads exceed i32".into()))?;
    let head_dim = i32::try_from(head_dim).map_err(|_| Error::Other("RoPE dim exceeds i32".into()))?;
    let seq_len = i32::try_from(seq_len).map_err(|_| Error::Other("RoPE seq exceeds i32".into()))?;
    let pos_offset =
        i32::try_from(pos_offset).map_err(|_| Error::Other("RoPE offset exceeds i32".into()))?;
    unsafe {
        status::check(abi::apxinf_rope_apply_batched_bf16(
            input_buffer.ptr(),
            output_buffer.ptr(),
            n_heads,
            head_dim,
            seq_len,
            theta,
            pos_offset,
            ctx.stream().handle(),
        ))?;
    }
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
        return Err(Error::Other("mRoPE requires a [seq, heads, dim] tensor".into()));
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
    let input_buffer = bf16_rank3_storage(ctx, input, "mRoPE")?;
    let output_buffer = bf16_rank3_storage(ctx, &output, "mRoPE")?;
    let ids = upload_u32(ctx, pos_ids, "mRoPE")?;
    let n_heads = i32::try_from(n_heads).map_err(|_| Error::Other("mRoPE heads exceed i32".into()))?;
    let head_dim = i32::try_from(head_dim).map_err(|_| Error::Other("mRoPE dim exceeds i32".into()))?;
    let seq_len = i32::try_from(seq_len).map_err(|_| Error::Other("mRoPE seq exceeds i32".into()))?;
    let sec_h = i32::try_from(sections[1]).map_err(|_| Error::Other("mRoPE section exceeds i32".into()))?;
    let sec_w = i32::try_from(sections[2]).map_err(|_| Error::Other("mRoPE section exceeds i32".into()))?;
    unsafe {
        status::check(abi::apxinf_rope_apply_mrope_bf16(
            input_buffer.ptr(),
            output_buffer.ptr(),
            n_heads,
            head_dim,
            seq_len,
            theta,
            ids.ptr(),
            sec_h,
            sec_w,
            ctx.stream().handle(),
        ))?;
    }
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
    let input_buffer = bf16_rank3_storage(ctx, input, "vision 2D RoPE")?;
    let output_buffer = bf16_rank3_storage(ctx, &output, "vision 2D RoPE")?;
    let ids = upload_u32(ctx, pos_ids, "vision 2D RoPE")?;
    let n_heads =
        i32::try_from(n_heads).map_err(|_| Error::Other("vision RoPE heads exceed i32".into()))?;
    let head_dim =
        i32::try_from(head_dim).map_err(|_| Error::Other("vision RoPE dim exceeds i32".into()))?;
    let seq_len =
        i32::try_from(seq_len).map_err(|_| Error::Other("vision RoPE seq exceeds i32".into()))?;
    unsafe {
        status::check(abi::apxinf_rope_apply_vision_2d_bf16(
            input_buffer.ptr(),
            output_buffer.ptr(),
            n_heads,
            head_dim,
            seq_len,
            theta,
            ids.ptr(),
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}


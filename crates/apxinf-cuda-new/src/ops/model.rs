//! Whole-model helpers: embedding gather and greedy token selection.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{model as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

fn storage_ranges_overlap(first: &crate::CudaBuffer, second: &crate::CudaBuffer) -> bool {
    let first_start = first.ptr() as usize;
    let second_start = second.ptr() as usize;
    let Some(first_end) = first_start.checked_add(first.len()) else {
        return true;
    };
    let Some(second_end) = second_start.checked_add(second.len()) else {
        return true;
    };
    first_start < second_end && second_start < first_end
}

/// Fused learned-position preparation for a spatial-merge vision tower.
///
/// The kernel bilinearly interpolates a square BF16 position table to
/// `height x width`, writes the embeddings in spatial-merge order, and emits
/// the matching `[row, column]` axial position IDs in the same launch.
pub struct VisionPositionArgs<'a> {
    pub table: &'a Tensor,
    pub embeddings: &'a mut Tensor,
    pub position_ids: &'a mut Tensor,
    pub height: usize,
    pub width: usize,
    pub merge_size: usize,
}

pub fn vision_position(ctx: &CudaContext, args: VisionPositionArgs<'_>) -> Result<()> {
    let table_dims = args.table.shape().dims();
    if table_dims.len() != 2 {
        return Err(invalid("vision position table must be rank 2"));
    }
    let (source_positions, hidden) = (table_dims[0], table_dims[1]);
    let source_side = (source_positions as f64).sqrt() as usize;
    let tokens = args
        .height
        .checked_mul(args.width)
        .ok_or_else(|| invalid("vision position target size overflow"))?;
    if source_side == 0
        || source_side.checked_mul(source_side) != Some(source_positions)
        || hidden == 0
        || args.height == 0
        || args.width == 0
        || args.merge_size == 0
        || args.height % args.merge_size != 0
        || args.width % args.merge_size != 0
    {
        return Err(invalid(
            "vision position requires a square table and merge-aligned target grid",
        ));
    }
    for extent in [
        source_side,
        args.height,
        args.width,
        hidden,
        args.merge_size,
        tokens,
    ] {
        i32::try_from(extent)
            .map_err(|_| invalid("vision position dimension exceeds native limits"))?;
    }

    let table = tensor_storage(ctx, args.table, DType::BF16, table_dims)?;
    let embeddings = tensor_storage(ctx, args.embeddings, DType::BF16, &[tokens, hidden])?;
    let position_ids = tensor_storage(ctx, args.position_ids, DType::I32, &[tokens, 2])?;
    if storage_ranges_overlap(&table, &embeddings)
        || storage_ranges_overlap(&table, &position_ids)
        || storage_ranges_overlap(&embeddings, &position_ids)
    {
        return Err(invalid("vision position tensor storage must not overlap"));
    }
    crate::workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize)?;
    unsafe {
        status::check(abi::apxinf_model_vision_position_bf16(
            table.ptr(),
            embeddings.ptr(),
            position_ids.ptr(),
            source_side as i64,
            args.height as i64,
            args.width as i64,
            hidden as i64,
            args.merge_size as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `output[t, :] = table[ids[t], :]`
pub fn embedding_gather(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let table_dims = table.shape().dims().to_vec();
    let output_dims = output.shape().dims().to_vec();
    if table_dims.len() != 2 || output_dims.len() != 2 {
        return Err(invalid("embedding gather expects rank-2 tensors"));
    }
    let (vocab, hidden) = (table_dims[0], table_dims[1]);
    if output_dims[1] != hidden {
        return Err(invalid("embedding output width must match the table"));
    }
    let tokens = output_dims[0];
    let table_buffer = tensor_storage(ctx, table, DType::BF16, &table_dims)?;
    let id_buffer = tensor_storage(ctx, ids, DType::I32, &[tokens])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_model_embedding_gather(
            table_buffer.ptr(),
            id_buffer.ptr(),
            output_buffer.ptr(),
            tokens as i64,
            hidden as i64,
            vocab as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Index of the largest logit, reduced on device.
pub fn argmax(ctx: &CudaContext, logits: &Tensor, index: &Tensor) -> Result<()> {
    let dims = logits.shape().dims().to_vec();
    let count = dims.iter().product::<usize>();
    let logits_buffer = tensor_storage(ctx, logits, DType::BF16, &dims)?;
    let index_buffer = tensor_storage(ctx, index, DType::I32, &[1])?;
    unsafe {
        status::check(abi::apxinf_model_argmax_bf16(
            logits_buffer.ptr(),
            index_buffer.ptr(),
            count as i64,
            ctx.stream().handle(),
        ))
    }
}

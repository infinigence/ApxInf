use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::volume as abi;
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::{CudaBuffer, CudaContext};

/// A contiguous THWC input is logically preceded by `history` frames. Padding
/// applies to that concatenation. Output rows enumerate T, H, W; columns
/// enumerate kernel T, H, W, C. All padding values are zero.
#[derive(Clone, Copy, Debug)]
pub struct Unfold3dGeometry {
    pub kernel: [usize; 3],
    pub stride: [usize; 3],
    pub padding_before: [usize; 3],
    pub padding_after: [usize; 3],
}

pub struct Unfold3dArgs<'a> {
    pub input: &'a Tensor,
    pub history: Option<&'a Tensor>,
    pub geometry: Unfold3dGeometry,
    /// First flattened output spatial/temporal row in this tile.
    pub row_start: usize,
    /// `[tile_rows, kernel_t * kernel_h * kernel_w * channels]`.
    pub out: &'a mut Tensor,
}

pub(super) struct Normalized {
    pub spec: abi::Spec,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
}

fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| invalid("Unfold3d element count overflow"))
}

pub(super) fn normalize(ctx: &CudaContext, args: Unfold3dArgs<'_>) -> Result<Normalized> {
    let dims = args.input.shape().dims();
    if dims.len() != 4 || dims.contains(&0) || args.input.dtype() != DType::BF16 {
        return Err(invalid("Unfold3d requires nonempty BF16 THWC input"));
    }
    let input = tensor_storage(ctx, args.input, DType::BF16, dims)?;
    let history_frames = args
        .history
        .map_or(0, |h| h.shape().dims().first().copied().unwrap_or(0));
    let history = args
        .history
        .map(|h| {
            if history_frames == 0 {
                return Err(invalid("Unfold3d history must be nonempty when supplied"));
            }
            tensor_storage(
                ctx,
                h,
                DType::BF16,
                &[history_frames, dims[1], dims[2], dims[3]],
            )
        })
        .transpose()?;
    let geometry = args.geometry;
    let mut extent = [0usize; 3];
    for axis in 0..3 {
        if geometry.kernel[axis] == 0 || geometry.stride[axis] == 0 {
            return Err(invalid("Unfold3d kernel and stride must be positive"));
        }
        let input_axis = dims[axis].checked_add(if axis == 0 { history_frames } else { 0 });
        let padded = input_axis
            .and_then(|v| v.checked_add(geometry.padding_before[axis]))
            .and_then(|v| v.checked_add(geometry.padding_after[axis]))
            .ok_or_else(|| invalid("Unfold3d padded extent overflow"))?;
        if padded < geometry.kernel[axis] {
            return Err(invalid("Unfold3d kernel exceeds padded input"));
        }
        extent[axis] = (padded - geometry.kernel[axis]) / geometry.stride[axis] + 1;
    }
    let total_rows = product(&extent)?;
    let columns = product(&[
        geometry.kernel[0],
        geometry.kernel[1],
        geometry.kernel[2],
        dims[3],
    ])?;
    let output_dims = args.out.shape().dims();
    if output_dims.len() != 2
        || output_dims[0] == 0
        || output_dims[1] != columns
        || args
            .row_start
            .checked_add(output_dims[0])
            .is_none_or(|end| end > total_rows)
    {
        return Err(invalid(
            "Unfold3d output tile disagrees with convolution geometry",
        ));
    }
    product(output_dims)?;
    let output = tensor_storage(ctx, args.out, DType::BF16, output_dims)?;
    let overlaps = |a: &CudaBuffer, b: &CudaBuffer| {
        let a0 = a.ptr() as usize;
        let b0 = b.ptr() as usize;
        a0 < b0.saturating_add(b.len()) && b0 < a0.saturating_add(a.len())
    };
    if overlaps(&input, &output) || history.as_ref().is_some_and(|h| overlaps(h, &output)) {
        return Err(invalid("Unfold3d output must not alias input/history"));
    }
    let abi_dim =
        |v: usize| i64::try_from(v).map_err(|_| invalid("Unfold3d extent exceeds ABI range"));
    let triple = |a: [usize; 3]| -> Result<[i64; 3]> {
        Ok([abi_dim(a[0])?, abi_dim(a[1])?, abi_dim(a[2])?])
    };
    let spec = abi::Spec {
        version: 1,
        reserved: 0,
        input: [
            abi_dim(dims[0])?,
            abi_dim(dims[1])?,
            abi_dim(dims[2])?,
            abi_dim(dims[3])?,
        ],
        history_frames: abi_dim(history_frames)?,
        kernel: triple(geometry.kernel)?,
        stride: triple(geometry.stride)?,
        padding_before: triple(geometry.padding_before)?,
        padding_after: triple(geometry.padding_after)?,
        output: triple(extent)?,
        row_start: abi_dim(args.row_start)?,
        rows: abi_dim(output_dims[0])?,
    };
    let bindings = abi::Bindings {
        input: input.ptr().cast_const(),
        history: history
            .as_ref()
            .map_or(std::ptr::null(), |h| h.ptr().cast_const()),
        output: output.ptr(),
        stream: ctx.stream().handle(),
    };
    let mut storage = vec![input, output];
    storage.extend(history);
    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

//! Exact data movement for spatial/temporal convolutions and tensor layouts.
use super::{
    contracts::{invalid, Normalized},
    launch,
};
use crate::{ffi::abi::gather as abi, CudaBuffer, CudaContext};
use apxinf_core::{DType, Device, Result, Tensor};

#[cfg(test)]
pub(crate) const DOC_IDS: &[&str] = &["permute", "unfold3d", "resize_bicubic"];

fn size(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        if d == 0 {
            return Err(invalid("layout dimensions must be positive"));
        }
        n.checked_mul(d)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or_else(|| invalid("layout size overflow"))
    })
}
fn number(n: usize) -> Result<i64> {
    i64::try_from(n).map_err(|_| invalid("layout dimension overflow"))
}
fn storage(ctx: &CudaContext, x: &Tensor) -> Result<CudaBuffer> {
    let bytes = size(x.shape().dims())?
        .checked_mul(x.dtype().size_in_bytes())
        .ok_or_else(|| invalid("layout bytes overflow"))?;
    if x.device() != Device::Cuda(ctx.device_id()) {
        return Err(invalid("layout tensor device mismatch"));
    }
    let buffer = CudaBuffer::from_tensor(x).map_err(apxinf_core::Error::Cuda)?;
    if buffer.len() < bytes || buffer.ptr() as usize % x.dtype().size_in_bytes() != 0 {
        return Err(invalid("layout storage size/alignment mismatch"));
    }
    Ok(buffer)
}
fn normalized(
    ctx: &CudaContext,
    input: &Tensor,
    out: &Tensor,
    semantic: u32,
) -> Result<Normalized> {
    let dtype = match input.dtype() {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        _ => return Err(invalid("layout supports F32/F16/BF16")),
    };
    if input.dtype() != out.dtype() {
        return Err(invalid("layout dtype mismatch"));
    }
    let source = storage(ctx, input)?;
    let destination = storage(ctx, out)?;
    let a = source.ptr() as usize;
    let b = destination.ptr() as usize;
    let ae = a
        .checked_add(source.len())
        .ok_or_else(|| invalid("layout address overflow"))?;
    let be = b
        .checked_add(destination.len())
        .ok_or_else(|| invalid("layout address overflow"))?;
    if a < be && b < ae {
        return Err(invalid("layout output overlaps input"));
    }
    let alignment = |p: usize| (1usize << p.trailing_zeros().min(8)) as u32;
    Ok(Normalized {
        spec: abi::Spec {
            version: abi::SPEC_VERSION,
            semantic,
            dtype,
            input_alignment: alignment(a),
            output_alignment: alignment(b),
            bias_alignment: 256,
            ..Default::default()
        },
        bindings: abi::Bindings {
            input: source.ptr(),
            ids: std::ptr::null(),
            replacement: std::ptr::null(),
            row_map: std::ptr::null(),
            bias: std::ptr::null(),
            position: std::ptr::null(),
            output: destination.ptr(),
            stream: ctx.stream().handle(),
        },
        storage: vec![source, destination],
    })
}

/// Physically permute a contiguous tensor, preserving the element bits.
/// `axes[i]` names the input axis that becomes output axis i. Ranks 1–8,
/// matching F32/F16/BF16 dtypes; output must not overlap input.
pub fn permute(ctx: &CudaContext, input: &Tensor, axes: &[usize], out: &mut Tensor) -> Result<()> {
    let shape = input.shape().dims();
    if shape.is_empty() || shape.len() > 8 || axes.len() != shape.len() {
        return Err(invalid("permutation rank must be 1–8"));
    }
    let mut seen = vec![false; shape.len()];
    for &axis in axes {
        if axis >= shape.len() || seen[axis] {
            return Err(invalid("permutation axes must be unique and in range"));
        }
        seen[axis] = true;
    }
    let expected: Vec<_> = axes.iter().map(|&i| shape[i]).collect();
    if out.shape().dims() != expected {
        return Err(invalid("permutation output shape mismatch"));
    }
    let mut n = normalized(ctx, input, out, 4)?;
    n.spec.rows = number(size(shape)?)?;
    n.spec.cols = 1;
    n.spec.rank = shape.len() as u32;
    let mut stride = 1usize;
    let mut strides = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride = stride
            .checked_mul(shape[i])
            .ok_or_else(|| invalid("permutation stride overflow"))?;
    }
    for (i, &axis) in axes.iter().enumerate() {
        n.spec.layout_shape[i] = number(shape[axis])?;
        n.spec.layout_strides[i] = number(strides[axis])?;
    }
    launch::execute(ctx, n)
}

/// Bicubic resize of NHWC F32/F16/BF16 data. Fixed half-pixel coordinates
/// (align_corners=false), edge replication, cubic coefficient -0.75, no
/// antialiasing. Accumulates in FP32 and casts once to the matching output dtype.
pub fn resize_bicubic(ctx: &CudaContext, input: &Tensor, out: &mut Tensor) -> Result<()> {
    let a = input.shape().dims();
    let b = out.shape().dims();
    if a.len() != 4
        || b.len() != 4
        || a[0] != b[0]
        || a[3] != b[3]
        || a.iter().chain(b).any(|d| *d == 0 || *d > i32::MAX as usize)
    {
        return Err(invalid(
            "bicubic resize expects matching nonempty NHWC batch/channels",
        ));
    }
    let mut n = normalized(ctx, input, out, 6)?;
    n.spec.rows = number(size(&b[..3])?)?;
    n.spec.cols = number(b[3])?;
    n.spec.rank = 4;
    for i in 0..4 {
        n.spec.layout_shape[i] = number(a[i])?;
    }
    n.spec.output_spatial[0] = number(b[1])?;
    n.spec.output_spatial[1] = number(b[2])?;
    launch::execute(ctx, n)
}

/// Causal or symmetric zero-padded 3D im2col geometry. Axes are time, height,
/// width. The channel-last input is `[batch,time,height,width,channels]`.
#[derive(Clone, Copy, Debug)]
pub struct Unfold3dGeometry {
    pub kernel: [usize; 3],
    pub stride: [usize; 3],
    pub dilation: [usize; 3],
    pub padding_before: [usize; 3],
    pub padding_after: [usize; 3],
}
impl Unfold3dGeometry {
    pub fn output_spatial(&self, input: [usize; 3]) -> Result<[usize; 3]> {
        let mut result = [0; 3];
        for i in 0..3 {
            if input[i] == 0 || self.kernel[i] == 0 || self.stride[i] == 0 || self.dilation[i] == 0
            {
                return Err(invalid("unfold geometry must be positive"));
            }
            let padded = input[i]
                .checked_add(self.padding_before[i])
                .and_then(|v| v.checked_add(self.padding_after[i]))
                .ok_or_else(|| invalid("unfold padding overflow"))?;
            let span = (self.kernel[i] - 1)
                .checked_mul(self.dilation[i])
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| invalid("unfold kernel overflow"))?;
            if padded > i64::MAX as usize || span > i64::MAX as usize {
                return Err(invalid("unfold coordinate range exceeds i64"));
            }
            if padded < span {
                return Err(invalid("unfold kernel exceeds padded input"));
            }
            result[i] = (padded - span) / self.stride[i] + 1;
        }
        Ok(result)
    }
}

/// Extract a contiguous range of output rows. Rows flatten batch/time/height/
/// width; columns flatten kernel-time/kernel-height/kernel-width/channel.
/// Row tiling bounds workspace for large temporal VAE convolutions. Padding
/// produces positive zero; values inside the image are copied bit-for-bit.
/// The output width is `kernel_elements * padded_channels`, with
/// `padded_channels >= input_channels`; extra channels at each tap are zero.
pub fn unfold3d(
    ctx: &CudaContext,
    input: &Tensor,
    geometry: Unfold3dGeometry,
    row_start: usize,
    out: &mut Tensor,
) -> Result<()> {
    unfold3d_impl(ctx, input, geometry, row_start, 0, out)
}

/// Extract columns in `[channel_block,kernel_time,kernel_height,kernel_width,
/// channel_in_block]` order. The padded channel count inferred from the output
/// width must be a multiple of the positive `channel_block`. Padding, row
/// tiling, dtypes and ownership have the same contract as [`unfold3d`].
pub fn unfold3d_blocked(
    ctx: &CudaContext,
    input: &Tensor,
    geometry: Unfold3dGeometry,
    row_start: usize,
    channel_block: usize,
    out: &mut Tensor,
) -> Result<()> {
    if channel_block == 0 {
        return Err(invalid("unfold channel block must be positive"));
    }
    unfold3d_impl(ctx, input, geometry, row_start, channel_block, out)
}

fn unfold3d_impl(
    ctx: &CudaContext,
    input: &Tensor,
    geometry: Unfold3dGeometry,
    row_start: usize,
    channel_block: usize,
    out: &mut Tensor,
) -> Result<()> {
    let shape = input.shape().dims();
    let output = out.shape().dims();
    if shape.len() != 5 || output.len() != 2 {
        return Err(invalid("unfold expects NTHWC input and matrix output"));
    }
    let spatial = geometry.output_spatial([shape[1], shape[2], shape[3]])?;
    let total = size(&[shape[0], spatial[0], spatial[1], spatial[2]])?;
    let taps = size(&geometry.kernel)?;
    if output[1] % taps != 0
        || output[1] / taps < shape[4]
        || (channel_block != 0 && (output[1] / taps) % channel_block != 0)
        || output[0] == 0
        || row_start
            .checked_add(output[0])
            .is_none_or(|end| end > total)
    {
        return Err(invalid("unfold output shape/row range mismatch"));
    }
    let mut n = normalized(ctx, input, out, 5)?;
    n.spec.rows = number(output[0])?;
    n.spec.cols = number(output[1])?;
    n.spec.row_start = number(row_start)?;
    n.spec.channel_block = number(channel_block)?;
    for (i, &v) in shape.iter().enumerate() {
        n.spec.layout_shape[i] = number(v)?;
    }
    for i in 0..3 {
        n.spec.output_spatial[i] = number(spatial[i])?;
        n.spec.kernel[i] = number(geometry.kernel[i])?;
        n.spec.stride[i] = number(geometry.stride[i])?;
        n.spec.dilation[i] = number(geometry.dilation[i])?;
        n.spec.padding[i] = number(geometry.padding_before[i])?;
    }
    launch::execute(ctx, n)
}

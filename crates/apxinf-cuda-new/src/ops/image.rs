//! Contiguous HWC spatial building blocks. These fixed kernels enqueue on the
//! active stream, allocate nothing, and retain storage during graph capture.
use crate::ffi::abi::{image as abi, status};
use crate::{CudaBuffer, CudaContext};

#[cfg(test)]
pub(crate) const DOC_IDS: [&str; 6] = [
    "image_rgb_indices",
    "image_im2col",
    "image_space_to_depth",
    "image_grouped_average",
    "channel_l2_normalize",
    "channel_standardize",
];
use apxinf_core::{DType, Device, Error, Result, Tensor};

fn invalid(message: &str) -> Error {
    Error::Other(message.into())
}
fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .filter(|_| b > 0)
            .ok_or_else(|| invalid("image geometry is empty or overflows"))
    })
}
fn u32dim(n: usize) -> Result<u32> {
    u32::try_from(n)
        .ok()
        .filter(|&x| x <= i32::MAX as u32)
        .ok_or_else(|| invalid("image dimension exceeds i32"))
}
fn dtype(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(2),
        _ => Err(invalid("image primitives support BF16 and F32")),
    }
}
fn buffer(ctx: &CudaContext, t: &Tensor, dt: DType, elements: usize) -> Result<CudaBuffer> {
    if t.device() != Device::Cuda(ctx.device_id())
        || t.dtype() != dt
        || t.shape().numel() != elements
    {
        return Err(invalid(
            "image tensor device, dtype or element count mismatch",
        ));
    }
    let b = CudaBuffer::from_tensor(t).map_err(Error::Cuda)?;
    let bytes = elements
        .checked_mul(dt.size_in_bytes())
        .ok_or_else(|| invalid("image storage overflow"))?;
    if b.len() < bytes {
        return Err(invalid("image tensor storage too small"));
    }
    if b.ptr() as usize % dt.size_in_bytes() != 0 {
        return Err(invalid("image tensor storage is not dtype-aligned"));
    }
    Ok(b)
}

/// Packed source layout. The leading dimension indexes independent views.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageMemoryLayout {
    /// `[views, height, width, channels]`.
    Nhwc,
    /// `[views, channels, height, width]`.
    Nchw,
}

/// Widen unsigned image bytes to I32 while concatenating views along width.
/// Output is a flat `[height * views * width * channels]` tensor with logical
/// HWC ordering. Values stay in `[0,255]`; normalization is deliberately not
/// part of this operation. Both source and output addresses remain fixed for
/// capture, while source contents may be replaced before replay.
pub fn image_rgb_indices(
    ctx: &CudaContext,
    images: &CudaBuffer,
    out: &mut Tensor,
    views: usize,
    height: usize,
    width: usize,
    channels: usize,
    layout: ImageMemoryLayout,
) -> Result<()> {
    let count = product(&[views, height, width, channels])?;
    let out_bytes = count
        .checked_mul(4)
        .ok_or_else(|| invalid("image indices storage overflow"))?;
    if count > i64::MAX as usize || images.device() != ctx.device_id() || images.len() != count {
        return Err(invalid("packed image buffer device or size mismatch"));
    }
    if out.shape().dims() != [count] {
        return Err(invalid("image indices output must be a flat I32 tensor"));
    }
    let ob = buffer(ctx, out, DType::I32, count)?;
    let ip = images.ptr() as usize;
    let op = ob.ptr() as usize;
    if ip < op.saturating_add(out_bytes) && op < ip.saturating_add(count) {
        return Err(invalid("image indices output may not alias source"));
    }
    let spec = abi::RgbIndicesSpec {
        version: 1,
        layout: match layout {
            ImageMemoryLayout::Nhwc => 0,
            ImageMemoryLayout::Nchw => 1,
        },
        views: u32dim(views)?,
        height: u32dim(height)?,
        width: u32dim(width)?,
        channels: u32dim(channels)?,
    };
    let bindings = abi::RgbIndicesBindings {
        input: images.ptr().cast(),
        output: ob.ptr().cast(),
        stream: ctx.stream().handle() as _,
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_image_rgb_indices_launch(
            ctx.runtime(),
            &spec,
            &bindings,
        ))?;
    }
    crate::workspace::retain_resource(&std::rc::Rc::new((
        images.retained_for_execution(),
        ob.retained_for_execution(),
    )));
    Ok(())
}
fn geometry(input: &Tensor, height: usize, width: usize) -> Result<(usize, abi::Spec)> {
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[0] != product(&[height, width])? || dims[0] > i32::MAX as usize {
        return Err(invalid("image input must be [height*width, channels]"));
    }
    let channels = dims[1];
    Ok((
        channels,
        abi::Spec {
            version: 1,
            semantic: 0,
            dtype: dtype(input.dtype())?,
            output_dtype: dtype(input.dtype())?,
            height: u32dim(height)?,
            width: u32dim(width)?,
            channels: u32dim(channels)?,
            output_height: u32dim(height)?,
            output_width: u32dim(width)?,
            output_channels: u32dim(channels)?,
            kernel: 1,
            stride: 1,
            padding: 0,
            temporal_factor: 1,
        },
    ))
}
fn execute(
    ctx: &CudaContext,
    s: abi::Spec,
    input: &Tensor,
    p: Option<&Tensor>,
    p2: Option<&Tensor>,
    out: &mut Tensor,
    scale: f32,
    epsilon: f32,
) -> Result<()> {
    let count = product(&[s.height as usize, s.width as usize, s.channels as usize])?;
    let out_count = product(&[
        s.output_height as usize,
        s.output_width as usize,
        s.output_channels as usize,
    ])?;
    if out.shape().dims()
        != [
            s.output_height as usize * s.output_width as usize,
            s.output_channels as usize,
        ]
    {
        return Err(invalid("image output shape mismatch"));
    }
    let ib = buffer(ctx, input, input.dtype(), count)?;
    let ob = buffer(
        ctx,
        out,
        if s.semantic == 4 {
            DType::F32
        } else {
            input.dtype()
        },
        out_count,
    )?;
    let bytes_in = count * input.dtype().size_in_bytes();
    let bytes_out = out_count * out.dtype().size_in_bytes();
    let ip = ib.ptr() as usize;
    let op = ob.ptr() as usize;
    if ip < op.saturating_add(bytes_out) && op < ip.saturating_add(bytes_in) {
        return Err(invalid("image output may not alias input"));
    }
    let pd = if s.semantic == 4 {
        DType::F32
    } else {
        input.dtype()
    };
    let pc = if s.semantic == 4 {
        s.output_channels
    } else {
        s.channels
    } as usize;
    let pb = p.map(|t| buffer(ctx, t, pd, pc)).transpose()?;
    let p2b = p2.map(|t| buffer(ctx, t, DType::F32, pc)).transpose()?;
    for b in [&pb, &p2b].into_iter().flatten() {
        let ptr = b.ptr() as usize;
        if ptr < op.saturating_add(bytes_out) && op < ptr.saturating_add(b.len()) {
            return Err(invalid("image output may not alias parameters"));
        }
    }
    let bindings = abi::Bindings {
        input: ib.ptr().cast(),
        parameter: pb.as_ref().map_or(std::ptr::null(), |b| b.ptr().cast()),
        parameter2: p2b.as_ref().map_or(std::ptr::null(), |b| b.ptr().cast()),
        output: ob.ptr().cast(),
        scale,
        epsilon,
        stream: ctx.stream().handle() as _,
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_image_launch(ctx.runtime(), &s, &bindings))?;
    }
    // Keep the allocation alive without extending workspace tensor liveness.
    crate::workspace::retain_resource(&std::rc::Rc::new((
        ib.retained_for_execution(),
        ob.retained_for_execution(),
        pb.as_ref().map(CudaBuffer::retained_for_execution),
        p2b.as_ref().map(CudaBuffer::retained_for_execution),
    )));
    Ok(())
}

/// Extract zero-padded square convolution windows, ordered `[kh,kw,channel]`.
/// Input is HWC flattened to `[H*W,C]`, output `[OH*OW,K*K*C]`.
pub fn image_im2col(
    ctx: &CudaContext,
    input: &Tensor,
    out: &mut Tensor,
    height: usize,
    width: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_height: usize,
    output_width: usize,
) -> Result<()> {
    let (channels, mut s) = geometry(input, height, width)?;
    if kernel == 0 || stride == 0 || output_height == 0 || output_width == 0 {
        return Err(invalid("invalid im2col geometry"));
    }
    // Explicit output extent permits asymmetric right/bottom padding. At
    // least the final window must overlap the input (no all-padding rows).
    for (n, o) in [(height, output_height), (width, output_width)] {
        let last = (o - 1)
            .checked_mul(stride)
            .ok_or_else(|| invalid("im2col extent overflow"))?;
        if padding >= kernel
            || last
                >= n.checked_add(padding)
                    .ok_or_else(|| invalid("im2col extent overflow"))?
        {
            return Err(invalid("im2col window lies outside input"));
        }
    }
    s.kernel = u32dim(kernel)?;
    s.stride = u32dim(stride)?;
    s.padding = u32dim(padding)?;
    s.output_height = u32dim(output_height)?;
    s.output_width = u32dim(output_width)?;
    s.output_channels = u32dim(product(&[kernel, kernel, channels])?)?;
    execute(ctx, s, input, None, None, out, 1., 0.)
}

/// Pixel unshuffle with output channel order `[channel,dx,dy]` (width first).
pub fn image_space_to_depth(
    ctx: &CudaContext,
    input: &Tensor,
    out: &mut Tensor,
    height: usize,
    width: usize,
    factor: usize,
) -> Result<()> {
    let (channels, mut s) = geometry(input, height, width)?;
    if factor == 0 || height % factor != 0 || width % factor != 0 {
        return Err(invalid("space-to-depth requires divisible dimensions"));
    }
    s.semantic = 1;
    s.kernel = u32dim(factor)?;
    s.output_height = u32dim(height / factor)?;
    s.output_width = u32dim(width / factor)?;
    s.output_channels = u32dim(product(&[channels, factor, factor])?)?;
    execute(ctx, s, input, None, None, out, 1., 0.)
}

/// Average contiguous groups after rearranging a single frame with zero
/// temporal padding on the left to `[C,T,dy,dx,H/f,W/f]`.
/// This is a stateless, explicit single-frame operation, not a temporal cache.
pub fn image_grouped_average(
    ctx: &CudaContext,
    input: &Tensor,
    out: &mut Tensor,
    height: usize,
    width: usize,
    spatial_factor: usize,
    temporal_factor: usize,
    output_channels: usize,
) -> Result<()> {
    let (channels, mut s) = geometry(input, height, width)?;
    let total = product(&[channels, spatial_factor, spatial_factor, temporal_factor])?;
    if output_channels == 0
        || total % output_channels != 0
        || height % spatial_factor != 0
        || width % spatial_factor != 0
    {
        return Err(invalid("invalid grouped spatial average geometry"));
    }
    s.semantic = 2;
    s.kernel = u32dim(spatial_factor)?;
    s.temporal_factor = u32dim(temporal_factor)?;
    s.output_height = u32dim(height / spatial_factor)?;
    s.output_width = u32dim(width / spatial_factor)?;
    s.output_channels = u32dim(output_channels)?;
    execute(ctx, s, input, None, None, out, 1., 0.)
}

/// `round(round(round(x / max(||x||_2,eps)) * scale) * gamma)`.
/// Reduction and division are F32; each indicated round uses the input dtype.
pub fn channel_l2_normalize(
    ctx: &CudaContext,
    input: &Tensor,
    gamma: &Tensor,
    out: &mut Tensor,
    scale: f32,
    epsilon: f32,
) -> Result<()> {
    if !scale.is_finite() || !epsilon.is_finite() || epsilon <= 0. {
        return Err(invalid("invalid channel normalization scalars"));
    }
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(invalid("channel normalization requires a matrix"));
    }
    let (_, mut s) = geometry(input, dims[0], 1)?;
    s.semantic = 3;
    execute(ctx, s, input, Some(gamma), None, out, scale, epsilon)
}

/// Select the leading channels, convert to F32, and compute `(x-mean)/std`.
pub fn channel_standardize(
    ctx: &CudaContext,
    input: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let dims = input.shape().dims();
    if dims.len() != 2 || out.shape().dims().len() != 2 {
        return Err(invalid("channel standardization requires matrices"));
    }
    let (_, mut s) = geometry(input, dims[0], 1)?;
    s.semantic = 4;
    s.output_dtype = 0;
    let oc = out.shape().dims()[1];
    if oc == 0 || oc > dims[1] {
        return Err(invalid("invalid standardization channel count"));
    }
    s.output_channels = u32dim(oc)?;
    execute(ctx, s, input, Some(mean), Some(std), out, 1., 0.)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::Shape;

    fn upload(ctx: &CudaContext, shape: Vec<usize>, values: &[f32], dtype: DType) -> Tensor {
        let host = if dtype == DType::F32 {
            Tensor::from_f32(shape, values).unwrap()
        } else {
            Tensor::from_bf16(
                shape,
                &values
                    .iter()
                    .map(|&x| half::bf16::from_f32(x))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        crate::transfers::to_cuda(&host, ctx.device_id()).unwrap()
    }
    fn values(x: &Tensor) -> Vec<f32> {
        crate::transfers::to_cpu(x).unwrap().to_f32_vec().unwrap()
    }
    fn zero(ctx: &CudaContext, rows: usize, cols: usize, dtype: DType) -> Tensor {
        ctx.allocate_output(Shape::new(vec![rows, cols]), dtype)
            .unwrap()
    }
    fn rounded(v: f32, dtype: DType) -> f32 {
        if dtype == DType::BF16 {
            half::bf16::from_f32(v).to_f32()
        } else {
            v
        }
    }

    #[test]
    fn image_rgb_indices_layout_graph_and_binding_validation() {
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        for (views, h, w, c) in [(2, 224, 224, 3), (3, 5, 7, 4), (1, 1, 9, 1)] {
            let count = views * h * w * c;
            for layout in [ImageMemoryLayout::Nhwc, ImageMemoryLayout::Nchw] {
                let mut raw = vec![0u8; count];
                let mut expected = vec![0f32; count];
                // Build fixtures by explicit source coordinates, independently
                // of the device kernel's output-index decomposition.
                for view in 0..views {
                    for y in 0..h {
                        for x in 0..w {
                            for ch in 0..c {
                                let value = ((view * 71 + y * 37 + x * 19 + ch * 113) % 256) as u8;
                                let source = match layout {
                                    ImageMemoryLayout::Nhwc => ((view * h + y) * w + x) * c + ch,
                                    ImageMemoryLayout::Nchw => ((view * c + ch) * h + y) * w + x,
                                };
                                raw[source] = value;
                                expected[(y * views * w + view * w + x) * c + ch] =
                                    f32::from(value);
                            }
                        }
                    }
                }
                let images = CudaBuffer::alloc(count, ctx.device_id()).unwrap();
                images.copy_from_host(&raw).unwrap();
                let mut out = ctx
                    .allocate_output(Shape::new(vec![count]), DType::I32)
                    .unwrap();
                image_rgb_indices(&ctx, &images, &mut out, views, h, w, c, layout).unwrap();
                assert_eq!(values(&out), expected);
                let graph = crate::capture(&ctx, || {
                    image_rgb_indices(&ctx, &images, &mut out, views, h, w, c, layout)
                })
                .unwrap();
                for _ in 0..3 {
                    graph.replay().unwrap();
                }
                ctx.synchronize().unwrap();
                assert_eq!(values(&out), expected);
                images.copy_from_host(&vec![255; count]).unwrap();
                drop(images);
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert!(values(&out).iter().all(|&x| x == 255.));
                let short = CudaBuffer::alloc(count - 1, ctx.device_id()).unwrap();
                assert!(image_rgb_indices(&ctx, &short, &mut out, views, h, w, c, layout).is_err());
                let raw = CudaBuffer::alloc(count, ctx.device_id()).unwrap();
                let mut rank2 = ctx
                    .allocate_output(Shape::new(vec![h * views * w, c]), DType::I32)
                    .unwrap();
                assert!(image_rgb_indices(&ctx, &raw, &mut rank2, views, h, w, c, layout).is_err());
                let mut wrong_dtype = ctx
                    .allocate_output(Shape::new(vec![count]), DType::F32)
                    .unwrap();
                assert!(
                    image_rgb_indices(&ctx, &raw, &mut wrong_dtype, views, h, w, c, layout)
                        .is_err()
                );
                let storage = CudaBuffer::alloc(count * 4 + 1, ctx.device_id()).unwrap();
                let mut unaligned = storage
                    .view(1, count * 4)
                    .unwrap()
                    .as_tensor(Shape::new(vec![count]), DType::I32)
                    .unwrap();
                assert!(
                    image_rgb_indices(&ctx, &raw, &mut unaligned, views, h, w, c, layout).is_err()
                );
                let shared = storage.view(0, count).unwrap();
                let mut alias = storage
                    .view(0, count * 4)
                    .unwrap()
                    .as_tensor(Shape::new(vec![count]), DType::I32)
                    .unwrap();
                assert!(
                    image_rgb_indices(&ctx, &shared, &mut alias, views, h, w, c, layout).is_err()
                );
            }
        }
    }

    #[test]
    fn image_float_bindings_reject_unaligned_storage() {
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        for dtype in [DType::BF16, DType::F32] {
            let x = upload(&ctx, vec![1, 4], &[1.; 4], dtype);
            let gamma = upload(&ctx, vec![4], &[1.; 4], dtype);
            let mut out = zero(&ctx, 1, 4, dtype);
            let bytes = 4 * dtype.size_in_bytes();
            let allocation = CudaBuffer::alloc(bytes + 1, ctx.device_id()).unwrap();
            let unaligned = allocation
                .view(1, bytes)
                .unwrap()
                .as_tensor(Shape::new(vec![1, 4]), dtype)
                .unwrap();
            assert!(channel_l2_normalize(&ctx, &unaligned, &gamma, &mut out, 1., 1e-12).is_err());
            assert!(channel_l2_normalize(&ctx, &x, &unaligned, &mut out, 1., 1e-12).is_err());
            let mut unaligned_out = unaligned;
            assert!(channel_l2_normalize(&ctx, &x, &gamma, &mut unaligned_out, 1., 1e-12).is_err());
        }
    }

    #[test]
    fn image_layout_real_shapes_and_temporal_padding() {
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        for dtype in [DType::F32, DType::BF16] {
            for (h, w, c, k, stride, padding) in [(28, 56, 320, 3, 2, 0), (14, 28, 48, 2, 2, 0)] {
                let (oh, ow) = (h / stride, w / stride);
                let data = (0..h * w * c)
                    .map(|i| rounded((i % 31) as f32 / 16. - 1., dtype))
                    .collect::<Vec<_>>();
                let x = upload(&ctx, vec![h * w, c], &data, dtype);
                let mut col = zero(&ctx, oh * ow, k * k * c, dtype);
                image_im2col(&ctx, &x, &mut col, h, w, k, stride, padding, oh, ow).unwrap();
                let got = values(&col);
                for row in 0..oh * ow {
                    for ky in 0..k {
                        for kx in 0..k {
                            for channel in [0, c - 1] {
                                let y = row / ow * stride + ky;
                                let x = row % ow * stride + kx;
                                let expected = if y < h && x < w {
                                    data[(y * w + x) * c + channel]
                                } else {
                                    0.
                                };
                                assert_eq!(
                                    got[(row * k * k + ky * k + kx) * c + channel],
                                    expected
                                );
                            }
                        }
                    }
                }
            }
            let (h, w, c) = (224, 448, 3);
            let data = (0..h * w * c)
                .map(|i| rounded((i % 97) as f32 / 32. - 1., dtype))
                .collect::<Vec<_>>();
            let x = upload(&ctx, vec![h * w, c], &data, dtype);
            let mut packed = zero(&ctx, h * w / 4, c * 4, dtype);
            image_space_to_depth(&ctx, &x, &mut packed, h, w, 2).unwrap();
            let got = values(&packed);
            for row in 0..h * w / 4 {
                for ch in 0..12 {
                    let source =
                        ((row / (w / 2) * 2 + ch % 2) * w + row % (w / 2) * 2 + (ch / 2) % 2) * 3
                            + ch / 4;
                    assert_eq!(got[row * 12 + ch], data[source]);
                }
            }
            for (h, w, c, oc, ft) in [
                (112, 224, 160, 160, 1),
                (56, 112, 160, 320, 2),
                (28, 56, 320, 640, 2),
                (14, 28, 640, 640, 1),
            ] {
                let fs = if h == 14 { 1 } else { 2 };
                let data = (0..h * w * c)
                    .map(|i| rounded(((i * 7) % 101) as f32 / 64. - 0.5, dtype))
                    .collect::<Vec<_>>();
                let x = upload(&ctx, vec![h * w, c], &data, dtype);
                let mut out = zero(&ctx, h * w / (fs * fs), oc, dtype);
                image_grouped_average(&ctx, &x, &mut out, h, w, fs, ft, oc).unwrap();
                let got = values(&out);
                let group = c * ft * fs * fs / oc;
                for row in 0..h * w / (fs * fs) {
                    for channel in 0..oc {
                        let mut sum = 0.;
                        for group_idx in 0..group {
                            let idx = channel * group + group_idx;
                            let src_c = idx / (ft * fs * fs);
                            let t = (idx / (fs * fs)) % ft;
                            if t == ft - 1 {
                                let y = (row / (w / fs)) * fs + (idx / fs) % fs;
                                let x = (row % (w / fs)) * fs + idx % fs;
                                sum += data[(y * w + x) * c + src_c];
                            }
                        }
                        assert_eq!(got[row * oc + channel], rounded(sum / group as f32, dtype));
                    }
                }
            }
        }
    }

    #[test]
    fn image_and_pairwise_capture_preserve_workspace_reuse() {
        use crate::ops::{pairwise_rope, PairwiseRopeArgs, PairwiseRopeTable};
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        let x = upload(&ctx, vec![1, 4], &[1., 2., 3., 4.], DType::BF16);
        let table = PairwiseRopeTable::new(&ctx, 1, 4, &[(1., 0.), (1., 0.)]).unwrap();
        for use_rope in [false, true] {
            let mut addresses = Vec::new();
            let phase = crate::PreparedPhase::prepare_and_capture(
                &ctx,
                crate::ExecutionSession::with_capacity(512, ctx.device_id()).unwrap(),
                || {
                    for _ in 0..2 {
                        let mut out = ctx.allocate_output(x.shape().clone(), DType::BF16)?;
                        addresses.push(CudaBuffer::from_tensor(&out).unwrap().ptr() as usize);
                        if use_rope {
                            pairwise_rope(
                                &ctx,
                                PairwiseRopeArgs {
                                    input: &x,
                                    table: &table,
                                    output: &mut out,
                                    heads: 1,
                                },
                            )?;
                        } else {
                            image_im2col(&ctx, &x, &mut out, 1, 1, 1, 1, 0, 1, 1)?;
                        }
                        // The tensor lease ends here in BOTH traversals. Graph
                        // retention must keep the arena, not pin this slice.
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(addresses.len(), 4);
            assert!(addresses.iter().all(|&p| p == addresses[0]), "{addresses:?}");
            phase.replay().unwrap();
            ctx.synchronize().unwrap();
        }
    }

    #[test]
    fn image_im2col_graph_rebind_and_retained_storage() {
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        let (h, w, c) = (14, 28, 640);
        let data = (0..h * w * c)
            .map(|i| ((i % 17) as f32 - 8.) / 8.)
            .collect::<Vec<_>>();
        let x = upload(&ctx, vec![h * w, c], &data, DType::BF16);
        let mut out = zero(&ctx, h * w, 9 * c, DType::BF16);
        let input_buffer = CudaBuffer::from_tensor(&x).unwrap();
        let graph = crate::capture(&ctx, || {
            image_im2col(&ctx, &x, &mut out, h, w, 3, 1, 1, h, w)
        })
        .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        let got = values(&out);
        for row in 0..h * w {
            for ky in 0..3 {
                for kx in 0..3 {
                    for ch in [0, 159, 639] {
                        let y = row / w + ky;
                        let x = row % w + kx;
                        let expected = if y >= 1 && y <= h && x >= 1 && x <= w {
                            data[((y - 1) * w + x - 1) * c + ch]
                        } else {
                            0.
                        };
                        assert_eq!(got[(row * 9 + ky * 3 + kx) * c + ch], expected);
                    }
                }
            }
        }
        let changed = vec![half::bf16::from_f32(2.); h * w * c];
        let changed_bytes: Vec<u8> = changed
            .iter()
            .flat_map(|x| x.to_bits().to_ne_bytes())
            .collect();
        input_buffer.copy_from_host(&changed_bytes).unwrap();
        drop(x);
        drop(input_buffer);
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(values(&out)[4 * c], 2.);
    }

    #[test]
    fn image_channel_norm_and_standardize_precision_and_validation() {
        let ctx = CudaContext::new_with_autotune(0, false).unwrap();
        for dtype in [DType::F32, DType::BF16] {
            for c in [160, 320, 640] {
                let data = (0..3 * c)
                    .map(|i| {
                        if i < c {
                            0.
                        } else {
                            rounded(((i * 3) % 53) as f32 / 64. - 0.4, dtype)
                        }
                    })
                    .collect::<Vec<_>>();
                let gamma = (0..c)
                    .map(|i| rounded(0.75 + (i % 7) as f32 / 16., dtype))
                    .collect::<Vec<_>>();
                let x = upload(&ctx, vec![3, c], &data, dtype);
                let gamma_t = upload(&ctx, vec![c], &gamma, dtype);
                let mut out = zero(&ctx, 3, c, dtype);
                channel_l2_normalize(&ctx, &x, &gamma_t, &mut out, (c as f32).sqrt(), 1e-12)
                    .unwrap();
                let got = values(&out);
                for row in 0..3 {
                    let norm = data[row * c..(row + 1) * c]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f32>()
                        .sqrt()
                        .max(1e-12);
                    for ch in 0..c {
                        let expected = rounded(
                            rounded(
                                rounded(data[row * c + ch] / norm, dtype) * (c as f32).sqrt(),
                                dtype,
                            ) * gamma[ch],
                            dtype,
                        );
                        assert!(
                            (got[row * c + ch] - expected).abs()
                                <= if dtype == DType::F32 { 2e-6 } else { 0. },
                            "{dtype:?} c={c} ch={ch}: {} vs {expected}",
                            got[row * c + ch]
                        );
                    }
                }
                let mean = upload(&ctx, vec![48], &vec![0.25; 48], DType::F32);
                let std = upload(&ctx, vec![48], &vec![0.75; 48], DType::F32);
                let mut standardized = zero(&ctx, 3, 48, DType::F32);
                channel_standardize(&ctx, &x, &mean, &std, &mut standardized).unwrap();
                let got = values(&standardized);
                for row in 0..3 {
                    for ch in 0..48 {
                        assert!(
                            (got[row * 48 + ch] - (data[row * c + ch] - 0.25) / 0.75).abs() < 1e-6
                        );
                    }
                }
                let mut alias = x.clone();
                assert!(channel_l2_normalize(&ctx, &x, &gamma_t, &mut alias, 1., 1e-12).is_err());
                assert!(channel_l2_normalize(&ctx, &x, &gamma_t, &mut out, 1., 0.).is_err());
            }
        }
    }
}

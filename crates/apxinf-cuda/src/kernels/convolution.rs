//! Model-neutral channels-first BF16 convolutions through the cuDNN v9 provider.
use super::contracts::{checked_bytes, gpu_ptr};
use crate::context::CudaContext;
use crate::cudnn::{api, check, Descriptor};
use crate::workspace::output_buffer;
use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct ConvSpec<const D: usize> {
    pub stride: [usize; D],
    pub padding: [usize; D],
    pub dilation: [usize; D],
    pub groups: usize,
}
pub type Conv2dSpec = ConvSpec<2>;
pub type Conv3dSpec = ConvSpec<3>;

impl<const D: usize> Default for ConvSpec<D> {
    fn default() -> Self {
        Self {
            stride: [1; D],
            padding: [0; D],
            dilation: [1; D],
            groups: 1,
        }
    }
}

/// NCHW input, [out_channels,in_channels/groups,kH,kW] filter, optional channel bias.
pub fn conv2d(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
) -> Result<Tensor> {
    convolve(ctx, input, weight, bias, spec, false)
}

/// NCDHW input and [out,in/groups,kD,kH,kW] filter, optional channel bias.
pub fn conv3d(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv3dSpec,
) -> Result<Tensor> {
    convolve(ctx, input, weight, bias, spec, false)
}

/// Transposed convolution with output_padding=0. Filter layout is [in,out/groups,kH,kW].
pub fn conv_transpose2d(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
) -> Result<Tensor> {
    convolve(ctx, input, weight, bias, spec, true)
}

fn shape_error() -> Error {
    Error::Other("invalid or overflowing BF16 convolution geometry".into())
}
fn integer(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| shape_error())
}

fn convolve<const D: usize>(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    spec: ConvSpec<D>,
    transpose: bool,
) -> Result<Tensor> {
    let x = input.shape().dims();
    let w = weight.shape().dims();
    if x.len() != D + 2
        || w.len() != D + 2
        || spec.groups == 0
        || spec.stride.contains(&0)
        || spec.dilation.contains(&0)
    {
        return Err(shape_error());
    }
    for t in [Some(input), Some(weight), bias].into_iter().flatten() {
        if t.dtype() != DType::BF16 || t.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "convolution requires BF16 tensors on the context device".into(),
            ));
        }
        checked_bytes(t.dtype(), t.shape().dims(), "convolution")?;
    }
    if ctx.caps().compute_major < 8 {
        return Err(Error::Other(
            "BF16 cuDNN convolution requires SM80 or newer".into(),
        ));
    }
    let output_channels = if transpose {
        w[1].checked_mul(spec.groups).ok_or_else(shape_error)?
    } else {
        w[0]
    };
    if x[1] % spec.groups != 0
        || output_channels % spec.groups != 0
        || (if transpose {
            w[0] != x[1]
        } else {
            w[1] != x[1] / spec.groups
        })
        || bias.is_some_and(|b| b.shape().dims() != [output_channels])
    {
        return Err(shape_error());
    }
    let mut y = vec![0; D + 2];
    y[0] = x[0];
    y[1] = output_channels;
    for axis in 0..D {
        let effective = spec.dilation[axis]
            .checked_mul(w[axis + 2] - 1)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(shape_error)?;
        let pad = spec.padding[axis].checked_mul(2).ok_or_else(shape_error)?;
        y[axis + 2] = if transpose {
            (x[axis + 2] - 1)
                .checked_mul(spec.stride[axis])
                .and_then(|v| v.checked_add(effective))
                .and_then(|v| v.checked_sub(pad))
                .ok_or_else(shape_error)?
        } else {
            x[axis + 2]
                .checked_add(pad)
                .and_then(|v| v.checked_sub(effective))
                .map(|v| v / spec.stride[axis] + 1)
                .ok_or_else(shape_error)?
        };
    }
    let bytes = checked_bytes(DType::BF16, &y, "convolution output")?;
    let xi = x.iter().map(|&v| integer(v)).collect::<Result<Vec<_>>>()?;
    let wi = w.iter().map(|&v| integer(v)).collect::<Result<Vec<_>>>()?;
    let yi = y.iter().map(|&v| integer(v)).collect::<Result<Vec<_>>>()?;
    // Descriptor/handle creation is not capture-safe. Until a prepared-plan
    // consumer exists, fail explicitly rather than creating resources in capture.
    if !crate::workspace::may_prepare_native_resources() {
        return Err(Error::Other(
            "cuDNN convolution needs prepared descriptors for CUDA Graph capture".into(),
        ));
    }
    let api = api().map_err(Error::Cuda)?;
    let handle = Descriptor::new(api, api.create, api.destroy).map_err(Error::Cuda)?;
    let xd = Descriptor::new(api, api.create_tensor, api.destroy_tensor).map_err(Error::Cuda)?;
    let yd = Descriptor::new(api, api.create_tensor, api.destroy_tensor).map_err(Error::Cuda)?;
    let wd = Descriptor::new(api, api.create_filter, api.destroy_filter).map_err(Error::Cuda)?;
    let cd = Descriptor::new(api, api.create_conv, api.destroy_conv).map_err(Error::Cuda)?;
    let output = output_buffer(ctx, bytes)?;
    unsafe {
        check(api, (api.set_stream)(handle.raw, ctx.stream().handle())).map_err(Error::Cuda)?;
        if D == 2 {
            check(
                api,
                (api.set_tensor)(xd.raw, 0, 9, xi[0], xi[1], xi[2], xi[3]),
            )
            .map_err(Error::Cuda)?;
            check(
                api,
                (api.set_tensor)(yd.raw, 0, 9, yi[0], yi[1], yi[2], yi[3]),
            )
            .map_err(Error::Cuda)?;
            check(
                api,
                (api.set_filter)(wd.raw, 9, 0, wi[0], wi[1], wi[2], wi[3]),
            )
            .map_err(Error::Cuda)?;
            check(
                api,
                (api.set_conv)(
                    cd.raw,
                    integer(spec.padding[0])?,
                    integer(spec.padding[1])?,
                    integer(spec.stride[0])?,
                    integer(spec.stride[1])?,
                    integer(spec.dilation[0])?,
                    integer(spec.dilation[1])?,
                    1,
                    0,
                ),
            )
            .map_err(Error::Cuda)?;
        } else {
            let xs = contiguous_strides(&xi)?;
            let ys = contiguous_strides(&yi)?;
            check(
                api,
                (api.set_tensor_nd)(xd.raw, 9, integer(D + 2)?, xi.as_ptr(), xs.as_ptr()),
            )
            .map_err(Error::Cuda)?;
            check(
                api,
                (api.set_tensor_nd)(yd.raw, 9, integer(D + 2)?, yi.as_ptr(), ys.as_ptr()),
            )
            .map_err(Error::Cuda)?;
            check(
                api,
                (api.set_filter_nd)(wd.raw, 9, 0, integer(D + 2)?, wi.as_ptr()),
            )
            .map_err(Error::Cuda)?;
            let pad = spec
                .padding
                .iter()
                .map(|&v| integer(v))
                .collect::<Result<Vec<_>>>()?;
            let stride = spec
                .stride
                .iter()
                .map(|&v| integer(v))
                .collect::<Result<Vec<_>>>()?;
            let dilation = spec
                .dilation
                .iter()
                .map(|&v| integer(v))
                .collect::<Result<Vec<_>>>()?;
            check(
                api,
                (api.set_conv_nd)(
                    cd.raw,
                    integer(D)?,
                    pad.as_ptr(),
                    stride.as_ptr(),
                    dilation.as_ptr(),
                    1,
                    0,
                ),
            )
            .map_err(Error::Cuda)?;
        }
        check(api, (api.set_math)(cd.raw, 1)).map_err(Error::Cuda)?;
        check(api, (api.set_groups)(cd.raw, integer(spec.groups)?)).map_err(Error::Cuda)?;
        let algorithm = 1; // forward implicit-precomp GEMM / backward-data algorithm 1.
        let mut workspace_bytes = 0;
        let status = if transpose {
            (api.backward_workspace)(
                handle.raw,
                wd.raw,
                xd.raw,
                cd.raw,
                yd.raw,
                algorithm,
                &mut workspace_bytes,
            )
        } else {
            (api.forward_workspace)(
                handle.raw,
                xd.raw,
                wd.raw,
                cd.raw,
                yd.raw,
                algorithm,
                &mut workspace_bytes,
            )
        };
        check(api, status).map_err(Error::Cuda)?;
        let workspace = output_buffer(ctx, workspace_bytes.max(1))?;
        let one = 1.0f32;
        let zero = 0.0f32;
        let alpha = (&one as *const f32).cast();
        let beta = (&zero as *const f32).cast();
        let status = if transpose {
            (api.backward)(
                handle.raw,
                alpha,
                wd.raw,
                gpu_ptr(weight)?,
                xd.raw,
                gpu_ptr(input)?,
                cd.raw,
                algorithm,
                workspace.ptr(),
                workspace_bytes,
                beta,
                yd.raw,
                output.ptr(),
            )
        } else {
            (api.forward)(
                handle.raw,
                alpha,
                xd.raw,
                gpu_ptr(input)?,
                wd.raw,
                gpu_ptr(weight)?,
                cd.raw,
                algorithm,
                workspace.ptr(),
                workspace_bytes,
                beta,
                yd.raw,
                output.ptr(),
            )
        };
        check(api, status).map_err(Error::Cuda)?;
        if let Some(bias) = bias {
            let bd =
                Descriptor::new(api, api.create_tensor, api.destroy_tensor).map_err(Error::Cuda)?;
            if D == 2 {
                check(api, (api.set_tensor)(bd.raw, 0, 9, 1, yi[1], 1, 1)).map_err(Error::Cuda)?;
            } else {
                let mut dims = vec![1; D + 2];
                dims[1] = yi[1];
                let strides = contiguous_strides(&dims)?;
                check(
                    api,
                    (api.set_tensor_nd)(
                        bd.raw,
                        9,
                        integer(D + 2)?,
                        dims.as_ptr(),
                        strides.as_ptr(),
                    ),
                )
                .map_err(Error::Cuda)?;
            }
            check(
                api,
                (api.add)(
                    handle.raw,
                    alpha,
                    bd.raw,
                    gpu_ptr(bias)?,
                    alpha,
                    yd.raw,
                    output.ptr(),
                ),
            )
            .map_err(Error::Cuda)?;
        }
    }
    Ok(output.into_tensor(Shape::new(y), DType::BF16))
}

fn contiguous_strides(dims: &[i32]) -> Result<Vec<i32>> {
    let mut strides = vec![1i32; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1])
            .ok_or_else(shape_error)?;
    }
    Ok(strides)
}

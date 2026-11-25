//! FP32 one-dimensional convolution family for waveform decoders: [`conv1d`], [`conv_transpose1d`] and [`snake_beta`].
//!
//! Tensors are contiguous `[batch, channels, time]` FP32 on the context's device. Accumulation is FP32 in input-channel-then-tap order with
//! exact FP32 arithmetic (no TF32). These are direct kernels, not GEMM lowerings: the shapes of a vocoder (a handful of channels at long
//! signals, or thousands of channels at a few frames) do not suit one tile shape, and the whole decoder is a few GFLOP.
use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::conv1d as abi;
use crate::ops::validation::check_cuda;
use crate::{CudaBuffer, CudaContext};

#[cfg(test)]
pub(crate) const DOC_IDS: &[&str] = &["Conv1d", "ConvTranspose1d", "SnakeBeta"];

/// How a convolution reads outside the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv1dPadding {
    /// `before` zero columns on the left; the output length fixes the right side.
    Zeros { before: usize },
    /// `before` copies of the first column on the left and copies of the last column on the right (`F.pad(mode="replicate")`).
    Replicate { before: usize },
}

/// `out[b, co, t] = bias[co] + sum_{ci, k} weight[co, ci, k] * x[b, ci, t * stride + k * dilation - before]`.
///
/// `weight` is `[out_channels, in_channels, kernel]`, or `[channels, 1, kernel]` with `depthwise` (one filter per channel). `out` is
/// `[batch, out_channels, out_len]`; its length is the caller's (the operator reads `out_len` columns, padding as described).
pub struct Conv1dArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub out: &'a Tensor,
    pub stride: usize,
    pub dilation: usize,
    pub padding: Conv1dPadding,
    pub depthwise: bool,
}

/// `out[b, co, j]` of the transposed convolution of the replicate-padded input, cropped on the left and scaled:
/// `scale * sum_{ci, k} x_padded[b, ci, p] * weight[ci, co, k]` over the taps with `p * stride + k = j + crop_left`, plus `bias[co]`.
///
/// `weight` is `[in_channels, out_channels / groups, kernel]` (`F.conv_transpose1d` layout). The input is padded with `replicate_pad`
/// copies of its edge columns on both sides first. `groups` is 1 or `in_channels` (depthwise, `out_channels == in_channels`).
pub struct ConvTranspose1dArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub out: &'a Tensor,
    pub stride: usize,
    pub replicate_pad: usize,
    pub crop_left: usize,
    pub groups: usize,
    pub scale: f32,
}

fn f32_buffer(
    ctx: &CudaContext,
    tensor: &Tensor,
    what: &str,
    elements: usize,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != DType::F32
        || tensor.numel() != elements
    {
        return Err(Error::Other(format!(
            "{what} must be an FP32 tensor of {elements} elements on CUDA{}, got {} {:?} elements on {:?}",
            ctx.device_id(),
            tensor.dtype(),
            tensor.numel(),
            tensor.device()
        )));
    }
    CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)
}

fn int(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Other(format!("{what} {value} exceeds i32::MAX")))
}

fn dims3(tensor: &Tensor, what: &str) -> Result<[usize; 3]> {
    match tensor.shape().dims() {
        [a, b, c] if *a > 0 && *b > 0 && *c > 0 => Ok([*a, *b, *c]),
        other => Err(Error::Other(format!(
            "{what} must be a non-empty [batch, channels, time] tensor, got {other:?}"
        ))),
    }
}

/// See [`Conv1dArgs`].
pub fn conv1d(ctx: &CudaContext, args: Conv1dArgs<'_>) -> Result<()> {
    let [batch, in_channels, in_len] = dims3(args.input, "conv1d input")?;
    let [out_batch, out_channels, out_len] = dims3(args.out, "conv1d output")?;
    let kernel = *args
        .weight
        .shape()
        .dims()
        .last()
        .ok_or_else(|| Error::Other("conv1d weight has no dimensions".into()))?;
    let (expected_weight, groups) = if args.depthwise {
        if out_channels != in_channels {
            return Err(Error::Other(format!(
                "depthwise conv1d needs equal channels, got {in_channels} -> {out_channels}"
            )));
        }
        ([out_channels, 1, kernel], in_channels)
    } else {
        ([out_channels, in_channels, kernel], 1)
    };
    if out_batch != batch
        || args.weight.shape().dims() != expected_weight
        || kernel == 0
        || args.stride == 0
        || args.dilation == 0
    {
        return Err(Error::Other(format!(
            "conv1d shapes disagree: input {:?}, weight {:?} (expected {expected_weight:?}), output {:?}, stride {}, dilation {}",
            args.input.shape().dims(),
            args.weight.shape().dims(),
            args.out.shape().dims(),
            args.stride,
            args.dilation
        )));
    }
    let (before, replicate) = match args.padding {
        Conv1dPadding::Zeros { before } => (before, 0),
        Conv1dPadding::Replicate { before } => (before, 1),
    };
    let x = f32_buffer(
        ctx,
        args.input,
        "conv1d input",
        batch * in_channels * in_len,
    )?;
    let w = f32_buffer(
        ctx,
        args.weight,
        "conv1d weight",
        expected_weight.iter().product(),
    )?;
    let o = f32_buffer(
        ctx,
        args.out,
        "conv1d output",
        batch * out_channels * out_len,
    )?;
    let bias = match args.bias {
        Some(bias) => Some(f32_buffer(ctx, bias, "conv1d bias", out_channels)?),
        None => None,
    };
    let geometry = abi::Geometry {
        batch: int(batch, "batch")?,
        in_channels: int(in_channels, "input channels")?,
        out_channels: int(out_channels, "output channels")?,
        in_len: int(in_len, "input length")?,
        out_len: int(out_len, "output length")?,
        kernel: int(kernel, "kernel")?,
        stride: int(args.stride, "stride")?,
        dilation: int(args.dilation, "dilation")?,
        pad_left: int(before, "padding")?,
        crop_left: 0,
        groups: int(groups, "groups")?,
        replicate,
        scale: 1.0,
    };
    unsafe {
        check_cuda(abi::apxinf_conv1d_f32(
            x.ptr().cast(),
            w.ptr().cast(),
            bias.as_ref().map_or(std::ptr::null(), |b| b.ptr().cast()),
            o.ptr().cast(),
            &geometry,
            ctx.stream().handle(),
        ))
    }
}

/// See [`ConvTranspose1dArgs`].
pub fn conv_transpose1d(ctx: &CudaContext, args: ConvTranspose1dArgs<'_>) -> Result<()> {
    let [batch, in_channels, in_len] = dims3(args.input, "conv_transpose1d input")?;
    let [out_batch, out_channels, out_len] = dims3(args.out, "conv_transpose1d output")?;
    let kernel = *args
        .weight
        .shape()
        .dims()
        .last()
        .ok_or_else(|| Error::Other("conv_transpose1d weight has no dimensions".into()))?;
    if args.groups != 1 && !(args.groups == in_channels && in_channels == out_channels) {
        return Err(Error::Other(format!(
            "conv_transpose1d groups must be 1 or the channel count (depthwise), got {}",
            args.groups
        )));
    }
    let expected_weight = [in_channels, out_channels / args.groups, kernel];
    if out_batch != batch
        || args.weight.shape().dims() != expected_weight
        || kernel == 0
        || args.stride == 0
        || !args.scale.is_finite()
    {
        return Err(Error::Other(format!(
            "conv_transpose1d shapes disagree: input {:?}, weight {:?} (expected {expected_weight:?}), output {:?}, stride {}",
            args.input.shape().dims(),
            args.weight.shape().dims(),
            args.out.shape().dims(),
            args.stride
        )));
    }
    let full = (in_len + 2 * args.replicate_pad - 1) * args.stride + kernel;
    if args.crop_left + out_len > full {
        return Err(Error::Other(format!("conv_transpose1d output of {out_len} columns after cropping {} exceeds the {full} the transposed convolution produces", args.crop_left)));
    }
    let x = f32_buffer(
        ctx,
        args.input,
        "conv_transpose1d input",
        batch * in_channels * in_len,
    )?;
    let w = f32_buffer(
        ctx,
        args.weight,
        "conv_transpose1d weight",
        expected_weight.iter().product(),
    )?;
    let o = f32_buffer(
        ctx,
        args.out,
        "conv_transpose1d output",
        batch * out_channels * out_len,
    )?;
    let bias = match args.bias {
        Some(bias) => Some(f32_buffer(
            ctx,
            bias,
            "conv_transpose1d bias",
            out_channels,
        )?),
        None => None,
    };
    let geometry = abi::Geometry {
        batch: int(batch, "batch")?,
        in_channels: int(in_channels, "input channels")?,
        out_channels: int(out_channels, "output channels")?,
        in_len: int(in_len, "input length")?,
        out_len: int(out_len, "output length")?,
        kernel: int(kernel, "kernel")?,
        stride: int(args.stride, "stride")?,
        dilation: 1,
        pad_left: int(args.replicate_pad, "replicate padding")?,
        crop_left: int(args.crop_left, "crop")?,
        groups: int(args.groups, "groups")?,
        replicate: 1,
        scale: args.scale,
    };
    unsafe {
        check_cuda(abi::apxinf_conv_transpose1d_f32(
            x.ptr().cast(),
            w.ptr().cast(),
            bias.as_ref().map_or(std::ptr::null(), |b| b.ptr().cast()),
            o.ptr().cast(),
            &geometry,
            ctx.stream().handle(),
        ))
    }
}

/// `out = x + sin(alpha[c] * x)^2 / (beta[c] + 1e-9)` per channel of `[batch, channels, time]`. `alpha` and `beta` are the exponentiated
/// per-channel parameters (`[channels]`). `out` may be `input` itself.
pub struct SnakeBetaArgs<'a> {
    pub input: &'a Tensor,
    pub alpha: &'a Tensor,
    pub beta: &'a Tensor,
    pub out: &'a Tensor,
}

/// See [`SnakeBetaArgs`].
pub fn snake_beta(ctx: &CudaContext, args: SnakeBetaArgs<'_>) -> Result<()> {
    let [batch, channels, len] = dims3(args.input, "snake_beta input")?;
    if args.out.shape().dims() != args.input.shape().dims() {
        return Err(Error::Other(
            "snake_beta output must have the input's shape".into(),
        ));
    }
    let total = batch * channels * len;
    let x = f32_buffer(ctx, args.input, "snake_beta input", total)?;
    let alpha = f32_buffer(ctx, args.alpha, "snake_beta alpha", channels)?;
    let beta = f32_buffer(ctx, args.beta, "snake_beta beta", channels)?;
    let o = f32_buffer(ctx, args.out, "snake_beta output", total)?;
    unsafe {
        check_cuda(abi::apxinf_snake_beta_f32(
            x.ptr().cast(),
            alpha.ptr().cast(),
            beta.ptr().cast(),
            o.ptr().cast(),
            int(channels, "channels")?,
            int(len, "length")?,
            i64::try_from(total)
                .map_err(|_| Error::Other("snake_beta element count overflow".into()))?,
            ctx.stream().handle(),
        ))
    }
}

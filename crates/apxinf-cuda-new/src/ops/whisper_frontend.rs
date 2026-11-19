//! Log-mel frontend nonlinear stages (framing, power and log-mel transforms).
//!
//! The DFT projections and the mel filterbank run through the shared GEMM
//! operator in F32; these launches cover the reflect-padded framing pass,
//! `power = re^2 + im^2`, and the log-mel transform
//! `clamp 1e-10 -> log10 -> floor(global_max - 8) -> (x + 4) / 4 -> f16`,
//! with the global max reduced on device.

use apxinf_core::{Device, Error, Result, Tensor};

use crate::ffi::abi::status;
use crate::ffi::abi::whisper_frontend as abi;
use crate::{transfers, CudaContext as Context};

fn tensor_ptr(tensor: &Tensor, ctx: &Context, label: &str) -> Result<*mut std::ffi::c_void> {
    if tensor.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(format!(
            "log-mel frontend {label} tensor is not on device {}",
            ctx.device_id()
        )));
    }
    let buffer = crate::CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    Ok(buffer.ptr() as *mut std::ffi::c_void)
}

/// Reflect-pad `waveform` by `pad`, gather `frames` rows of `n_fft` at
/// `hop` stride, and multiply by the `hann` window, all on device.
/// `frames_out` is `[frames, n_fft]` F32.
pub fn frames(
    ctx: &Context,
    waveform: &Tensor,
    hann: &Tensor,
    frames_out: &mut Tensor,
    n_fft: usize,
    hop: usize,
    pad: usize,
) -> Result<()> {
    let n_samples = waveform.shape().numel();
    let frames = frames_out.shape().dims().first().copied().unwrap_or(0);
    if frames_out.shape().numel() != frames * n_fft {
        return Err(Error::Other(format!(
            "frames output has {} elements, expected {}",
            frames_out.shape().numel(),
            frames * n_fft
        )));
    }
    for (tensor, label) in [(waveform, "waveform"), (hann, "hann"), (frames_out, "frames")] {
        if tensor.dtype() != apxinf_core::DType::F32 {
            return Err(Error::Other(format!(
                "frames {label} must be F32, got {:?}",
                tensor.dtype()
            )));
        }
    }
    if hann.shape().numel() != n_fft {
        return Err(Error::Other(format!(
            "frames hann has {} elements, expected n_fft={n_fft}",
            hann.shape().numel()
        )));
    }
    let bindings = abi::FramesBindings {
        waveform: tensor_ptr(waveform, ctx, "waveform")?,
        hann: tensor_ptr(hann, ctx, "hann")?,
        frames: tensor_ptr(frames_out, ctx, "frames")?,
        stream: ctx.stream().handle(),
    };
    unsafe {
        status::check(abi::apxinf_whisper_frames_launch(
            ctx.runtime(),
            n_samples as i64,
            n_fft as i64,
            hop as i64,
            pad as i64,
            frames as i64,
            &bindings,
        ))?;
    }
    Ok(())
}

/// `power[i] = re[i]^2 + im[i]^2` over `count` F32 elements.
pub fn power(ctx: &Context, re: &Tensor, im: &Tensor, power: &mut Tensor) -> Result<()> {
    let count = re.shape().numel();
    if im.shape().numel() != count || power.shape().numel() != count {
        return Err(Error::Other(format!(
            "power element count mismatch: re={} im={} out={}",
            count,
            im.shape().numel(),
            power.shape().numel()
        )));
    }
    for (tensor, label) in [(re, "re"), (im, "im"), (power, "power")] {
        if tensor.dtype() != apxinf_core::DType::F32 {
            return Err(Error::Other(format!(
                "power {label} must be F32, got {:?}",
                tensor.dtype()
            )));
        }
    }
    let bindings = abi::PowerBindings {
        re: tensor_ptr(re, ctx, "re")?,
        im: tensor_ptr(im, ctx, "im")?,
        power: tensor_ptr(power, ctx, "power")?,
        stream: ctx.stream().handle(),
    };
    unsafe {
        status::check(abi::apxinf_whisper_power_launch(
            ctx.runtime(),
            count as i64,
            &bindings,
        ))?;
    }
    Ok(())
}

/// Log-mel transform over an F32 `spec` `[rows, cols]` into an F16 output:
/// `out = (max(log10(clamp(spec, 1e-10)), global_max - 8) + 4) / 4`.
///
/// `max_scratch` is a device F32 tensor with at least `blocks + 1` elements
/// where `blocks = ceil(rows * cols / 256)` capped at 4096; the caller may
/// reuse one scratch allocation across calls.
pub fn log_mel(
    ctx: &Context,
    spec: &Tensor,
    out_f16: &mut Tensor,
    max_scratch: &mut Tensor,
) -> Result<()> {
    let dims = spec.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "log-mel expects a rank-2 spec, got {dims:?}"
        )));
    }
    let count = dims[0] * dims[1];
    if out_f16.shape().numel() != count {
        return Err(Error::Other(format!(
            "log-mel output has {} elements, expected {count}",
            out_f16.shape().numel()
        )));
    }
    if spec.dtype() != apxinf_core::DType::F32 || out_f16.dtype() != apxinf_core::DType::F16 {
        return Err(Error::Other(format!(
            "log-mel dtypes: spec {:?} (want F32), out {:?} (want F16)",
            spec.dtype(),
            out_f16.dtype()
        )));
    }
    let blocks = count.div_ceil(256).min(4096).max(1);
    if max_scratch.shape().numel() < blocks + 1 {
        return Err(Error::Other(format!(
            "log-mel scratch has {} elements, needs {}",
            max_scratch.shape().numel(),
            blocks + 1
        )));
    }
    if max_scratch.dtype() != apxinf_core::DType::F32 {
        return Err(Error::Other(format!(
            "log-mel scratch must be F32, got {:?}",
            max_scratch.dtype()
        )));
    }
    let bindings = abi::LogMelBindings {
        spec: tensor_ptr(spec, ctx, "spec")?,
        out_f16: tensor_ptr(out_f16, ctx, "out_f16")?,
        max_scratch: tensor_ptr(max_scratch, ctx, "max_scratch")?,
        stream: ctx.stream().handle(),
    };
    unsafe {
        status::check(abi::apxinf_whisper_log_mel_launch(
            ctx.runtime(),
            dims[0] as i64,
            dims[1] as i64,
            &bindings,
        ))?;
    }
    Ok(())
}

/// Upload a host F32 tensor to the context device.
pub fn to_device(ctx: &Context, shape: &[usize], values: &[f32]) -> Result<Tensor> {
    let host = Tensor::from_f32(shape.to_vec(), values)?;
    transfers::to_cuda(&host, ctx.device_id())
}

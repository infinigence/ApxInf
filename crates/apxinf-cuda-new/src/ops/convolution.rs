//! Valid 3D convolution, with session-owned cuDNN plans and caller-owned scratch.
use crate::{
    ffi::abi::{convolution as abi, status},
    CudaBuffer, CudaContext,
};
use apxinf_core::{DType, Device, Error, Result, Tensor};
#[cfg(test)]
use std::ffi::CStr;
use std::{rc::Rc, sync::Arc};

#[cfg(test)]
pub(crate) const DOC_IDS: &[&str] = &["convolution3d"];
fn invalid(message: &str) -> Error {
    Error::Other(message.into())
}

/// Available cuDNN runtime version, or None when the optional provider cannot
/// load (including builds without its development headers).
pub fn convolution_cudnn_version() -> Option<u64> {
    let version = unsafe { abi::apxinf_convolution_cudnn_version() };
    (version != 0).then_some(version)
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    spec: abi::Spec,
    device: usize,
    pointers: [usize; 4],
    workspace_bytes: usize,
    stream: usize,
    configuration: i32,
}
struct Execution {
    raw: abi::Execution,
    stream: Arc<crate::CudaStream>,
    _storage: Vec<CudaBuffer>,
}
impl Drop for Execution {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe { abi::apxinf_convolution_destroy(self.raw) };
    }
}
fn storage(ctx: &CudaContext, tensor: &Tensor) -> Result<CudaBuffer> {
    let bytes = tensor
        .shape()
        .dims()
        .iter()
        .try_fold(tensor.dtype().size_in_bytes(), |a, &b| a.checked_mul(b))
        .filter(|&n| n > 0 && n <= i64::MAX as usize)
        .ok_or_else(|| invalid("convolution tensor size overflow"))?;
    if tensor.device() != Device::Cuda(ctx.device_id()) {
        return Err(invalid("convolution tensor device mismatch"));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < bytes {
        return Err(invalid("convolution tensor storage too short"));
    }
    Ok(buffer)
}
fn shape(x: &Tensor) -> Result<[i64; 5]> {
    let dims = x.shape().dims();
    if dims.len() != 5 || dims.iter().any(|&d| d == 0 || d > i32::MAX as usize) {
        return Err(invalid(
            "convolution expects nonempty contiguous rank-five tensors",
        ));
    }
    Ok(std::array::from_fn(|i| dims[i] as i64))
}
fn prepare(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    stride: [usize; 3],
    output: &Tensor,
    workspace: &Tensor,
    configuration: i32,
) -> Result<Rc<Execution>> {
    let dtype = match input.dtype() {
        DType::F16 => 1,
        DType::BF16 => 2,
        _ => return Err(invalid("convolution supports F16/BF16")),
    };
    if weight.dtype() != input.dtype() || output.dtype() != input.dtype() {
        return Err(invalid("convolution dtype mismatch"));
    }
    let x = shape(input)?;
    let w = shape(weight)?;
    let y = shape(output)?;
    if x[0] != y[0] || x[1] != w[1] || w[0] != y[1] {
        return Err(invalid("convolution batch/channel mismatch"));
    }
    for i in 0..3 {
        if stride[i] == 0
            || stride[i] > i32::MAX as usize
            || x[i + 2] < w[i + 2]
            || y[i + 2] != (x[i + 2] - w[i + 2]) / stride[i] as i64 + 1
        {
            return Err(invalid("convolution output/stride mismatch"));
        }
    }
    let buffers = [input, weight, output, workspace]
        .map(|x| storage(ctx, x))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let pointers = std::array::from_fn(|i| buffers[i].ptr() as usize);
    if pointers[3] % 256 != 0 {
        return Err(invalid("convolution workspace requires 256-byte alignment"));
    }
    for i in 0..4 {
        for j in i + 1..4 {
            if (i == 0 && j == 1) || (i == 1 && j == 0) {
                continue;
            }
            let a_end = pointers[i]
                .checked_add(buffers[i].len())
                .ok_or_else(|| invalid("convolution address overflow"))?;
            let b_end = pointers[j]
                .checked_add(buffers[j].len())
                .ok_or_else(|| invalid("convolution address overflow"))?;
            if pointers[i] < b_end && pointers[j] < a_end {
                return Err(invalid("convolution writable storage overlaps operands"));
            }
        }
    }
    let alignment = std::array::from_fn(|i| (1usize << pointers[i].trailing_zeros().min(5)) as u32);
    if alignment.iter().any(|&a| a < 2) {
        return Err(invalid("convolution operand alignment mismatch"));
    }
    let spec = abi::Spec {
        version: 1,
        dtype,
        input: x,
        weight: w,
        output: y,
        stride: stride.map(|s| s as i64),
        alignment,
    };
    let key = Key {
        spec,
        device: ctx.device_id(),
        pointers,
        workspace_bytes: buffers[3].len(),
        stream: ctx.stream().handle() as usize,
        configuration,
    };
    if let Some(execution) = crate::workspace::lookup_execution(&key) {
        crate::workspace::use_execution(&execution)?;
        return Ok(execution);
    }
    if !crate::workspace::may_prepare_native_resources() {
        return Err(invalid(
            "convolution execution cache miss during capture; prepare the same bindings first",
        ));
    }
    let bindings = abi::Bindings {
        input: buffers[0].ptr(),
        weight: buffers[1].ptr(),
        output: buffers[2].ptr(),
        workspace: buffers[3].ptr(),
        workspace_bytes: buffers[3].len() as u64,
        stream: ctx.stream().handle(),
    };
    let mut raw = std::ptr::null_mut();
    unsafe {
        status::check(abi::apxinf_convolution_prepare(
            ctx.runtime(),
            &spec,
            &bindings,
            configuration,
            &mut raw,
        ))?;
    }
    if raw.is_null() {
        return Err(invalid("native convolution prepare returned null"));
    }
    let execution = Rc::new(Execution {
        raw,
        stream: ctx.shared_stream(),
        _storage: buffers,
    });
    crate::workspace::store_execution(key, Rc::clone(&execution));
    crate::workspace::use_execution(&execution)?;
    Ok(execution)
}
fn enqueue(execution: &Rc<Execution>) -> Result<()> {
    crate::workspace::validate_capture_target(
        execution.stream.device(),
        execution.stream.handle() as usize,
    )?;
    execution.stream.set_current_device().map_err(Error::Cuda)?;
    crate::workspace::retain_resource(execution);
    unsafe { status::check(abi::apxinf_convolution_enqueue(execution.raw)) }
}

/// Ungrouped, valid 3D cross-correlation with explicit positive strides.
/// Input/output are contiguous NCDHW, weights OIDHW; matching F16/BF16 with
/// F32 accumulation and one output rounding. No bias, implicit padding or
/// dilation. Caller-owned workspace may be reused by sequential operations on
/// this stream, and must not overlap operands. Plans reject nondeterministic
/// input-downconversion and reduced-precision-reduction engines. Preparation selects the first supported
/// cuDNN heuristic within the workspace capacity; no persistent tuning recipe.
/// A session must prepare the same bindings before CUDA graph capture.
pub fn convolution3d(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    stride: [usize; 3],
    output: &mut Tensor,
    workspace: &Tensor,
) -> Result<()> {
    let execution = prepare(ctx, input, weight, stride, output, workspace, -1)?;
    enqueue(&execution)?;
    if !crate::workspace::is_capturing()
        && (!crate::workspace::has_active_session() || crate::workspace::is_preparing_session())
    {
        ctx.synchronize().map_err(Error::Cuda)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_configurations(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    stride: [usize; 3],
    output: &Tensor,
    workspace: &Tensor,
    mut check: impl FnMut(&str) -> Result<()>,
) -> Result<usize> {
    let mut tested = 0;
    let baseline = prepare(ctx, input, weight, stride, output, workspace, -1)?;
    let count = unsafe { abi::apxinf_convolution_configuration_count(baseline.raw) };
    drop(baseline);
    for configuration in 0..count {
        match prepare(ctx, input, weight, stride, output, workspace, configuration) {
            Ok(e) => {
                enqueue(&e)?;
                ctx.synchronize().map_err(Error::Cuda)?;
                let summary = unsafe { CStr::from_ptr(abi::apxinf_convolution_summary(e.raw)) }
                    .to_string_lossy();
                check(&summary)?;
                tested += 1;
            }
            Err(Error::Cuda(s)) if s.contains("native status 2:") => {}
            Err(e) => return Err(e),
        }
    }
    Ok(tested)
}

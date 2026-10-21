use crate::ffi::abi::{reduction as abi, status};
use crate::{CudaBuffer, CudaContext, CudaStream};
use apxinf_core::{DType, Device, Error, Result, Tensor};
use std::{ptr::NonNull, rc::Rc, sync::Arc};
fn invalid(s: &str) -> Error {
    Error::Other(s.into())
}
/// Static indexed column means over an FP32 tensor. `indices` is a row-major
/// [rows, output.numel()] matrix of flat input indices, all checked at prepare.
/// No index upload or allocation occurs in enqueue. Captures retain storage.
pub struct PreparedIndexedMean(Rc<Execution>);
struct Execution {
    raw: NonNull<std::ffi::c_void>,
    _storage: Vec<CudaBuffer>,
    stream: Arc<CudaStream>,
}
impl PreparedIndexedMean {
    pub fn new(ctx: &CudaContext, input: &Tensor, out: &Tensor, indices: &[u32]) -> Result<Self> {
        if !crate::workspace::may_prepare_native_resources() {
            return Err(invalid("indexed mean preparation during capture"));
        }
        let (n, cols) = (input.numel(), out.numel());
        if n == 0
            || n > u32::MAX as usize
            || cols == 0
            || indices.is_empty()
            || indices.len() % cols != 0
            || indices.len() / cols > 16777216
            || indices.iter().any(|&i| i as usize >= n)
        {
            return Err(invalid("invalid indexed mean geometry/indices"));
        }
        let get = |t: &Tensor| -> Result<CudaBuffer> {
            if t.dtype() != DType::F32 || t.device() != Device::Cuda(ctx.device_id()) {
                return Err(invalid("indexed mean requires FP32 device tensors"));
            }
            let b = CudaBuffer::from_tensor(t).map_err(Error::Cuda)?;
            if b.len() < t.numel() * 4 || (b.ptr() as usize) % 4 != 0 {
                return Err(invalid("invalid indexed mean storage"));
            }
            Ok(b)
        };
        let input = get(input)?;
        let output = get(out)?;
        let (a, b) = (input.ptr() as usize, output.ptr() as usize);
        if a < b.saturating_add(cols * 4) && b < a.saturating_add(n * 4) {
            return Err(invalid("indexed mean output must not overlap input"));
        }
        let map = CudaBuffer::alloc(indices.len() * 4, ctx.device_id()).map_err(Error::Cuda)?;
        let bytes: Vec<_> = indices.iter().flat_map(|i| i.to_ne_bytes()).collect();
        map.copy_from_host(&bytes).map_err(Error::Cuda)?;
        let spec = abi::Spec {
            version: 1,
            reserved: 0,
            input_elements: n as i64,
            rows: (indices.len() / cols) as i64,
            cols: cols as i64,
        };
        let bindings = abi::Bindings {
            input: input.ptr().cast(),
            indices: map.ptr().cast(),
            output: output.ptr().cast(),
            stream: ctx.stream().handle(),
        };
        let mut raw = std::ptr::null_mut();
        unsafe {
            status::check(abi::apxinf_indexed_mean_prepare(
                ctx.runtime(),
                &spec,
                &bindings,
                &mut raw,
            ))?;
        }
        Ok(Self(Rc::new(Execution {
            raw: NonNull::new(raw).ok_or_else(|| invalid("null indexed mean execution"))?,
            _storage: vec![input, output, map],
            stream: ctx.shared_stream(),
        })))
    }
    pub fn enqueue(&self) -> Result<()> {
        crate::workspace::validate_capture_target(
            self.0.stream.device(),
            self.0.stream.handle() as usize,
        )?;
        crate::workspace::use_execution(&self.0)?;
        unsafe { status::check(abi::apxinf_indexed_mean_enqueue(self.0.raw.as_ptr())) }
    }
}
impl Drop for Execution {
    fn drop(&mut self) {
        unsafe {
            abi::apxinf_indexed_mean_destroy(self.raw.as_ptr());
        }
    }
}

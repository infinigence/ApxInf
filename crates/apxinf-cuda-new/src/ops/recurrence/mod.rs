//! FP64 recurrent storage with explicit FP32 input/output boundaries.
use crate::ffi::abi::{recurrence as abi, status};
use crate::{CudaBuffer, CudaContext, CudaStream};
use apxinf_core::{DType, Device, Error, Result, Tensor};
use std::{ptr::NonNull, rc::Rc, sync::Arc};
fn invalid(s: &str) -> Error {
    Error::Other(s.into())
}

/// Coefficients for `a*state + d*(b*input + c*previous_input)` in FP64.
/// Every multiply/add rounds separately, with no implicit FMA contraction.
#[derive(Clone, Copy, Debug)]
pub struct RecurrenceCoefficients {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

/// Opaque, typed FP64 storage. It cannot be reinterpreted as a core tensor.
/// The state is tied to one device stream; each graph/request must reset it.
pub struct F64RecurrenceState {
    state: CudaBuffer,
    previous: CudaBuffer,
    count: usize,
    stream: Arc<CudaStream>,
}
impl F64RecurrenceState {
    pub fn new(ctx: &CudaContext, count: usize) -> Result<Self> {
        let bytes = count
            .checked_mul(8)
            .filter(|&n| n > 0 && n <= i64::MAX as usize)
            .ok_or_else(|| invalid("invalid recurrence extent"))?;
        if !crate::workspace::may_prepare_native_resources() {
            return Err(invalid("recurrence allocation during capture"));
        }
        Ok(Self {
            state: CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?,
            previous: CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?,
            count,
            stream: ctx.shared_stream(),
        })
    }
    pub fn prepare_reset(
        &self,
        ctx: &CudaContext,
        input: &Tensor,
        out: &Tensor,
    ) -> Result<PreparedRecurrence> {
        self.prepare(ctx, input, out, None)
    }
    pub fn prepare_update(
        &self,
        ctx: &CudaContext,
        input: &Tensor,
        out: &Tensor,
        coefficients: RecurrenceCoefficients,
    ) -> Result<PreparedRecurrence> {
        self.prepare(ctx, input, out, Some(coefficients))
    }
    fn prepare(
        &self,
        ctx: &CudaContext,
        input: &Tensor,
        out: &Tensor,
        coefficients: Option<RecurrenceCoefficients>,
    ) -> Result<PreparedRecurrence> {
        if !crate::workspace::may_prepare_native_resources() {
            return Err(invalid("recurrence preparation during capture"));
        }
        if ctx.device_id() != self.stream.device() || ctx.stream().handle() != self.stream.handle()
        {
            return Err(invalid("recurrence stream mismatch"));
        }
        if input.shape() != out.shape() {
            return Err(invalid("recurrence shape mismatch"));
        }
        let get = |t: &Tensor| -> Result<CudaBuffer> {
            if t.device() != Device::Cuda(ctx.device_id())
                || t.dtype() != DType::F32
                || t.numel() != self.count
            {
                return Err(invalid("recurrence requires matching FP32 tensors"));
            }
            let b = CudaBuffer::from_tensor(t).map_err(Error::Cuda)?;
            if b.len() < self.count * 4 || (b.ptr() as usize) % 4 != 0 {
                return Err(invalid("invalid recurrence tensor storage"));
            }
            Ok(b)
        };
        let input = get(input)?;
        let output = get(out)?;
        let (a, b) = (input.ptr() as usize, output.ptr() as usize);
        let bytes = self.count * 4;
        if a != b && a < b.saturating_add(bytes) && b < a.saturating_add(bytes) {
            return Err(invalid("recurrence rejects partial aliasing"));
        }
        let c = coefficients.unwrap_or(RecurrenceCoefficients {
            a: 0.,
            b: 0.,
            c: 0.,
            d: 0.,
        });
        if [c.a, c.b, c.c, c.d].iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite recurrence coefficient"));
        }
        let spec = abi::Spec {
            version: 1,
            reset: u32::from(coefficients.is_none()),
            count: self.count as i64,
            a: c.a,
            b: c.b,
            c: c.c,
            d: c.d,
        };
        let bindings = abi::Bindings {
            input: input.ptr().cast(),
            output: output.ptr().cast(),
            state: self.state.ptr().cast(),
            previous: self.previous.ptr().cast(),
            stream: self.stream.handle(),
        };
        let mut raw = std::ptr::null_mut();
        unsafe {
            status::check(abi::apxinf_recurrence_prepare(
                ctx.runtime(),
                &spec,
                &bindings,
                &mut raw,
            ))?;
        }
        Ok(PreparedRecurrence(Rc::new(Execution {
            raw: NonNull::new(raw).ok_or_else(|| invalid("null recurrence execution"))?,
            _storage: vec![input, output, self.state.clone(), self.previous.clone()],
            stream: self.stream.clone(),
        })))
    }
}
struct Execution {
    raw: NonNull<std::ffi::c_void>,
    _storage: Vec<CudaBuffer>,
    stream: Arc<CudaStream>,
}
impl Drop for Execution {
    fn drop(&mut self) {
        unsafe {
            abi::apxinf_recurrence_destroy(self.raw.as_ptr());
        }
    }
}
/// Fixed bindings retained by captured graphs, including the FP64 state.
pub struct PreparedRecurrence(Rc<Execution>);
impl PreparedRecurrence {
    pub fn enqueue(&self) -> Result<()> {
        crate::workspace::validate_capture_target(
            self.0.stream.device(),
            self.0.stream.handle() as usize,
        )?;
        crate::workspace::use_execution(&self.0)?;
        unsafe { status::check(abi::apxinf_recurrence_enqueue(self.0.raw.as_ptr())) }
    }
}

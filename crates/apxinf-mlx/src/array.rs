use crate::ffi;
use apxinf_core::{Error, Result};
use std::{
    cell::{Cell, RefCell},
    ffi::{c_void, CStr, CString},
    ptr::NonNull,
    rc::Rc,
};

fn error(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}
fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(error(
            unsafe { CStr::from_ptr(ffi::apx_mlx_error()) }
                .to_string_lossy()
                .into_owned(),
        ))
    }
}
fn dims(shape: &[usize]) -> Result<Vec<i32>> {
    shape
        .iter()
        .map(|&x| i32::try_from(x).map_err(|_| error("MLX shape exceeds i32")))
        .collect()
}

/// Process-wide MLX allocator observations, in bytes. This is not per stream,
/// model, Rust ownership graph, or system resident memory. Cached buffers are
/// separate from active allocations. Synchronize before lifecycle comparisons.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub active_bytes: usize,
    pub cache_bytes: usize,
    pub peak_bytes: usize,
}
pub fn memory_stats() -> Result<MemoryStats> {
    let mut stats = MemoryStats::default();
    check(unsafe {
        ffi::apx_mlx_memory_stats(
            &mut stats.active_bytes,
            &mut stats.cache_bytes,
            &mut stats.peak_bytes,
        )
    })?;
    Ok(stats)
}
/// Reset the process-wide peak allocator observation. Use an isolated process
/// for qualification; another model in this process shares the observation.
pub fn reset_peak_memory() -> Result<()> {
    check(unsafe { ffi::apx_mlx_reset_peak_memory() })
}
/// Release cached allocator buffers process-wide. Active arrays remain valid.
/// Synchronize first when measuring release, and use an isolated test process.
pub fn clear_cache() -> Result<()> {
    check(unsafe { ffi::apx_mlx_clear_cache() })
}
fn count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| error("MLX shape overflow"))
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum MlxDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I32 = 3,
    U32 = 4,
    Bool = 5,
}
impl MlxDType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::Bool => 1,
            _ => 4,
        }
    }
    fn from_code(code: i32) -> Result<Self> {
        match code {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::BF16),
            3 => Ok(Self::I32),
            4 => Ok(Self::U32),
            5 => Ok(Self::Bool),
            _ => Err(error("invalid native dtype")),
        }
    }
}

#[derive(Clone)]
pub struct Stream(Rc<StreamInner>);
/// Cumulative observations for one logical execution stream and its clones.
/// Host byte counts cover successful explicit bridge uploads/downloads only;
/// they do not estimate driver transfers or MLX's internal allocations.
/// `eval_calls` counts attempted evaluation boundaries, including downloads.
/// Counters saturate instead of wrapping and never reset implicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamCounters {
    pub trace_callbacks: usize,
    pub uploads_bytes: usize,
    pub downloads_bytes: usize,
    pub eval_calls: usize,
}
struct StreamInner {
    raw: NonNull<c_void>,
    metal: Option<usize>,
    counters: Cell<StreamCounters>,
}
impl Drop for StreamInner {
    fn drop(&mut self) {
        unsafe { ffi::apx_mlx_stream_free(self.raw.as_ptr()) }
    }
}
impl Stream {
    /// CPU stream for semantic tests and explicit diagnostics only.
    pub fn cpu() -> Result<Self> {
        Self::create(false, 0)
    }
    pub fn metal(index: usize) -> Result<Self> {
        Self::create(true, index)
    }
    fn create(gpu: bool, index: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        let index_i = i32::try_from(index).map_err(|_| error("device index exceeds i32"))?;
        check(unsafe { ffi::apx_mlx_stream_new(gpu as i32, index_i, &mut raw) })?;
        Ok(Self(Rc::new(StreamInner {
            raw: NonNull::new(raw).ok_or_else(|| error("null MLX stream"))?,
            metal: gpu.then_some(index),
            counters: Cell::new(StreamCounters::default()),
        })))
    }
    pub fn metal_index(&self) -> Option<usize> {
        self.0.metal
    }
    pub fn counters(&self) -> StreamCounters {
        self.0.counters.get()
    }
    fn record(&self, update: impl FnOnce(&mut StreamCounters)) {
        let mut counters = self.counters();
        update(&mut counters);
        self.0.counters.set(counters);
    }
    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { ffi::apx_mlx_sync(self.raw()) })
    }
    pub fn eval(&self, arrays: &[Array]) -> Result<()> {
        let p = self.inputs(arrays.iter())?;
        self.record(|c| c.eval_calls = c.eval_calls.saturating_add(1));
        check(unsafe { ffi::apx_mlx_eval(self.raw(), p.as_ptr(), p.len()) })
    }
    fn raw(&self) -> ffi::Handle {
        self.0.raw.as_ptr()
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    fn inputs<'a>(&self, arrays: impl Iterator<Item = &'a Array>) -> Result<Vec<*const c_void>> {
        arrays
            .map(|a| {
                if self.same(a.stream()) {
                    Ok(a.raw() as *const c_void)
                } else {
                    Err(error("MLX operands belong to different execution streams"))
                }
            })
            .collect()
    }
    fn op(&self, op: i32, arrays: &[&Array], ints: &[i32], floats: &[f32]) -> Result<Array> {
        let p = self.inputs(arrays.iter().copied())?;
        let mut out = std::ptr::null_mut();
        check(unsafe {
            ffi::apx_mlx_op(
                self.raw(),
                op,
                p.as_ptr(),
                p.len(),
                ints.as_ptr(),
                ints.len(),
                floats.as_ptr(),
                floats.len(),
                &mut out,
            )
        })?;
        unsafe { Array::own(self.clone(), out) }
    }
}

/// Immutable lazy array, bound to one thread and retained execution stream.
/// Clone shares the native graph; every exposed operation is functional.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<apxinf_mlx::Array>();
/// ```
#[derive(Clone)]
pub struct Array(Rc<ArrayInner>);
struct ArrayInner {
    raw: NonNull<c_void>,
    stream: Stream,
    shape: Vec<usize>,
    dtype: MlxDType,
}
impl Drop for ArrayInner {
    fn drop(&mut self) {
        unsafe { ffi::apx_mlx_array_free(self.raw.as_ptr()) }
    }
}
impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxArray")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}
impl Array {
    unsafe fn own(stream: Stream, raw: ffi::Handle) -> Result<Self> {
        let raw = NonNull::new(raw).ok_or_else(|| error("null MLX array"))?;
        let mut shape = [0i32; 32];
        let mut rank = 0;
        let mut dtype = 0;
        if let Err(e) = check(ffi::apx_mlx_array_info(
            raw.as_ptr(),
            shape.as_mut_ptr(),
            shape.len(),
            &mut rank,
            &mut dtype,
        )) {
            ffi::apx_mlx_array_free(raw.as_ptr());
            return Err(e);
        }
        let dt = match MlxDType::from_code(dtype) {
            Ok(v) => v,
            Err(e) => {
                ffi::apx_mlx_array_free(raw.as_ptr());
                return Err(e);
            }
        };
        Ok(Self(Rc::new(ArrayInner {
            raw,
            stream,
            shape: shape[..rank].iter().map(|&x| x as usize).collect(),
            dtype: dt,
        })))
    }
    unsafe fn copy_handle(stream: Stream, raw: *const c_void) -> Result<Self> {
        let mut owned = std::ptr::null_mut();
        check(ffi::apx_mlx_array_clone(raw, &mut owned))?;
        Self::own(stream, owned)
    }
    fn raw(&self) -> ffi::Handle {
        self.0.raw.as_ptr()
    }
    pub fn stream(&self) -> &Stream {
        &self.0.stream
    }
    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }
    pub fn dtype(&self) -> MlxDType {
        self.0.dtype
    }
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }
    pub fn from_bytes(
        stream: &Stream,
        shape: &[usize],
        dtype: MlxDType,
        bytes: &[u8],
    ) -> Result<Self> {
        let expected = count(shape)?
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| error("upload length overflow"))?;
        if expected != bytes.len() {
            return Err(error("upload length differs from shape/dtype"));
        }
        let d = dims(shape)?;
        let mut out = std::ptr::null_mut();
        check(unsafe {
            ffi::apx_mlx_array_new(
                stream.raw(),
                d.as_ptr(),
                d.len(),
                dtype as i32,
                bytes.as_ptr(),
                bytes.len(),
                &mut out,
            )
        })?;
        let array = unsafe { Self::own(stream.clone(), out) }?;
        stream.record(|c| c.uploads_bytes = c.uploads_bytes.saturating_add(bytes.len()));
        Ok(array)
    }
    pub fn from_f32(stream: &Stream, shape: &[usize], data: &[f32]) -> Result<Self> {
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
        Self::from_bytes(stream, shape, MlxDType::F32, &bytes)
    }
    pub fn from_i32(stream: &Stream, shape: &[usize], data: &[i32]) -> Result<Self> {
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
        Self::from_bytes(stream, shape, MlxDType::I32, &bytes)
    }
    pub fn scalar(stream: &Stream, value: f32, dtype: MlxDType) -> Result<Self> {
        Self::from_f32(stream, &[], &[value])?.cast(dtype)
    }
    pub fn zeros(stream: &Stream, shape: &[usize], dtype: MlxDType) -> Result<Self> {
        count(shape)?
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| error("zero array byte extent overflow"))?;
        let mut p = vec![dtype as i32];
        p.extend(dims(shape)?);
        stream.op(28, &[], &p, &[])
    }
    pub fn arange(
        stream: &Stream,
        start: f32,
        stop: f32,
        step: f32,
        dtype: MlxDType,
    ) -> Result<Self> {
        if !start.is_finite() || !stop.is_finite() || !step.is_finite() || step == 0.0 {
            return Err(error("invalid arange"));
        }
        stream.op(18, &[], &[dtype as i32], &[start, stop, step])
    }
    pub fn eval(&self) -> Result<()> {
        self.stream().eval(&[self.clone()])
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = vec![0; self.numel() * self.dtype().size_in_bytes()];
        self.stream()
            .record(|c| c.eval_calls = c.eval_calls.saturating_add(1));
        check(unsafe {
            ffi::apx_mlx_array_read(self.stream().raw(), self.raw(), out.as_mut_ptr(), out.len())
        })?;
        self.stream()
            .record(|c| c.downloads_bytes = c.downloads_bytes.saturating_add(out.len()));
        Ok(out)
    }
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(self
            .cast(MlxDType::F32)?
            .to_bytes()?
            .chunks_exact(4)
            .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
            .collect())
    }
    pub fn to_u32_scalar(&self) -> Result<u32> {
        if self.numel() != 1 {
            return Err(error("scalar read requires one element"));
        }
        let bytes = self.cast(MlxDType::U32)?.to_bytes()?;
        Ok(u32::from_ne_bytes(bytes.try_into().unwrap()))
    }
    pub fn cast(&self, dtype: MlxDType) -> Result<Self> {
        self.stream().op(0, &[self], &[dtype as i32], &[])
    }
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        if count(shape)? != self.numel() {
            return Err(error("reshape changes element count"));
        }
        self.stream().op(1, &[self], &dims(shape)?, &[])
    }
    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.shape().len()
            || axes.iter().any(|&x| x >= axes.len())
            || (0..axes.len()).any(|x| axes.iter().filter(|&&v| v == x).count() != 1)
        {
            return Err(error("invalid permutation"));
        }
        self.stream().op(2, &[self], &dims(axes)?, &[])
    }
    pub fn contiguous(&self) -> Result<Self> {
        self.stream().op(3, &[self], &[], &[])
    }
    pub fn copy(&self) -> Result<Self> {
        self.stream().op(33, &[self], &[], &[])
    }
    pub fn slice(&self, start: &[usize], stop: &[usize]) -> Result<Self> {
        if start.len() != self.shape().len()
            || stop.len() != start.len()
            || start
                .iter()
                .zip(stop)
                .zip(self.shape())
                .any(|((&a, &b), &d)| a > b || b > d)
        {
            return Err(error("invalid array slice"));
        }
        let mut p = dims(start)?;
        p.extend(dims(stop)?);
        self.stream().op(4, &[self], &p, &[])
    }
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> Result<Self> {
        if axis >= self.shape().len() {
            return Err(error("invalid slice axis"));
        }
        let mut begin = vec![0; self.shape().len()];
        let mut stop = self.shape().to_vec();
        begin[axis] = start;
        stop[axis] = end;
        self.slice(&begin, &stop)
    }
    pub fn concat(arrays: &[&Array], axis: usize) -> Result<Self> {
        let a = arrays.first().ok_or_else(|| error("concat needs inputs"))?;
        a.stream().op(
            5,
            arrays,
            &[i32::try_from(axis).map_err(|_| error("axis exceeds i32"))?],
            &[],
        )
    }
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        count(shape)?
            .checked_mul(self.dtype().size_in_bytes())
            .ok_or_else(|| error("broadcast byte extent overflow"))?;
        self.stream().op(6, &[self], &dims(shape)?, &[])
    }
    pub fn matmul(&self, b: &Self) -> Result<Self> {
        self.binary(7, b)
    }
    pub fn add(&self, b: &Self) -> Result<Self> {
        self.binary(8, b)
    }
    pub fn mul(&self, b: &Self) -> Result<Self> {
        self.binary(9, b)
    }
    pub fn div(&self, b: &Self) -> Result<Self> {
        self.binary(10, b)
    }
    pub fn pow(&self, b: &Self) -> Result<Self> {
        self.binary(11, b)
    }
    pub fn sub(&self, b: &Self) -> Result<Self> {
        self.binary(22, b)
    }
    pub fn less_equal(&self, b: &Self) -> Result<Self> {
        self.binary(26, b)
    }
    pub fn equal(&self, b: &Self) -> Result<Self> {
        self.binary(36, b)
    }
    fn binary(&self, op: i32, b: &Self) -> Result<Self> {
        self.stream().op(op, &[self, b], &[], &[])
    }
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary(12)
    }
    pub fn exp(&self) -> Result<Self> {
        self.unary(13)
    }
    pub fn sigmoid(&self) -> Result<Self> {
        self.unary(14)
    }
    pub fn neg(&self) -> Result<Self> {
        self.unary(23)
    }
    pub fn cos(&self) -> Result<Self> {
        self.unary(24)
    }
    pub fn sin(&self) -> Result<Self> {
        self.unary(25)
    }
    pub fn isfinite(&self) -> Result<Self> {
        self.unary(34)
    }
    fn unary(&self, op: i32) -> Result<Self> {
        self.stream().op(op, &[self], &[], &[])
    }
    pub fn softmax(&self, axis: i32) -> Result<Self> {
        self.stream().op(15, &[self], &[axis], &[])
    }
    pub fn sum(&self, axes: &[i32], keepdims: bool) -> Result<Self> {
        self.reduce(16, axes, keepdims)
    }
    pub fn mean(&self, axes: &[i32], keepdims: bool) -> Result<Self> {
        self.reduce(17, axes, keepdims)
    }
    fn reduce(&self, op: i32, axes: &[i32], keepdims: bool) -> Result<Self> {
        let mut p = vec![keepdims as i32];
        p.extend(axes);
        self.stream().op(op, &[self], &p, &[])
    }
    pub fn max(&self, axis: i32) -> Result<Self> {
        self.stream().op(35, &[self], &[axis], &[])
    }
    pub fn take(&self, indices: &Self, axis: i32) -> Result<Self> {
        self.stream().op(19, &[self, indices], &[axis], &[])
    }
    pub fn slice_update(&self, update: &Self, index: &Self, axes: &[i32]) -> Result<Self> {
        self.stream().op(20, &[self, update, index], axes, &[])
    }
    pub fn argmax(&self, axis: i32) -> Result<Self> {
        self.stream().op(21, &[self], &[axis], &[])
    }
    pub fn where_select(&self, yes: &Self, no: &Self) -> Result<Self> {
        self.stream().op(27, &[self, yes, no], &[], &[])
    }
    pub fn nan_to_num(&self, nan: f32, posinf: f32, neginf: f32) -> Result<Self> {
        self.stream().op(37, &[self], &[], &[nan, posinf, neginf])
    }
    /// Stock MLX fused RMSNorm; callers must select its rounding contract
    /// deliberately instead of substituting it for an ordered family norm.
    pub fn fast_rms_norm(&self, weight: &Self, epsilon: f32) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(error("RMSNorm epsilon must be finite and positive"));
        }
        self.stream().op(38, &[self, weight], &[], &[epsilon])
    }
    /// MLX fast attention semantics; model caller owns its numerical contract.
    pub fn sdpa(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        causal: bool,
        mask: Option<&Self>,
    ) -> Result<Self> {
        let mut a = vec![self, k, v];
        if let Some(m) = mask {
            a.push(m)
        }
        self.stream().op(29, &a, &[causal as i32], &[scale])
    }
    pub fn rope(
        &self,
        offset: &Self,
        dims: usize,
        traditional: bool,
        base: f32,
        scale: f32,
    ) -> Result<Self> {
        self.stream().op(
            30,
            &[self, offset],
            &[
                i32::try_from(dims).map_err(|_| error("rope dims exceed i32"))?,
                traditional as i32,
            ],
            &[base, scale],
        )
    }
    /// Safe nonnegative-stride view of a contiguous, owned source extent.
    pub fn as_strided(&self, shape: &[usize], strides: &[usize], offset: usize) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(error("strides rank mismatch"));
        }
        let end = shape.iter().zip(strides).try_fold(offset, |n, (&d, &s)| {
            n.checked_add(
                d.saturating_sub(1)
                    .checked_mul(s)
                    .ok_or_else(|| error("strided extent overflow"))?,
            )
            .ok_or_else(|| error("strided extent overflow"))
        })?;
        let elements = count(shape)?;
        if offset > self.numel() || (elements > 0 && end >= self.numel()) {
            return Err(error("strided view exceeds source extent"));
        }
        let mut p = vec![shape.len() as i32];
        p.extend(dims(shape)?);
        p.extend(dims(strides)?);
        p.push(i32::try_from(offset).map_err(|_| error("stride offset exceeds i32"))?);
        let source = self.contiguous()?;
        self.stream().op(32, &[&source], &p, &[])
    }
    pub fn quantize(&self, group_size: i32, bits: i32) -> Result<[Self; 3]> {
        let mut out = [std::ptr::null_mut(); 3];
        check(unsafe {
            ffi::apx_mlx_quantize(
                self.stream().raw(),
                self.raw(),
                group_size,
                bits,
                out.as_mut_ptr(),
            )
        })?;
        let values = own_many(self.stream(), &mut out)?;
        Ok(values.try_into().unwrap())
    }
    pub fn quantized_matmul(
        &self,
        weight: &Self,
        scales: &Self,
        biases: &Self,
        transpose: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        self.stream().op(
            31,
            &[self, weight, scales, biases],
            &[transpose as i32, group_size, bits],
            &[],
        )
    }
    /// Affine dequantization with MLX's inferred output dtype (the scales dtype).
    /// The receiver contains packed uint32 weights, including selected rows.
    pub fn dequantize(
        &self,
        scales: &Self,
        biases: &Self,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        self.stream()
            .op(39, &[self, scales, biases], &[group_size, bits], &[])
    }
}
fn own_many(stream: &Stream, raw: &mut [ffi::Handle]) -> Result<Vec<Array>> {
    let mut out = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let handle = std::mem::replace(&mut raw[i], std::ptr::null_mut());
        match unsafe { Array::own(stream.clone(), handle) } {
            Ok(v) => out.push(v),
            Err(e) => {
                for p in &mut raw[i + 1..] {
                    unsafe { ffi::apx_mlx_array_free(*p) };
                    *p = std::ptr::null_mut();
                }
                return Err(e);
            }
        }
    }
    Ok(out)
}

type TraceFn = dyn Fn(&[Array]) -> Result<Vec<Array>>;
struct Trace {
    stream: Stream,
    body: Box<TraceFn>,
    error: RefCell<Option<String>>,
}
/// Owns a pure tracing closure. Changing state is explicit array input/output.
pub struct Compiled {
    raw: NonNull<c_void>,
    trace: Box<Trace>,
    outputs: usize,
}
impl Compiled {
    pub fn new(
        stream: &Stream,
        outputs: usize,
        body: impl Fn(&[Array]) -> Result<Vec<Array>> + 'static,
    ) -> Result<Self> {
        Self::with_shapeless(stream, outputs, false, body)
    }
    pub fn with_shapeless(
        stream: &Stream,
        outputs: usize,
        shapeless: bool,
        body: impl Fn(&[Array]) -> Result<Vec<Array>> + 'static,
    ) -> Result<Self> {
        if outputs == 0 {
            return Err(error("compiled callable needs outputs"));
        }
        let mut trace = Box::new(Trace {
            stream: stream.clone(),
            body: Box::new(body),
            error: RefCell::new(None),
        });
        let mut out = std::ptr::null_mut();
        check(unsafe {
            ffi::apx_mlx_compile_new(
                trace_callback,
                (&mut *trace as *mut Trace).cast(),
                outputs,
                shapeless as i32,
                &mut out,
            )
        })?;
        Ok(Self {
            raw: NonNull::new(out).ok_or_else(|| error("null compiled function"))?,
            trace,
            outputs,
        })
    }
    pub fn call(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let p = self.trace.stream.inputs(inputs.iter())?;
        let mut out = vec![std::ptr::null_mut(); self.outputs];
        *self.trace.error.borrow_mut() = None;
        let status = unsafe {
            ffi::apx_mlx_compile_call(
                self.raw.as_ptr(),
                p.as_ptr(),
                p.len(),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        if status != 0 {
            if let Some(e) = self.trace.error.borrow_mut().take() {
                return Err(error(e));
            }
            check(status)?;
        }
        own_many(&self.trace.stream, &mut out)
    }
    /// Evaluate all returned arrays before publishing any of them as new state.
    pub fn call_and_eval(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        let out = self.call(inputs)?;
        self.trace.stream.eval(&out)?;
        Ok(out)
    }

    /// Warm one exact input profile, then verify it replays without tracing.
    /// Use only during preparation: this evaluates the pure function twice and
    /// returns the second result. It does not publish model state. A disabled
    /// or unavailable compiler that returns the original eager body fails here.
    /// This observes Rust trace callbacks, not Metal pipeline/JIT compilation.
    pub fn prepare(&self, inputs: &[Array]) -> Result<Vec<Array>> {
        self.call_and_eval(inputs)?;
        let before = self.trace.stream.counters().trace_callbacks;
        let output = self.call_and_eval(inputs)?;
        if self.trace.stream.counters().trace_callbacks != before {
            return Err(error("MLX prepared profile retraced on replay; required compiled execution is unavailable"));
        }
        Ok(output)
    }
}
impl Drop for Compiled {
    fn drop(&mut self) {
        unsafe { ffi::apx_mlx_compile_free(self.raw.as_ptr()) }
    }
}
unsafe extern "C" fn trace_callback(
    ctx: ffi::Handle,
    inputs: *const *const c_void,
    count: usize,
    outputs: *mut ffi::Handle,
    nout: usize,
) -> i32 {
    let trace = &*(ctx as *const Trace);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
        trace
            .stream
            .record(|c| c.trace_callbacks = c.trace_callbacks.saturating_add(1));
        // An empty C++ vector may have a null data pointer. Rust slices still
        // require a non-null pointer at length zero.
        let handles = if count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(inputs, count)
        };
        let input = handles
            .iter()
            .map(|&p| Array::copy_handle(trace.stream.clone(), p))
            .collect::<Result<Vec<_>>>()?;
        let out = (trace.body)(&input)?;
        if out.len() != nout {
            return Err(error("trace output count differs from declared contract"));
        }
        trace.stream.inputs(out.iter())?;
        // Native owns each cloned handle, including partial-output failure cleanup.
        for (i, a) in out.iter().enumerate() {
            check(ffi::apx_mlx_array_clone(a.raw(), outputs.add(i)))?;
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            *trace.error.borrow_mut() = Some(e.to_string());
            -1
        }
        Err(_) => {
            *trace.error.borrow_mut() = Some("panic in MLX tracing closure".into());
            -1
        }
    }
}

/// Prepared custom Metal primitive. Source validity and operand geometry remain
/// the safe model-neutral wrapper's contract; no Python execution is involved.
pub struct MetalKernel {
    raw: NonNull<c_void>,
    stream: Stream,
    input_count: usize,
    output_count: usize,
}
impl MetalKernel {
    pub fn new(
        stream: &Stream,
        name: &str,
        inputs: &[&str],
        outputs: &[&str],
        source: &str,
        header: &str,
    ) -> Result<Self> {
        if stream.metal_index().is_none() {
            return Err(error("custom Metal kernels require a Metal stream"));
        }
        let cs = |s: &str| CString::new(s).map_err(|_| error("NUL in Metal source/name"));
        let name = cs(name)?;
        let source = cs(source)?;
        let header = cs(header)?;
        let ins = inputs.iter().map(|s| cs(s)).collect::<Result<Vec<_>>>()?;
        let outs = outputs.iter().map(|s| cs(s)).collect::<Result<Vec<_>>>()?;
        let ip: Vec<_> = ins.iter().map(|s| s.as_ptr()).collect();
        let op: Vec<_> = outs.iter().map(|s| s.as_ptr()).collect();
        let mut raw = std::ptr::null_mut();
        check(unsafe {
            ffi::apx_mlx_metal_new(
                name.as_ptr(),
                source.as_ptr(),
                header.as_ptr(),
                ip.as_ptr(),
                ip.len(),
                op.as_ptr(),
                op.len(),
                &mut raw,
            )
        })?;
        Ok(Self {
            raw: NonNull::new(raw).ok_or_else(|| error("null Metal kernel"))?,
            stream: stream.clone(),
            input_count: inputs.len(),
            output_count: outputs.len(),
        })
    }
    /// # Safety
    /// The source must bound every read/write by the supplied tensor geometry,
    /// including every indirect index. It must match declared types and avoid
    /// races. Arbitrary Metal source can access raw memory; callers expose only
    /// reviewed, shape-checked model-neutral wrappers around this primitive.
    pub unsafe fn call(
        &self,
        inputs: &[Array],
        output_specs: &[(&[usize], MlxDType)],
        grid: [usize; 3],
        threadgroup: [usize; 3],
        template_dtype: Option<MlxDType>,
    ) -> Result<Vec<Array>> {
        if inputs.len() != self.input_count || output_specs.len() != self.output_count {
            return Err(error("Metal kernel arity mismatch"));
        }
        let p = self.stream.inputs(inputs.iter())?;
        let mut shapes = Vec::new();
        let mut ranks = Vec::new();
        let mut types = Vec::new();
        for &(s, t) in output_specs {
            count(s)?;
            shapes.extend(dims(s)?);
            ranks.push(s.len());
            types.push(t as i32)
        }
        let g = dims(&grid)?;
        let tg = dims(&threadgroup)?;
        if g.contains(&0) || tg.contains(&0) {
            return Err(error("zero Metal launch geometry"));
        }
        let mut out = vec![std::ptr::null_mut(); self.output_count];
        check(ffi::apx_mlx_metal_call(
            self.stream.raw(),
            self.raw.as_ptr(),
            p.as_ptr(),
            p.len(),
            shapes.as_ptr(),
            ranks.as_ptr(),
            types.as_ptr(),
            self.output_count,
            g.as_ptr(),
            tg.as_ptr(),
            template_dtype.map_or(-1, |d| d as i32),
            out.as_mut_ptr(),
        ))?;
        own_many(&self.stream, &mut out)
    }
}
impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { ffi::apx_mlx_metal_free(self.raw.as_ptr()) }
    }
}

//! Qwen3.5 gated delta-network CUDA primitives.

use std::mem::size_of;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::cublas::CublasTranspose;
use crate::ffi;

const FLAG_NON_FINITE_INPUT: u32 = 1;
const FLAG_NON_FINITE_OUTPUT: u32 = 2;

/// Deferred-status mode: instead of allocating a fresh flags buffer and doing
/// a full stream synchronize + device-to-host read after every GDN op (about
/// four synchronizations per GDN layer per token), ops latch their non-finite
/// flags into one resident per-device buffer via `atomicOr` and the caller
/// drains it once per decoded token / prefill block with
/// [`drain_deferred_status`]. Numerics are unchanged (same kernels, same
/// order); only the point at which a non-finite value aborts the request
/// moves from per-op to per-token/per-block granularity, after which the
/// failed session is dropped as before. Read from the environment on every
/// call (matching the debug-capture hooks) so tests can flip modes; default
/// off, i.e. the eager per-op behaviour is preserved bit-for-bit.
fn deferred_status_enabled() -> bool {
    std::env::var("APXINF_Q35_DEFERRED_STATUS").is_ok_and(|value| value == "1")
}

fn shared_status_flags(device: usize) -> Result<&'static CudaBuffer> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static FLAGS: OnceLock<Mutex<HashMap<usize, &'static CudaBuffer>>> = OnceLock::new();
    let registry = FLAGS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry
        .lock()
        .map_err(|_| Error::Other("deferred status registry poisoned".into()))?;
    if let Some(buffer) = guard.get(&device) {
        return Ok(buffer);
    }
    let buffer: &'static CudaBuffer = Box::leak(Box::new(alloc_zeroed(size_of::<u32>(), device)?));
    guard.insert(device, buffer);
    Ok(buffer)
}

/// Per-op status flag handle. Eager mode owns a freshly zeroed buffer and
/// checks it synchronously; deferred mode shares the resident latch buffer
/// and defers the check to [`drain_deferred_status`].
enum StatusFlags {
    Eager(CudaBuffer),
    Deferred(&'static CudaBuffer),
}

impl StatusFlags {
    fn acquire(ctx: &CudaContext) -> Result<Self> {
        if deferred_status_enabled() {
            Ok(Self::Deferred(shared_status_flags(ctx.device_id())?))
        } else {
            Ok(Self::Eager(alloc_zeroed(
                size_of::<u32>(),
                ctx.device_id(),
            )?))
        }
    }

    fn ptr(&self) -> *mut std::ffi::c_void {
        match self {
            Self::Eager(buffer) => buffer.ptr(),
            Self::Deferred(buffer) => buffer.ptr(),
        }
    }

    fn finish(self, ctx: &CudaContext, operation: &str) -> Result<()> {
        match self {
            Self::Eager(buffer) => {
                let flag = read_status(ctx, &buffer)?;
                if flag != 0 {
                    return Err(status_error(operation, flag));
                }
                Ok(())
            }
            Self::Deferred(_) => Ok(()),
        }
    }
}

/// Synchronize once, surface any latched deferred non-finite status, and
/// clear the latch so the next request starts clean. No-op unless
/// `APXINF_Q35_DEFERRED_STATUS=1`.
pub fn drain_deferred_status(ctx: &CudaContext, operation: &str) -> Result<()> {
    if !deferred_status_enabled() {
        return Ok(());
    }
    let flags = shared_status_flags(ctx.device_id())?;
    let flag = read_status(ctx, flags)?;
    if flag != 0 {
        unsafe {
            ffi::check_cuda(ffi::cudaMemset(flags.ptr(), 0, size_of::<u32>()))
                .map_err(Error::Cuda)?;
        }
        return Err(status_error(operation, flag));
    }
    Ok(())
}

/// Check a BF16 tensor for NaN/Inf values on the selected CUDA device.
pub fn require_finite_bf16(ctx: &CudaContext, tensor: &Tensor, operation: &str) -> Result<()> {
    let dims = tensor.shape().dims();
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != DType::BF16
        || dims.is_empty()
        || dims.iter().any(|dimension| *dimension == 0)
    {
        return Err(Error::Other(format!(
            "{operation} finite check expects non-empty CUDA{} BF16 tensor, got {} {:?}",
            ctx.device_id(),
            tensor.dtype(),
            dims
        )));
    }
    let input = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let flags = StatusFlags::acquire(ctx)?;
    let elements = dims.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| Error::Other(format!("{operation} finite check size overflow")))
    })?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_gdn_check_finite_bf16(
            input.ptr(),
            flags.ptr(),
            checked_i32(elements, "finite check elements")?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    if let StatusFlags::Eager(buffer) = flags {
        let flag = read_status(ctx, &buffer)?;
        if flag != 0 {
            return Err(Error::Other(format!(
                "{operation} contains a non-finite value"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35GdnLayout {
    pub conv_kernel: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl Qwen35GdnLayout {
    pub fn new(
        conv_kernel: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<Self> {
        if conv_kernel == 0
            || key_heads == 0
            || value_heads == 0
            || key_dim == 0
            || value_dim == 0
            || value_heads % key_heads != 0
        {
            return Err(Error::Other("Qwen3.5 GDN dimensions are invalid".into()));
        }
        Ok(Self {
            conv_kernel,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
        })
    }

    pub const fn conv_channels(self) -> usize {
        self.key_heads * self.key_dim * 2 + self.value_heads * self.value_dim
    }

    pub const fn query_width(self) -> usize {
        self.key_heads * self.key_dim
    }

    pub const fn value_width(self) -> usize {
        self.value_heads * self.value_dim
    }

    fn conv_elements(self) -> Result<usize> {
        self.conv_channels()
            .checked_mul(self.conv_kernel)
            .ok_or_else(|| Error::Other("Qwen3.5 GDN convolution size overflow".into()))
    }

    fn recurrent_elements(self) -> Result<usize> {
        self.value_heads
            .checked_mul(self.key_dim)
            .and_then(|value| value.checked_mul(self.value_dim))
            .ok_or_else(|| Error::Other("Qwen3.5 GDN state size overflow".into()))
    }
}

/// Device-resident request state for one GDN layer.
///
/// Each operator writes the scratch buffer and reports a device-side finite
/// status. Rust swaps the buffers only after synchronization and a clean
/// status, keeping failed requests transactional.
pub struct Qwen35GdnState {
    layout: Qwen35GdnLayout,
    device_id: usize,
    conv_current: CudaBuffer,
    conv_scratch: CudaBuffer,
    conv_backup: CudaBuffer,
    recurrent_current: CudaBuffer,
    recurrent_scratch: CudaBuffer,
    recurrent_backup: CudaBuffer,
    conv_cursor: usize,
    position: usize,
    conv_commit_pending_rollback: bool,
    recurrent_commit_pending_rollback: bool,
    recurrent_commit_tokens: usize,
    conv_commit_tokens: usize,
}

impl Qwen35GdnState {
    pub fn new(ctx: &CudaContext, layout: Qwen35GdnLayout) -> Result<Self> {
        let conv_bytes = checked_bytes(layout.conv_elements()?, DType::BF16)?;
        let recurrent_bytes = checked_bytes(layout.recurrent_elements()?, DType::F32)?;
        Ok(Self {
            layout,
            device_id: ctx.device_id(),
            conv_current: alloc_zeroed(conv_bytes, ctx.device_id())?,
            conv_scratch: alloc_zeroed(conv_bytes, ctx.device_id())?,
            conv_backup: alloc_zeroed(conv_bytes, ctx.device_id())?,
            recurrent_current: alloc_zeroed(recurrent_bytes, ctx.device_id())?,
            recurrent_scratch: alloc_zeroed(recurrent_bytes, ctx.device_id())?,
            recurrent_backup: alloc_zeroed(recurrent_bytes, ctx.device_id())?,
            conv_cursor: 0,
            position: 0,
            conv_commit_pending_rollback: false,
            recurrent_commit_pending_rollback: false,
            recurrent_commit_tokens: 0,
            conv_commit_tokens: 0,
        })
    }

    /// Test/support constructor that validates a caller-provided recurrent
    /// allocation before taking ownership of it.
    pub fn from_buffers_for_test(
        ctx: &CudaContext,
        layout: Qwen35GdnLayout,
        recurrent_current: CudaBuffer,
    ) -> Result<Self> {
        let expected = checked_bytes(layout.recurrent_elements()?, DType::F32)?;
        if recurrent_current.device() != ctx.device_id() || recurrent_current.len() != expected {
            return Err(Error::Other(format!(
                "Qwen3.5 GDN recurrent state requires {expected} bytes on CUDA{}",
                ctx.device_id()
            )));
        }
        let conv_bytes = checked_bytes(layout.conv_elements()?, DType::BF16)?;
        let recurrent_scratch = alloc_zeroed(expected, ctx.device_id())?;
        let recurrent_backup = alloc_zeroed(expected, ctx.device_id())?;
        Ok(Self {
            layout,
            device_id: ctx.device_id(),
            conv_current: alloc_zeroed(conv_bytes, ctx.device_id())?,
            conv_scratch: alloc_zeroed(conv_bytes, ctx.device_id())?,
            conv_backup: alloc_zeroed(conv_bytes, ctx.device_id())?,
            recurrent_current,
            recurrent_scratch,
            recurrent_backup,
            conv_cursor: 0,
            position: 0,
            conv_commit_pending_rollback: false,
            recurrent_commit_pending_rollback: false,
            recurrent_commit_tokens: 0,
            conv_commit_tokens: 0,
        })
    }

    pub const fn layout(&self) -> Qwen35GdnLayout {
        self.layout
    }

    /// Overwrite this state's buffers with `source`'s contents
    /// (stream-ordered copies; caller synchronizes once per session). The
    /// recycled-session fast path for the prefix cache: reusing a retired
    /// state's allocations avoids ~6 cudaMallocs per layer per fork.
    pub fn copy_state_from(&mut self, source: &Self, ctx: &CudaContext) -> Result<()> {
        if self.layout != source.layout || self.device_id != source.device_id {
            return Err(Error::Other(
                "GDN state refill requires an identical layout and device".into(),
            ));
        }
        let stream = ctx.stream();
        self.conv_current
            .copy_from_device_async(&source.conv_current, stream)
            .map_err(Error::Cuda)?;
        self.conv_scratch
            .copy_from_device_async(&source.conv_scratch, stream)
            .map_err(Error::Cuda)?;
        self.conv_backup
            .copy_from_device_async(&source.conv_backup, stream)
            .map_err(Error::Cuda)?;
        self.recurrent_current
            .copy_from_device_async(&source.recurrent_current, stream)
            .map_err(Error::Cuda)?;
        self.recurrent_scratch
            .copy_from_device_async(&source.recurrent_scratch, stream)
            .map_err(Error::Cuda)?;
        self.recurrent_backup
            .copy_from_device_async(&source.recurrent_backup, stream)
            .map_err(Error::Cuda)?;
        self.conv_cursor = source.conv_cursor;
        self.position = source.position;
        // "Pending rollback" means "the last commit is still revocable" —
        // the normal post-step state, not an error. The backup buffers were
        // copied above, so the clone can honour the same rollback.
        self.conv_commit_pending_rollback = source.conv_commit_pending_rollback;
        self.recurrent_commit_pending_rollback = source.recurrent_commit_pending_rollback;
        self.recurrent_commit_tokens = source.recurrent_commit_tokens;
        self.conv_commit_tokens = source.conv_commit_tokens;
        Ok(())
    }

    /// Deep-clone this state (stream-ordered device-to-device copies of
    /// every buffer plus the host-side cursors). Used by the prefix cache to
    /// fork a session whose prompt state matches a completed prefill; the
    /// clone is bit-identical, so a forked request decodes exactly like the
    /// original. The copies are asynchronous on the context stream — the
    /// caller synchronizes once after cloning the whole session.
    pub fn deep_clone(&self, ctx: &CudaContext) -> Result<Self> {
        let stream = ctx.stream();
        let clone = |buffer: &CudaBuffer| buffer.try_clone_async(stream).map_err(Error::Cuda);
        Ok(Self {
            layout: self.layout,
            device_id: self.device_id,
            conv_current: clone(&self.conv_current)?,
            conv_scratch: clone(&self.conv_scratch)?,
            conv_backup: clone(&self.conv_backup)?,
            recurrent_current: clone(&self.recurrent_current)?,
            recurrent_scratch: clone(&self.recurrent_scratch)?,
            recurrent_backup: clone(&self.recurrent_backup)?,
            conv_cursor: self.conv_cursor,
            position: self.position,
            // The clone can honour the same last-commit rollback as the
            // original: its backup buffers are copies of the same snapshot.
            conv_commit_pending_rollback: self.conv_commit_pending_rollback,
            recurrent_commit_pending_rollback: self.recurrent_commit_pending_rollback,
            recurrent_commit_tokens: self.recurrent_commit_tokens,
            conv_commit_tokens: self.conv_commit_tokens,
        })
    }

    pub const fn device_id(&self) -> usize {
        self.device_id
    }

    pub const fn conv_cursor(&self) -> usize {
        self.conv_cursor
    }

    /// Number of tokens committed through the causal convolution primitive.
    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn recurrent_dtype(&self) -> &'static str {
        "f32"
    }

    pub fn causal_conv_silu(
        &mut self,
        ctx: &CudaContext,
        input: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        self.check_context(ctx)?;
        require_matrix(
            ctx,
            input,
            1,
            self.layout.conv_channels(),
            "GDN convolution input",
        )?;
        if weights.device() != Device::Cuda(ctx.device_id())
            || weights.dtype() != DType::BF16
            || weights.shape().dims() != [self.layout.conv_channels(), 1, self.layout.conv_kernel]
        {
            return Err(Error::Other(format!(
                "GDN convolution weights must be CUDA BF16 [{},1,{}], got {:?} {} on {:?}",
                self.layout.conv_channels(),
                self.layout.conv_kernel,
                weights.shape().dims(),
                weights.dtype(),
                weights.device()
            )));
        }
        let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
        let weight_buffer = CudaBuffer::from_tensor(weights).map_err(Error::Cuda)?;
        let output_bytes = checked_bytes(self.layout.conv_channels(), DType::BF16)?;
        let output = alloc_zeroed(output_bytes, ctx.device_id())?;
        let flags = StatusFlags::acquire(ctx)?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_qwen35_gdn_conv_bf16(
                self.conv_current.ptr(),
                self.conv_scratch.ptr(),
                input_buffer.ptr(),
                weight_buffer.ptr(),
                output.ptr(),
                flags.ptr(),
                checked_i32(self.layout.conv_channels(), "GDN channels")?,
                checked_i32(self.layout.conv_kernel, "GDN kernel")?,
                checked_i32(self.conv_cursor, "GDN cursor")?,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        flags.finish(ctx, "GDN convolution")?;
        std::mem::swap(&mut self.conv_current, &mut self.conv_scratch);
        copy_device_buffer(&self.conv_scratch, &self.conv_backup)?;
        self.conv_cursor = (self.conv_cursor + 1) % self.layout.conv_kernel;
        self.position = self
            .position
            .checked_add(1)
            .ok_or_else(|| Error::Other("GDN position overflow".into()))?;
        self.conv_commit_pending_rollback = true;
        self.conv_commit_tokens = 1;
        Ok(output.into_tensor(
            Shape::new(vec![1, self.layout.conv_channels()]),
            DType::BF16,
        ))
    }

    /// Apply causal depthwise convolution to a complete prefill sequence.
    /// Input rows are processed in order with zero-left padding inherited from
    /// the request ring. The final ring is committed atomically after the
    /// device finite-status check succeeds.
    pub fn causal_conv_silu_prefill(
        &mut self,
        ctx: &CudaContext,
        input: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor> {
        self.check_context(ctx)?;
        let dims = input.shape().dims();
        if dims.len() != 2 || dims[1] != self.layout.conv_channels() || dims[0] == 0 {
            return Err(Error::Other(format!(
                "GDN convolution prefill input must be [rows,{}] with rows > 0, got {dims:?}",
                self.layout.conv_channels()
            )));
        }
        if input.device() != Device::Cuda(ctx.device_id()) || input.dtype() != DType::BF16 {
            return Err(Error::Other(format!(
                "GDN convolution prefill input must be CUDA BF16, got {} on {:?}",
                input.dtype(),
                input.device()
            )));
        }
        if weights.device() != Device::Cuda(ctx.device_id())
            || weights.dtype() != DType::BF16
            || weights.shape().dims() != [self.layout.conv_channels(), 1, self.layout.conv_kernel]
        {
            return Err(Error::Other(format!(
                "GDN convolution weights must be CUDA BF16 [{},1,{}], got {:?} {} on {:?}",
                self.layout.conv_channels(),
                self.layout.conv_kernel,
                weights.shape().dims(),
                weights.dtype(),
                weights.device()
            )));
        }
        let rows = dims[0];
        let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
        let weight_buffer = CudaBuffer::from_tensor(weights).map_err(Error::Cuda)?;
        let output_bytes = checked_bytes(
            rows.checked_mul(self.layout.conv_channels())
                .ok_or_else(|| Error::Other("GDN prefill output size overflow".into()))?,
            DType::BF16,
        )?;
        let output = alloc_zeroed(output_bytes, ctx.device_id())?;
        let flags = StatusFlags::acquire(ctx)?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_qwen35_gdn_conv_prefill_bf16(
                self.conv_current.ptr(),
                self.conv_scratch.ptr(),
                input_buffer.ptr(),
                weight_buffer.ptr(),
                output.ptr(),
                flags.ptr(),
                checked_i32(rows, "GDN prefill rows")?,
                checked_i32(self.layout.conv_channels(), "GDN channels")?,
                checked_i32(self.layout.conv_kernel, "GDN kernel")?,
                checked_i32(self.conv_cursor, "GDN cursor")?,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        flags.finish(ctx, "GDN convolution prefill")?;
        std::mem::swap(&mut self.conv_current, &mut self.conv_scratch);
        copy_device_buffer(&self.conv_scratch, &self.conv_backup)?;
        self.conv_cursor = (self.conv_cursor + rows) % self.layout.conv_kernel;
        self.position = self
            .position
            .checked_add(rows)
            .ok_or_else(|| Error::Other("GDN position overflow".into()))?;
        self.conv_commit_pending_rollback = true;
        self.conv_commit_tokens = rows;
        Ok(output.into_tensor(
            Shape::new(vec![rows, self.layout.conv_channels()]),
            DType::BF16,
        ))
    }

    pub fn gated_delta_step(
        &mut self,
        ctx: &CudaContext,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Tensor> {
        self.check_context(ctx)?;
        require_matrix(ctx, query, 1, self.layout.query_width(), "GDN query")?;
        require_matrix(ctx, key, 1, self.layout.query_width(), "GDN key")?;
        require_matrix(ctx, value, 1, self.layout.value_width(), "GDN value")?;
        require_vector(ctx, a, self.layout.value_heads, "GDN a")?;
        require_vector(ctx, b, self.layout.value_heads, "GDN b")?;
        require_vector(ctx, a_log, self.layout.value_heads, "GDN A_log")?;
        require_vector(ctx, dt_bias, self.layout.value_heads, "GDN dt_bias")?;

        let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
        let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
        let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
        let a = CudaBuffer::from_tensor(a).map_err(Error::Cuda)?;
        let b = CudaBuffer::from_tensor(b).map_err(Error::Cuda)?;
        let a_log = CudaBuffer::from_tensor(a_log).map_err(Error::Cuda)?;
        let dt_bias = CudaBuffer::from_tensor(dt_bias).map_err(Error::Cuda)?;
        let output = alloc_zeroed(
            checked_bytes(self.layout.value_width(), DType::BF16)?,
            ctx.device_id(),
        )?;
        let flags = StatusFlags::acquire(ctx)?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_qwen35_gdn_recurrent_bf16_f32(
                self.recurrent_current.ptr(),
                self.recurrent_scratch.ptr(),
                query.ptr(),
                key.ptr(),
                value.ptr(),
                a.ptr(),
                b.ptr(),
                a_log.ptr(),
                dt_bias.ptr(),
                output.ptr(),
                flags.ptr(),
                checked_i32(self.layout.key_heads, "GDN key heads")?,
                checked_i32(self.layout.value_heads, "GDN value heads")?,
                checked_i32(self.layout.key_dim, "GDN key dimension")?,
                checked_i32(self.layout.value_dim, "GDN value dimension")?,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        flags.finish(ctx, "GDN recurrent update")?;
        std::mem::swap(&mut self.recurrent_current, &mut self.recurrent_scratch);
        copy_device_buffer(&self.recurrent_scratch, &self.recurrent_backup)?;
        self.recurrent_commit_pending_rollback = true;
        self.recurrent_commit_tokens = 1;
        Ok(output.into_tensor(Shape::new(vec![1, self.layout.value_width()]), DType::BF16))
    }

    /// Apply the gated-delta recurrence to a complete sequence.  Inputs are
    /// BF16 activations; q/k normalization and beta sigmoid are materialized
    /// at the BF16 boundary while the recurrent matrix remains FP32.  The
    /// scratch state is committed only after the device finite-status check.
    pub fn gated_delta_prefill(
        &mut self,
        ctx: &CudaContext,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Tensor> {
        self.check_context(ctx)?;
        let query_dims = query.shape().dims();
        if query_dims.len() != 2 || query_dims[1] != self.layout.query_width() || query_dims[0] == 0
        {
            return Err(Error::Other(format!(
                "GDN query prefill must be [rows,{}] with rows > 0, got {query_dims:?}",
                self.layout.query_width()
            )));
        }
        let rows = query_dims[0];
        require_matrix(ctx, key, rows, self.layout.query_width(), "GDN key prefill")?;
        require_matrix(
            ctx,
            value,
            rows,
            self.layout.value_width(),
            "GDN value prefill",
        )?;
        require_matrix(ctx, a, rows, self.layout.value_heads, "GDN a prefill")?;
        require_matrix(ctx, b, rows, self.layout.value_heads, "GDN b prefill")?;
        require_vector(ctx, a_log, self.layout.value_heads, "GDN A_log")?;
        require_vector(ctx, dt_bias, self.layout.value_heads, "GDN dt_bias")?;

        let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
        let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
        let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
        let a = CudaBuffer::from_tensor(a).map_err(Error::Cuda)?;
        let b = CudaBuffer::from_tensor(b).map_err(Error::Cuda)?;
        let a_log = CudaBuffer::from_tensor(a_log).map_err(Error::Cuda)?;
        let dt_bias = CudaBuffer::from_tensor(dt_bias).map_err(Error::Cuda)?;
        let output_elements = rows
            .checked_mul(self.layout.value_width())
            .ok_or_else(|| Error::Other("GDN recurrent prefill output size overflow".into()))?;
        let output = alloc_zeroed(
            checked_bytes(output_elements, DType::BF16)?,
            ctx.device_id(),
        )?;
        let workspace_floats =
            checked_workspace_floats(self.layout.key_dim, self.layout.value_dim)?;
        let workspace_stride = checked_i64(workspace_floats, "GDN workspace stride")?;
        let workspace = alloc_zeroed(
            checked_bytes(
                self.layout
                    .value_heads
                    .checked_mul(workspace_floats)
                    .ok_or_else(|| Error::Other("GDN workspace size overflow".into()))?,
                DType::F32,
            )?,
            ctx.device_id(),
        )?;
        let flags = StatusFlags::acquire(ctx)?;
        let chunk_count = rows
            .checked_add(63)
            .and_then(|value| value.checked_div(64))
            .ok_or_else(|| Error::Other("GDN chunk count overflow".into()))?;
        let qk_elements = self
            .layout
            .value_heads
            .checked_mul(64)
            .and_then(|value| value.checked_mul(64))
            .ok_or_else(|| Error::Other("GDN qk workspace size overflow".into()))?;
        let qk_scores = alloc_zeroed(checked_bytes(qk_elements, DType::F32)?, ctx.device_id())?;
        let transition_scores =
            alloc_zeroed(checked_bytes(qk_elements, DType::F32)?, ctx.device_id())?;
        let workspace_bytes = workspace.len();
        let q_norm = workspace.view(0, workspace_bytes).map_err(Error::Cuda)?;
        let k_norm = workspace
            .view(
                checked_bytes(
                    64usize
                        .checked_mul(self.layout.key_dim)
                        .ok_or_else(|| Error::Other("GDN qk offset overflow".into()))?,
                    DType::F32,
                )?,
                workspace_bytes
                    .checked_sub(checked_bytes(64 * self.layout.key_dim, DType::F32)?)
                    .ok_or_else(|| Error::Other("GDN qk view underflow".into()))?,
            )
            .map_err(Error::Cuda)?;
        let qk_bytes = checked_bytes(
            64usize
                .checked_mul(self.layout.key_dim)
                .ok_or_else(|| Error::Other("GDN k_beta offset overflow".into()))?,
            DType::F32,
        )?;
        let k_beta_offset = qk_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Other("GDN k_beta offset overflow".into()))?;
        let k_beta = workspace
            .view(
                k_beta_offset,
                workspace_bytes
                    .checked_sub(k_beta_offset)
                    .ok_or_else(|| Error::Other("GDN k_beta view underflow".into()))?,
            )
            .map_err(Error::Cuda)?;
        let decayed_k_beta = workspace
            .view(
                qk_bytes
                    .checked_mul(3)
                    .ok_or_else(|| Error::Other("GDN decayed-k offset overflow".into()))?,
                workspace_bytes
                    .checked_sub(
                        qk_bytes
                            .checked_mul(3)
                            .ok_or_else(|| Error::Other("GDN decayed-k offset overflow".into()))?,
                    )
                    .ok_or_else(|| Error::Other("GDN decayed-k view underflow".into()))?,
            )
            .map_err(Error::Cuda)?;
        let v_beta_offset = qk_bytes
            .checked_mul(4)
            .ok_or_else(|| Error::Other("GDN v_beta offset overflow".into()))?;
        let v_beta_bytes = checked_bytes(
            64usize
                .checked_mul(self.layout.value_dim)
                .ok_or_else(|| Error::Other("GDN v_beta size overflow".into()))?,
            DType::F32,
        )?;
        let v_beta = workspace
            .view(
                v_beta_offset,
                workspace_bytes
                    .checked_sub(v_beta_offset)
                    .ok_or_else(|| Error::Other("GDN v_beta view underflow".into()))?,
            )
            .map_err(Error::Cuda)?;
        let attn_offset = v_beta_offset
            .checked_add(v_beta_bytes)
            .and_then(|value| value.checked_add(128 * std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("GDN attn offset overflow".into()))?;
        let attn = workspace
            .view(
                attn_offset,
                64usize
                    .checked_mul(64)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| Error::Other("GDN attn size overflow".into()))?,
            )
            .map_err(Error::Cuda)?;
        let transformed = workspace
            .view(
                attn_offset + 64 * 64 * std::mem::size_of::<f32>(),
                v_beta_bytes,
            )
            .map_err(Error::Cuda)?;
        let k_cumdecay = workspace
            .view(
                attn_offset + 64 * 64 * std::mem::size_of::<f32>() + v_beta_bytes,
                qk_bytes,
            )
            .map_err(Error::Cuda)?;
        let v_new = workspace
            .view(
                attn_offset + 64 * 64 * std::mem::size_of::<f32>() + v_beta_bytes + qk_bytes,
                v_beta_bytes,
            )
            .map_err(Error::Cuda)?;
        let qk_stride = checked_i64(64usize * 64, "GDN qk stride")?;
        let state_stride = checked_i64(
            self.layout
                .key_dim
                .checked_mul(self.layout.value_dim)
                .ok_or_else(|| Error::Other("GDN recurrent state stride overflow".into()))?,
            "GDN recurrent state stride",
        )?;
        for chunk in 0..chunk_count {
            unsafe {
                ffi::check_cuda(ffi::apxinf_qwen35_gdn_sequence_recurrent_bf16_f32(
                    self.recurrent_current.ptr(),
                    self.recurrent_scratch.ptr(),
                    query.ptr(),
                    key.ptr(),
                    value.ptr(),
                    a.ptr(),
                    b.ptr(),
                    a_log.ptr(),
                    dt_bias.ptr(),
                    output.ptr(),
                    flags.ptr(),
                    checked_i32(rows, "GDN recurrent prefill rows")?,
                    checked_i32(self.layout.key_heads, "GDN key heads")?,
                    checked_i32(self.layout.value_heads, "GDN value heads")?,
                    checked_i32(self.layout.key_dim, "GDN key dimension")?,
                    checked_i32(self.layout.value_dim, "GDN value dimension")?,
                    workspace.ptr(),
                    workspace_stride,
                    qk_scores.ptr(),
                    transition_scores.ptr(),
                    checked_i32(chunk, "GDN chunk index")?,
                    0,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)?;
            }
            ctx.cublas()
                .batched_gemm_ex_f32(
                    CublasTranspose::None,
                    CublasTranspose::Transpose,
                    64,
                    64,
                    self.layout.key_dim,
                    1.0,
                    &q_norm,
                    checked_i32(self.layout.key_dim, "GDN qk A leading dimension")?,
                    workspace_stride,
                    &k_norm,
                    checked_i32(self.layout.key_dim, "GDN qk B leading dimension")?,
                    workspace_stride,
                    0.0,
                    &qk_scores,
                    checked_i32(64, "GDN qk output leading dimension")?,
                    qk_stride,
                    checked_i32(self.layout.value_heads, "GDN qk batch count")?,
                )
                .map_err(Error::Cuda)?;
            ctx.cublas()
                .batched_gemm_ex_f32(
                    CublasTranspose::None,
                    CublasTranspose::Transpose,
                    64,
                    64,
                    self.layout.key_dim,
                    1.0,
                    &k_beta,
                    checked_i32(self.layout.key_dim, "GDN transition A leading dimension")?,
                    workspace_stride,
                    &k_norm,
                    checked_i32(self.layout.key_dim, "GDN transition B leading dimension")?,
                    workspace_stride,
                    0.0,
                    &transition_scores,
                    checked_i32(64, "GDN transition output leading dimension")?,
                    qk_stride,
                    checked_i32(self.layout.value_heads, "GDN transition batch count")?,
                )
                .map_err(Error::Cuda)?;
            unsafe {
                ffi::check_cuda(ffi::apxinf_qwen35_gdn_sequence_recurrent_bf16_f32(
                    self.recurrent_current.ptr(),
                    self.recurrent_scratch.ptr(),
                    query.ptr(),
                    key.ptr(),
                    value.ptr(),
                    a.ptr(),
                    b.ptr(),
                    a_log.ptr(),
                    dt_bias.ptr(),
                    output.ptr(),
                    flags.ptr(),
                    checked_i32(rows, "GDN recurrent prefill rows")?,
                    checked_i32(self.layout.key_heads, "GDN key heads")?,
                    checked_i32(self.layout.value_heads, "GDN value heads")?,
                    checked_i32(self.layout.key_dim, "GDN key dimension")?,
                    checked_i32(self.layout.value_dim, "GDN value dimension")?,
                    workspace.ptr(),
                    workspace_stride,
                    qk_scores.ptr(),
                    transition_scores.ptr(),
                    checked_i32(chunk, "GDN chunk index")?,
                    1,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)?;
            }
            ctx.cublas()
                .batched_gemm_ex_f32(
                    CublasTranspose::None,
                    CublasTranspose::None,
                    64,
                    self.layout.value_dim,
                    64,
                    1.0,
                    &attn,
                    64,
                    workspace_stride,
                    &v_beta,
                    checked_i32(self.layout.value_dim, "GDN v_beta leading dimension")?,
                    workspace_stride,
                    0.0,
                    &transformed,
                    checked_i32(self.layout.value_dim, "GDN transformed leading dimension")?,
                    workspace_stride,
                    checked_i32(self.layout.value_heads, "GDN transformed batch count")?,
                )
                .map_err(Error::Cuda)?;
            ctx.cublas()
                .batched_gemm_ex_f32(
                    CublasTranspose::None,
                    CublasTranspose::None,
                    64,
                    self.layout.key_dim,
                    64,
                    1.0,
                    &attn,
                    64,
                    workspace_stride,
                    &decayed_k_beta,
                    checked_i32(self.layout.key_dim, "GDN decayed-k leading dimension")?,
                    workspace_stride,
                    0.0,
                    &k_cumdecay,
                    checked_i32(self.layout.key_dim, "GDN k-cumdecay leading dimension")?,
                    workspace_stride,
                    checked_i32(self.layout.value_heads, "GDN k-cumdecay batch count")?,
                )
                .map_err(Error::Cuda)?;
            unsafe {
                ffi::check_cuda(ffi::apxinf_qwen35_gdn_sequence_recurrent_bf16_f32(
                    self.recurrent_current.ptr(),
                    self.recurrent_scratch.ptr(),
                    query.ptr(),
                    key.ptr(),
                    value.ptr(),
                    a.ptr(),
                    b.ptr(),
                    a_log.ptr(),
                    dt_bias.ptr(),
                    output.ptr(),
                    flags.ptr(),
                    checked_i32(rows, "GDN recurrent prefill rows")?,
                    checked_i32(self.layout.key_heads, "GDN key heads")?,
                    checked_i32(self.layout.value_heads, "GDN value heads")?,
                    checked_i32(self.layout.key_dim, "GDN key dimension")?,
                    checked_i32(self.layout.value_dim, "GDN value dimension")?,
                    workspace.ptr(),
                    workspace_stride,
                    qk_scores.ptr(),
                    transition_scores.ptr(),
                    checked_i32(chunk, "GDN chunk index")?,
                    2,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)?;
            }
            ctx.cublas()
                .batched_gemm_ex_f32(
                    CublasTranspose::Transpose,
                    CublasTranspose::None,
                    self.layout.key_dim,
                    self.layout.value_dim,
                    64,
                    1.0,
                    &k_cumdecay,
                    checked_i32(self.layout.key_dim, "GDN state-update A leading dimension")?,
                    workspace_stride,
                    &v_new,
                    checked_i32(
                        self.layout.value_dim,
                        "GDN state-update B leading dimension",
                    )?,
                    workspace_stride,
                    1.0,
                    &self.recurrent_scratch,
                    checked_i32(self.layout.value_dim, "GDN state-update leading dimension")?,
                    state_stride,
                    checked_i32(self.layout.value_heads, "GDN state-update batch count")?,
                )
                .map_err(Error::Cuda)?;
        }
        flags.finish(ctx, "GDN recurrent prefill")?;
        std::mem::swap(&mut self.recurrent_current, &mut self.recurrent_scratch);
        copy_device_buffer(&self.recurrent_scratch, &self.recurrent_backup)?;
        self.recurrent_commit_pending_rollback = true;
        self.recurrent_commit_tokens = rows;
        Ok(output.into_tensor(
            Shape::new(vec![rows, self.layout.value_width()]),
            DType::BF16,
        ))
    }

    /// Undo the most recently committed causal convolution step. Callers use
    /// this when a later operator in the same token fails.
    pub fn rollback_last_convolution(&mut self) -> Result<()> {
        if !self.conv_commit_pending_rollback {
            return Err(Error::Other(
                "Qwen3.5 GDN has no convolution commit to roll back".into(),
            ));
        }
        self.position = self
            .position
            .checked_sub(self.conv_commit_tokens)
            .ok_or_else(|| Error::Other("Qwen3.5 GDN convolution position underflow".into()))?;
        std::mem::swap(&mut self.conv_current, &mut self.conv_backup);
        self.conv_cursor = (self.conv_cursor + self.layout.conv_kernel
            - (self.conv_commit_tokens % self.layout.conv_kernel))
            % self.layout.conv_kernel;
        self.conv_commit_pending_rollback = false;
        self.conv_commit_tokens = 0;
        Ok(())
    }

    /// Undo the most recently committed FP32 recurrent update. The prior
    /// matrix remains in the alternating scratch allocation until the next
    /// successful recurrent step.
    pub fn rollback_last_recurrent(&mut self) -> Result<()> {
        if !self.recurrent_commit_pending_rollback {
            return Err(Error::Other(
                "Qwen3.5 GDN has no recurrent commit to roll back".into(),
            ));
        }
        std::mem::swap(&mut self.recurrent_current, &mut self.recurrent_backup);
        self.recurrent_commit_pending_rollback = false;
        self.recurrent_commit_tokens = 0;
        Ok(())
    }

    pub fn recurrent_host(&self, ctx: &CudaContext) -> Result<Vec<f32>> {
        self.check_context(ctx)?;
        let mut bytes = vec![0u8; self.recurrent_current.len()];
        self.recurrent_current
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    pub fn conv_ring_channel_host(&self, ctx: &CudaContext, channel: usize) -> Result<Vec<f32>> {
        self.check_context(ctx)?;
        if channel >= self.layout.conv_channels() {
            return Err(Error::Other("GDN convolution channel out of range".into()));
        }
        let mut bytes = vec![0u8; self.conv_current.len()];
        self.conv_current
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        let values = bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>();
        let base = channel * self.layout.conv_kernel;
        Ok((0..self.layout.conv_kernel)
            .map(|offset| values[base + (self.conv_cursor + offset) % self.layout.conv_kernel])
            .collect())
    }

    pub fn reset(&mut self, ctx: &CudaContext) -> Result<()> {
        self.check_context(ctx)?;
        unsafe {
            ffi::check_cuda(ffi::cudaMemset(
                self.conv_current.ptr(),
                0,
                self.conv_current.len(),
            ))
            .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaMemset(
                self.conv_scratch.ptr(),
                0,
                self.conv_scratch.len(),
            ))
            .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaMemset(
                self.conv_backup.ptr(),
                0,
                self.conv_backup.len(),
            ))
            .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaMemset(
                self.recurrent_current.ptr(),
                0,
                self.recurrent_current.len(),
            ))
            .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaMemset(
                self.recurrent_scratch.ptr(),
                0,
                self.recurrent_scratch.len(),
            ))
            .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaMemset(
                self.recurrent_backup.ptr(),
                0,
                self.recurrent_backup.len(),
            ))
            .map_err(Error::Cuda)?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        self.conv_cursor = 0;
        self.position = 0;
        self.conv_commit_pending_rollback = false;
        self.recurrent_commit_pending_rollback = false;
        self.recurrent_commit_tokens = 0;
        self.conv_commit_tokens = 0;
        Ok(())
    }

    fn check_context(&self, ctx: &CudaContext) -> Result<()> {
        if self.device_id != ctx.device_id() {
            return Err(Error::DeviceMismatch {
                expected: Device::Cuda(self.device_id),
                got: Device::Cuda(ctx.device_id()),
            });
        }
        Ok(())
    }
}

pub fn gated_rms_norm_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Tensor> {
    if heads == 0 || head_dim == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Other(
            "GDN gated RMSNorm dimensions/epsilon are invalid".into(),
        ));
    }
    let width = heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("GDN gated RMSNorm width overflow".into()))?;
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[1] != width {
        return Err(Error::Other(format!(
            "GDN gated RMSNorm input must be [rows,{width}], got {dims:?}"
        )));
    }
    let rows = dims[0];
    if gate.shape() != input.shape()
        || gate.dtype() != DType::BF16
        || gate.device() != Device::Cuda(ctx.device_id())
        || weight.dtype() != DType::BF16
        || weight.device() != Device::Cuda(ctx.device_id())
        || weight.shape().dims() != [head_dim]
        || input.dtype() != DType::BF16
        || input.device() != Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "GDN gated RMSNorm tensor layout mismatch".into(),
        ));
    }
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output = alloc_zeroed(checked_bytes(rows * width, DType::BF16)?, ctx.device_id())?;
    let flags = StatusFlags::acquire(ctx)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_gdn_gated_rms_norm_bf16(
            input.ptr(),
            gate.ptr(),
            weight.ptr(),
            output.ptr(),
            flags.ptr(),
            checked_i32(rows, "GDN RMSNorm rows")?,
            checked_i32(heads, "GDN RMSNorm heads")?,
            checked_i32(head_dim, "GDN RMSNorm head dimension")?,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    flags.finish(ctx, "GDN gated RMSNorm")?;
    Ok(output.into_tensor(Shape::new(vec![rows, width]), DType::BF16))
}

fn require_matrix(
    ctx: &CudaContext,
    tensor: &Tensor,
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<()> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != DType::BF16
        || tensor.shape().dims() != [rows, cols]
    {
        return Err(Error::Other(format!(
            "{name} must be CUDA BF16 [{rows},{cols}], got {:?} {} on {:?}",
            tensor.shape().dims(),
            tensor.dtype(),
            tensor.device()
        )));
    }
    Ok(())
}

fn require_vector(ctx: &CudaContext, tensor: &Tensor, length: usize, name: &str) -> Result<()> {
    let valid_shape = tensor.shape().dims() == [length] || tensor.shape().dims() == [1, length];
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != DType::BF16
        || !valid_shape
    {
        return Err(Error::Other(format!(
            "{name} must be CUDA BF16 [{length}] or [1,{length}], got {:?} {} on {:?}",
            tensor.shape().dims(),
            tensor.dtype(),
            tensor.device()
        )));
    }
    Ok(())
}

/// Upload a small host array of device pointers/cursors for a batched
/// launch. Synchronous H2D by design: the copy also orders the host write
/// before the kernel that consumes it.
fn upload_control<T: Copy>(ctx: &CudaContext, values: &[T]) -> Result<CudaBuffer> {
    let bytes = std::mem::size_of_val(values);
    let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let raw = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, bytes) };
    buffer.copy_from_host(raw).map_err(Error::Cuda)?;
    Ok(buffer)
}

/// Batched single-token causal convolution across independent request
/// states: one `blockIdx.y` per request, each with its own ring buffers.
/// Per-request math is bit-identical to `causal_conv_silu` (same kernel
/// body), and each state's swap/backup/cursor commit matches the serial
/// method.
pub fn causal_conv_silu_batch(
    ctx: &CudaContext,
    states: &mut [&mut Qwen35GdnState],
    input: &Tensor,
    weights: &Tensor,
) -> Result<Tensor> {
    let batch = states.len();
    if batch == 0 {
        return Err(Error::Other("batched GDN convolution needs states".into()));
    }
    let layout = states[0].layout;
    for state in states.iter() {
        state.check_context(ctx)?;
        if state.layout != layout {
            return Err(Error::Other("batched GDN convolution layout mismatch".into()));
        }
    }
    require_matrix(
        ctx,
        input,
        batch,
        layout.conv_channels(),
        "batched GDN convolution input",
    )?;
    if weights.device() != Device::Cuda(ctx.device_id())
        || weights.dtype() != DType::BF16
        || weights.shape().dims() != [layout.conv_channels(), 1, layout.conv_kernel]
    {
        return Err(Error::Other(
            "batched GDN convolution weights have the wrong shape".into(),
        ));
    }
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weights).map_err(Error::Cuda)?;

    let ring_in: Vec<u64> = states
        .iter()
        .map(|state| state.conv_current.ptr() as u64)
        .collect();
    let ring_out: Vec<u64> = states
        .iter()
        .map(|state| state.conv_scratch.ptr() as u64)
        .collect();
    let cursors: Vec<i32> = states
        .iter()
        .map(|state| state.conv_cursor as i32)
        .collect();
    let ring_in_buf = upload_control(ctx, &ring_in)?;
    let ring_out_buf = upload_control(ctx, &ring_out)?;
    let cursor_buf = upload_control(ctx, &cursors)?;

    let output = alloc_zeroed(
        checked_bytes(
            batch
                .checked_mul(layout.conv_channels())
                .ok_or_else(|| Error::Other("batched GDN conv output overflow".into()))?,
            DType::BF16,
        )?,
        ctx.device_id(),
    )?;
    let flags = StatusFlags::acquire(ctx)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_gdn_conv_batch_bf16(
            ring_in_buf.ptr(),
            ring_out_buf.ptr(),
            input_buffer.ptr(),
            weight_buffer.ptr(),
            output.ptr(),
            cursor_buf.ptr(),
            flags.ptr(),
            checked_i32(batch, "GDN batch")?,
            checked_i32(layout.conv_channels(), "GDN channels")?,
            checked_i32(layout.conv_kernel, "GDN kernel")?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    flags.finish(ctx, "batched GDN convolution")?;
    let stream = ctx.stream();
    for state in states.iter_mut() {
        std::mem::swap(&mut state.conv_current, &mut state.conv_scratch);
        state
            .conv_backup
            .copy_from_device_async(&state.conv_scratch, stream)
            .map_err(Error::Cuda)?;
        state.conv_cursor = (state.conv_cursor + 1) % state.layout.conv_kernel;
        state.position = state
            .position
            .checked_add(1)
            .ok_or_else(|| Error::Other("GDN position overflow".into()))?;
        state.conv_commit_pending_rollback = true;
        state.conv_commit_tokens = 1;
    }
    Ok(output.into_tensor(
        Shape::new(vec![batch, layout.conv_channels()]),
        DType::BF16,
    ))
}

/// Batched single-token gated-delta update across independent request
/// states. Per-request math is bit-identical to `gated_delta_step`.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_step_batch(
    ctx: &CudaContext,
    states: &mut [&mut Qwen35GdnState],
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<Tensor> {
    let batch = states.len();
    if batch == 0 {
        return Err(Error::Other("batched GDN recurrence needs states".into()));
    }
    let layout = states[0].layout;
    for state in states.iter() {
        state.check_context(ctx)?;
        if state.layout != layout {
            return Err(Error::Other("batched GDN recurrence layout mismatch".into()));
        }
    }
    require_matrix(ctx, query, batch, layout.query_width(), "batched GDN query")?;
    require_matrix(ctx, key, batch, layout.query_width(), "batched GDN key")?;
    require_matrix(ctx, value, batch, layout.value_width(), "batched GDN value")?;
    require_matrix(ctx, a, batch, layout.value_heads, "batched GDN a")?;
    require_matrix(ctx, b, batch, layout.value_heads, "batched GDN b")?;
    require_vector(ctx, a_log, layout.value_heads, "GDN A_log")?;
    require_vector(ctx, dt_bias, layout.value_heads, "GDN dt_bias")?;

    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
    let a = CudaBuffer::from_tensor(a).map_err(Error::Cuda)?;
    let b = CudaBuffer::from_tensor(b).map_err(Error::Cuda)?;
    let a_log = CudaBuffer::from_tensor(a_log).map_err(Error::Cuda)?;
    let dt_bias = CudaBuffer::from_tensor(dt_bias).map_err(Error::Cuda)?;

    let state_in: Vec<u64> = states
        .iter()
        .map(|state| state.recurrent_current.ptr() as u64)
        .collect();
    let state_out: Vec<u64> = states
        .iter()
        .map(|state| state.recurrent_scratch.ptr() as u64)
        .collect();
    let state_in_buf = upload_control(ctx, &state_in)?;
    let state_out_buf = upload_control(ctx, &state_out)?;

    let output = alloc_zeroed(
        checked_bytes(
            batch
                .checked_mul(layout.value_width())
                .ok_or_else(|| Error::Other("batched GDN output overflow".into()))?,
            DType::BF16,
        )?,
        ctx.device_id(),
    )?;
    let flags = StatusFlags::acquire(ctx)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_gdn_recurrent_batch_bf16_f32(
            state_in_buf.ptr(),
            state_out_buf.ptr(),
            query.ptr(),
            key.ptr(),
            value.ptr(),
            a.ptr(),
            b.ptr(),
            a_log.ptr(),
            dt_bias.ptr(),
            output.ptr(),
            flags.ptr(),
            checked_i32(batch, "GDN batch")?,
            checked_i32(layout.key_heads, "GDN key heads")?,
            checked_i32(layout.value_heads, "GDN value heads")?,
            checked_i32(layout.key_dim, "GDN key dimension")?,
            checked_i32(layout.value_dim, "GDN value dimension")?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    flags.finish(ctx, "batched GDN recurrent update")?;
    let stream = ctx.stream();
    for state in states.iter_mut() {
        std::mem::swap(&mut state.recurrent_current, &mut state.recurrent_scratch);
        state
            .recurrent_backup
            .copy_from_device_async(&state.recurrent_scratch, stream)
            .map_err(Error::Cuda)?;
        state.recurrent_commit_pending_rollback = true;
        state.recurrent_commit_tokens = 1;
    }
    Ok(output.into_tensor(
        Shape::new(vec![batch, layout.value_width()]),
        DType::BF16,
    ))
}

fn checked_bytes(elements: usize, dtype: DType) -> Result<usize> {
    elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("Qwen3.5 GDN byte size overflow".into()))
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Other(format!("{name} exceeds CUDA i32 range")))
}

fn checked_i64(value: usize, name: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::Other(format!("{name} exceeds CUDA i64 range")))
}

fn checked_workspace_floats(key_dim: usize, value_dim: usize) -> Result<usize> {
    const CHUNK: usize = 64;
    let qk = CHUNK
        .checked_mul(key_dim)
        .ok_or_else(|| Error::Other("GDN workspace size overflow".into()))?;
    let values = CHUNK
        .checked_mul(value_dim)
        .ok_or_else(|| Error::Other("GDN workspace size overflow".into()))?;
    let matrix = CHUNK
        .checked_mul(CHUNK)
        .ok_or_else(|| Error::Other("GDN workspace size overflow".into()))?;
    qk.checked_mul(4)
        .and_then(|size| size.checked_add(values))
        .and_then(|size| size.checked_add(CHUNK * 2))
        .and_then(|size| size.checked_add(matrix))
        .and_then(|size| size.checked_add(values))
        .and_then(|size| size.checked_add(qk))
        .and_then(|size| size.checked_add(values))
        .ok_or_else(|| Error::Other("GDN workspace size overflow".into()))
}

fn alloc_zeroed(bytes: usize, device: usize) -> Result<CudaBuffer> {
    CudaBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)
}

fn copy_device_buffer(source: &CudaBuffer, destination: &CudaBuffer) -> Result<()> {
    if source.device() != destination.device() || source.len() != destination.len() {
        return Err(Error::Other(
            "Qwen3.5 GDN device buffer copy mismatch".into(),
        ));
    }
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpy(
            destination.ptr(),
            source.ptr(),
            source.len(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        ))
        .map_err(Error::Cuda)
    }
}

fn read_status(ctx: &CudaContext, flags: &CudaBuffer) -> Result<u32> {
    ctx.synchronize().map_err(Error::Cuda)?;
    let mut bytes = [0u8; size_of::<u32>()];
    flags.copy_to_host(&mut bytes).map_err(Error::Cuda)?;
    Ok(u32::from_le_bytes(bytes))
}

fn status_error(operation: &str, flag: u32) -> Error {
    if flag & FLAG_NON_FINITE_INPUT != 0 {
        Error::Other(format!("{operation} received a non-finite input or state"))
    } else if flag & FLAG_NON_FINITE_OUTPUT != 0 {
        Error::Other(format!("{operation} produced a non-finite output"))
    } else {
        Error::Other(format!("{operation} failed with CUDA status flag {flag}"))
    }
}

//! Fail-closed Qwen3.5 asymmetric packed-W4 reference projection.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::output_tensor;
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::cublas::CublasTranspose;
use crate::ffi;
use crate::workspace::output_buffer;

const REQUIRED_GROUP_SIZE: usize = 32;
const FLAG_NON_FINITE_SCALE: u32 = 1;
const FLAG_NON_FINITE_OUTPUT: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35W4Layout {
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
}

impl Qwen35W4Layout {
    pub fn new(out_features: usize, in_features: usize, group_size: usize) -> Result<Self> {
        if out_features == 0 || in_features == 0 {
            return Err(Error::Other(
                "Qwen3.5 W4 projection dimensions must be non-zero".into(),
            ));
        }
        if group_size != REQUIRED_GROUP_SIZE {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 group size must be {REQUIRED_GROUP_SIZE}, got {group_size}"
            )));
        }
        Ok(Self {
            out_features,
            in_features,
            group_size,
        })
    }

    pub const fn groups(self) -> usize {
        self.in_features.div_ceil(self.group_size)
    }

    pub const fn packed_k_columns(self) -> usize {
        self.in_features.div_ceil(8)
    }

    pub const fn packed_n_rows(self) -> usize {
        self.out_features.div_ceil(8)
    }

    fn weight_bytes(self) -> Result<usize> {
        checked_product(
            &[self.out_features, self.packed_k_columns(), size_of::<u32>()],
            "packed weight",
        )
    }

    fn zero_point_bytes(self) -> Result<usize> {
        checked_product(
            &[self.packed_n_rows(), self.groups(), size_of::<u32>()],
            "packed zero-point",
        )
    }
}

pub struct Qwen35W4Buffers<'a> {
    pub weight_packed: &'a CudaBuffer,
    pub scales: &'a Tensor,
    pub zero_points: &'a CudaBuffer,
}

/// Owns the device-side buffers for one bounded W4 projection.
///
/// Packed weights and zero-points remain raw I32-compatible bytes because the
/// core tensor dtype set intentionally has no I32 variant. Scales are uploaded
/// as native BF16 and are never expanded to a full-precision model copy.
pub struct Qwen35W4DeviceProjection {
    layout: Qwen35W4Layout,
    weight_packed: CudaBuffer,
    scales: Tensor,
    zero_points: CudaBuffer,
}

impl Qwen35W4DeviceProjection {
    pub fn upload(
        ctx: &CudaContext,
        layout: Qwen35W4Layout,
        weight_packed: &[u8],
        scales_cpu: &Tensor,
        zero_points: &[u8],
    ) -> Result<Self> {
        let expected_weight_bytes = layout.weight_bytes()?;
        let expected_zero_point_bytes = layout.zero_point_bytes()?;
        if weight_packed.len() != expected_weight_bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 packed weight requires {expected_weight_bytes} bytes, got {}",
                weight_packed.len()
            )));
        }
        if zero_points.len() != expected_zero_point_bytes {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 zero-point requires {expected_zero_point_bytes} bytes, got {}",
                zero_points.len()
            )));
        }
        let expected_scale_shape = [layout.out_features, layout.groups()];
        if scales_cpu.device() != Device::Cpu
            || scales_cpu.dtype() != DType::BF16
            || scales_cpu.shape().dims() != expected_scale_shape
        {
            return Err(Error::Other(format!(
                "Qwen3.5 W4 CPU scales must be BF16 {:?}, got {:?} {} on {:?}",
                expected_scale_shape,
                scales_cpu.shape().dims(),
                scales_cpu.dtype(),
                scales_cpu.device()
            )));
        }
        let weight_buffer =
            CudaBuffer::alloc(expected_weight_bytes, ctx.device_id()).map_err(Error::Cuda)?;
        weight_buffer
            .copy_from_host(weight_packed)
            .map_err(Error::Cuda)?;
        let zero_point_buffer =
            CudaBuffer::alloc(expected_zero_point_bytes, ctx.device_id()).map_err(Error::Cuda)?;
        zero_point_buffer
            .copy_from_host(zero_points)
            .map_err(Error::Cuda)?;
        let scales = crate::transfers::to_cuda(scales_cpu, ctx.device_id())?;
        Ok(Self {
            layout,
            weight_packed: weight_buffer,
            scales,
            zero_points: zero_point_buffer,
        })
    }

    pub const fn layout(&self) -> Qwen35W4Layout {
        self.layout
    }

    pub fn project(&self, ctx: &CudaContext, activation: &Tensor) -> Result<Tensor> {
        project_bf16(
            ctx,
            activation,
            Qwen35W4Buffers {
                weight_packed: &self.weight_packed,
                scales: &self.scales,
                zero_points: &self.zero_points,
            },
            self.layout,
        )
    }
}

pub fn project_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    buffers: Qwen35W4Buffers<'_>,
    layout: Qwen35W4Layout,
) -> Result<Tensor> {
    let dims = activation.shape().dims();
    if activation.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: activation.dtype(),
        });
    }
    if dims.len() != 2 || dims[0] == 0 || dims[1] != layout.in_features {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 activation shape must be [rows, {}], got {dims:?}",
            layout.in_features
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: activation.device(),
        });
    }
    let expected_scale_shape = [layout.out_features, layout.groups()];
    if buffers.scales.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: buffers.scales.dtype(),
        });
    }
    if buffers.scales.shape().dims() != expected_scale_shape {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 scale shape must be {expected_scale_shape:?}, got {:?}",
            buffers.scales.shape().dims()
        )));
    }
    if buffers.scales.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: buffers.scales.device(),
        });
    }
    let rows = dims[0];
    let activation_bytes = checked_product(
        &[rows, layout.in_features, DType::BF16.size_in_bytes()],
        "activation",
    )?;
    let activation_buffer = require_tensor_buffer(ctx, "activation", activation, activation_bytes)?;
    let scale_bytes = checked_product(
        &[
            layout.out_features,
            layout.groups(),
            DType::BF16.size_in_bytes(),
        ],
        "scale",
    )?;
    let scale_buffer = require_tensor_buffer(ctx, "scale", buffers.scales, scale_bytes)?;
    require_exact_buffer(
        ctx,
        "packed weight",
        buffers.weight_packed,
        layout.weight_bytes()?,
    )?;
    require_exact_buffer(
        ctx,
        "N-packed zero-point",
        buffers.zero_points,
        layout.zero_point_bytes()?,
    )?;

    // Large-M path: the per-output GEMV kernel re-reads the whole packed
    // weight once per activation row, so an M-row prefill block costs M times
    // the weight traffic of a single decode step and extracts no batching
    // benefit. Dequantize once into a BF16 scratch matrix and let the
    // tensor-core GEMM reuse it across all rows instead.
    if rows >= dequantize_row_threshold() {
        return project_bf16_via_dequantized_gemm(ctx, activation, buffers, layout, rows);
    }

    let output_bytes = checked_product(
        &[rows, layout.out_features, DType::BF16.size_in_bytes()],
        "output",
    )?;
    let output = output_buffer(ctx, output_bytes)?;
    let flags = CudaBuffer::alloc_zeros(size_of::<u32>(), ctx.device_id()).map_err(Error::Cuda)?;
    let rows_i32 = checked_i32(rows, "rows")?;
    let out_i32 = checked_i32(layout.out_features, "out_features")?;
    let in_i32 = checked_i32(layout.in_features, "in_features")?;
    let group_i32 = checked_i32(layout.group_size, "group_size")?;
    checked_i32(
        rows.checked_mul(layout.out_features)
            .ok_or_else(|| Error::Other("Qwen3.5 W4 CUDA grid size overflow".into()))?,
        "CUDA grid blocks",
    )?;

    // Diagnostic kernels (measurement only; variants 1 and 2 are numerically
    // wrong by construction and exist solely to attribute cost).
    if let Some(variant) = diag_kernel_variant() {
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_qwen35_w4_project_bf16_diag(
                activation_buffer.ptr(),
                buffers.weight_packed.ptr(),
                scale_buffer.ptr(),
                buffers.zero_points.ptr(),
                output.ptr(),
                flags.ptr(),
                rows_i32,
                out_i32,
                in_i32,
                group_i32,
                variant,
                ctx.stream().handle(),
            ))
            .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 diag launch failed: {error}")))?;
        }
    } else {
        // Kernel selection for decode-shaped calls. The warp-per-output
        // kernel needs the activation row in shared memory (4 bytes per
        // in_feature), so it only applies below the 48 KB budget; wider
        // shapes (e.g. the MLP down projection at 17408) fall back.
        let warp_fits_shared = layout.in_features * size_of::<f32>() <= 48 * 1024;
        let launch = if warp_gemv_enabled() && warp_fits_shared {
            ffi::apxinf_static_qwen35_w4_project_bf16_warp
        } else if packed_gemv_enabled() {
            ffi::apxinf_static_qwen35_w4_project_bf16_packed
        } else {
            ffi::apxinf_static_qwen35_w4_project_bf16
        };
        unsafe {
            ffi::check_cuda(launch(
                activation_buffer.ptr(),
                buffers.weight_packed.ptr(),
                scale_buffer.ptr(),
                buffers.zero_points.ptr(),
                output.ptr(),
                flags.ptr(),
                rows_i32,
                out_i32,
                in_i32,
                group_i32,
                ctx.stream().handle(),
            ))
            .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 launch failed: {error}")))?;
        }
    }
    ctx.synchronize()
        .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 synchronize failed: {error}")))?;
    let mut flag_bytes = [0u8; size_of::<u32>()];
    flags
        .copy_to_host(&mut flag_bytes)
        .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 status copy failed: {error}")))?;
    let flag = u32::from_le_bytes(flag_bytes);
    if flag & FLAG_NON_FINITE_SCALE != 0 {
        return Err(Error::Other(
            "Qwen3.5 W4 projection encountered a non-finite W4 scale".into(),
        ));
    }
    if flag & FLAG_NON_FINITE_OUTPUT != 0 {
        return Err(Error::Other(
            "Qwen3.5 W4 projection produced a non-finite W4 output".into(),
        ));
    }
    Ok(output_tensor(
        ctx,
        Shape::new(vec![rows, layout.out_features]),
        DType::BF16,
        output,
    ))
}

/// Reusable device scratch for W4 dequantization. `cudaMalloc`+`cudaFree` of
/// a 178 MB scratch costs ~3.2 ms on this host — measured at 82% of a whole
/// MLP projection — so per-call allocation dominated prefill. The pool keeps
/// at most one buffer per (device, byte-size) class; the dequantize kernel
/// overwrites every element, so returned buffers need no zeroing. Disabled
/// with `APXINF_Q35_SCRATCH_POOL=0` as the paired A/B control.
mod scratch_pool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use super::{CudaBuffer, Error, Result};

    fn pool() -> &'static Mutex<HashMap<(usize, usize), Vec<CudaBuffer>>> {
        static POOL: OnceLock<Mutex<HashMap<(usize, usize), Vec<CudaBuffer>>>> = OnceLock::new();
        POOL.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn enabled() -> bool {
        !matches!(std::env::var("APXINF_Q35_SCRATCH_POOL").as_deref(), Ok("0"))
    }

    /// A pooled device buffer that returns to the pool on drop.
    pub(super) struct PooledBuffer {
        buffer: Option<CudaBuffer>,
        key: (usize, usize),
        pooled: bool,
    }

    impl PooledBuffer {
        pub(super) fn ptr(&self) -> *mut std::ffi::c_void {
            self.buffer.as_ref().expect("pooled buffer present").ptr()
        }

        pub(super) fn buffer(&self) -> &CudaBuffer {
            self.buffer.as_ref().expect("pooled buffer present")
        }

        /// Permanently take the buffer out of pool circulation (used when the
        /// caller must keep it alive beyond the projection call).
        pub(super) fn into_inner(mut self) -> CudaBuffer {
            self.pooled = false;
            self.buffer.take().expect("pooled buffer present")
        }
    }

    impl Drop for PooledBuffer {
        fn drop(&mut self) {
            if !self.pooled {
                return;
            }
            if let Some(buffer) = self.buffer.take() {
                if let Ok(mut guard) = pool().lock() {
                    let slot = guard.entry(self.key).or_default();
                    // One resident buffer per size class bounds pool growth to
                    // the distinct projection shapes (~765 MB worst case).
                    if slot.is_empty() {
                        slot.push(buffer);
                    }
                }
            }
        }
    }

    pub(super) fn acquire(device: usize, bytes: usize) -> Result<PooledBuffer> {
        let key = (device, bytes);
        let pooled = enabled();
        if pooled {
            if let Ok(mut guard) = pool().lock() {
                if let Some(buffer) = guard.get_mut(&key).and_then(Vec::pop) {
                    return Ok(PooledBuffer {
                        buffer: Some(buffer),
                        key,
                        pooled,
                    });
                }
            }
        }
        Ok(PooledBuffer {
            buffer: Some(CudaBuffer::alloc(bytes, device).map_err(Error::Cuda)?),
            key,
            pooled,
        })
    }
}

/// Selects the bandwidth-oriented packed-W4 GEMV kernel (one packed uint32 per
/// thread, warp-shuffle reduction) over the baseline one-K-per-thread kernel.
/// Default on; `APXINF_Q35_W4_PACKED_GEMV=0` is the paired A/B control.
fn packed_gemv_enabled() -> bool {
    !matches!(
        std::env::var("APXINF_Q35_W4_PACKED_GEMV").as_deref(),
        Ok("0")
    )
}

/// Diagnostic kernel selector, measurement only. `1` drops dequantization
/// arithmetic and `2` drops the activation entirely, so both produce WRONG
/// results and exist to attribute cost; `3` is the vectorized-load candidate
/// with production-equivalent arithmetic. Unset in every normal run.
fn diag_kernel_variant() -> Option<i32> {
    std::env::var("APXINF_Q35_W4_DIAG_KERNEL")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .filter(|variant| (1..=6).contains(variant))
}

/// Selects the warp-per-output packed-W4 GEMV kernel (one warp per output,
/// eight outputs per block, activation row staged in shared memory), which
/// raises per-thread work from ~2.5 to ~20 packed uint32 values. Default on;
/// `APXINF_Q35_W4_WARP_GEMV=0` is the paired A/B control.
fn warp_gemv_enabled() -> bool {
    !matches!(std::env::var("APXINF_Q35_W4_WARP_GEMV").as_deref(), Ok("0"))
}

/// Row count at or above which a packed-W4 projection switches from the
/// per-output GEMV kernel to dequantize-then-BF16-GEMM. `usize::MAX` disables
/// the large-M path entirely, which is the paired A/B control.
fn dequantize_row_threshold() -> usize {
    match std::env::var("APXINF_Q35_W4_PREFILL_GEMM") {
        Ok(value) if value == "0" => usize::MAX,
        Ok(value) => value.parse().unwrap_or(8),
        Err(_) => 8,
    }
}

/// Launch the dequantize kernel into caller-provided scratch and surface the
/// finite-status flags. Each weight is rounded to BF16 exactly once, matching
/// the GEMV kernel's decompression boundary.
fn dequantize_into(
    ctx: &CudaContext,
    buffers: &Qwen35W4Buffers<'_>,
    layout: Qwen35W4Layout,
    destination: *mut std::ffi::c_void,
) -> Result<()> {
    let flags = CudaBuffer::alloc_zeros(size_of::<u32>(), ctx.device_id()).map_err(Error::Cuda)?;
    let scale_bytes = checked_product(
        &[
            layout.out_features,
            layout.groups(),
            DType::BF16.size_in_bytes(),
        ],
        "scale",
    )?;
    let scale_buffer = require_tensor_buffer(ctx, "scale", buffers.scales, scale_bytes)?;
    require_exact_buffer(
        ctx,
        "packed weight",
        buffers.weight_packed,
        layout.weight_bytes()?,
    )?;
    require_exact_buffer(
        ctx,
        "N-packed zero-point",
        buffers.zero_points,
        layout.zero_point_bytes()?,
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_w4_dequantize_bf16(
            buffers.weight_packed.ptr(),
            scale_buffer.ptr(),
            buffers.zero_points.ptr(),
            destination,
            flags.ptr(),
            checked_i32(layout.out_features, "out_features")?,
            checked_i32(layout.in_features, "in_features")?,
            checked_i32(layout.group_size, "group_size")?,
            ctx.stream().handle(),
        ))
        .map_err(|error| Error::Cuda(format!("Qwen3.5 W4 dequantize launch failed: {error}")))?;
    }
    ctx.synchronize().map_err(|error| {
        Error::Cuda(format!("Qwen3.5 W4 dequantize synchronize failed: {error}"))
    })?;
    let mut flag_bytes = [0u8; size_of::<u32>()];
    flags.copy_to_host(&mut flag_bytes).map_err(|error| {
        Error::Cuda(format!("Qwen3.5 W4 dequantize status copy failed: {error}"))
    })?;
    let flag = u32::from_le_bytes(flag_bytes);
    if flag & FLAG_NON_FINITE_SCALE != 0 {
        return Err(Error::Other(
            "Qwen3.5 W4 projection encountered a non-finite W4 scale".into(),
        ));
    }
    if flag & FLAG_NON_FINITE_OUTPUT != 0 {
        return Err(Error::Other(
            "Qwen3.5 W4 projection produced a non-finite W4 output".into(),
        ));
    }
    Ok(())
}

/// Decompress the packed asymmetric W4 matrix into a dense BF16
/// checkpoint-layout `[out_features, in_features]` tensor.
pub fn dequantize_bf16(
    ctx: &CudaContext,
    buffers: Qwen35W4Buffers<'_>,
    layout: Qwen35W4Layout,
) -> Result<Tensor> {
    let dequantized_bytes = checked_product(
        &[
            layout.out_features,
            layout.in_features,
            DType::BF16.size_in_bytes(),
        ],
        "dequantized",
    )?;
    let scratch = scratch_pool::acquire(ctx.device_id(), dequantized_bytes)?;
    dequantize_into(ctx, &buffers, layout, scratch.ptr())?;
    Ok(scratch.into_inner().into_tensor(
        Shape::new(vec![layout.out_features, layout.in_features]),
        DType::BF16,
    ))
}

/// Dequantize the packed weight into pooled BF16 scratch once, then compute
/// `activation @ weight^T` with the tensor-core BF16 GEMM. Weight
/// decompression rounds to BF16 exactly once, identical to the GEMV kernel;
/// only the dot-product accumulation order differs.
fn project_bf16_via_dequantized_gemm(
    ctx: &CudaContext,
    activation: &Tensor,
    buffers: Qwen35W4Buffers<'_>,
    layout: Qwen35W4Layout,
    rows: usize,
) -> Result<Tensor> {
    let dequantized_bytes = checked_product(
        &[
            layout.out_features,
            layout.in_features,
            DType::BF16.size_in_bytes(),
        ],
        "dequantized",
    )?;
    let scratch = scratch_pool::acquire(ctx.device_id(), dequantized_bytes)?;
    dequantize_into(ctx, &buffers, layout, scratch.ptr())?;

    let activation_bytes = checked_product(
        &[rows, layout.in_features, DType::BF16.size_in_bytes()],
        "activation",
    )?;
    let activation_buffer = require_tensor_buffer(ctx, "activation", activation, activation_bytes)?;
    let output_bytes = checked_product(
        &[rows, layout.out_features, DType::BF16.size_in_bytes()],
        "output",
    )?;
    let output = output_buffer(ctx, output_bytes)?;
    // activation [rows, K] @ dequantized [N, K]^T -> output [rows, N], the
    // same call `gemm::project_checkpoint_bf16` makes, expressed on raw
    // buffers so the pooled scratch never enters `Tensor` ownership.
    ctx.cublas()
        .gemm_ex(
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            rows,
            layout.out_features,
            layout.in_features,
            1.0,
            &activation_buffer,
            layout.in_features as i32,
            scratch.buffer(),
            layout.in_features as i32,
            0.0,
            &output,
            layout.out_features as i32,
        )
        .map_err(Error::Cuda)?;
    Ok(output_tensor(
        ctx,
        Shape::new(vec![rows, layout.out_features]),
        DType::BF16,
        output,
    ))
}

fn require_tensor_buffer(
    ctx: &CudaContext,
    name: &str,
    tensor: &Tensor,
    required_bytes: usize,
) -> Result<CudaBuffer> {
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.device() != ctx.device_id() || buffer.len() < required_bytes {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 {name} requires {required_bytes} bytes on CUDA{}, got {} bytes on CUDA{}",
            ctx.device_id(),
            buffer.len(),
            buffer.device()
        )));
    }
    Ok(buffer)
}

fn require_exact_buffer(
    ctx: &CudaContext,
    name: &str,
    buffer: &CudaBuffer,
    expected_bytes: usize,
) -> Result<()> {
    if buffer.device() != ctx.device_id() {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 {name} is on CUDA{}, expected CUDA{}",
            buffer.device(),
            ctx.device_id()
        )));
    }
    if buffer.len() != expected_bytes {
        return Err(Error::Other(format!(
            "Qwen3.5 W4 {name} must contain exactly {expected_bytes} bytes, got {}",
            buffer.len()
        )));
    }
    Ok(())
}

fn checked_product(factors: &[usize], name: &str) -> Result<usize> {
    factors.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or_else(|| Error::Other(format!("Qwen3.5 W4 {name} size overflow")))
    })
}

fn checked_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Other(format!("Qwen3.5 W4 {name} exceeds CUDA i32 range")))
}

use apxinf_core::{DType, Result, Tensor};

use super::contracts::{invalid, tensor_storage};
use crate::ffi::abi::{gemm as abi, status};
use crate::CudaContext;

/// Bytes an NVFP4 block-scale buffer occupies in the layout the kernel reads.
///
/// `rows` is M for an activation operand and N for a weight. The layout pads
/// the logical `[rows, k / block_size]` grid up to the kernel's atom
/// boundaries, so the answer is larger than the checkpoint's own scale tensor
/// and has to be queried rather than derived.
pub fn nvfp4_scale_buffer_bytes(rows: usize, k: usize, block_size: u32) -> Result<usize> {
    if rows == 0 || k == 0 || rows > i32::MAX as usize || k > i32::MAX as usize {
        return Err(invalid("NVFP4 scale buffer needs a nonempty shape"));
    }
    let bytes =
        unsafe { abi::apxinf_gemm_nvfp4_scale_buffer_bytes(rows as i64, k as i64, block_size) };
    if bytes == 0 {
        return Err(invalid(format!(
            "unsupported NVFP4 block size {block_size}"
        )));
    }
    Ok(bytes as usize)
}

/// Rewrite checkpoint-order block scales into the layout the kernel reads.
///
/// `source` is the row-major `[rows, k / block_size]` E4M3 tensor as a
/// checkpoint stores it. `destination` must be an E4M3 tensor holding at least
/// [`nvfp4_scale_buffer_bytes`] bytes.
///
/// The target layout depends only on `block_size`, not on which tactic the
/// autotuner picks, so this runs once when weights are loaded and stays valid
/// for the life of the model.
pub fn nvfp4_pack_block_scales(
    ctx: &CudaContext,
    source: &Tensor,
    destination: &Tensor,
    rows: usize,
    k: usize,
    block_size: u32,
) -> Result<()> {
    let required = nvfp4_scale_buffer_bytes(rows, k, block_size)?;
    let blocks = k.div_ceil(block_size as usize);

    let source_dims = source.shape().dims().to_vec();
    let source_buffer = tensor_storage(ctx, source, DType::F8E4M3, &source_dims)?;
    if source_buffer.len() < rows * blocks {
        return Err(invalid(format!(
            "NVFP4 scale source holds {} bytes, needs {}",
            source_buffer.len(),
            rows * blocks
        )));
    }

    let destination_dims = destination.shape().dims().to_vec();
    let destination_buffer = tensor_storage(ctx, destination, DType::F8E4M3, &destination_dims)?;
    if destination_buffer.len() < required {
        return Err(invalid(format!(
            "NVFP4 scale destination holds {} bytes, needs {required}",
            destination_buffer.len()
        )));
    }

    unsafe {
        status::check(abi::apxinf_gemm_nvfp4_pack_block_scales(
            source_buffer.ptr(),
            destination_buffer.ptr(),
            rows as i64,
            k as i64,
            block_size,
            ctx.stream().handle(),
        ))
    }
}

/// Which layout a quantizer should write its block scales in.
///
/// The block-scaled GEMM reads a tcgen05 atom layout; the GEMV indexes a
/// plain row-major grid. Producing the right one directly costs nothing and
/// saves the decode path a relayout pass per projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    /// `[rows, k / block_size]`, the layout a checkpoint stores and the GEMV
    /// reads.
    RowMajor,
    /// The kernel atom layout the block-scaled GEMM reads. Size it with
    /// [`nvfp4_scale_buffer_bytes`].
    GemmAtom,
}

impl ScaleLayout {
    fn flag(self) -> i32 {
        match self {
            ScaleLayout::RowMajor => 1,
            ScaleLayout::GemmAtom => 0,
        }
    }

    fn required_bytes(self, rows: usize, k: usize, block_size: u32) -> Result<usize> {
        match self {
            ScaleLayout::RowMajor => Ok(rows * k.div_ceil(block_size as usize)),
            ScaleLayout::GemmAtom => nvfp4_scale_buffer_bytes(rows, k, block_size),
        }
    }
}

/// Quantize a BF16 activation into the operand pair an NVFP4 GEMM consumes.
///
/// `activation` is `[M, K]` BF16. `packed` must be `[M, K/2]` of
/// [`DType::E2M1Pair`], and `scales` an E4M3 buffer of at least
/// [`nvfp4_scale_buffer_bytes`] bytes.
///
/// `input_scale` must be the checkpoint's per-tensor activation scale for this
/// projection. Block scales are stored relative to it, so the GEMM recovers
/// absolute magnitudes by folding `input_scale * weight_scale_2` into `alpha`.
/// Passing a different value silently rescales the layer rather than failing,
/// which is why it is a required argument and not a default.
pub fn nvfp4_quantize_activation(
    ctx: &CudaContext,
    activation: &Tensor,
    packed: &Tensor,
    scales: &Tensor,
    input_scale: f32,
    block_size: u32,
    layout: ScaleLayout,
) -> Result<()> {
    let dims = activation.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("NVFP4 activation quantization requires a rank-2 tensor"));
    }
    let (rows, k) = (dims[0], dims[1]);
    if block_size == 0 || k % block_size as usize != 0 {
        return Err(invalid("NVFP4 K must be a multiple of the block size"));
    }
    if !(input_scale > 0.0) || !input_scale.is_finite() {
        return Err(invalid("NVFP4 input_scale must be finite and positive"));
    }
    let required = layout.required_bytes(rows, k, block_size)?;

    let source = tensor_storage(ctx, activation, DType::BF16, &dims)?;
    let destination = tensor_storage(ctx, packed, DType::E2M1Pair, &[rows, k / 2])?;
    let scale_dims = scales.shape().dims().to_vec();
    let scale_buffer = tensor_storage(ctx, scales, DType::F8E4M3, &scale_dims)?;
    if scale_buffer.len() < required {
        return Err(invalid(format!(
            "NVFP4 scale destination holds {} bytes, needs {required}",
            scale_buffer.len()
        )));
    }

    unsafe {
        status::check(abi::apxinf_gemm_nvfp4_quantize_activation(
            source.ptr(),
            destination.ptr(),
            scale_buffer.ptr(),
            rows as i64,
            k as i64,
            block_size,
            input_scale,
            layout.flag(),
            ctx.stream().handle(),
        ))
    }
}

/// RMSNorm fused with NVFP4 quantization.
///
/// Equivalent to [`crate::ops::rms_norm`] followed by
/// [`nvfp4_quantize_activation`], but never materializes the normalized BF16
/// tensor. At prefill widths that round trip dominates both ops' arithmetic.
pub fn nvfp4_quantize_rms_norm(
    ctx: &CudaContext,
    input: &Tensor,
    norm_weight: &Tensor,
    packed: &Tensor,
    scales: &Tensor,
    epsilon: f32,
    input_scale: f32,
    block_size: u32,
    layout: ScaleLayout,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("fused RMSNorm quantization requires a rank-2 tensor"));
    }
    let (rows, k) = (dims[0], dims[1]);
    if block_size == 0 || k % block_size as usize != 0 {
        return Err(invalid("NVFP4 K must be a multiple of the block size"));
    }
    if !(input_scale > 0.0) || !input_scale.is_finite() {
        return Err(invalid("NVFP4 input_scale must be finite and positive"));
    }
    let required = layout.required_bytes(rows, k, block_size)?;
    let source = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let weight = tensor_storage(ctx, norm_weight, DType::BF16, &[k])?;
    let destination = tensor_storage(ctx, packed, DType::E2M1Pair, &[rows, k / 2])?;
    let scale_dims = scales.shape().dims().to_vec();
    let scale_buffer = tensor_storage(ctx, scales, DType::F8E4M3, &scale_dims)?;
    if scale_buffer.len() < required {
        return Err(invalid(format!(
            "NVFP4 scale destination holds {} bytes, needs {required}",
            scale_buffer.len()
        )));
    }
    unsafe {
        status::check(abi::apxinf_gemm_nvfp4_quantize_rms_norm(
            source.ptr(),
            weight.ptr(),
            destination.ptr(),
            scale_buffer.ptr(),
            rows as i64,
            k as i64,
            block_size,
            epsilon,
            input_scale,
            layout.flag(),
            ctx.stream().handle(),
        ))
    }
}

/// SwiGLU fused with NVFP4 quantization.
///
/// `fused` is the `[rows, 2*k]` gate/up projection, gate first. Equivalent to
/// [`crate::ops::swiglu`] followed by [`nvfp4_quantize_activation`] without the
/// `[rows, k]` BF16 intermediate.
pub fn nvfp4_quantize_swiglu(
    ctx: &CudaContext,
    fused: &Tensor,
    packed: &Tensor,
    scales: &Tensor,
    input_scale: f32,
    block_size: u32,
    layout: ScaleLayout,
) -> Result<()> {
    let dims = fused.shape().dims().to_vec();
    if dims.len() != 2 || dims[1] % 2 != 0 {
        return Err(invalid("fused SwiGLU input must be [rows, 2*k]"));
    }
    let (rows, k) = (dims[0], dims[1] / 2);
    if block_size == 0 || k % block_size as usize != 0 {
        return Err(invalid("NVFP4 K must be a multiple of the block size"));
    }
    if !(input_scale > 0.0) || !input_scale.is_finite() {
        return Err(invalid("NVFP4 input_scale must be finite and positive"));
    }
    let required = layout.required_bytes(rows, k, block_size)?;
    let source = tensor_storage(ctx, fused, DType::BF16, &dims)?;
    let destination = tensor_storage(ctx, packed, DType::E2M1Pair, &[rows, k / 2])?;
    let scale_dims = scales.shape().dims().to_vec();
    let scale_buffer = tensor_storage(ctx, scales, DType::F8E4M3, &scale_dims)?;
    if scale_buffer.len() < required {
        return Err(invalid(format!(
            "NVFP4 scale destination holds {} bytes, needs {required}",
            scale_buffer.len()
        )));
    }
    unsafe {
        status::check(abi::apxinf_gemm_nvfp4_quantize_swiglu(
            source.ptr(),
            destination.ptr(),
            scale_buffer.ptr(),
            rows as i64,
            k as i64,
            block_size,
            input_scale,
            layout.flag(),
            ctx.stream().handle(),
        ))
    }
}

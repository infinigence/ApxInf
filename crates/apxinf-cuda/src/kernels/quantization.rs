//! Quantization operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{check_cuda, gpu_ptr, make_gpu_tensor};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;
use crate::cublas::CublasTranspose;

/// Quantize an FP16 device tensor to E4M3 using a pre-calibrated scale.
pub fn quantize_f16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        return Err(Error::DTypeMismatch {
            expected: DType::F16,
            got: input.dtype(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!("invalid FP8 scale {scale}")));
    }
    let output = output_buffer(ctx, input.numel())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_f16_e4m3(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}


/// BF16 activation times asymmetric group-wise packed INT4 weights.
///
/// Activation shape is `[rows, in_cols]`. Packed weights are Hugging Face
/// compressed-tensors W4A16 layout with logical shape `[out_cols, in_cols]`,
/// packed low-nibble-first across input columns and zero-points packed across
/// output rows. The output is BF16 `[rows, out_cols]`.

/// BF16 activation times a row-major transposed dense BF16 weight.
///
/// Activation shape is `[rows, in_cols]`; weight shape is `[out_cols, in_cols]`.
/// The output is BF16 `[rows, out_cols]`. This matches HuggingFace linear
/// weight layout without materializing a `[in_cols, out_cols]` transpose.

/// W4A16 prefill GEMM writing into caller-owned dense scratch + output, so
/// repeated prefill GEMMs never touch cudaMalloc.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_w4a16_asym_prefill_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight_packed: &CudaBuffer,
    weight_scale: &CudaBuffer,
    weight_zero_point: &CudaBuffer,
    dense: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
    groups: usize,
) -> Result<()> {
    let dense_bytes = out_cols
        .checked_mul(in_cols)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("prefill GEMM dense overflow".into()))?;
    if dense.len() < dense_bytes {
        return Err(Error::Other("prefill GEMM: dense scratch too small".into()));
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_dequant_w4a16_bf16_row(
            weight_packed.ptr(),
            weight_scale.ptr(),
            weight_zero_point.ptr(),
            dense.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    matmul_bf16_transposed_into(ctx, activation, dense, output, rows, in_cols, out_cols)
}

/// BF16 activation (row-major `[rows, in_cols]`) times transposed dense BF16
/// weight (`[out_cols, in_cols]`) into caller-owned output storage.
pub fn matmul_bf16_transposed_into(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    in_cols: usize,
    out_cols: usize,
) -> Result<()> {
    let out_bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("matmul_bf16_transposed_into output overflow".into()))?;
    if output.len() < out_bytes {
        return Err(Error::Other("matmul_bf16_transposed_into: buffer too small".into()));
    }
    super::gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        rows,
        out_cols,
        in_cols,
        1.0,
        activation,
        in_cols as i32,
        weight,
        in_cols as i32,
        0.0,
        output,
        out_cols as i32,
    )
}

pub fn matmul_bf16_transposed(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return Err(Error::Other("matmul_bf16_transposed dtype mismatch".into()));
    }
    let a_dims = activation.shape().dims();
    let w_dims = weight.shape().dims();
    if a_dims.len() != 2 || w_dims.len() != 2 || a_dims[1] != w_dims[1] {
        return Err(Error::Other(format!(
            "matmul_bf16_transposed activation {a_dims:?} incompatible with weight {w_dims:?}"
        )));
    }
    let rows = a_dims[0];
    let in_cols = a_dims[1];
    let out_cols = w_dims[0];
    let bytes = rows
        .checked_mul(out_cols)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("matmul_bf16_transposed output overflow".into()))?;
    let output = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let a = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let w = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    super::gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        rows,
        out_cols,
        in_cols,
        1.0,
        &a,
        in_cols as i32,
        &w,
        in_cols as i32,
        0.0,
        &output,
        out_cols as i32,
    )?;
    Ok(output.into_tensor(
        apxinf_core::Shape::from(vec![rows, out_cols]),
        DType::BF16,
    ))
}

    // Allocation-free single-row variant: writes straight into caller storage
    // so decode steps never touch cudaMalloc (which degrades after prefill
    // churns large transient buffers).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_bf16_w4a16_asym_into(
        ctx: &CudaContext,
        activation: &CudaBuffer,
        weight_packed: &CudaBuffer,
        weight_scale: &CudaBuffer,
        weight_zero_point: &CudaBuffer,
        output: &CudaBuffer,
        rows: usize,
        in_cols: usize,
        out_cols: usize,
        groups: usize,
    ) -> Result<()> {
        let out_bytes = out_cols
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("matmul_bf16_w4a16_asym_into output overflow".into()))?;
        if output.len() < out_bytes {
            return Err(Error::Other("matmul_bf16_w4a16_asym_into: buffer too small".into()));
        }
        if out_cols % 128 != 0 {
            return Err(Error::Other(
                "matmul_bf16_w4a16_asym_into: out_cols must be a multiple of 128".into(),
            ));
        }
        check_cuda(unsafe {
            ffi::apxinf_qwen35_gemm_w4a16_bf16(
                activation.ptr(),
                weight_packed.ptr(),
                weight_scale.ptr(),
                weight_zero_point.ptr(),
                output.ptr(),
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            )
        })
    }

pub fn matmul_bf16_w4a16_asym(
    ctx: &CudaContext,
    activation: &Tensor,
    weight_packed: &Tensor,
    weight_scale: &Tensor,
    weight_zero_point: &Tensor,
    out_cols: usize,
    in_cols: usize,
    groups: usize,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16
        || weight_packed.dtype() != DType::I32
        || weight_scale.dtype() != DType::BF16
        || weight_zero_point.dtype() != DType::I32
    {
        return Err(Error::Other("matmul_bf16_w4a16_asym dtype mismatch".into()));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[1] != in_cols {
        return Err(Error::Other(format!(
            "matmul_bf16_w4a16_asym activation shape {dims:?} incompatible with in_cols {in_cols}"
        )));
    }
    let _rows = dims[0];
    if weight_packed.shape().dims() != [out_cols, in_cols.div_ceil(8)] {
        return Err(Error::Other("matmul_bf16_w4a16_asym packed shape mismatch".into()));
    }
    if weight_scale.shape().dims() != [out_cols, groups]
        || weight_zero_point.shape().dims() != [out_cols.div_ceil(8), groups]
    {
        return Err(Error::Other("matmul_bf16_w4a16_asym scale/zero-point shape mismatch".into()));
    }
    // Decode (single row) is latency-bound: the static dequant + cublas path
    // would stream the entire weight matrix through HBM twice per call. Use
    // the fused dequant-GEMM kernel, which reads the packed weights once.
    // Prefill keeps the cublas path: batched GEMM amortizes the dequant cost
    // and tensor-core GEMM dominates the fused kernel's scalar loop.
    if dims[0] == 1 {
        let out_bytes = out_cols
            .checked_mul(DType::BF16.size_in_bytes())
            .ok_or_else(|| Error::Other("matmul_bf16_w4a16_asym output overflow".into()))?;
        let output = CudaBuffer::alloc_zeros(out_bytes, ctx.device_id()).map_err(Error::Cuda)?;
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_matmul_bf16_w4a16_asym(
                gpu_ptr(activation)?,
                gpu_ptr(weight_packed)?,
                gpu_ptr(weight_scale)?,
                gpu_ptr(weight_zero_point)?,
                output.ptr(),
                1,
                in_cols as i32,
                out_cols as i32,
                groups as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return Ok(output.into_tensor(
            apxinf_core::Shape::from(vec![1, out_cols]),
            DType::BF16,
        ));
    }

    let _t0 = std::time::Instant::now();
    let dense_bytes = out_cols
        .checked_mul(in_cols)
        .and_then(|v| v.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("matmul_bf16_w4a16_asym dense weight overflow".into()))?;
    let dense = CudaBuffer::alloc_zeros(dense_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_qwen35_dequant_w4a16_bf16_row(
            gpu_ptr(weight_packed)?,
            gpu_ptr(weight_scale)?,
            gpu_ptr(weight_zero_point)?,
            dense.ptr(),
            in_cols as i32,
            out_cols as i32,
            groups as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    let t_dequant = _t0.elapsed().as_secs_f32() * 1000.0;
    let dense = dense.into_tensor(
        apxinf_core::Shape::from(vec![out_cols, in_cols]),
        DType::BF16,
    );
    let result = matmul_bf16_transposed(ctx, activation, &dense);
    if std::env::var_os("APXINF_GEMM_PROF").is_some() {
        let ms = _t0.elapsed().as_secs_f32() * 1000.0;
        if ms > 5.0 {
            eprintln!(
                "[gemm] {}x{} rows={} : {:.1} ms (dequant {:.1} ms)",
                out_cols, in_cols, dims[0], ms, t_dequant
            );
        }
    }
    result
}


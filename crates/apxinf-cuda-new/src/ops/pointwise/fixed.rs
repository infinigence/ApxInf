//! Fixed pointwise operators and direct-launch specializations.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::CudaContext;

/// Whether SwiGLU rounds the SiLU intermediate before multiplying by `up`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwiGluRounding {
    FullPrecision,
    RoundedToInput,
}

/// SwiGLU with its numerical rounding contract expressed explicitly.
pub fn swiglu(ctx: &CudaContext, gate_up: &Tensor, rounding: SwiGluRounding) -> Result<Tensor> {
    match (gate_up.dtype(), rounding) {
        (DType::BF16, SwiGluRounding::FullPrecision) => swiglu_bf16(ctx, gate_up),
        (DType::BF16, SwiGluRounding::RoundedToInput) => swiglu_bf16_rounded(ctx, gate_up),
        (DType::F16, SwiGluRounding::FullPrecision) => swiglu_f16(ctx, gate_up),
        (DType::F16, SwiGluRounding::RoundedToInput) => Err(Error::Other(
            "rounded SwiGLU is not available for F16 input".into(),
        )),
        _ => Err(Error::Other("SwiGLU requires BF16 or F16 input".into())),
    }
}

/// `swiglu_bf16`: `[rows, 2*cols]` gate/up projection to `[rows, cols]` SwiGLU.
///
/// Uses the legacy unrounded form (`silu` at full f32 before the multiply).
/// `ops::pointwise::swiglu` is the rounded variant (bf16(silu) first) that the
/// qwen_drive checkpoint path needs; routing this name there would shift
/// every result by up to 1 ULP.
fn swiglu_bf16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::activation as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || gate_up.dtype() != DType::BF16 || dims[1] % 2 != 0 {
        return Err(Error::Other(
            "SwiGLU requires a [rows, 2*cols] BF16 input".into(),
        ));
    }
    let (rows, inner) = (dims[0], dims[1] / 2);
    let r = i32::try_from(rows).map_err(|_| Error::Other("SwiGLU rows exceed i32".into()))?;
    let n = i32::try_from(inner).map_err(|_| Error::Other("SwiGLU inner exceeds i32".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![rows, inner]), DType::BF16)?;
    let input_buffer = CudaBuffer::from_tensor(gate_up).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_activation_swiglu_bf16(
            input_buffer.ptr(),
            output_buffer.ptr(),
            r,
            n,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// Fused `BF16(SiLU(gate)) * up` over separate,
/// equally-shaped inputs. Mirrors the legacy `silu_mul_separate_bf16` path,
/// whose SiLU intermediate rounds to BF16 before the multiply.
pub fn silu_mul(ctx: &CudaContext, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::activation as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    if gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 {
        return Err(Error::Other("SiLU multiply requires BF16 inputs".into()));
    }
    if gate.shape() != up.shape() || gate.numel() == 0 {
        return Err(Error::Other(
            "SiLU multiply requires equal non-empty input shapes".into(),
        ));
    }
    let count = u32::try_from(gate.numel())
        .map_err(|_| Error::Other("SiLU multiply count exceeds u32".into()))?;
    let output = ctx.allocate_output(Shape::new(gate.shape().dims().to_vec()), DType::BF16)?;
    let gate_buffer = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let up_buffer = CudaBuffer::from_tensor(up).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_activation_silu_mul_separate_bf16(
            gate_buffer.ptr(),
            up_buffer.ptr(),
            output_buffer.ptr(),
            count,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// Fused `BF16(BF16(SiLU(gate)) * up) -> calibrated
/// E4M3` over separate, equally-shaped inputs.
pub fn silu_mul_quantized(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::activation as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    if gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 {
        return Err(Error::Other("silu_mul_quant requires BF16 inputs".into()));
    }
    if gate.shape() != up.shape() || gate.numel() == 0 || !(scale > 0.0) {
        return Err(Error::Other(
            "silu_mul_quant requires equal non-empty shapes and a positive scale".into(),
        ));
    }
    let count = i64::try_from(gate.numel())
        .map_err(|_| Error::Other("silu_mul_quant count exceeds i64".into()))?;
    let output = ctx.allocate_output(Shape::new(gate.shape().dims().to_vec()), DType::F8E4M3)?;
    let gate_buffer = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let up_buffer = CudaBuffer::from_tensor(up).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_activation_silu_mul_quant_bf16_e4m3(
            gate_buffer.ptr(),
            up_buffer.ptr(),
            output_buffer.ptr(),
            count,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// Fused `BF16(bias + tanh-GELU(input)) ->
/// calibrated E4M3` using packed-4 IO.
pub fn bias_gelu_quantized(
    ctx: &CudaContext,
    input: &Tensor,
    bias: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::activation as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(
            "bias GELU quantization expects a rank-2 input".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if input.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || bias.shape().dims() != [cols]
        || cols % 4 != 0
        || !(scale > 0.0)
    {
        return Err(Error::Other(
            "bias GELU quantization expects a BF16 matrix, matching bias, width divisible by 4, and positive scale".into(),
        ));
    }
    let rows_i32 =
        i32::try_from(rows).map_err(|_| Error::Other("bias GELU rows exceed i32".into()))?;
    let cols_i32 =
        i32::try_from(cols).map_err(|_| Error::Other("bias GELU cols exceed i32".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_activation_bias_gelu_quant_bf16_e4m3(
            input_buffer.ptr(),
            bias_buffer.ptr(),
            output_buffer.ptr(),
            rows_i32,
            cols_i32,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// Broadcast BF16 bias, apply tanh-GELU, then quantize each row to signed
/// INT8 with an F32 scale. The output representation is carried by the return
/// type rather than encoded in the L3 function name.
pub fn bias_gelu_quantize_rows(
    ctx: &CudaContext,
    input: &Tensor,
    bias: &Tensor,
) -> Result<crate::ops::quantization::DynamicI8Tensor> {
    use crate::ffi::abi::quantization as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;

    let dims = input.shape().dims();
    if input.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || dims.len() != 2
        || dims[0] == 0
        || dims[1] == 0
        || bias.shape().dims() != [dims[1]]
    {
        return Err(Error::Other(
            "fused bias-GELU row quantization expects [rows,cols] BF16 input and [cols] bias"
                .into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let values = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::I8)?;
    let scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let values_buffer = CudaBuffer::from_tensor(&values).map_err(Error::Cuda)?;
    let scales_buffer = CudaBuffer::from_tensor(&scales).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_quantization_bias_gelu_quantize_rows_bf16_int8(
            input_buffer.ptr(),
            bias_buffer.ptr(),
            values_buffer.ptr(),
            scales_buffer.ptr(),
            i32::try_from(rows).map_err(|_| Error::Other("rows exceed i32".into()))?,
            i32::try_from(cols).map_err(|_| Error::Other("cols exceed i32".into()))?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(crate::ops::quantization::DynamicI8Tensor { values, scales })
}

/// Apply the legacy rounded SiLU-multiply semantic and quantize each row to
/// signed INT8. The packed-4 implementation is selected internally when its
/// existing SM87/exact-shape support predicate is satisfied.
pub fn silu_mul_quantize_rows(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
) -> Result<crate::ops::quantization::DynamicI8Tensor> {
    if gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 || gate.shape() != up.shape() {
        return Err(Error::Other(
            "fused SiLU-multiply row quantization expects equal BF16 tensors".into(),
        ));
    }
    let dims = gate.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(
            "fused SiLU-multiply row quantization expects a non-empty matrix".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let mut values = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::I8)?;
    let mut scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let mut args = crate::ops::QuantizationArgs::new(
        crate::ops::QuantizationSemantic::SiluMulRowwiseI8,
        gate,
        &mut values,
    );
    args.secondary = Some(up);
    args.scales = Some(&mut scales);
    crate::ops::quantization(ctx, args)?;
    Ok(crate::ops::quantization::DynamicI8Tensor { values, scales })
}

/// SwiGLU over a packed gate/up projection
/// followed by rowwise E4M3 quantization, zero-padded to `output_cols`.
pub fn swiglu_quantize_rows(
    ctx: &CudaContext,
    gate_up: &Tensor,
    bias: Option<&Tensor>,
    logical_inner: usize,
    output_cols: usize,
) -> Result<crate::ops::quantization::DynamicFp8Tensor> {
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || gate_up.dtype() != DType::BF16 {
        return Err(Error::Other(
            "fused SwiGLU quantization requires a rank-2 BF16 projection".into(),
        ));
    }
    let (rows, input_cols) = (dims[0], dims[1]);
    if logical_inner == 0 || input_cols < 2 * logical_inner || output_cols < logical_inner {
        return Err(Error::Other(
            "fused SwiGLU quantization shape mismatch".into(),
        ));
    }
    let mut values = ctx.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
    let mut scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let mut args = crate::ops::PointwiseArgs::new(
        crate::ops::PointwiseSemantic::SwigluRowwiseE4m3,
        gate_up,
        &mut values,
    );
    args.bias = bias;
    args.scales = Some(&mut scales);
    args.logical_cols = logical_inner;
    crate::ops::pointwise(ctx, args)?;
    Ok(crate::ops::quantization::DynamicFp8Tensor { values, scales })
}

/// `swiglu_bf16_rounded`: SwiGLU whose SiLU intermediate rounds to BF16
/// before the multiply — matching an unfused activation-then-multiply.
fn swiglu_bf16_rounded(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::activation as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || gate_up.dtype() != DType::BF16 || dims[1] == 0 || dims[1] % 2 != 0 {
        return Err(Error::Other(
            "rounded SwiGLU expects nonempty BF16 [rows,2*inner]".into(),
        ));
    }
    let (rows, inner) = (dims[0], dims[1] / 2);
    let r = i32::try_from(rows).map_err(|_| Error::Other("SwiGLU rows overflow".into()))?;
    let n = i32::try_from(inner).map_err(|_| Error::Other("SwiGLU inner overflow".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![rows, inner]), DType::BF16)?;
    let input_buffer = CudaBuffer::from_tensor(gate_up).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_activation_swiglu_bf16_rounded(
            input_buffer.ptr(),
            output_buffer.ptr(),
            r,
            n,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

// ── smolvla FP16 surface ────────────────────────────────────────────────────
//
// Ported verbatim from the legacy FP16 activation wrappers over the
// `family-specific FP16` launchers (native/adapters/activation/fixed_f16.cu).

/// `swiglu_f16`: FP16 SwiGLU over packed `[rows, 2*inner]` gate/up halves.
fn swiglu_f16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::activation as abi;
    use crate::ops::validation::{check_cuda, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape};
    let (rows, twice_inner) = matrix_shape(gate_up, "SwiGLU")?;
    if gate_up.dtype() != DType::F16 || twice_inner % 2 != 0 {
        return Err(Error::Other(
            "static inference FP16 SwiGLU expects [rows,2*inner]".into(),
        ));
    }
    let inner = twice_inner / 2;
    let output = f16_output(ctx, rows, inner)?;
    unsafe {
        check_cuda(abi::apxinf_activation_swiglu_f16(
            gpu_ptr(gate_up)?,
            output.ptr(),
            rows as i32,
            inner as i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, inner]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───
//
// Direct fixed decode launchers: no allocation, no resource
// creation, writes into caller-owned decode workspace buffers. Arithmetic is
// the legacy kernel's, copied verbatim into native/adapters/fixed_decode.

/// Allocation-free SiLU into caller-owned decode storage.
pub fn silu_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    count: usize,
) -> Result<()> {
    use crate::ffi::abi::activation as abi;
    use crate::ops::validation::{check_cuda, checked_bytes, require_buffers};
    let bytes = checked_bytes(dtype, &[count], "decode SiLU")?;
    require_buffers(
        ctx,
        "decode SiLU",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => abi::apxinf_activation_fixed_decode_silu_f32(
                input.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_activation_fixed_decode_silu_bf16(
                input.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode SiLU does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Fused BF16 SiLU(gate) * up into caller-owned decode storage.
pub fn silu_mul_into(
    ctx: &CudaContext,
    gate_up: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    intermediate: usize,
) -> Result<()> {
    use crate::ffi::abi::activation as abi;
    use crate::ops::validation::{check_cuda, checked_bytes, require_buffers};
    require_buffers(
        ctx,
        "decode SiLU multiply",
        &[
            (
                "gate_up",
                gate_up,
                checked_bytes(DType::BF16, &[2, intermediate], "decode SiLU multiply")?,
            ),
            (
                "output",
                output,
                checked_bytes(DType::BF16, &[intermediate], "decode SiLU multiply")?,
            ),
        ],
    )?;
    check_cuda(unsafe {
        abi::apxinf_activation_fixed_decode_silu_mul_bf16(
            gate_up.ptr(),
            output.ptr(),
            intermediate as u32,
            ctx.stream().handle(),
        )
    })
}

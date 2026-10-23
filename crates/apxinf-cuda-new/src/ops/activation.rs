//! Activation operators and direct-launch specializations.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// `silu`: elementwise SiLU into a fresh tensor.
pub fn silu(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_activation(ctx, input, &output, ops::ElementwiseActivation::Silu)?;
    Ok(output)
}

/// `gelu_tanh`: elementwise tanh-approximated GELU into a fresh tensor.
pub fn gelu_tanh(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_activation(ctx, input, &output, ops::ElementwiseActivation::GeluTanh)?;
    Ok(output)
}

/// `geglu_bf16`: `[rows, 2*cols]` gate/up projection to `[rows, cols]` GeGLU.
pub fn geglu_bf16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || dims[1] % 2 != 0 {
        return Err(Error::Other("GeGLU requires a [rows, 2*cols] input".into()));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![dims[0], dims[1] / 2]), DType::BF16)?;
    let args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Geglu, gate_up, &mut output);
    ops::pointwise(ctx, args)?;
    Ok(output)
}

/// `swiglu_bf16`: `[rows, 2*cols]` gate/up projection to `[rows, cols]` SwiGLU.
///
/// Uses the legacy unrounded form (`silu` at full f32 before the multiply).
/// `ops::mlp::swiglu` is the rounded variant (bf16(silu) first) that the
/// qwen_drive checkpoint path needs; routing this name there would shift
/// every result by up to 1 ULP.
pub fn swiglu_bf16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::hybrid as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || gate_up.dtype() != DType::BF16 || dims[1] % 2 != 0 {
        return Err(Error::Other("SwiGLU requires a [rows, 2*cols] BF16 input".into()));
    }
    let (rows, inner) = (dims[0], dims[1] / 2);
    let r = i32::try_from(rows).map_err(|_| Error::Other("SwiGLU rows exceed i32".into()))?;
    let n = i32::try_from(inner).map_err(|_| Error::Other("SwiGLU inner exceeds i32".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![rows, inner]), DType::BF16)?;
    let input_buffer = CudaBuffer::from_tensor(gate_up).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_hybrid_swiglu_bf16(
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

/// `bias_gelu_bf16`: broadcast bias then tanh-GELU. A missing bias is a plain
/// GELU, matching the legacy contract.
pub fn bias_gelu_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = value;
    args.activation = ops::PointwiseActivation::Gelu;
    ops::pointwise(ctx, args)?;
    Ok(output)
}

/// `bias_silu_bf16`: broadcast bias then SiLU. A missing bias is a plain SiLU,
/// matching the legacy `bias_activation(.., activation = 2)` contract.
pub fn bias_silu_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = value;
    args.activation = ops::PointwiseActivation::Silu;
    ops::pointwise(ctx, args)?;
    Ok(output)
}

/// `bias_relu_bf16`: broadcast bias then ReLU (`fmaxf(value, 0.0f)` after the
/// bias add, matching the legacy `bias_relu_bf16_kernel`). A missing bias is a
/// plain ReLU.
pub fn bias_relu_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = value;
    args.activation = ops::PointwiseActivation::Relu;
    ops::pointwise(ctx, args)?;
    Ok(output)
}

/// `silu_mul_bf16`: fused BF16 `BF16(SiLU(gate)) * up` over separate,
/// equally-shaped inputs. Mirrors the legacy `silu_mul_separate_bf16` path,
/// whose SiLU intermediate rounds to BF16 before the multiply.
pub fn silu_mul_bf16(ctx: &CudaContext, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::transformer_bf16 as abi;
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
        raw::check_cuda(abi::apxinf_transformer_bf16_silu_mul_separate_bf16(
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

/// `silu_mul_quant_bf16_e4m3`: fused `BF16(BF16(SiLU(gate)) * up) -> calibrated
/// E4M3` over separate, equally-shaped inputs.
pub fn silu_mul_quant_bf16_e4m3(
    ctx: &CudaContext,
    gate: &Tensor,
    up: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::transformer_bf16 as abi;
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
        raw::check_cuda(abi::apxinf_transformer_bf16_silu_mul_quant_bf16_e4m3(
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

/// `bias_gelu_quant_bf16_e4m3`: fused `BF16(bias + tanh-GELU(input)) ->
/// calibrated E4M3` using packed-4 IO.
pub fn bias_gelu_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    bias: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::transformer_bf16 as abi;
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
        raw::check_cuda(abi::apxinf_transformer_bf16_bias_gelu_quant_bf16_e4m3(
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

/// `bias_gelu_bf16_packed8`: broadcast bias then tanh-GELU through a 16-byte
/// packed-8 IO path. Requires a BF16 matrix with a width divisible by eight.
pub fn bias_gelu_bf16_packed8(ctx: &CudaContext, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::transformer_bf16 as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("packed8 bias GELU expects a rank-2 input".into()));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if input.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || bias.shape().dims() != [cols]
        || cols % 8 != 0
    {
        return Err(Error::Other(
            "packed8 BF16 bias GELU has incompatible dtype or shape".into(),
        ));
    }
    let rows_i32 =
        i32::try_from(rows).map_err(|_| Error::Other("packed8 bias GELU rows exceed i32".into()))?;
    let cols_i32 =
        i32::try_from(cols).map_err(|_| Error::Other("packed8 bias GELU cols exceed i32".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_transformer_bf16_bias_gelu_bf16_packed8(
            input_buffer.ptr(),
            bias_buffer.ptr(),
            output_buffer.ptr(),
            rows_i32,
            cols_i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// `try_bias_activation_bf16_packed8`: the legacy packed-8 bias activation
/// policy hook. cuda-new does not expose this fast path; the caller falls back
/// to the generic `bias_gelu_bf16` / `bias_silu_bf16` contract.
pub fn try_bias_activation_bf16_packed8(
    _ctx: &CudaContext,
    _input: &Tensor,
    _bias: &Tensor,
    _activation: i32,
) -> Result<Option<Tensor>> {
    Ok(None)
}

/// `swiglu_quantize_rows_bf16_e4m3`: SwiGLU over a packed gate/up projection
/// followed by rowwise E4M3 quantization, zero-padded to `output_cols`.
pub fn swiglu_quantize_rows_bf16_e4m3(
    ctx: &CudaContext,
    gate_up: &Tensor,
    bias: Option<&Tensor>,
    logical_inner: usize,
    output_cols: usize,
) -> Result<super::quantization::DynamicFp8Tensor> {
    use crate::ffi::abi::{quant_fused as abi, status};
    use crate::CudaBuffer;
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
    let values = ctx.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
    let scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let gate_up_buffer = CudaBuffer::from_tensor(gate_up).map_err(Error::Cuda)?;
    let bias_buffer = bias.map(CudaBuffer::from_tensor).transpose().map_err(Error::Cuda)?;
    let values_buffer = CudaBuffer::from_tensor(&values).map_err(Error::Cuda)?;
    let scales_buffer = CudaBuffer::from_tensor(&scales).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_quant_swiglu_rows_bf16_e4m3(
            gate_up_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            values_buffer.ptr(),
            scales_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(input_cols, "input cols")?,
            to_i32(logical_inner, "inner")?,
            to_i32(output_cols, "output cols")?,
            ctx.stream().handle(),
        ))?;
    }
    Ok(super::quantization::DynamicFp8Tensor { values, scales })
}

/// `swiglu_bf16_rounded`: SwiGLU whose SiLU intermediate rounds to BF16
/// before the multiply — matching an unfused activation-then-multiply.
pub fn swiglu_bf16_rounded(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    use crate::ffi::abi::hybrid as abi;
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
        raw::check_cuda(abi::apxinf_hybrid_swiglu_bf16_rounded(
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
// `apxinf_transformer_f16_*` launchers (native/adapters/activation/fixed_f16.cu).

/// `swiglu_f16`: FP16 SwiGLU over packed `[rows, 2*inner]` gate/up halves.
pub fn swiglu_f16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    use super::validation::{check_cuda, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape};
    use crate::ffi::abi::transformer_f16 as abi;
    let (rows, twice_inner) = matrix_shape(gate_up, "SwiGLU")?;
    if gate_up.dtype() != DType::F16 || twice_inner % 2 != 0 {
        return Err(Error::Other(
            "static inference FP16 SwiGLU expects [rows,2*inner]".into(),
        ));
    }
    let inner = twice_inner / 2;
    let output = f16_output(ctx, rows, inner)?;
    unsafe {
        check_cuda(abi::apxinf_transformer_f16_swiglu_f16(
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

/// `bias_gelu_f16`: optional-bias tanh-GELU over an FP16 matrix.
pub fn bias_gelu_f16(
    ctx: &CudaContext,
    input: &Tensor,
    value: Option<&Tensor>,
) -> Result<Tensor> {
    use super::validation::{
        check_cuda, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape, optional_ptr,
    };
    use crate::ffi::abi::transformer_f16 as abi;
    let (rows, cols) = matrix_shape(input, "bias GELU")?;
    if input.dtype() != DType::F16
        || value.is_some_and(|bias| bias.dtype() != DType::F16 || bias.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference FP16 bias GELU has incompatible dtype or shape".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(abi::apxinf_transformer_f16_bias_gelu_f16(
            gpu_ptr(input)?,
            optional_ptr(value)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

/// `bias_silu_f16`: optional-bias SiLU over an FP16 matrix.
pub fn bias_silu_f16(ctx: &CudaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    use super::validation::{
        check_cuda, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape, optional_ptr,
    };
    use crate::ffi::abi::transformer_f16 as abi;
    let (rows, cols) = matrix_shape(input, "bias SiLU")?;
    if input.dtype() != DType::F16
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference bias SiLU expects an FP16 matrix and matching bias".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(abi::apxinf_transformer_f16_bias_silu_f16(
            gpu_ptr(input)?,
            optional_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───
//
// Direct launchers over the llm_decode adapter: no allocation, no resource
// creation, writes into caller-owned decode workspace buffers. Arithmetic is
// the legacy kernel's, copied verbatim into native/adapters/llm_decode.

/// Allocation-free SiLU into caller-owned decode storage.
pub fn silu_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    count: usize,
) -> Result<()> {
    use super::validation::{check_cuda, checked_bytes, require_buffers};
    use crate::ffi::abi::llm_decode as abi;
    let bytes = checked_bytes(dtype, &[count], "decode SiLU")?;
    require_buffers(
        ctx,
        "decode SiLU",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => abi::apxinf_llmdec_silu_f32(
                input.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_llmdec_silu_bf16(
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
pub fn silu_mul_bf16_into(
    ctx: &CudaContext,
    gate_up: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    intermediate: usize,
) -> Result<()> {
    use super::validation::{check_cuda, checked_bytes, require_buffers};
    use crate::ffi::abi::llm_decode as abi;
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
        abi::apxinf_llmdec_silu_mul_bf16(
            gate_up.ptr(),
            output.ptr(),
            intermediate as u32,
            ctx.stream().handle(),
        )
    })
}

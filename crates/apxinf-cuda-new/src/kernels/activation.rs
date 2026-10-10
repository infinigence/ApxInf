//! Legacy `kernels::activation` names over cuda-new pointwise operators.

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
pub fn swiglu_bf16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || dims[1] % 2 != 0 {
        return Err(Error::Other("SwiGLU requires a [rows, 2*cols] input".into()));
    }
    let output = ctx.allocate_output(Shape::new(vec![dims[0], dims[1] / 2]), DType::BF16)?;
    ops::mlp::swiglu(ctx, gate_up, &output)?;
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
    use crate::ffi::abi::vla_la as abi;
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
        raw::check_cuda(abi::apxinf_cn_swiglu_bf16_rounded(
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

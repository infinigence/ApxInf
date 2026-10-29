//! Out-of-place elementwise and activation operators.
//!
//! These back the portable [`apxinf_core::Backend`] trait so a model written
//! against `dyn Backend` (llama, qwen3-vl) can run on this runtime. Each has a
//! single fixed implementation with nothing to select, so per
//! `doc/adding-new-kernels.md` section 6 they are direct entry points rather
//! than registry-backed operators.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::ops::validation::check_cuda;
use crate::CudaContext;

/// Which elementwise activation a call applies. Matches the native encoding
/// in `elementwise_ops.cuh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementwiseActivation {
    None,
    GeluTanh,
    Silu,
}

impl ElementwiseActivation {
    fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::GeluTanh => 1,
            Self::Silu => 2,
        }
    }
}

fn element_count(tensor: &Tensor, what: &str) -> Result<i64> {
    let numel = tensor.shape().numel();
    i64::try_from(numel).map_err(|_| invalid(format!("{what} element count exceeds i64")))
}

/// `output[i] = activation(input[i])`, elementwise, F32 or BF16 in and out.
///
/// `output` and `input` must be distinct buffers with equal shapes; the
/// operator is out-of-place by contract, matching the trait it backs.
pub fn activation(
    ctx: &CudaContext,
    input: &Tensor,
    output: &Tensor,
    activation: ElementwiseActivation,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if output.shape().dims() != dims.as_slice() {
        return Err(invalid("activation input/output shape mismatch"));
    }
    if activation == ElementwiseActivation::None {
        return Err(invalid("activation requires gelu_tanh or silu"));
    }
    let count = element_count(input, "activation")?;
    if output.dtype() != input.dtype() {
        return Err(invalid("activation input/output dtype mismatch"));
    }
    let dtype = input.dtype();
    let input_buffer = tensor_storage(ctx, input, dtype, &dims)?;
    let output_buffer = tensor_storage(ctx, output, dtype, &dims)?;
    match dtype {
        DType::BF16 => unsafe {
            status::check(abi::apxinf_elementwise_activation_bf16(
                input_buffer.ptr(),
                output_buffer.ptr(),
                count,
                activation.code(),
                ctx.stream().handle(),
            ))
        },
        DType::F32 if activation == ElementwiseActivation::Silu => unsafe {
            check_cuda(
                crate::ffi::abi::activation::apxinf_activation_fixed_decode_silu_f32(
                    input_buffer.ptr(),
                    output_buffer.ptr(),
                    u32::try_from(count)
                        .map_err(|_| invalid("activation element count exceeds u32"))?,
                    ctx.stream().handle(),
                ),
            )
        },
        DType::F32 => Err(invalid("F32 activation currently supports SiLU")),
        _ => Err(invalid("activation currently supports F32 and BF16")),
    }
}

/// `output[i] = a[i] * b[i]`, with dtype inferred from the tensors.
pub fn mul(ctx: &CudaContext, a: &Tensor, b: &Tensor, output: &Tensor) -> Result<()> {
    let dims = a.shape().dims().to_vec();
    let dtype = a.dtype();
    if b.shape().dims() != dims.as_slice()
        || output.shape().dims() != dims.as_slice()
        || b.dtype() != dtype
        || output.dtype() != dtype
    {
        return Err(invalid(
            "mul requires three equally shaped tensors with one dtype",
        ));
    }
    let count = element_count(a, "mul")?;
    let a_buffer = tensor_storage(ctx, a, dtype, &dims)?;
    let b_buffer = tensor_storage(ctx, b, dtype, &dims)?;
    let output_buffer = tensor_storage(ctx, output, dtype, &dims)?;
    match dtype {
        DType::BF16 => unsafe {
            status::check(abi::apxinf_elementwise_mul_bf16(
                a_buffer.ptr(),
                b_buffer.ptr(),
                output_buffer.ptr(),
                count,
                ctx.stream().handle(),
            ))
        },
        DType::F32 => unsafe {
            check_cuda(abi::apxinf_elementwise_fixed_decode_mul_f32(
                a_buffer.ptr(),
                b_buffer.ptr(),
                output_buffer.ptr(),
                u32::try_from(count).map_err(|_| invalid("mul element count exceeds u32"))?,
                ctx.stream().handle(),
            ))
        },
        _ => Err(invalid("mul currently supports F32 and BF16")),
    }
}

/// `output[i] = a[i] + b[i]`, with dtype inferred from the tensors.
pub fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor, output: &Tensor) -> Result<()> {
    let dims = a.shape().dims().to_vec();
    let dtype = a.dtype();
    if b.shape().dims() != dims.as_slice()
        || output.shape().dims() != dims.as_slice()
        || b.dtype() != dtype
        || output.dtype() != dtype
    {
        return Err(invalid(
            "add requires three equally shaped tensors with one dtype",
        ));
    }
    let count = element_count(a, "add")?;
    let a_buffer = tensor_storage(ctx, a, dtype, &dims)?;
    let b_buffer = tensor_storage(ctx, b, dtype, &dims)?;
    let output_buffer = tensor_storage(ctx, output, dtype, &dims)?;
    match dtype {
        DType::BF16 => unsafe {
            status::check(abi::apxinf_elementwise_add_bf16(
                a_buffer.ptr(),
                b_buffer.ptr(),
                output_buffer.ptr(),
                count,
                ctx.stream().handle(),
            ))
        },
        DType::F16 => unsafe {
            check_cuda(abi::apxinf_elementwise_add_f16(
                a_buffer.ptr(),
                b_buffer.ptr(),
                output_buffer.ptr(),
                u32::try_from(count).map_err(|_| invalid("add element count exceeds u32"))?,
                ctx.stream().handle(),
            ))
        },
        DType::F32 => unsafe {
            check_cuda(abi::apxinf_elementwise_fixed_decode_add_f32(
                a_buffer.ptr(),
                b_buffer.ptr(),
                output_buffer.ptr(),
                u32::try_from(count).map_err(|_| invalid("add element count exceeds u32"))?,
                ctx.stream().handle(),
            ))
        },
        _ => Err(invalid("add currently supports F32, F16, and BF16")),
    }
}

/// `output[i] = input[i] * factor`, with dtype inferred from the tensors.
pub fn scale(ctx: &CudaContext, input: &Tensor, output: &Tensor, factor: f32) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    let dtype = input.dtype();
    if output.shape().dims() != dims.as_slice() || output.dtype() != dtype {
        return Err(invalid("scale input/output shape or dtype mismatch"));
    }
    if !factor.is_finite() {
        return Err(invalid("scale factor must be finite"));
    }
    let count = element_count(input, "scale")?;
    let input_buffer = tensor_storage(ctx, input, dtype, &dims)?;
    let output_buffer = tensor_storage(ctx, output, dtype, &dims)?;
    match dtype {
        DType::BF16 => unsafe {
            status::check(abi::apxinf_elementwise_scale_bf16(
                input_buffer.ptr(),
                output_buffer.ptr(),
                count,
                factor,
                ctx.stream().handle(),
            ))
        },
        DType::F16 => unsafe {
            check_cuda(abi::apxinf_elementwise_scale_f16(
                input_buffer.ptr(),
                output_buffer.ptr(),
                u32::try_from(count).map_err(|_| invalid("scale element count exceeds u32"))?,
                factor,
                ctx.stream().handle(),
            ))
        },
        DType::F32 => unsafe {
            check_cuda(abi::apxinf_elementwise_fixed_decode_scale_f32(
                input_buffer.ptr(),
                output_buffer.ptr(),
                u32::try_from(count).map_err(|_| invalid("scale element count exceeds u32"))?,
                factor,
                ctx.stream().handle(),
            ))
        },
        _ => Err(invalid("scale currently supports F32, F16, and BF16")),
    }
}

/// `output[r, c] = input[r, c] + bias[c]`, broadcasting a length-`cols` BF16
/// bias vector over `rows` rows.
pub fn add_bias(ctx: &CudaContext, input: &Tensor, bias: &Tensor, output: &Tensor) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 || output.shape().dims() != dims.as_slice() {
        return Err(invalid("add_bias expects equally shaped rank-2 tensors"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if bias.shape().dims() != [cols] {
        return Err(invalid("add_bias bias must be a length-cols vector"));
    }
    let rows_i64 = i64::try_from(rows).map_err(|_| invalid("add_bias rows exceed i64"))?;
    let cols_i64 = i64::try_from(cols).map_err(|_| invalid("add_bias cols exceed i64"))?;
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let bias_buffer = tensor_storage(ctx, bias, DType::BF16, &[cols])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_elementwise_add_bias_bf16(
            input_buffer.ptr(),
            bias_buffer.ptr(),
            output_buffer.ptr(),
            rows_i64,
            cols_i64,
            ctx.stream().handle(),
        ))
    }
}

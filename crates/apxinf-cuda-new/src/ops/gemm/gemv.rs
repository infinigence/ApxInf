//! Fixed single-token GEMV operations.

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::cublas::CublasTranspose;
use crate::ffi::abi::{gemm as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::ops::validation::checked_bytes;
use crate::CudaBuffer;
use crate::CudaContext;

#[derive(Clone, Copy)]
pub struct GemvBlockScales<'a> {
    pub weight: &'a Tensor,
    pub activation: &'a Tensor,
    pub block_size: u32,
}

pub struct GemvArgs<'a> {
    pub weight: &'a Tensor,
    pub activation: &'a Tensor,
    pub output: &'a Tensor,
    pub block_scales: Option<GemvBlockScales<'a>>,
    pub alpha: f32,
}

/// Single-token projection with a bias:
/// `output[n] = bias[n] + sum_k weight[n,k] * activation[k]`.
///
/// The current implementation accepts BF16 tensors and preserves the legacy
/// FP32-accumulator contract: the bias and dot product are combined in FP32,
/// then rounded once when written to the caller-owned output.
pub struct GemvBiasArgs<'a> {
    pub weight: &'a Tensor,
    pub activation: &'a Tensor,
    pub bias: &'a Tensor,
    pub output: &'a Tensor,
}

/// Single-token projection: `y[n] = alpha * sum_k weight[n,k] * a[k]`.
///
/// Unit-scale E4M3 operands use no extra binding. Packed E2M1 operands carry
/// their block scales through `block_scales`. Both preserved CUDA
/// implementations remain private dispatch targets of this one semantic L3.
pub fn gemv(ctx: &CudaContext, args: GemvArgs<'_>) -> Result<()> {
    match args.block_scales {
        None => gemv_unit_scale(ctx, &args),
        Some(scales) => gemv_block_quantized(ctx, &args, scales),
    }
}

pub fn gemv_bias(ctx: &CudaContext, args: GemvBiasArgs<'_>) -> Result<()> {
    let weight_dims = args.weight.shape().dims();
    if weight_dims.len() != 2
        || weight_dims.contains(&0)
        || args.activation.shape().dims() != [weight_dims[1]]
        || args.bias.shape().dims() != [weight_dims[0]]
        || args.output.shape().dims() != [weight_dims[0]]
    {
        return Err(Error::Other(
            "GEMV+bias expects weight[N,K], activation[K], bias[N], output[N]".into(),
        ));
    }
    for tensor in [args.weight, args.activation, args.bias, args.output] {
        if tensor.dtype() != DType::BF16 || tensor.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "GEMV+bias currently requires BF16 tensors on the context device".into(),
            ));
        }
        checked_bytes(DType::BF16, tensor.shape().dims(), "GEMV+bias")?;
    }

    let k = i32::try_from(weight_dims[1])
        .map_err(|_| Error::Other("GEMV+bias input width overflow".into()))?;
    let n = i32::try_from(weight_dims[0])
        .map_err(|_| Error::Other("GEMV+bias output width overflow".into()))?;
    let accumulator = crate::workspace::output_buffer(
        ctx,
        checked_bytes(DType::F32, &[weight_dims[0]], "GEMV+bias accumulator")?,
    )?
    .into_tensor(apxinf_core::Shape::new(vec![weight_dims[0]]), DType::F32);
    crate::ops::quantization::cast(ctx, args.bias, &accumulator)?;

    let weight = CudaBuffer::from_tensor(args.weight).map_err(Error::Cuda)?;
    let activation = CudaBuffer::from_tensor(args.activation).map_err(Error::Cuda)?;
    let accumulator_buffer = CudaBuffer::from_tensor(&accumulator).map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm_bf16_f32_ex(
            CublasTranspose::None,
            CublasTranspose::Transpose,
            1,
            weight_dims[0],
            weight_dims[1],
            1.0,
            &activation,
            k,
            &weight,
            k,
            1.0,
            &accumulator_buffer,
            n,
        )
        .map_err(Error::Cuda)?;
    crate::ops::quantization::cast(ctx, &accumulator, args.output)
}

fn gemv_unit_scale(ctx: &CudaContext, args: &GemvArgs<'_>) -> Result<()> {
    let weight_dims = args.weight.shape().dims().to_vec();
    if weight_dims.len() != 2 {
        return Err(invalid("FP8 GEMV weight must be [N, K]"));
    }
    let (n, k) = (weight_dims[0], weight_dims[1]);
    let weight_buffer = tensor_storage(ctx, args.weight, DType::F8E4M3, &weight_dims)?;
    let activation_dims = args.activation.shape().dims().to_vec();
    if activation_dims.iter().product::<usize>() != k {
        return Err(invalid("FP8 GEMV activation must hold K elements"));
    }
    let activation_buffer = tensor_storage(ctx, args.activation, DType::F8E4M3, &activation_dims)?;
    let output_dims = args.output.shape().dims().to_vec();
    if output_dims.iter().product::<usize>() != n {
        return Err(invalid("FP8 GEMV output must hold N elements"));
    }
    let output_buffer = tensor_storage(ctx, args.output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_gemm_fp8_gemv(
            weight_buffer.ptr(),
            activation_buffer.ptr(),
            output_buffer.ptr(),
            n as i64,
            k as i64,
            args.alpha,
            ctx.stream().handle(),
        ))
    }
}

fn gemv_block_quantized(
    ctx: &CudaContext,
    args: &GemvArgs<'_>,
    scales: GemvBlockScales<'_>,
) -> Result<()> {
    let weight_dims = args.weight.shape().dims().to_vec();
    if weight_dims.len() != 2 {
        return Err(invalid("NVFP4 GEMV weight must be [N, K/2]"));
    }
    if scales.block_size != 16 {
        return Err(invalid(
            "block-quantized GEMV currently requires block_size 16",
        ));
    }
    let (n, k) = (weight_dims[0], weight_dims[1] * 2);
    let weight_buffer = tensor_storage(ctx, args.weight, DType::E2M1Pair, &weight_dims)?;
    let weight_scale_dims = scales.weight.shape().dims().to_vec();
    let weight_scale_buffer =
        tensor_storage(ctx, scales.weight, DType::F8E4M3, &weight_scale_dims)?;
    if weight_scale_buffer.len() < n * k / scales.block_size as usize {
        return Err(invalid("block-quantized GEMV weight scales are too short"));
    }
    let activation_dims = args.activation.shape().dims().to_vec();
    let activation_buffer =
        tensor_storage(ctx, args.activation, DType::E2M1Pair, &activation_dims)?;
    if activation_buffer.len() < k / 2 {
        return Err(invalid("NVFP4 GEMV activation must hold K/2 bytes"));
    }
    let activation_scale_dims = scales.activation.shape().dims().to_vec();
    let activation_scale_buffer = tensor_storage(
        ctx,
        scales.activation,
        DType::F8E4M3,
        &activation_scale_dims,
    )?;
    if activation_scale_buffer.len() < k / scales.block_size as usize {
        return Err(invalid(
            "block-quantized GEMV activation scales are too short",
        ));
    }
    let output_dims = args.output.shape().dims().to_vec();
    if output_dims.iter().product::<usize>() != n {
        return Err(invalid("NVFP4 GEMV output must hold N elements"));
    }
    let output_buffer = tensor_storage(ctx, args.output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_gemm_nvfp4_gemv(
            weight_buffer.ptr(),
            weight_scale_buffer.ptr(),
            activation_buffer.ptr(),
            activation_scale_buffer.ptr(),
            output_buffer.ptr(),
            n as i64,
            k as i64,
            args.alpha,
            ctx.stream().handle(),
        ))
    }
}

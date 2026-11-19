use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::conv2d as abi;
use crate::{CudaBuffer, CudaContext};

/// Channel-last two-dimensional convolution (groups 1, dilation 1).
///
/// Input is `[batch, in_h, in_w, in_channels]` with contiguous row-major
/// storage. `weight` is `[out_channels, in_channels, kernel_h, kernel_w]`.
/// Bias is `[out_channels]`. The output is `[batch, out_h, out_w, out_channels]`
/// with `out = (in + 2*pad - kernel) / stride + 1` per spatial axis.
pub struct Conv2dArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
}

impl<'a> Conv2dArgs<'a> {
    pub fn new(input: &'a Tensor, weight: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            input,
            weight,
            bias: None,
            out,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        }
    }

    pub fn with_bias(mut self, bias: &'a Tensor) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn with_stride(mut self, stride_h: usize, stride_w: usize) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    pub fn with_padding(mut self, pad_h: usize, pad_w: usize) -> Self {
        self.pad_h = pad_h;
        self.pad_w = pad_w;
        self
    }
}

pub(crate) struct Normalized {
    pub spec: abi::Spec,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Conv2d supports F32, F16, and BF16 tensors only")),
    }
}

fn alignment_of(pointer: usize) -> u32 {
    if pointer == 0 {
        return 256;
    }
    (1usize << pointer.trailing_zeros().min(8)) as u32
}

fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
    label: &str,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != dtype {
        return Err(invalid(format!("Conv2d {label} device/dtype mismatch")));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid(format!("Conv2d {label} element count mismatch")));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("Conv2d size overflow"))?;
    if buffer.len() < expected {
        return Err(invalid(format!("Conv2d {label} storage is too small")));
    }
    Ok(buffer)
}

fn conv_output_size(input: usize, kernel: usize, stride: usize, padding: usize) -> Option<usize> {
    let numerator = input
        .checked_add(padding.checked_mul(2)?)?
        .checked_sub(kernel)?;
    Some(numerator / stride + 1)
}

fn product(factors: &[usize], label: &str) -> Result<usize> {
    factors
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| invalid(format!("Conv2d {label} size overflow")))
}

fn abi_i64(value: usize, label: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| invalid(format!("Conv2d {label} exceeds i64")))
}

pub(crate) fn normalize(ctx: &CudaContext, args: Conv2dArgs<'_>) -> Result<Normalized> {
    if args.stride_h == 0 || args.stride_w == 0 {
        return Err(invalid("Conv2d strides must be nonzero"));
    }
    let input_dims = args.input.shape().dims().to_vec();
    let weight_dims = args.weight.shape().dims().to_vec();
    let out_dims = args.out.shape().dims().to_vec();
    if input_dims.len() != 4 || weight_dims.len() != 4 || out_dims.len() != 4 {
        return Err(invalid(
            "Conv2d expects rank-4 input, weight, and output tensors",
        ));
    }
    let (batch, in_h, in_w, in_channels) =
        (input_dims[0], input_dims[1], input_dims[2], input_dims[3]);
    let (kernel_h, kernel_w) = (weight_dims[2], weight_dims[3]);
    let (out_h, out_w, out_channels) = (out_dims[1], out_dims[2], out_dims[3]);
    if batch == 0 || in_h == 0 || in_w == 0 || in_channels == 0 || kernel_h == 0 || kernel_w == 0 {
        return Err(invalid("Conv2d input geometry must be non-empty"));
    }
    if out_dims[0] != batch {
        return Err(invalid("Conv2d output batch must match the input"));
    }
    if weight_dims[0] != out_channels || weight_dims[1] != in_channels {
        return Err(invalid(
            "Conv2d weight must be [out_channels, in_channels, kernel_h, kernel_w]",
        ));
    }
    let expected_h = conv_output_size(in_h, kernel_h, args.stride_h, args.pad_h)
        .ok_or_else(|| invalid("Conv2d output height underflows"))?;
    let expected_w = conv_output_size(in_w, kernel_w, args.stride_w, args.pad_w)
        .ok_or_else(|| invalid("Conv2d output width underflows"))?;
    if expected_h != out_h || expected_w != out_w {
        return Err(invalid(format!(
            "Conv2d output {out_h}x{out_w} does not match the geometry {expected_h}x{expected_w}"
        )));
    }
    if args.input.dtype() != args.weight.dtype() || args.input.dtype() != args.out.dtype() {
        return Err(invalid("Conv2d input, weight, and output dtypes must match"));
    }
    let dtype = args.input.dtype();
    let dtype_code = dtype_code(dtype)?;

    let input_count = product(&[batch, in_h, in_w, in_channels], "input")?;
    let weight_count = product(&weight_dims, "weight")?;
    let output_count = product(&[batch, out_h, out_w, out_channels], "output")?;

    let mut storage = Vec::new();

    let input_buffer = tensor_storage(ctx, args.input, dtype, input_count, "input")?;
    let input = input_buffer.ptr() as *const std::ffi::c_void;
    let input_alignment = alignment_of(input_buffer.ptr() as usize);
    storage.push(input_buffer);

    let weight_buffer = tensor_storage(ctx, args.weight, dtype, weight_count, "weight")?;
    let weight = weight_buffer.ptr() as *const std::ffi::c_void;
    let weight_alignment = alignment_of(weight_buffer.ptr() as usize);
    storage.push(weight_buffer);

    let (bias, bias_alignment) = match args.bias {
        Some(tensor) => {
            let buffer = tensor_storage(ctx, tensor, dtype, out_channels, "bias")?;
            let pointer = buffer.ptr() as *const std::ffi::c_void;
            let alignment = alignment_of(buffer.ptr() as usize);
            storage.push(buffer);
            (pointer, alignment)
        }
        None => (std::ptr::null(), 256),
    };

    let output_buffer = tensor_storage(ctx, args.out, dtype, output_count, "output")?;
    let output = output_buffer.ptr() as *mut std::ffi::c_void;
    let output_alignment = alignment_of(output_buffer.ptr() as usize);
    storage.push(output_buffer);

    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        dtype: dtype_code,
        has_bias: u32::from(!bias.is_null()),
        input_alignment,
        weight_alignment,
        bias_alignment,
        output_alignment,
        reserved: 0,
        batch: abi_i64(batch, "batch")?,
        in_h: abi_i64(in_h, "in_h")?,
        in_w: abi_i64(in_w, "in_w")?,
        in_channels: abi_i64(in_channels, "in_channels")?,
        out_channels: abi_i64(out_channels, "out_channels")?,
        kernel_h: abi_i64(kernel_h, "kernel_h")?,
        kernel_w: abi_i64(kernel_w, "kernel_w")?,
        stride_h: abi_i64(args.stride_h, "stride_h")?,
        stride_w: abi_i64(args.stride_w, "stride_w")?,
        pad_h: abi_i64(args.pad_h, "pad_h")?,
        pad_w: abi_i64(args.pad_w, "pad_w")?,
        out_h: abi_i64(out_h, "out_h")?,
        out_w: abi_i64(out_w, "out_w")?,
    };
    let bindings = abi::Bindings {
        input,
        weight,
        bias,
        output,
        stream: ctx.stream().handle(),
    };
    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    super::execution::execute(ctx, normalized)
}

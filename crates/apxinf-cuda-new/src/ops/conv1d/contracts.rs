use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::conv1d as abi;
use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Conv1dSemantic {
    Conv1d,
    ConvTranspose1d,
}

impl Conv1dSemantic {
    pub const ALL: [Self; 2] = [Self::Conv1d, Self::ConvTranspose1d];
    pub fn doc_id(self) -> &'static str {
        match self {
            Self::Conv1d => "conv1d",
            Self::ConvTranspose1d => "conv-transpose1d",
        }
    }
    fn code(self) -> u32 {
        match self {
            Self::Conv1d => abi::SEMANTIC_CONV1D,
            Self::ConvTranspose1d => abi::SEMANTIC_CONV_TRANSPOSE1D,
        }
    }
}

/// Accumulation constraint for a convolution feeding a discrete decision.
/// OrderedF32 preserves the direct baseline's sequential FP32 accumulation;
/// Default permits deterministic optimized accumulation orders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Conv1dAccumulation {
    #[default]
    Default,
    OrderedF32,
}

/// Channel-last one-dimensional convolution.
///
/// Input is `[batch, in_length, in_channels]` with contiguous row-major
/// storage. `weight` is `[out_channels, in_channels / groups, kernel]` for
/// `Conv1d` and `[in_channels, out_channels / groups, kernel]` for
/// `ConvTranspose1d`. Bias is `[out_channels]`. The output is
/// `[batch, out_length, out_channels]`.
pub struct Conv1dArgs<'a> {
    pub semantic: Conv1dSemantic,
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub output_padding: usize,
    pub accumulation: Conv1dAccumulation,
    /// Bind channel-last buffers directly, skipping both layout transforms, and
    /// leave the bias to a downstream kernel. The caller must then add the bias
    /// with the same rounding the ordinary path performs when it writes the
    /// result: `round(dtype_value(x + bias))` after a float add.
    pub direct_channel_last: bool,
}

impl<'a> Conv1dArgs<'a> {
    pub fn new(
        semantic: Conv1dSemantic,
        input: &'a Tensor,
        weight: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        Self {
            semantic,
            input,
            weight,
            bias: None,
            out,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            output_padding: 0,
            accumulation: Conv1dAccumulation::Default,
            direct_channel_last: false,
        }
    }

    pub fn with_bias(mut self, bias: &'a Tensor) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Bind the caller's channel-last buffers in place and defer the bias.
    pub fn with_direct_channel_last(mut self) -> Self {
        self.direct_channel_last = true;
        self
    }
}

pub(crate) struct Normalized {
    pub spec: abi::Spec,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
    pub accumulation: Conv1dAccumulation,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Conv1d supports F32, F16, and BF16 tensors only")),
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
        return Err(invalid(format!("Conv1d {label} device/dtype mismatch")));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid(format!("Conv1d {label} element count mismatch")));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("Conv1d size overflow"))?;
    if buffer.len() < expected {
        return Err(invalid(format!("Conv1d {label} storage is too small")));
    }
    Ok(buffer)
}

fn conv_output_length(
    in_length: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let span = dilation
        .checked_mul(kernel.checked_sub(1)?)?
        .checked_add(1)?;
    let numerator = in_length
        .checked_add(padding.checked_mul(2)?)?
        .checked_sub(span)?;
    Some(numerator / stride + 1)
}

/// Expected `ConvTranspose1d` output length, matching PyTorch:
/// `(in_length - 1) * stride - 2 * padding + dilation * (kernel - 1)
/// + output_padding + 1`. `None` means the geometry yields a non-positive
/// output length, the same case PyTorch rejects.
pub fn transpose_output_length(
    in_length: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_padding: usize,
) -> Option<usize> {
    // PyTorch: (in_length - 1) * stride - 2 * padding
    //     + dilation * (kernel - 1) + output_padding + 1.
    // The dilation term is added before the padding is subtracted: an
    // intermediate value of (in_length - 1) * stride - 2 * padding may be
    // negative for valid geometry (e.g. in_length = 1 with padding covering
    // the kernel), while a negative total is rejected, as PyTorch does.
    let base = in_length
        .checked_sub(1)?
        .checked_mul(stride)?
        .checked_add(dilation.checked_mul(kernel.checked_sub(1)?)?)?
        .checked_add(output_padding)?
        .checked_add(1)?;
    let length = base.checked_sub(padding.checked_mul(2)?)?;
    if length == 0 {
        // Zero is still a non-positive output length: callers reject `None`
        // rather than allocating an empty tensor, and PyTorch likewise
        // refuses a zero-size conv_transpose output.
        return None;
    }
    Some(length)
}

pub(crate) fn normalize(ctx: &CudaContext, args: Conv1dArgs<'_>) -> Result<Normalized> {
    if args.semantic == Conv1dSemantic::Conv1d && args.output_padding != 0 {
        return Err(invalid("Conv1d output_padding must be zero"));
    }
    if args.direct_channel_last && args.bias.is_some() {
        return Err(invalid(
            "direct channel-last Conv1d defers the bias to a downstream kernel and takes none",
        ));
    }
    if args.stride == 0 || args.dilation == 0 || args.groups == 0 {
        return Err(invalid(
            "Conv1d stride, dilation, and groups must be nonzero",
        ));
    }
    let input_dims = args.input.shape().dims().to_vec();
    let weight_dims = args.weight.shape().dims().to_vec();
    let out_dims = args.out.shape().dims().to_vec();
    if input_dims.len() != 3 || weight_dims.len() != 3 || out_dims.len() != 3 {
        return Err(invalid(
            "Conv1d expects rank-3 input, weight, and output tensors",
        ));
    }
    let (batch, in_length, in_channels) = (input_dims[0], input_dims[1], input_dims[2]);
    let kernel = weight_dims[2];
    let (out_channels, out_length) = (out_dims[2], out_dims[1]);
    if batch == 0 || in_length == 0 || in_channels == 0 || kernel == 0 {
        return Err(invalid("Conv1d input geometry must be non-empty"));
    }
    if out_dims[0] != batch {
        return Err(invalid("Conv1d output batch must match the input"));
    }
    let in_per_group = in_channels / args.groups;
    let out_per_group = out_channels / args.groups;
    if in_channels % args.groups != 0 || out_channels % args.groups != 0 {
        return Err(invalid("Conv1d channels must be divisible by groups"));
    }
    match args.semantic {
        Conv1dSemantic::Conv1d => {
            if weight_dims[0] != out_channels || weight_dims[1] != in_per_group {
                return Err(invalid(
                    "Conv1d weight must be [out_channels, in_channels/groups, kernel]",
                ));
            }
        }
        Conv1dSemantic::ConvTranspose1d => {
            if weight_dims[0] != in_channels || weight_dims[1] != out_per_group {
                return Err(invalid(
                    "ConvTranspose1d weight must be [in_channels, out_channels/groups, kernel]",
                ));
            }
            if args.output_padding >= args.stride {
                return Err(invalid(
                    "ConvTranspose1d output_padding must be smaller than stride",
                ));
            }
            if args.dilation != 1 {
                return Err(invalid(
                    "ConvTranspose1d currently supports dilation 1 only",
                ));
            }
        }
    }
    let expected = match args.semantic {
        Conv1dSemantic::Conv1d => {
            conv_output_length(in_length, kernel, args.stride, args.padding, args.dilation)
        }
        Conv1dSemantic::ConvTranspose1d => transpose_output_length(
            in_length,
            kernel,
            args.stride,
            args.padding,
            args.dilation,
            args.output_padding,
        ),
    }
    .ok_or_else(|| invalid("Conv1d output length underflows"))?;
    if expected != out_length {
        return Err(invalid(format!(
            "Conv1d output length {out_length} does not match the geometry {expected}"
        )));
    }
    if args.input.dtype() != args.weight.dtype() || args.input.dtype() != args.out.dtype() {
        return Err(invalid(
            "Conv1d input, weight, and output dtypes must match",
        ));
    }
    let dtype = args.input.dtype();
    let _ = dtype_code(dtype)?;

    let mut storage = Vec::new();
    let input_count = batch
        .checked_mul(in_length)
        .and_then(|value| value.checked_mul(in_channels))
        .ok_or_else(|| invalid("Conv1d input size overflow"))?;
    let weight_count = weight_dims
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| invalid("Conv1d weight size overflow"))?;
    let output_count = batch
        .checked_mul(out_length)
        .and_then(|value| value.checked_mul(out_channels))
        .ok_or_else(|| invalid("Conv1d output size overflow"))?;

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
        semantic: args.semantic.code(),
        dtype: dtype_code(dtype)?,
        has_bias: u32::from(!bias.is_null()),
        direct_channel_last: u32::from(args.direct_channel_last),
        input_alignment,
        weight_alignment,
        bias_alignment,
        output_alignment,
        batch: abi_i64(batch, "batch")?,
        in_length: abi_i64(in_length, "in_length")?,
        in_channels: abi_i64(in_channels, "in_channels")?,
        out_channels: abi_i64(out_channels, "out_channels")?,
        kernel: abi_i64(kernel, "kernel")?,
        stride: abi_i64(args.stride, "stride")?,
        padding: abi_i64(args.padding, "padding")?,
        dilation: abi_i64(args.dilation, "dilation")?,
        groups: abi_i64(args.groups, "groups")?,
        output_padding: abi_i64(args.output_padding, "output_padding")?,
        out_length: abi_i64(out_length, "out_length")?,
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
        accumulation: args.accumulation,
    })
}

fn abi_i64(value: usize, label: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| invalid(format!("Conv1d {label} exceeds i64")))
}

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    super::execution::execute(ctx, normalized)
}

#[cfg(test)]
mod transpose_output_length_tests {
    use super::transpose_output_length;

    /// PyTorch reference: `(in_length - 1) * stride - 2 * padding
    /// + dilation * (kernel - 1) + output_padding + 1`. Raw formula value:
    /// this keeps `Some(0)` for a zero total, while
    /// `transpose_output_length` itself maps non-positive totals to `None`,
    /// so the cross-checks below avoid zero-length cases.
    fn expected(
        in_length: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_padding: usize,
    ) -> Option<usize> {
        let total = (in_length as i64 - 1) * stride as i64 - 2 * padding as i64
            + dilation as i64 * (kernel as i64 - 1)
            + output_padding as i64
            + 1;
        usize::try_from(total).ok()
    }

    #[test]
    fn matches_pytorch_formula() {
        let cases = [
            (1usize, 3usize, 1usize, 1usize, 1usize, 0usize),
            (4, 3, 2, 1, 1, 0),
            (5, 3, 2, 1, 1, 0),
            (4, 3, 2, 1, 1, 1),
            (8, 5, 3, 2, 1, 0),
            (12, 4, 4, 2, 1, 3),
        ];
        for &(in_length, kernel, stride, padding, dilation, output_padding) in &cases {
            assert_eq!(
                transpose_output_length(
                    in_length,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    output_padding
                ),
                expected(in_length, kernel, stride, padding, dilation, output_padding),
                "in_length={in_length} kernel={kernel} stride={stride} padding={padding} \
                 dilation={dilation} output_padding={output_padding}"
            );
        }
    }

    #[test]
    fn padding_larger_than_the_strided_input_still_admits_valid_geometry() {
        // (1 - 1) * 1 - 2 * 1 + 1 * (3 - 1) + 0 + 1 = 1. The intermediate
        // (in_length - 1) * stride - 2 * padding is negative here; only the
        // total decides.
        assert_eq!(transpose_output_length(1, 3, 1, 1, 1, 0), Some(1));
    }

    #[test]
    fn non_positive_output_length_is_rejected() {
        // (1 - 1) * 1 - 2 * 3 + 1 * 2 + 0 + 1 = -3, which PyTorch rejects.
        assert_eq!(transpose_output_length(1, 3, 1, 3, 1, 0), None);
        // (1 - 1) * 4 - 2 * 4 + 1 * 1 + 0 + 1 = -6.
        assert_eq!(transpose_output_length(1, 2, 4, 4, 1, 0), None);
        // (1 - 1) * 1 - 2 * 1 + 1 * 1 + 0 + 1 = 0: a zero total is also
        // non-positive, so callers refuse it instead of allocating an empty
        // tensor.
        assert_eq!(transpose_output_length(1, 2, 1, 1, 1, 0), None);
    }
}

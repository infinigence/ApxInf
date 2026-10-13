use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::pointwise as abi;
use crate::{CudaBuffer, CudaContext};

/// Which element-wise operation the bindings describe. See
/// `pointwise_types.h` for the binding table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointwiseSemantic {
    /// `input [rows, 2 * cols]` -> `output [rows, cols]`.
    Geglu,
    BiasActivation,
    EulerUpdate,
}

impl PointwiseSemantic {
    fn code(self) -> u32 {
        match self {
            Self::Geglu => 0,
            Self::BiasActivation => 1,
            Self::EulerUpdate => 2,
        }
    }

    fn reads_secondary(self) -> bool {
        matches!(self, Self::EulerUpdate)
    }

    fn may_have_bias(self) -> bool {
        matches!(self, Self::BiasActivation)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointwiseActivation {
    None,
    Gelu,
    Silu,
}

impl PointwiseActivation {
    fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Gelu => 1,
            Self::Silu => 2,
        }
    }
}

/// Element-wise operation over a contiguous `[rows, cols]` activation, where
/// `cols` is the *output* width.
pub struct PointwiseArgs<'a> {
    pub semantic: PointwiseSemantic,
    pub input: &'a Tensor,
    /// Euler-update velocity; unused by the other semantics.
    pub secondary: Option<&'a Tensor>,
    pub bias: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub activation: PointwiseActivation,
    pub dt: f32,
    pub output_scale: f32,
}

impl<'a> PointwiseArgs<'a> {
    pub fn new(semantic: PointwiseSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input,
            secondary: None,
            bias: None,
            out,
            activation: PointwiseActivation::None,
            dt: 0.0,
            output_scale: 1.0,
        }
    }
}

pub(crate) struct Normalized {
    pub spec: abi::Spec,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn input_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Pointwise input currently supports F16 and BF16")),
    }
}

fn output_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::F8E4M3 => Ok(3),
        _ => Err(invalid(
            "Pointwise output currently supports F16, BF16, and E4M3",
        )),
    }
}

fn alignment_of(pointer: usize) -> u32 {
    if pointer == 0 {
        return 256;
    }
    (1usize << pointer.trailing_zeros().min(8)) as u32
}

fn storage_ranges_overlap(
    left: usize,
    left_bytes: usize,
    right: usize,
    right_bytes: usize,
) -> Result<bool> {
    let left_end = left
        .checked_add(left_bytes)
        .ok_or_else(|| invalid("Pointwise input storage address range overflow"))?;
    let right_end = right
        .checked_add(right_bytes)
        .ok_or_else(|| invalid("Pointwise output storage address range overflow"))?;
    Ok(left < right_end && right < left_end)
}

fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != dtype {
        return Err(invalid("Pointwise tensor device/dtype mismatch"));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid("Pointwise tensor element count mismatch"));
    }
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("Pointwise size overflow"))?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < expected || (buffer.ptr() as usize) % dtype.size_in_bytes() != 0 {
        return Err(invalid("Pointwise tensor storage is invalid"));
    }
    Ok(buffer)
}

pub(crate) fn normalize(ctx: &CudaContext, args: PointwiseArgs<'_>) -> Result<Normalized> {
    let semantic = args.semantic;
    let dims = args.out.shape().dims();
    if dims.len() != 2 {
        return Err(invalid("Pointwise output must be rank 2 [rows, cols]"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if rows == 0 || cols == 0 {
        return Err(invalid("Pointwise output must be non-empty"));
    }
    let rows_abi = i32::try_from(rows)
        .map_err(|_| invalid("Pointwise row count exceeds the CUDA kernel range"))?;
    let cols_abi = i32::try_from(cols)
        .map_err(|_| invalid("Pointwise column count exceeds the CUDA kernel range"))?;
    let input_dtype = args.input.dtype();
    let output_dtype = args.out.dtype();
    let quantized_geglu = matches!(semantic, PointwiseSemantic::Geglu)
        && input_dtype == DType::F16
        && output_dtype == DType::F8E4M3;
    if !quantized_geglu && input_dtype != output_dtype {
        return Err(invalid(
            "Pointwise input/output dtypes must match except for F16-to-E4M3 GeGLU",
        ));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Pointwise size overflow"))?;
    if quantized_geglu {
        i32::try_from(count)
            .map_err(|_| invalid("F16-to-E4M3 GeGLU element count exceeds kernel range"))?;
    }
    // GeGLU consumes a gate and an up half per output element.
    let input_count = if matches!(semantic, PointwiseSemantic::Geglu) {
        count
            .checked_mul(2)
            .ok_or_else(|| invalid("Pointwise size overflow"))?
    } else {
        count
    };

    if !args.dt.is_finite() {
        return Err(invalid("Pointwise dt must be finite"));
    }
    if !(args.output_scale.is_finite() && args.output_scale > 0.0) {
        return Err(invalid(
            "Pointwise output_scale must be finite and positive",
        ));
    }
    if !quantized_geglu && args.output_scale != 1.0 {
        return Err(invalid(
            "Pointwise output_scale is only supported for F16-to-E4M3 GeGLU",
        ));
    }
    if quantized_geglu && cols % 2 != 0 {
        return Err(invalid(
            "F16-to-E4M3 GeGLU requires an even output width",
        ));
    }
    if args.bias.is_some() && !semantic.may_have_bias() {
        return Err(invalid("Pointwise semantic does not take a bias"));
    }
    if args.secondary.is_some() && !semantic.reads_secondary() {
        return Err(invalid(
            "Pointwise semantic does not take a secondary input",
        ));
    }
    if args.activation != PointwiseActivation::None
        && !matches!(semantic, PointwiseSemantic::BiasActivation)
    {
        return Err(invalid("Pointwise semantic does not take an activation"));
    }
    if matches!(semantic, PointwiseSemantic::BiasActivation)
        && args.bias.is_none()
        && args.activation == PointwiseActivation::None
    {
        return Err(invalid("Pointwise bias-activation is a no-op"));
    }

    let mut storage = Vec::new();

    let input_buffer = tensor_storage(ctx, args.input, input_dtype, input_count)?;
    let input = input_buffer.ptr() as *const std::ffi::c_void;
    let input_alignment = alignment_of(input_buffer.ptr() as usize);
    storage.push(input_buffer);

    let (secondary, secondary_alignment) = match args.secondary {
        Some(tensor) => {
            let buffer = tensor_storage(ctx, tensor, input_dtype, count)?;
            let pointer = buffer.ptr() as *const std::ffi::c_void;
            let alignment = alignment_of(buffer.ptr() as usize);
            storage.push(buffer);
            (pointer, alignment)
        }
        None if semantic.reads_secondary() => {
            return Err(invalid("Pointwise semantic requires a secondary input"))
        }
        None => (std::ptr::null(), 256),
    };

    let (bias, bias_alignment) = match args.bias {
        Some(tensor) => {
            let buffer = tensor_storage(ctx, tensor, input_dtype, cols)?;
            let pointer = buffer.ptr() as *const std::ffi::c_void;
            let alignment = alignment_of(buffer.ptr() as usize);
            storage.push(buffer);
            (pointer, alignment)
        }
        None => (std::ptr::null(), 256),
    };

    let output_buffer = tensor_storage(ctx, args.out, output_dtype, count)?;
    let output = output_buffer.ptr() as *mut std::ffi::c_void;
    let output_alignment = alignment_of(output_buffer.ptr() as usize);
    if quantized_geglu {
        if input_alignment < 4 || output_alignment < 2 {
            return Err(invalid(
                "F16-to-E4M3 GeGLU requires 4-byte input and 2-byte output alignment",
            ));
        }
        let input_bytes = input_count
            .checked_mul(input_dtype.size_in_bytes())
            .ok_or_else(|| invalid("Pointwise input storage size overflow"))?;
        let output_bytes = count
            .checked_mul(output_dtype.size_in_bytes())
            .ok_or_else(|| invalid("Pointwise output storage size overflow"))?;
        if storage_ranges_overlap(
            input as usize,
            input_bytes,
            output_buffer.ptr() as usize,
            output_bytes,
        )? {
            return Err(invalid(
                "F16-to-E4M3 GeGLU input and output storage must not overlap",
            ));
        }
    }
    storage.push(output_buffer);

    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: semantic.code(),
        dtype: input_dtype_code(input_dtype)?,
        output_dtype: output_dtype_code(output_dtype)?,
        activation: args.activation.code(),
        has_bias: u32::from(!bias.is_null()),
        input_alignment,
        secondary_alignment,
        bias_alignment,
        output_alignment,
        rows: i64::from(rows_abi),
        cols: i64::from(cols_abi),
        output_scale_is_unit: u32::from(args.output_scale == 1.0),
    };

    let bindings = abi::Bindings {
        input,
        secondary,
        bias,
        output,
        stream: ctx.stream().handle() as abi::CudaStream,
        dt: args.dt,
        output_scale: args.output_scale,
    };

    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

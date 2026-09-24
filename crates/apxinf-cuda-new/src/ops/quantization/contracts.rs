use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::quantization as abi;
use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantizationSemantic {
    /// F16 or BF16 to E4M3 with one positive, pre-calibrated scale.
    FixedScaleE4m3,
    /// BF16 to E4M3 with one computed F32 scale per row and optional zero padding.
    RowwiseE4m3,
    /// Shape-preserving F16 to BF16 conversion.
    CastF16ToBf16,
    /// Keep the leading columns of a contiguous BF16 matrix.
    SliceColumnsBf16,
    /// BF16 to signed INT8 with one computed F32 scale per row.
    RowwiseI8,
}

impl QuantizationSemantic {
    fn code(self) -> u32 {
        match self {
            Self::FixedScaleE4m3 => 0,
            Self::RowwiseE4m3 => 1,
            Self::CastF16ToBf16 => 2,
            Self::SliceColumnsBf16 => 3,
            Self::RowwiseI8 => 4,
        }
    }

    fn has_row_scales(self) -> bool {
        matches!(self, Self::RowwiseE4m3 | Self::RowwiseI8)
    }
}

pub struct QuantizationArgs<'a> {
    pub semantic: QuantizationSemantic,
    pub input: &'a Tensor,
    pub out: &'a mut Tensor,
    /// Required for rowwise quantization and otherwise forbidden.
    pub scales: Option<&'a mut Tensor>,
    /// Used only by [`QuantizationSemantic::FixedScaleE4m3`].
    pub scale: f32,
}

impl<'a> QuantizationArgs<'a> {
    pub fn new(semantic: QuantizationSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input,
            out,
            scales: None,
            scale: 1.0,
        }
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

fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::F8E4M3 => 3,
        DType::I8 => 4,
        DType::I32 => 5,
    }
}

fn alignment_of(pointer: usize) -> u32 {
    if pointer == 0 {
        return 256;
    }
    (1usize << pointer.trailing_zeros().min(8)) as u32
}

fn tensor_buffer(ctx: &CudaContext, tensor: &Tensor, elements: usize) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) {
        return Err(invalid("Quantization tensor is not on this CUDA device"));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid("Quantization tensor element count mismatch"));
    }
    let bytes = elements
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| invalid("Quantization storage size overflow"))?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < bytes || (buffer.ptr() as usize) % tensor.dtype().size_in_bytes() != 0 {
        return Err(invalid("Quantization tensor storage is invalid"));
    }
    Ok(buffer)
}

pub(crate) fn normalize(ctx: &CudaContext, args: QuantizationArgs<'_>) -> Result<Normalized> {
    let semantic = args.semantic;
    let input_numel = args.input.shape().numel();
    let output_numel = args.out.shape().numel();
    if input_numel == 0 || output_numel == 0 {
        return Err(invalid("Quantization tensors must be non-empty"));
    }

    let (rows, input_cols, output_cols) = match semantic {
        QuantizationSemantic::RowwiseE4m3
        | QuantizationSemantic::SliceColumnsBf16
        | QuantizationSemantic::RowwiseI8 => {
            let input_shape = args.input.shape().dims();
            let output_shape = args.out.shape().dims();
            if input_shape.len() != 2
                || output_shape.len() != 2
                || input_shape[0] != output_shape[0]
            {
                return Err(invalid(
                    "rowwise/slice Quantization requires matching rank-2 row counts",
                ));
            }
            (input_shape[0], input_shape[1], output_shape[1])
        }
        QuantizationSemantic::FixedScaleE4m3 | QuantizationSemantic::CastF16ToBf16 => {
            if args.input.shape() != args.out.shape() {
                return Err(invalid(
                    "fixed-scale/cast Quantization requires identical shapes",
                ));
            }
            (1, input_numel, output_numel)
        }
    };
    if rows == 0 || input_cols == 0 || output_cols == 0 {
        return Err(invalid("Quantization geometry must be non-empty"));
    }
    i32::try_from(rows).map_err(|_| invalid("Quantization row count exceeds kernel range"))?;
    i32::try_from(input_cols)
        .map_err(|_| invalid("Quantization input width exceeds kernel range"))?;
    i32::try_from(output_cols)
        .map_err(|_| invalid("Quantization output width exceeds kernel range"))?;

    let (expected_input, expected_output) = match semantic {
        QuantizationSemantic::FixedScaleE4m3 => {
            if !matches!(args.input.dtype(), DType::F16 | DType::BF16) {
                return Err(invalid(
                    "fixed E4M3 quantization requires F16 or BF16 input",
                ));
            }
            (args.input.dtype(), DType::F8E4M3)
        }
        QuantizationSemantic::RowwiseE4m3 => {
            if output_cols < input_cols {
                return Err(invalid("rowwise E4M3 output width cannot shrink input"));
            }
            (DType::BF16, DType::F8E4M3)
        }
        QuantizationSemantic::CastF16ToBf16 => (DType::F16, DType::BF16),
        QuantizationSemantic::SliceColumnsBf16 => {
            if output_cols > input_cols {
                return Err(invalid("BF16 column slice cannot widen input"));
            }
            (DType::BF16, DType::BF16)
        }
        QuantizationSemantic::RowwiseI8 => {
            if output_cols != input_cols {
                return Err(invalid("rowwise INT8 output shape must match input"));
            }
            (DType::BF16, DType::I8)
        }
    };
    if args.input.dtype() != expected_input || args.out.dtype() != expected_output {
        return Err(invalid("Quantization tensor dtype disagrees with semantic"));
    }
    if semantic == QuantizationSemantic::FixedScaleE4m3 {
        if !(args.scale.is_finite() && args.scale > 0.0) {
            return Err(invalid("fixed E4M3 scale must be finite and positive"));
        }
    } else if args.scale != 1.0 {
        return Err(invalid("this Quantization semantic does not take a scale"));
    }

    let input_buffer = tensor_buffer(ctx, args.input, input_numel)?;
    let output_buffer = tensor_buffer(ctx, args.out, output_numel)?;
    if input_buffer.ptr() == output_buffer.ptr() {
        return Err(invalid("Quantization input and output must not alias"));
    }
    let input = input_buffer.ptr() as *const std::ffi::c_void;
    let output = output_buffer.ptr() as *mut std::ffi::c_void;
    let input_alignment = alignment_of(input_buffer.ptr() as usize);
    let output_alignment = alignment_of(output_buffer.ptr() as usize);

    let mut storage = vec![input_buffer, output_buffer];
    let (scales, scales_alignment) = match args.scales {
        Some(scales) if semantic.has_row_scales() => {
            if scales.dtype() != DType::F32 || scales.shape().dims() != [rows] {
                return Err(invalid("row scales must be F32 [rows]"));
            }
            let buffer = tensor_buffer(ctx, scales, rows)?;
            if buffer.ptr() as *const std::ffi::c_void == input
                || buffer.ptr() as *mut std::ffi::c_void == output
            {
                return Err(invalid(
                    "Quantization scales must not alias input or output",
                ));
            }
            let pointer = buffer.ptr() as *mut f32;
            let alignment = alignment_of(buffer.ptr() as usize);
            storage.push(buffer);
            (pointer, alignment)
        }
        Some(_) => return Err(invalid("Quantization semantic does not produce row scales")),
        None if semantic.has_row_scales() => {
            return Err(invalid(
                "rowwise Quantization requires an F32 scales output",
            ))
        }
        None => (std::ptr::null_mut(), 256),
    };

    Ok(Normalized {
        spec: abi::Spec {
            version: abi::SPEC_VERSION,
            semantic: semantic.code(),
            input_dtype: dtype_code(expected_input),
            output_dtype: dtype_code(expected_output),
            scale_dtype: dtype_code(DType::F32),
            input_alignment,
            output_alignment,
            scales_alignment,
            rows: rows as i64,
            input_cols: input_cols as i64,
            output_cols: output_cols as i64,
        },
        bindings: abi::Bindings {
            input,
            output,
            scales,
            stream: ctx.stream().handle(),
            scale: args.scale,
        },
        storage,
    })
}

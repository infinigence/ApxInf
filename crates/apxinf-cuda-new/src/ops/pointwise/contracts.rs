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
    Swiglu,
    AddInto,
    SwigluRowwiseE4m3,
    Add,
    Subtract,
    Multiply,
    /// Add a `[1, cols]` secondary tensor to every input row.
    BroadcastAdd,
    /// Multiply by a `[1, cols]` secondary tensor.
    BroadcastMultiply,
    /// Multiply by a `[rows, 1]` secondary tensor.
    RowMultiply,
    /// Multiply by the scalar `dt`, rounding once to the output dtype.
    Scale,
    /// Add the scalar `dt`, rounding once to the output dtype.
    ScalarAdd,
    /// Divide by the finite nonzero scalar `dt` with FP32 division rounding.
    ScalarDivide,
    /// FP32 exponential, cast once to the output dtype.
    Exp,
    /// FP32 sine in radians, cast once to the output dtype.
    Sin,
    /// FP32 cosine in radians, cast once to the output dtype.
    Cos,
    /// Dtype-rounded `input * (1 + scale) + shift`. `secondary` is [1,cols]
    /// scale and the required `bias` binding is [cols] shift. Every elementary
    /// operation rounds to the input dtype; exact input/output alias is allowed.
    StagedModulation,
    /// Dtype-rounded `residual + input * gate`. `secondary` is [rows,cols]
    /// residual and the required `bias` binding is [cols] gate. The product
    /// rounds before addition; exact output alias with either matrix is allowed.
    StagedGatedResidual,
}

impl PointwiseSemantic {
    #[cfg(test)]
    pub(crate) const ALL: &'static [Self] = &[
        Self::Geglu,
        Self::BiasActivation,
        Self::EulerUpdate,
        Self::Swiglu,
        Self::AddInto,
        Self::SwigluRowwiseE4m3,
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::BroadcastAdd,
        Self::BroadcastMultiply,
        Self::RowMultiply,
        Self::Scale,
        Self::ScalarAdd,
        Self::ScalarDivide,
        Self::Exp,
        Self::Sin,
        Self::Cos,
        Self::StagedModulation,
        Self::StagedGatedResidual,
    ];

    #[cfg(test)]
    pub(crate) const fn doc_id(self) -> &'static str {
        match self {
            Self::Geglu => "pointwise_geglu",
            Self::BiasActivation => "pointwise_bias_activation",
            Self::EulerUpdate => "pointwise_euler_update",
            Self::Swiglu => "pointwise_swiglu",
            Self::AddInto => "pointwise_add_into",
            Self::SwigluRowwiseE4m3 => "pointwise_swiglu_rowwise_e4m3",
            Self::Add => "pointwise_add",
            Self::Subtract => "pointwise_subtract",
            Self::Multiply => "pointwise_multiply",
            Self::BroadcastAdd => "pointwise_broadcast_add",
            Self::BroadcastMultiply => "pointwise_broadcast_multiply",
            Self::RowMultiply => "pointwise_row_multiply",
            Self::Scale => "pointwise_scale",
            Self::ScalarAdd => "pointwise_scalar_add",
            Self::ScalarDivide => "pointwise_scalar_divide",
            Self::Exp => "pointwise_exp",
            Self::Sin => "pointwise_sin",
            Self::Cos => "pointwise_cos",
            Self::StagedModulation => "pointwise_staged_modulation",
            Self::StagedGatedResidual => "pointwise_staged_gated_residual",
        }
    }

    fn code(self) -> u32 {
        match self {
            Self::Geglu => 0,
            Self::BiasActivation => 1,
            Self::EulerUpdate => 2,
            Self::Swiglu => 3,
            Self::AddInto => 4,
            Self::SwigluRowwiseE4m3 => 5,
            Self::Add => 6,
            Self::Subtract => 7,
            Self::Multiply => 8,
            Self::BroadcastAdd => 9,
            Self::BroadcastMultiply => 10,
            Self::RowMultiply => 11,
            Self::Scale => 12,
            Self::ScalarAdd => 13,
            Self::ScalarDivide => 14,
            Self::Exp => 15,
            Self::Sin => 16,
            Self::Cos => 17,
            Self::StagedModulation => 18,
            Self::StagedGatedResidual => 19,
        }
    }

    fn reads_secondary(self) -> bool {
        matches!(
            self,
            Self::EulerUpdate
                | Self::Add
                | Self::Subtract
                | Self::Multiply
                | Self::BroadcastAdd
                | Self::BroadcastMultiply
                | Self::RowMultiply
                | Self::StagedModulation
                | Self::StagedGatedResidual
        )
    }

    fn may_have_bias(self) -> bool {
        matches!(
            self,
            Self::BiasActivation
                | Self::SwigluRowwiseE4m3
                | Self::StagedModulation
                | Self::StagedGatedResidual
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PointwiseActivation {
    None,
    Gelu,
    Silu,
    GeluExact,
    Relu,
    /// Tanh GELU with a dtype rounding boundary after every elementary
    /// operation, including the cubic power. Unlike fused tanh GELU.
    GeluRounded,
    /// Explicit sigmoid tensor rounded before multiplication by input.
    SwishRounded,
}

impl PointwiseActivation {
    fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Gelu => 1,
            Self::Silu => 2,
            Self::GeluExact => 3,
            Self::Relu => 4,
            Self::GeluRounded => 5,
            Self::SwishRounded => 6,
        }
    }
}

/// Element-wise operation over a contiguous `[rows, cols]` activation, where
/// `cols` is the *output* width.
pub struct PointwiseArgs<'a> {
    pub semantic: PointwiseSemantic,
    pub input: &'a Tensor,
    /// Binary operand; broadcasting follows the selected semantic.
    pub secondary: Option<&'a Tensor>,
    /// Per-column bias, or shift/gate for the two staged affine semantics.
    pub bias: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    /// One F32 scale per row for dynamic rowwise quantization.
    pub scales: Option<&'a mut Tensor>,
    /// Logical, unpadded SwiGLU width. Zero means the physical output width.
    pub logical_cols: usize,
    pub activation: PointwiseActivation,
    /// Step size for EulerUpdate or scalar for Scale/ScalarAdd.
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
            scales: None,
            logical_cols: 0,
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
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Pointwise input supports F32, F16 and BF16")),
    }
}

fn output_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::F8E4M3 => Ok(3),
        _ => Err(invalid(
            "Pointwise output supports F32, F16, BF16, and E4M3",
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
        return Err(invalid(format!(
            "Pointwise tensor element count mismatch: expected {elements}, got {} for shape {:?}",
            tensor.shape().numel(),
            tensor.shape().dims()
        )));
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
    if semantic == PointwiseSemantic::ScalarDivide && args.dt == 0. {
        return Err(invalid("Pointwise scalar divisor must be nonzero"));
    }
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
    let input_dims = args.input.shape().dims();
    if input_dims.len() != 2 || input_dims[0] != rows {
        return Err(invalid(
            "Pointwise input and output must be rank-2 tensors with matching rows",
        ));
    }
    let input_cols = input_dims[1];
    let logical_cols = if args.logical_cols == 0 {
        cols
    } else {
        args.logical_cols
    };
    let quantized_geglu = matches!(semantic, PointwiseSemantic::Geglu)
        && input_dtype == DType::F16
        && output_dtype == DType::F8E4M3;
    let quantized_swiglu = matches!(semantic, PointwiseSemantic::SwigluRowwiseE4m3)
        && input_dtype == DType::BF16
        && output_dtype == DType::F8E4M3;
    if !quantized_geglu && !quantized_swiglu && input_dtype != output_dtype {
        return Err(invalid(
            "Pointwise input/output dtypes must match except for quantized GeGLU/SwiGLU",
        ));
    }
    match semantic {
        PointwiseSemantic::Geglu | PointwiseSemantic::Swiglu if input_cols != 2 * cols => {
            return Err(invalid(
                "Pointwise gated activation input width must be twice the output width",
            ));
        }
        PointwiseSemantic::BiasActivation
        | PointwiseSemantic::EulerUpdate
        | PointwiseSemantic::AddInto
            if input_cols != cols =>
        {
            return Err(invalid("Pointwise input/output width mismatch"));
        }
        PointwiseSemantic::SwigluRowwiseE4m3
            if logical_cols == 0
                || input_cols < 2 * logical_cols
                || cols < logical_cols
                || !quantized_swiglu =>
        {
            return Err(invalid(
                "Pointwise rowwise SwiGLU quantization shape or dtype mismatch",
            ));
        }
        _ => {}
    }
    if matches!(
        semantic,
        PointwiseSemantic::Swiglu | PointwiseSemantic::AddInto
    ) && input_dtype != DType::BF16
    {
        return Err(invalid("Pointwise SwiGLU/add-into currently require BF16"));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Pointwise size overflow"))?;
    if quantized_geglu {
        i32::try_from(count)
            .map_err(|_| invalid("F16-to-E4M3 GeGLU element count exceeds kernel range"))?;
    }
    let input_count = rows
        .checked_mul(input_cols)
        .ok_or_else(|| invalid("Pointwise size overflow"))?;

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
        return Err(invalid("F16-to-E4M3 GeGLU requires an even output width"));
    }
    let staged_affine = matches!(
        semantic,
        PointwiseSemantic::StagedModulation | PointwiseSemantic::StagedGatedResidual
    );
    if staged_affine && args.bias.is_none() {
        return Err(invalid(
            "Staged affine requires a per-column shift/gate binding",
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
    if args.scales.is_some() != matches!(semantic, PointwiseSemantic::SwigluRowwiseE4m3) {
        return Err(invalid(
            "Pointwise scales binding is required only for rowwise SwiGLU quantization",
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

    let (secondary_count, secondary_shape) = match semantic {
        PointwiseSemantic::BroadcastAdd
        | PointwiseSemantic::BroadcastMultiply
        | PointwiseSemantic::StagedModulation => (cols, [1, cols]),
        PointwiseSemantic::RowMultiply => (rows, [rows, 1]),
        _ => (count, [rows, cols]),
    };
    let (secondary, secondary_alignment) = match args.secondary {
        Some(tensor) => {
            if semantic != PointwiseSemantic::EulerUpdate
                && tensor.shape().dims() != secondary_shape
            {
                return Err(invalid(
                    "Pointwise secondary shape disagrees with its broadcast contract",
                ));
            }
            let buffer = tensor_storage(ctx, tensor, input_dtype, secondary_count)?;
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
            if staged_affine && tensor.shape().dims() != [cols] {
                return Err(invalid("Staged affine shift/gate must have shape [cols]"));
            }
            let bias_elements = if matches!(semantic, PointwiseSemantic::SwigluRowwiseE4m3) {
                logical_cols
                    .checked_mul(2)
                    .ok_or_else(|| invalid("Pointwise SwiGLU bias size overflow"))?
            } else {
                cols
            };
            let buffer = tensor_storage(ctx, tensor, input_dtype, bias_elements)?;
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
    // Broadcast operands may be reused by many threads. Reject overlap rather
    // than allowing an order-dependent in-place update. Only the two staged
    // affine semantics permit exact aliases with their full-size matrices.
    if semantic.code() >= PointwiseSemantic::Add.code() {
        if args.input.shape().dims() != [rows, cols] {
            return Err(invalid("Pointwise input must match output [rows, cols]"));
        }
        let bytes = count
            .checked_mul(output_dtype.size_in_bytes())
            .ok_or_else(|| invalid("Pointwise output size overflow"))?;
        for (index, buffer) in storage.iter().enumerate() {
            let read_bytes = if staged_affine {
                (match index {
                    0 => count,
                    1 => secondary_count,
                    _ => cols,
                }) * input_dtype.size_in_bytes()
            } else {
                buffer.len()
            };
            let exact_matrix_alias = staged_affine
                && (index == 0
                    || (index == 1 && semantic == PointwiseSemantic::StagedGatedResidual))
                && buffer.ptr() as usize == output as usize
                && read_bytes == bytes;
            if !exact_matrix_alias
                && storage_ranges_overlap(
                    buffer.ptr() as usize,
                    read_bytes,
                    output as usize,
                    bytes,
                )?
            {
                return Err(invalid(
                    "Pointwise binary/scalar input and output storage must not overlap",
                ));
            }
        }
    }
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

    let scales = match args.scales {
        Some(tensor) => {
            let buffer = tensor_storage(ctx, tensor, DType::F32, rows)?;
            let pointer = buffer.ptr() as *mut std::ffi::c_void;
            storage.push(buffer);
            pointer
        }
        None => std::ptr::null_mut(),
    };

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
        input_cols: i64::try_from(input_cols)
            .map_err(|_| invalid("Pointwise input width exceeds i64"))?,
        logical_cols: i64::try_from(logical_cols)
            .map_err(|_| invalid("Pointwise logical width exceeds i64"))?,
        output_scale_is_unit: u32::from(args.output_scale == 1.0),
    };

    let bindings = abi::Bindings {
        input,
        secondary,
        bias,
        output,
        scales,
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

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::norm as abi;
use crate::{CudaBuffer, CudaContext};

/// Public L3 operation selected by the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Operation {
    Norm,
    FusedAddNorm,
    ResidualAdd,
}

impl Operation {
    fn code(self) -> u32 {
        match self {
            Self::Norm => 0,
            Self::FusedAddNorm => 1,
            Self::ResidualAdd => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Statistic {
    None,
    Rms,
    Layer,
}

impl Statistic {
    fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Rms => 1,
            Self::Layer => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AffineKind {
    None,
    Weight,
    Adaptive,
}

impl AffineKind {
    fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Weight => 1,
            Self::Adaptive => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GateKind {
    None,
    Style,
}

impl GateKind {
    fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Style => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Quantization {
    None,
    StaticE4m3,
    RowwiseE4m3,
    RowwiseI8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum NormNumerics {
    /// The family may select any candidate that satisfies the mathematical
    /// contract.
    #[default]
    Flexible,
    /// Preserve the reduction and BF16 update order used by the decode
    /// kernels. This is a numerical contract, not a provider request.
    StableDecode,
    /// Preserve the reduction and explicitly staged BF16 rounding used by
    /// the planning kernels. This is a numerical contract, not a provider
    /// request.
    StablePlanning,
}

impl NormNumerics {
    fn code(self) -> u32 {
        match self {
            Self::Flexible => 0,
            Self::StableDecode => 1,
            Self::StablePlanning => 2,
        }
    }
}

impl Quantization {
    fn code(self) -> u32 {
        match self {
            Self::None => 0,
            Self::StaticE4m3 => 1,
            Self::RowwiseE4m3 => 2,
            Self::RowwiseI8 => 3,
        }
    }
}

fn writes_hidden(operation: Operation) -> bool {
    operation != Operation::Norm
}

fn writes_normalized(operation: Operation) -> bool {
    operation != Operation::ResidualAdd
}

fn reads_residual(operation: Operation) -> bool {
    writes_hidden(operation)
}

fn reads_weight(affine: AffineKind) -> bool {
    affine == AffineKind::Weight
}

fn reads_norm_bias(statistic: Statistic, affine: AffineKind) -> bool {
    statistic == Statistic::Layer && affine == AffineKind::Weight
}

fn reads_norm_style(operation: Operation, affine: AffineKind) -> bool {
    writes_normalized(operation) && affine == AffineKind::Adaptive
}

fn reads_gate_style(gate: GateKind) -> bool {
    gate == GateKind::Style
}

fn may_have_bias(operation: Operation, gate: GateKind) -> bool {
    writes_hidden(operation) && gate == GateKind::None
}

fn valid_contract(
    operation: Operation,
    statistic: Statistic,
    affine: AffineKind,
    gate: GateKind,
) -> bool {
    matches!(
        (operation, statistic, affine, gate),
        (
            Operation::Norm,
            Statistic::Rms | Statistic::Layer,
            AffineKind::Weight,
            GateKind::None
        ) | (
            Operation::Norm,
            Statistic::Rms | Statistic::Layer,
            AffineKind::Adaptive,
            GateKind::None
        ) | (
            Operation::ResidualAdd,
            Statistic::None,
            AffineKind::None,
            GateKind::None | GateKind::Style
        ) | (
            Operation::FusedAddNorm,
            Statistic::Rms | Statistic::Layer,
            AffineKind::Weight,
            GateKind::None
        ) | (
            Operation::FusedAddNorm,
            Statistic::Rms,
            AffineKind::Adaptive,
            GateKind::Style
        ) | (
            Operation::FusedAddNorm,
            Statistic::Layer,
            AffineKind::Adaptive,
            GateKind::None
        )
    )
}

/// Rounding order for the optional bias and residual combination.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BiasOrder {
    /// Accumulate input, bias, and residual before rounding to the output dtype.
    #[default]
    SingleRound,
    /// Round input plus bias to BF16 before adding the residual and rounding again.
    BiasThenResidual,
}

impl BiasOrder {
    fn code(self) -> u32 {
        match self {
            Self::SingleRound => 0,
            Self::BiasThenResidual => 1,
        }
    }
}

/// Row-wise normalization over a contiguous `[rows, cols]` activation.
///
/// `norm_style` is `[2 * cols]` (scale then shift) and `gate_style` is
/// `[3 * cols]` whose third segment is the gate; they are not interchangeable.
pub(crate) struct RawArgs<'a> {
    pub(crate) operation: Operation,
    pub(crate) statistic: Statistic,
    pub(crate) affine: AffineKind,
    pub(crate) gate: GateKind,
    pub(crate) quantization: Quantization,
    pub(crate) numerics: NormNumerics,
    pub(crate) bias_order: BiasOrder,
    pub(crate) weight_offset: bool,
    pub(crate) input: &'a Tensor,
    pub(crate) bias: Option<&'a Tensor>,
    pub(crate) residual: Option<&'a Tensor>,
    pub(crate) weight: Option<&'a Tensor>,
    pub(crate) norm_bias: Option<&'a Tensor>,
    pub(crate) norm_style: Option<&'a Tensor>,
    pub(crate) gate_style: Option<&'a Tensor>,
    pub(crate) hidden: Option<&'a Tensor>,
    pub(crate) normalized: Option<&'a Tensor>,
    pub(crate) quantized_output: Option<&'a Tensor>,
    pub(crate) quantized_scales: Option<&'a Tensor>,
    pub(crate) eps: f32,
    pub(crate) output_scale: f32,
}

pub(crate) struct Normalized {
    pub spec: abi::Spec,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Norm currently supports F32, F16 and BF16")),
    }
}

fn output_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::F8E4M3 => Ok(3),
        DType::I8 => Ok(4),
        _ => Err(invalid("Norm output supports F32, F16, BF16, E4M3 and I8")),
    }
}

/// Largest power-of-two byte alignment of a device pointer, capped at 256 to
/// match the Spec contract.
fn alignment_of(pointer: usize) -> u32 {
    if pointer == 0 {
        return 256;
    }
    let alignment = 1usize << pointer.trailing_zeros().min(8);
    alignment as u32
}

fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != dtype {
        return Err(invalid("Norm tensor device/dtype mismatch"));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid("Norm tensor element count mismatch"));
    }
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("Norm size overflow"))?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < expected || (buffer.ptr() as usize) % dtype.size_in_bytes() != 0 {
        return Err(invalid("Norm tensor storage is invalid"));
    }
    Ok(buffer)
}

pub(crate) fn normalize(ctx: &CudaContext, args: RawArgs<'_>) -> Result<Normalized> {
    let operation = args.operation;
    let statistic = args.statistic;
    let affine = args.affine;
    let gate = args.gate;
    if !valid_contract(operation, statistic, affine, gate) {
        return Err(invalid(
            "unsupported Norm operation/statistic/affine/gate contract",
        ));
    }
    let dims = args.input.shape().dims();
    if dims.len() != 2 {
        return Err(invalid("Norm input must be rank 2 [rows, cols]"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if rows == 0 || cols == 0 {
        return Err(invalid("Norm input must be non-empty"));
    }
    let rows_abi =
        i32::try_from(rows).map_err(|_| invalid("Norm row count exceeds the CUDA kernel range"))?;
    let cols_abi = i32::try_from(cols)
        .map_err(|_| invalid("Norm column count exceeds the CUDA kernel range"))?;
    let dtype = args.input.dtype();
    if args.weight_offset {
        let plain = operation == Operation::Norm && args.numerics != NormNumerics::StableDecode;
        let fused =
            operation == Operation::FusedAddNorm && args.numerics == NormNumerics::StablePlanning;
        if statistic != Statistic::Rms
            || affine != AffineKind::Weight
            || gate != GateKind::None
            || dtype != DType::BF16
            || args.quantization != Quantization::None
            || args.bias.is_some()
            || args.norm_bias.is_some()
            || args.bias_order != BiasOrder::SingleRound
            || (!plain && !fused)
        {
            return Err(invalid(
                "weight-offset affine requires unquantized BF16 RMSNorm, no bias/gate, SingleRound order, and Flexible/StablePlanning numerics",
            ));
        }
    } else if args.numerics == NormNumerics::StablePlanning {
        return Err(invalid(
            "StablePlanning numerics currently requires weight-offset affine",
        ));
    }
    if dtype == DType::F32
        && !(operation == Operation::Norm
            && statistic == Statistic::Rms
            && affine == AffineKind::Weight
            && gate == GateKind::None
            && !args.weight_offset
            && args.quantization == Quantization::None
            && args.numerics == NormNumerics::StableDecode)
    {
        return Err(invalid(
            "F32 Norm is supported only for stable decode RMSNorm",
        ));
    }
    if args.numerics == NormNumerics::StableDecode
        && !((operation == Operation::Norm
            && statistic == Statistic::Rms
            && affine == AffineKind::Weight
            && gate == GateKind::None)
            || (operation == Operation::FusedAddNorm
                && statistic == Statistic::Rms
                && affine == AffineKind::Weight
                && gate == GateKind::None
                && dtype == DType::BF16
                && !args.weight_offset
                && args.bias.is_none()))
    {
        return Err(invalid(
            "StableDecode numerics supports plain RMSNorm or unbiased BF16 residual RMSNorm only",
        ));
    }
    if args.bias_order == BiasOrder::BiasThenResidual && dtype != DType::BF16 {
        return Err(invalid(
            "BiasThenResidual order is BF16-only because its intermediate rounding is part of the contract",
        ));
    }
    if statistic == Statistic::Layer && affine == AffineKind::Adaptive && dtype != DType::BF16 {
        return Err(invalid(
            "adaptive LayerNorm currently requires BF16 input and output",
        ));
    }
    if args.bias_order == BiasOrder::BiasThenResidual
        && (operation == Operation::Norm || gate != GateKind::None || statistic == Statistic::Rms)
    {
        return Err(invalid(
            "BiasThenResidual requires ungated residual add or fused LayerNorm",
        ));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Norm size overflow"))?;

    if !(args.eps.is_finite() && args.eps > 0.0) {
        return Err(invalid("Norm eps must be finite and positive"));
    }
    if !(args.output_scale.is_finite() && args.output_scale > 0.0) {
        return Err(invalid("Norm output_scale must be finite and positive"));
    }
    if args.bias.is_some() && !may_have_bias(operation, gate) {
        return Err(invalid("Norm contract does not take a bias"));
    }
    if args.residual.is_some() && !reads_residual(operation) {
        return Err(invalid("Norm contract does not take a residual"));
    }
    if args.weight.is_some() && !reads_weight(affine) {
        return Err(invalid("Norm contract does not take a weight"));
    }
    if args.norm_bias.is_some() && !reads_norm_bias(statistic, affine) {
        return Err(invalid("Norm contract does not take a norm bias"));
    }
    if args.norm_style.is_some() && !reads_norm_style(operation, affine) {
        return Err(invalid("Norm contract does not take a norm style"));
    }
    if args.gate_style.is_some() && !reads_gate_style(gate) {
        return Err(invalid("Norm contract does not take a gate style"));
    }
    if args.hidden.is_some() && !writes_hidden(operation) {
        return Err(invalid("Norm contract does not write a hidden output"));
    }
    if args.normalized.is_some() && !writes_normalized(operation) {
        return Err(invalid("Norm contract does not write a normalized output"));
    }
    if args.quantized_output.is_some() && !writes_normalized(operation) {
        return Err(invalid("Norm contract does not write a quantized output"));
    }

    let mut storage = Vec::new();
    let bind_vector = |tensor: Option<&Tensor>,
                       required: bool,
                       elements: usize,
                       label: &str,
                       storage: &mut Vec<CudaBuffer>|
     -> Result<(*const std::ffi::c_void, u32)> {
        match tensor {
            Some(tensor) => {
                let buffer = tensor_storage(ctx, tensor, dtype, elements)?;
                let pointer = buffer.ptr() as *const std::ffi::c_void;
                let alignment = alignment_of(buffer.ptr() as usize);
                storage.push(buffer);
                Ok((pointer, alignment))
            }
            None if required => Err(invalid(format!("Norm contract requires {label}"))),
            None => Ok((std::ptr::null(), 256)),
        }
    };

    let input_buffer = tensor_storage(ctx, args.input, dtype, count)?;
    let input_pointer = input_buffer.ptr() as *const std::ffi::c_void;
    let input_alignment = alignment_of(input_buffer.ptr() as usize);
    storage.push(input_buffer);

    let (bias, bias_alignment) = bind_vector(args.bias, false, cols, "bias", &mut storage)?;
    let (residual, residual_alignment) = bind_vector(
        args.residual,
        reads_residual(operation),
        count,
        "residual",
        &mut storage,
    )?;
    let (weight, weight_alignment) = bind_vector(
        args.weight,
        reads_weight(affine),
        cols,
        "weight",
        &mut storage,
    )?;
    let (norm_bias, norm_bias_alignment) = bind_vector(
        args.norm_bias,
        reads_norm_bias(statistic, affine),
        cols,
        "norm_bias",
        &mut storage,
    )?;
    let norm_style_elements = cols
        .checked_mul(2)
        .ok_or_else(|| invalid("Norm style size overflow"))?;
    let (norm_style, norm_style_alignment) = bind_vector(
        args.norm_style,
        reads_norm_style(operation, affine),
        norm_style_elements,
        "norm_style",
        &mut storage,
    )?;
    let gate_style_elements = cols
        .checked_mul(3)
        .ok_or_else(|| invalid("Norm gate style size overflow"))?;
    let (gate_style, gate_style_alignment) = bind_vector(
        args.gate_style,
        reads_gate_style(gate),
        gate_style_elements,
        "gate_style",
        &mut storage,
    )?;

    let bind_output = |tensor: Option<&Tensor>,
                       required: bool,
                       output_dtype: DType,
                       elements: usize,
                       label: &str,
                       storage: &mut Vec<CudaBuffer>|
     -> Result<(*mut std::ffi::c_void, u32)> {
        match tensor {
            Some(tensor) => {
                let buffer = tensor_storage(ctx, tensor, output_dtype, elements)?;
                let pointer = buffer.ptr() as *mut std::ffi::c_void;
                let alignment = alignment_of(buffer.ptr() as usize);
                storage.push(buffer);
                Ok((pointer, alignment))
            }
            None if required => Err(invalid(format!("Norm contract requires {label}"))),
            None => Ok((std::ptr::null_mut(), 256)),
        }
    };

    let (hidden, hidden_alignment) = bind_output(
        args.hidden,
        writes_hidden(operation),
        dtype,
        count,
        "hidden",
        &mut storage,
    )?;
    let has_native_output = args.normalized.is_some();
    let contract = (operation, statistic, affine, gate);
    let (output_dtype, quantized_cols, quantized_elements) = match args.quantization {
        Quantization::None => {
            if has_native_output != writes_normalized(operation)
                || args.quantized_output.is_some()
                || args.quantized_scales.is_some()
                || args.output_scale != 1.0
            {
                return Err(invalid(
                    "Non-quantized Norm requires a native output, no quantized bindings, and unit output_scale",
                ));
            }
            (dtype, 0, 0)
        }
        Quantization::StaticE4m3 => {
            let bf16_contract = dtype == DType::BF16
                && matches!(
                    contract,
                    (
                        Operation::Norm,
                        Statistic::Rms | Statistic::Layer,
                        AffineKind::Weight,
                        GateKind::None
                    ) | (
                        Operation::FusedAddNorm,
                        Statistic::Layer,
                        AffineKind::Weight,
                        GateKind::None
                    ) | (
                        Operation::Norm,
                        Statistic::Layer,
                        AffineKind::Adaptive,
                        GateKind::None
                    )
                );
            let dual_output_contract = dtype == DType::BF16
                && matches!(
                    contract,
                    (
                        Operation::Norm,
                        Statistic::Layer,
                        AffineKind::Adaptive,
                        GateKind::None
                    )
                );
            if (dtype != DType::F16 && !bf16_contract)
                || !writes_normalized(operation)
                || args.quantized_output.is_none()
                || args.quantized_scales.is_some()
                || (has_native_output && !dual_output_contract)
            {
                return Err(invalid(
                    "Static E4M3 Norm output has an unsupported dtype or semantic contract",
                ));
            }
            (DType::F8E4M3, cols, count)
        }
        Quantization::RowwiseE4m3 => {
            if dtype != DType::BF16
                || !matches!(
                    contract,
                    (
                        Operation::Norm,
                        Statistic::Rms,
                        AffineKind::Weight,
                        GateKind::None
                    ) | (
                        Operation::FusedAddNorm,
                        Statistic::Rms,
                        AffineKind::Weight,
                        GateKind::None
                    )
                )
                || has_native_output
                || args.quantized_output.is_none()
                || args.quantized_scales.is_none()
                || args.output_scale != 1.0
            {
                return Err(invalid(
                    "Rowwise E4M3 Norm requires the BF16 weighted RMS contract, values/scales bindings, no native output, and unit output_scale",
                ));
            }
            let dims = args
                .quantized_output
                .expect("validated rowwise output binding")
                .shape()
                .dims();
            if dims.len() != 2 || dims[0] != rows || dims[1] < cols {
                return Err(invalid(
                    "Rowwise E4M3 Norm output must be [rows, output_cols] with output_cols >= input cols",
                ));
            }
            let elements = rows
                .checked_mul(dims[1])
                .ok_or_else(|| invalid("rowwise Norm output size overflow"))?;
            (DType::F8E4M3, dims[1], elements)
        }
        Quantization::RowwiseI8 => {
            if dtype != DType::BF16
                || !matches!(
                    contract,
                    (
                        Operation::Norm,
                        Statistic::Layer,
                        AffineKind::Weight | AffineKind::Adaptive,
                        GateKind::None
                    )
                )
                || !has_native_output
                || args.quantized_output.is_none()
                || args.quantized_scales.is_none()
                || args.output_scale != 1.0
            {
                return Err(invalid(
                    "Rowwise I8 Norm requires BF16 LayerNorm, native/values/scales outputs, and unit output_scale",
                ));
            }
            let dims = args
                .quantized_output
                .expect("validated rowwise output binding")
                .shape()
                .dims();
            if dims != [rows, cols] {
                return Err(invalid(
                    "Rowwise I8 Norm output must match the input [rows, cols] shape",
                ));
            }
            (DType::I8, cols, count)
        }
    };
    let (normalized, normalized_alignment) = bind_output(
        args.normalized,
        writes_normalized(operation) && args.quantization == Quantization::None,
        dtype,
        count,
        "normalized",
        &mut storage,
    )?;
    let (quantized, quantized_alignment) = bind_output(
        args.quantized_output,
        args.quantization != Quantization::None,
        output_dtype,
        quantized_elements,
        "quantized",
        &mut storage,
    )?;
    let (quantized_scales, quantized_scales_alignment) = bind_output(
        args.quantized_scales,
        matches!(
            args.quantization,
            Quantization::RowwiseE4m3 | Quantization::RowwiseI8
        ),
        DType::F32,
        rows,
        "quantized scales",
        &mut storage,
    )?;
    if args.numerics == NormNumerics::StableDecode
        && operation == Operation::FusedAddNorm
        && hidden as usize != residual as usize
    {
        return Err(invalid(
            "stable decode residual RMSNorm must update residual in place",
        ));
    }

    let dtype_value = dtype_code(dtype)?;
    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        operation: operation.code(),
        statistic: statistic.code(),
        affine: affine.code(),
        gate: gate.code(),
        quantization: args.quantization.code(),
        numerics: args.numerics.code(),
        dtype: dtype_value,
        output_dtype: output_dtype_code(output_dtype)?,
        has_native_output: u32::from(has_native_output),
        has_bias: u32::from(!bias.is_null()),
        bias_order: args.bias_order.code(),
        weight_offset: u32::from(args.weight_offset),
        input_alignment,
        weight_alignment,
        bias_alignment: bias_alignment.min(norm_bias_alignment),
        residual_alignment,
        style_alignment: norm_style_alignment.min(gate_style_alignment),
        hidden_alignment,
        normalized_alignment,
        quantized_alignment,
        quantized_scales_alignment,
        rows: i64::from(rows_abi),
        cols: i64::from(cols_abi),
        quantized_cols: i64::try_from(quantized_cols)
            .map_err(|_| invalid("Norm quantized column count exceeds the ABI range"))?,
        output_scale_is_unit: u32::from(args.output_scale == 1.0),
    };

    let bindings = abi::Bindings {
        input: input_pointer,
        bias,
        residual,
        weight,
        norm_bias,
        norm_style,
        gate_style,
        hidden,
        normalized,
        quantized,
        quantized_scales,
        stream: ctx.stream().handle() as abi::CudaStream,
        eps: args.eps,
        output_scale: args.output_scale,
    };

    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

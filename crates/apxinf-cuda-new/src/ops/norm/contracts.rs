use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::norm as abi;
use crate::{CudaBuffer, CudaContext};

/// The stage fields of one native Norm Spec, minus the quantization kind,
/// which follows from the output tensor's dtype rather than from the pipeline
/// shape. Mirrors the field group in `norm_types.h`, and the predicates below
/// mirror the ones the native adapter derives from the same fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Fields {
    pub semantic: u32,
    pub norm_kind: u32,
    pub affine_kind: u32,
    pub has_gate: bool,
    pub bias_order: u32,
}

impl Fields {
    fn writes_hidden(self) -> bool {
        self.semantic != abi::SEMANTIC_NORM
    }

    fn writes_normalized(self) -> bool {
        self.semantic != abi::SEMANTIC_RESIDUAL_ADD
    }

    fn reads_residual(self) -> bool {
        self.writes_hidden()
    }

    fn reads_weight(self) -> bool {
        self.affine_kind == abi::AFFINE_WEIGHT || self.affine_kind == abi::AFFINE_WEIGHT_OFFSET
    }

    fn reads_norm_bias(self) -> bool {
        self.norm_kind == abi::NORM_KIND_LAYER
    }

    fn reads_norm_style(self) -> bool {
        self.affine_kind == abi::AFFINE_ADA_SCALE_SHIFT
    }

    fn reads_gate_style(self) -> bool {
        self.has_gate
    }

    /// The gate replaces the bias in the combine, so a gated pipeline never
    /// takes one, and a pipeline with no combine has nothing to add it to.
    fn may_have_bias(self) -> bool {
        self.writes_hidden() && !self.has_gate
    }

    fn rounds_after_bias(self) -> bool {
        self.bias_order == abi::BIAS_ORDER_ROUND_AFTER_BIAS
    }
}

/// Row-wise normalization over a contiguous `[rows, cols]` activation.
///
/// `norm_style` is `[2 * cols]` (scale then shift) and `gate_style` is
/// `[3 * cols]` whose third segment is the gate; they are not interchangeable.
pub(crate) struct RawArgs<'a> {
    pub(crate) fields: Fields,
    pub(crate) input: &'a Tensor,
    pub(crate) bias: Option<&'a Tensor>,
    pub(crate) residual: Option<&'a Tensor>,
    pub(crate) weight: Option<&'a Tensor>,
    pub(crate) norm_bias: Option<&'a Tensor>,
    pub(crate) norm_style: Option<&'a Tensor>,
    pub(crate) gate_style: Option<&'a Tensor>,
    pub(crate) hidden: Option<&'a mut Tensor>,
    pub(crate) normalized: Option<&'a mut Tensor>,
    pub(crate) eps: f32,
    /// The `QuantOut` scale, or `None` when the caller asked for no output
    /// quantization. Deliberately not a bare `f32` defaulting to 1.0: that
    /// made `post: Some(E4m3 { scale: 1.0 })` indistinguishable from `post:
    /// None`, so a quantization request against a non-E4M3 output was
    /// silently dropped instead of refused.
    pub(crate) post_scale: Option<f32>,
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
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("Norm currently supports F16 and BF16")),
    }
}

fn output_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::F8E4M3 => Ok(3),
        _ => Err(invalid("Norm output supports F16, BF16 and E4M3")),
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
    let fields = args.fields;
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
    if fields.rounds_after_bias() && dtype != DType::BF16 {
        return Err(invalid(
            "BiasThenResidual is a BF16-only semantic because its intermediate rounding is part of the contract",
        ));
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid("Norm size overflow"))?;

    if !(args.eps.is_finite() && args.eps > 0.0) {
        return Err(invalid("Norm eps must be finite and positive"));
    }
    let output_scale = args.post_scale.unwrap_or(1.0);
    if !(output_scale.is_finite() && output_scale > 0.0) {
        return Err(invalid("Norm output_scale must be finite and positive"));
    }
    if args.bias.is_some() && !fields.may_have_bias() {
        return Err(invalid("Norm semantic does not take a bias"));
    }
    if args.residual.is_some() && !fields.reads_residual() {
        return Err(invalid("Norm semantic does not take a residual"));
    }
    if args.weight.is_some() && !fields.reads_weight() {
        return Err(invalid("Norm semantic does not take a weight"));
    }
    if args.norm_bias.is_some() && !fields.reads_norm_bias() {
        return Err(invalid("Norm semantic does not take a norm bias"));
    }
    if args.norm_style.is_some() && !fields.reads_norm_style() {
        return Err(invalid("Norm semantic does not take a norm style"));
    }
    if args.gate_style.is_some() && !fields.reads_gate_style() {
        return Err(invalid("Norm semantic does not take a gate style"));
    }
    if args.hidden.is_some() && !fields.writes_hidden() {
        return Err(invalid("Norm semantic does not write a hidden output"));
    }
    if args.normalized.is_some() && !fields.writes_normalized() {
        return Err(invalid("Norm semantic does not write a normalized output"));
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
            None if required => Err(invalid(format!("Norm semantic requires {label}"))),
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
        fields.reads_residual(),
        count,
        "residual",
        &mut storage,
    )?;
    let (weight, weight_alignment) = bind_vector(
        args.weight,
        fields.reads_weight(),
        cols,
        "weight",
        &mut storage,
    )?;
    let (norm_bias, norm_bias_alignment) = bind_vector(
        args.norm_bias,
        fields.reads_norm_bias(),
        cols,
        "norm_bias",
        &mut storage,
    )?;
    let norm_style_elements = cols
        .checked_mul(2)
        .ok_or_else(|| invalid("Norm style size overflow"))?;
    let (norm_style, norm_style_alignment) = bind_vector(
        args.norm_style,
        fields.reads_norm_style(),
        norm_style_elements,
        "norm_style",
        &mut storage,
    )?;
    let gate_style_elements = cols
        .checked_mul(3)
        .ok_or_else(|| invalid("Norm gate style size overflow"))?;
    let (gate_style, gate_style_alignment) = bind_vector(
        args.gate_style,
        fields.reads_gate_style(),
        gate_style_elements,
        "gate_style",
        &mut storage,
    )?;

    let bind_output = |tensor: Option<&mut Tensor>,
                       required: bool,
                       output_dtype: DType,
                       label: &str,
                       storage: &mut Vec<CudaBuffer>|
     -> Result<(*mut std::ffi::c_void, u32)> {
        match tensor {
            Some(tensor) => {
                let buffer = tensor_storage(ctx, tensor, output_dtype, count)?;
                let pointer = buffer.ptr() as *mut std::ffi::c_void;
                let alignment = alignment_of(buffer.ptr() as usize);
                storage.push(buffer);
                Ok((pointer, alignment))
            }
            None if required => Err(invalid(format!("Norm semantic requires {label}"))),
            None => Ok((std::ptr::null_mut(), 256)),
        }
    };

    let (hidden, hidden_alignment) = bind_output(
        args.hidden,
        fields.writes_hidden(),
        dtype,
        "hidden",
        &mut storage,
    )?;
    let output_dtype = args
        .normalized
        .as_ref()
        .map_or(dtype, |tensor| tensor.dtype());
    // The output dtype and `post` are one decision stated twice, so they must
    // agree. Deriving `quant_kind` from the dtype alone let a `post` on a
    // non-E4M3 output be dropped whenever its scale happened to be 1.0 -- the
    // caller asked for quantization and got a plain norm, silently.
    let quant_kind = match (output_dtype == DType::F8E4M3, args.post_scale.is_some()) {
        (true, true) => {
            if dtype != DType::F16 || !fields.writes_normalized() {
                return Err(invalid(
                    "Quantized Norm output requires an F16 normalization semantic",
                ));
            }
            abi::QUANT_E4M3
        }
        (false, false) => {
            if output_dtype != dtype {
                return Err(invalid(
                    "Non-quantized Norm output must preserve dtype and use unit output_scale",
                ));
            }
            abi::QUANT_NONE
        }
        (true, false) => {
            return Err(invalid(
                "E4M3 Norm output requires a post quantization scale",
            ))
        }
        (false, true) => {
            return Err(invalid(
                "Non-quantized Norm output must preserve dtype and use unit output_scale",
            ))
        }
    };
    let (normalized, normalized_alignment) = bind_output(
        args.normalized,
        fields.writes_normalized(),
        output_dtype,
        "normalized",
        &mut storage,
    )?;

    let dtype_value = dtype_code(dtype)?;
    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        semantic: fields.semantic,
        norm_kind: fields.norm_kind,
        affine_kind: fields.affine_kind,
        has_gate: u32::from(fields.has_gate),
        bias_order: fields.bias_order,
        quant_kind,
        dtype: dtype_value,
        output_dtype: output_dtype_code(output_dtype)?,
        has_bias: u32::from(!bias.is_null()),
        input_alignment,
        weight_alignment,
        bias_alignment: bias_alignment.min(norm_bias_alignment),
        residual_alignment,
        style_alignment: norm_style_alignment.min(gate_style_alignment),
        hidden_alignment,
        normalized_alignment,
        rows: i64::from(rows_abi),
        cols: i64::from(cols_abi),
        output_scale_is_unit: u32::from(output_scale == 1.0),
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
        stream: ctx.stream().handle() as abi::CudaStream,
        eps: args.eps,
        output_scale,
    };

    Ok(Normalized {
        spec,
        bindings,
        storage,
    })
}

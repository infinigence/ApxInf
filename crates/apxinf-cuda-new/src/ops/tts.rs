//! Speech-model elementwise, statistics, and layout operators.
//!
//! These are the semantics GPT-SoVITS needs that the language-model Pointwise
//! ABI does not expose: extended activations, GRN, per-column normalization,
//! reference RoPE tables, row statistics, and the KV-cache token append. They
//! live in a separate adapter so the stable Pointwise encoding is untouched.

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::{status, tts as abi};
use crate::{CudaBuffer, CudaContext};

/// One FP32 normalizer per column has to fit the 48 KiB dynamic shared-memory
/// window a `grn` launch may request.
const GRN_MAX_COLUMNS: usize = 48 * 1024 / std::mem::size_of::<f32>();

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TtsActivation {
    None,
    Gelu,
    Silu,
    GeluExact,
    Mish,
    LeakyRelu,
    Tanh,
    Sigmoid,
    Relu,
    LeakyReluOnePercent,
}

impl TtsActivation {
    fn code(self) -> u32 {
        match self {
            Self::None => abi::ACTIVATION_NONE,
            Self::Gelu => abi::ACTIVATION_GELU,
            Self::Silu => abi::ACTIVATION_SILU,
            Self::GeluExact => abi::ACTIVATION_GELU_EXACT,
            Self::Mish => abi::ACTIVATION_MISH,
            Self::LeakyRelu => abi::ACTIVATION_LEAKY_RELU,
            Self::Tanh => abi::ACTIVATION_TANH,
            Self::Sigmoid => abi::ACTIVATION_SIGMOID,
            Self::Relu => abi::ACTIVATION_RELU,
            Self::LeakyReluOnePercent => abi::ACTIVATION_LEAKY_RELU_ONE_PERCENT,
        }
    }
}

/// Catalog identifier of every public semantic in this family. The doc test
/// compares `ALL` with the markers in `cuda-operator.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TtsSemantic {
    BiasActivation,
    ResidualAddBias,
    Affine,
    GluSeparate,
    TanhSigmoidGate,
    Grn,
    ColumnNorm,
    RopeApply,
    SinusoidalEmbedding,
    MeanRows,
    ArgmaxRows,
    WriteKvCacheToken,
    FillZero,
}

impl TtsSemantic {
    pub(crate) const ALL: [Self; 13] = [
        Self::BiasActivation,
        Self::ResidualAddBias,
        Self::Affine,
        Self::GluSeparate,
        Self::TanhSigmoidGate,
        Self::Grn,
        Self::ColumnNorm,
        Self::RopeApply,
        Self::SinusoidalEmbedding,
        Self::MeanRows,
        Self::ArgmaxRows,
        Self::WriteKvCacheToken,
        Self::FillZero,
    ];

    pub(crate) fn doc_id(self) -> &'static str {
        match self {
            Self::BiasActivation => "bias-activation",
            Self::ResidualAddBias => "residual-add-bias",
            Self::Affine => "affine",
            Self::GluSeparate => "glu-separate",
            Self::TanhSigmoidGate => "tanh-sigmoid-gate",
            Self::Grn => "grn",
            Self::ColumnNorm => "column-norm",
            Self::RopeApply => "rope-apply",
            Self::SinusoidalEmbedding => "sinusoidal-embedding",
            Self::MeanRows => "mean-rows",
            Self::ArgmaxRows => "argmax-rows",
            Self::WriteKvCacheToken => "write-kv-cache-token",
            Self::FillZero => "fill-zero",
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid("tts operators support F32, F16, and BF16 only")),
    }
}

fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
    label: &str,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != dtype {
        return Err(invalid(format!("tts {label} device/dtype mismatch")));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid(format!(
            "tts {label} element count mismatch: expected {elements}, got {}",
            tensor.shape().numel()
        )));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("tts size overflow"))?;
    if buffer.len() < expected {
        return Err(invalid(format!("tts {label} storage is too small")));
    }
    Ok(buffer)
}

/// Like [`tensor_storage`], but accepts a tensor with at least `elements`
/// contiguous leading elements. Rotary tables are allocated once for the
/// maximum position count; a shorter sequence consumes only a prefix.
fn tensor_storage_at_least(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
    label: &str,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != dtype {
        return Err(invalid(format!("tts {label} device/dtype mismatch")));
    }
    if tensor.shape().numel() < elements {
        return Err(invalid(format!(
            "tts {label} element count mismatch: expected at least {elements}, got {}",
            tensor.shape().numel()
        )));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid("tts size overflow"))?;
    if buffer.len() < expected {
        return Err(invalid(format!("tts {label} storage is too small")));
    }
    Ok(buffer)
}

fn same_layout(ctx: &CudaContext, input: &Tensor, out: &Tensor, label: &str) -> Result<usize> {
    if input.dtype() != out.dtype() {
        return Err(invalid(format!("tts {label} dtype mismatch")));
    }
    if input.device() != Device::Cuda(ctx.device_id())
        || out.device() != Device::Cuda(ctx.device_id())
    {
        return Err(invalid(format!("tts {label} device mismatch")));
    }
    if input.shape() != out.shape() {
        return Err(invalid(format!("tts {label} shape mismatch")));
    }
    Ok(input.numel())
}

/// `output = activation(input + bias)`, where `bias` repeats every `cols`
/// elements. `out` has the same shape as `input`.
///
/// With `round_bias`, the sum is rounded to the tensor dtype before the
/// activation. Use it when the bias a producer deferred has to reproduce the
/// value that producer would have written itself.
pub fn bias_activation(
    ctx: &CudaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    out: &Tensor,
    activation: TtsActivation,
    round_bias: bool,
) -> Result<()> {
    let count = same_layout(ctx, input, out, "bias activation")?;
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let cols = *input
        .shape()
        .dims()
        .last()
        .ok_or_else(|| invalid("tts bias activation needs a rank"))?;
    if cols == 0 || count % cols != 0 {
        return Err(invalid("tts bias activation geometry is invalid"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, count, "bias activation input")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "bias activation output")?;
    let bias_buffer = bias
        .map(|tensor| tensor_storage(ctx, tensor, dtype, cols, "bias activation bias"))
        .transpose()?;
    unsafe {
        status::check(abi::apxinf_tts_bias_activation(
            ctx.runtime(),
            input_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr()),
            out_buffer.ptr(),
            count as i64,
            cols as i64,
            code,
            activation.code(),
            u32::from(round_bias),
            ctx.stream().handle(),
        ))
    }
}

/// `output = round_dtype(round_dtype(input + bias) + residual)`.
///
/// This is the arithmetic of a deferred bias followed by the ordinary residual
/// add, in one launch: the intermediate sum is rounded to the tensor dtype
/// before the residual is added.
pub fn residual_add_bias(
    ctx: &CudaContext,
    input: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let count = same_layout(ctx, input, out, "residual add bias")?;
    if residual.shape() != input.shape() {
        return Err(invalid("tts residual add bias residual shape mismatch"));
    }
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let cols = *input
        .shape()
        .dims()
        .last()
        .ok_or_else(|| invalid("tts residual add bias needs a rank"))?;
    if cols == 0 || count % cols != 0 {
        return Err(invalid("tts residual add bias geometry is invalid"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, count, "residual add bias input")?;
    let bias_buffer = tensor_storage(ctx, bias, dtype, cols, "residual add bias bias")?;
    let residual_buffer =
        tensor_storage(ctx, residual, dtype, count, "residual add bias residual")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "residual add bias output")?;
    unsafe {
        status::check(abi::apxinf_tts_residual_add_bias(
            ctx.runtime(),
            input_buffer.ptr(),
            bias_buffer.ptr(),
            residual_buffer.ptr(),
            out_buffer.ptr(),
            count as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// `output = input * style[col] + style[cols + col]`.
pub fn affine(ctx: &CudaContext, input: &Tensor, style: &Tensor, out: &Tensor) -> Result<()> {
    let count = same_layout(ctx, input, out, "affine")?;
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let cols = *input
        .shape()
        .dims()
        .last()
        .ok_or_else(|| invalid("tts affine needs a rank"))?;
    if cols == 0 {
        return Err(invalid("tts affine width is zero"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, count, "affine input")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "affine output")?;
    let style_buffer = tensor_storage(ctx, style, dtype, 2 * cols, "affine style")?;
    unsafe {
        status::check(abi::apxinf_tts_affine(
            ctx.runtime(),
            input_buffer.ptr(),
            style_buffer.ptr(),
            out_buffer.ptr(),
            count as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// `output = input * sigmoid(gate)`.
pub fn glu_separate(ctx: &CudaContext, input: &Tensor, gate: &Tensor, out: &Tensor) -> Result<()> {
    let count = same_layout(ctx, input, out, "glu separate")?;
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let input_buffer = tensor_storage(ctx, input, dtype, count, "glu separate input")?;
    let gate_buffer = tensor_storage(ctx, gate, dtype, count, "glu separate gate")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "glu separate output")?;
    unsafe {
        status::check(abi::apxinf_tts_glu_separate(
            ctx.runtime(),
            input_buffer.ptr(),
            gate_buffer.ptr(),
            out_buffer.ptr(),
            count as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// `tanh(a) * sigmoid(b)` over halves of the last axis of `input`, optionally
/// after adding the same-shaped `secondary`.
pub fn tanh_sigmoid_gate(
    ctx: &CudaContext,
    input: &Tensor,
    secondary: Option<&Tensor>,
    out: &Tensor,
) -> Result<()> {
    if input.dtype() != out.dtype() {
        return Err(invalid("tts tanh-sigmoid gate dtype mismatch"));
    }
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let cols = *input
        .shape()
        .dims()
        .last()
        .ok_or_else(|| invalid("tts tanh-sigmoid gate needs a rank"))?;
    if cols % 2 != 0 || out.numel() * 2 != input.numel() {
        return Err(invalid("tts tanh-sigmoid gate geometry is invalid"));
    }
    let input_count = input.numel();
    let input_buffer = tensor_storage(ctx, input, dtype, input_count, "tanh-sigmoid input")?;
    let secondary_buffer = secondary
        .map(|tensor| tensor_storage(ctx, tensor, dtype, input_count, "tanh-sigmoid secondary"))
        .transpose()?;
    let out_buffer = tensor_storage(ctx, out, dtype, out.numel(), "tanh-sigmoid output")?;
    unsafe {
        status::check(abi::apxinf_tts_tanh_sigmoid_gate(
            ctx.runtime(),
            input_buffer.ptr(),
            secondary_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr()),
            out_buffer.ptr(),
            out.numel() as i64,
            (cols / 2) as i64,
            code,
            u32::from(secondary_buffer.is_some()),
            ctx.stream().handle(),
        ))
    }
}

/// Global response normalization over `[rows, cols]`.
pub fn grn(ctx: &CudaContext, input: &Tensor, style: &Tensor, out: &Tensor) -> Result<()> {
    let count = same_layout(ctx, input, out, "grn")?;
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("tts grn expects a rank-2 input"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if rows == 0 || cols == 0 {
        return Err(invalid("tts grn geometry is invalid"));
    }
    // The kernel stages one FP32 normalizer per column in dynamic shared
    // memory and does not opt into the larger per-block limit.
    if cols > GRN_MAX_COLUMNS {
        return Err(invalid(
            "tts grn columns exceed the 48 KiB shared-memory window",
        ));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, count, "grn input")?;
    let style_buffer = tensor_storage(ctx, style, dtype, 2 * cols, "grn style")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "grn output")?;
    unsafe {
        status::check(abi::apxinf_tts_grn(
            ctx.runtime(),
            input_buffer.ptr(),
            style_buffer.ptr(),
            out_buffer.ptr(),
            rows as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// Per-column LayerNorm over rows.
pub fn column_norm(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    out: &Tensor,
    eps: f32,
) -> Result<()> {
    let count = same_layout(ctx, input, out, "column norm")?;
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("tts column norm expects a rank-2 input"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if rows == 0 || cols == 0 {
        return Err(invalid("tts column norm geometry is invalid"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, count, "column norm input")?;
    let style_buffer = tensor_storage(ctx, style, dtype, 2 * cols, "column norm style")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "column norm output")?;
    unsafe {
        status::check(abi::apxinf_tts_column_norm(
            ctx.runtime(),
            input_buffer.ptr(),
            style_buffer.ptr(),
            out_buffer.ptr(),
            rows as i64,
            cols as i64,
            code,
            eps,
            ctx.stream().handle(),
        ))
    }
}

/// Interleaved rotary embedding over `[batch, time, heads, head_dim]` with
/// `cos`/`sin` tables of shape `[time, head_dim]`.
pub fn rope_apply(
    ctx: &CudaContext,
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let count = same_layout(ctx, input, out, "rope")?;
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let dims = input.shape().dims().to_vec();
    if dims.len() != 4 {
        return Err(invalid("tts rope expects [batch, time, heads, head_dim]"));
    }
    let (batch, time, heads, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
    if batch == 0 || time == 0 || heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
        return Err(invalid("tts rope geometry is invalid"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, count, "rope input")?;
    let cos_buffer = tensor_storage_at_least(ctx, cos, dtype, time * head_dim, "rope cos")?;
    let sin_buffer = tensor_storage_at_least(ctx, sin, dtype, time * head_dim, "rope sin")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "rope output")?;
    unsafe {
        status::check(abi::apxinf_tts_rope_apply(
            ctx.runtime(),
            input_buffer.ptr(),
            cos_buffer.ptr(),
            sin_buffer.ptr(),
            out_buffer.ptr(),
            (batch * time) as i64,
            time as i64,
            heads as i64,
            head_dim as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// Sin/cos features from scalar `times`, matching the reference table formula.
/// [`sinusoidal_embedding`] wraps this with the reference frequency sweep.
fn sinusoidal(
    ctx: &CudaContext,
    times: &Tensor,
    out: &Tensor,
    scale: f32,
    log_step: f32,
) -> Result<()> {
    let dtype = times.dtype();
    let code = dtype_code(dtype)?;
    if out.dtype() != dtype {
        return Err(invalid("tts sinusoidal dtype mismatch"));
    }
    let dims = out.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("tts sinusoidal output must be rank 2"));
    }
    let (rows, width) = (dims[0], dims[1]);
    if width < 4 || width % 2 != 0 {
        return Err(invalid("tts sinusoidal width must be even and at least 4"));
    }
    let times_buffer = tensor_storage(ctx, times, dtype, rows, "sinusoidal times")?;
    let out_buffer = tensor_storage(ctx, out, dtype, rows * width, "sinusoidal output")?;
    unsafe {
        status::check(abi::apxinf_tts_sinusoidal(
            ctx.runtime(),
            times_buffer.ptr(),
            out_buffer.ptr(),
            rows as i64,
            width as i64,
            scale,
            log_step,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// Reference sinusoidal table: `scale` multiplies each scalar time before a
/// frequency sweep derived from `max_period`, matching the speech reference's
/// `sin`-then-`cos` column layout.
pub fn sinusoidal_embedding(
    ctx: &CudaContext,
    times: &Tensor,
    output: &Tensor,
    scale: f32,
    max_period: f64,
) -> Result<()> {
    let dims = output.shape().dims().to_vec();
    if dims.len() != 2
        || dims[0] == 0
        || dims[1] < 4
        || dims[1] % 2 != 0
        || times.shape().dims() != [dims[0]]
        || !scale.is_finite()
        || !max_period.is_finite()
        || max_period <= 1.0
    {
        return Err(invalid(
            "invalid sinusoidal embedding geometry or frequency",
        ));
    }
    let log_step = (max_period.ln() / (dims[1] / 2 - 1) as f64) as f32;
    sinusoidal(ctx, times, output, scale, log_step)
}

/// Mean over rows, producing one row.
pub fn mean_rows(ctx: &CudaContext, input: &Tensor, out: &Tensor) -> Result<()> {
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    if out.dtype() != dtype {
        return Err(invalid("tts mean rows dtype mismatch"));
    }
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(invalid("tts mean rows expects a non-empty rank-2 input"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if out.numel() != cols {
        return Err(invalid("tts mean rows output must have one row"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, rows * cols, "mean rows input")?;
    let out_buffer = tensor_storage(ctx, out, dtype, cols, "mean rows output")?;
    unsafe {
        status::check(abi::apxinf_tts_mean_rows(
            ctx.runtime(),
            input_buffer.ptr(),
            out_buffer.ptr(),
            rows as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// Per-row argmax with first-index tie breaking.
pub fn argmax_rows(ctx: &CudaContext, logits: &Tensor, index: &Tensor) -> Result<()> {
    let dtype = logits.dtype();
    let code = dtype_code(dtype)?;
    let dims = logits.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(invalid("tts argmax rows expects a non-empty rank-2 input"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if index.dtype() != DType::I32
        || index.device() != Device::Cuda(ctx.device_id())
        || index.numel() != rows
    {
        return Err(invalid("tts argmax rows index must be one I32 per row"));
    }
    let logits_buffer = tensor_storage(ctx, logits, dtype, rows * cols, "argmax rows logits")?;
    let index_buffer = tensor_storage(ctx, index, DType::I32, rows, "argmax rows index")?;
    unsafe {
        status::check(abi::apxinf_tts_argmax_rows(
            ctx.runtime(),
            logits_buffer.ptr(),
            index_buffer.ptr(),
            rows as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// Copy one input token into the cache row named by `meta[1]`.
///
/// A row index at or past the cache capacity writes nothing, which mirrors the
/// clamp `kv_cache_attention` applies to the same metadata: the write position
/// is device-side state, so enforcing it on the host would require a sync.
pub fn write_kv_cache_token(
    ctx: &CudaContext,
    input: &Tensor,
    cache: &Tensor,
    meta: &crate::ops::KvCacheDecodeMeta,
) -> Result<()> {
    let dims = cache.shape().dims().to_vec();
    let input_dims = input.shape().dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[2] == 0 || dims[3] == 0 {
        return Err(invalid(
            "dynamic KV token write requires [1, capacity, heads, head_dim] cache",
        ));
    }
    if input_dims != [1, 1, dims[2], dims[3]] {
        return Err(invalid(
            "dynamic KV token write requires a single token row",
        ));
    }
    let dtype = cache.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16) || input.dtype() != dtype {
        return Err(invalid(
            "dynamic KV token write supports matching F16 or BF16 tensors",
        ));
    }
    if meta.capacity() != dims[1] || meta.device() != ctx.device_id() {
        return Err(invalid("dynamic KV token write meta geometry mismatch"));
    }
    let capacity = dims[1];
    let row_elements = dims[2] * dims[3];
    let input_buffer = tensor_storage(ctx, input, dtype, row_elements, "kv token input")?;
    let cache_buffer =
        tensor_storage(ctx, cache, dtype, capacity * row_elements, "kv token cache")?;
    meta.retain();
    crate::workspace::retain_buffer(&input_buffer);
    crate::workspace::retain_buffer(&cache_buffer);
    crate::workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize)?;
    let code = dtype_code(dtype)?;
    unsafe {
        status::check(abi::apxinf_tts_write_kv_cache_token(
            ctx.runtime(),
            input_buffer.ptr(),
            cache_buffer.ptr(),
            meta.device_address().cast_const().cast(),
            capacity as i64,
            row_elements as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// Set every element of `tensor` to zero.
pub fn fill_zero(ctx: &CudaContext, tensor: &Tensor) -> Result<()> {
    let dtype = tensor.dtype();
    let code = dtype_code(dtype)?;
    if tensor.numel() == 0 {
        return Ok(());
    }
    if tensor.device() != Device::Cuda(ctx.device_id()) {
        return Err(invalid("tts fill zero device mismatch"));
    }
    let buffer = tensor_storage(ctx, tensor, dtype, tensor.numel(), "fill zero")?;
    unsafe {
        status::check(abi::apxinf_tts_fill_zero(
            ctx.runtime(),
            buffer.ptr(),
            tensor.numel() as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

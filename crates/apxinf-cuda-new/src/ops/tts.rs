//! Speech-model elementwise, statistics, and layout operators.
//!
//! These are the semantics GPT-SoVITS needs that the language-model Pointwise
//! ABI does not expose: extended activations, GRN, per-column normalization,
//! reference RoPE tables, and row/column layout copies. They live in a separate
//! adapter so the stable Pointwise encoding is untouched.

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::{status, tts as abi};
use crate::{CudaBuffer, CudaContext};

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

fn same_layout(
    ctx: &CudaContext,
    input: &Tensor,
    out: &Tensor,
    label: &str,
) -> Result<usize> {
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
pub fn bias_activation(
    ctx: &CudaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    out: &Tensor,
    activation: TtsActivation,
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
            ctx.stream().handle(),
        ))
    }
}

/// `output = input * style[col] + style[cols + col]`.
pub fn affine(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    out: &Tensor,
) -> Result<()> {
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

/// GLU over the last axis: `[rows, 2 * cols] -> [rows, cols]`.
pub fn glu(ctx: &CudaContext, input: &Tensor, out: &Tensor) -> Result<()> {
    if input.dtype() != out.dtype() {
        return Err(invalid("tts glu dtype mismatch"));
    }
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let cols = *out
        .shape()
        .dims()
        .last()
        .ok_or_else(|| invalid("tts glu needs a rank"))?;
    let count = out.numel();
    if cols == 0 || input.numel() != count * 2 {
        return Err(invalid("tts glu geometry is invalid"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, count * 2, "glu input")?;
    let out_buffer = tensor_storage(ctx, out, dtype, count, "glu output")?;
    unsafe {
        status::check(abi::apxinf_tts_glu(
            ctx.runtime(),
            input_buffer.ptr(),
            out_buffer.ptr(),
            count as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// `output = input * sigmoid(gate)`.
pub fn glu_separate(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    out: &Tensor,
) -> Result<()> {
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
        .map(|tensor| {
            tensor_storage(ctx, tensor, dtype, input_count, "tanh-sigmoid secondary")
        })
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
pub fn grn(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    out: &Tensor,
) -> Result<()> {
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
pub fn sinusoidal(
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
        return Err(invalid("dynamic KV token write requires a single token row"));
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
    crate::workspace::retain_resource(&std::rc::Rc::new(input_buffer.clone()));
    crate::workspace::retain_resource(&std::rc::Rc::new(cache_buffer.clone()));
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        ctx.stream().handle() as usize,
    )?;
    let code = dtype_code(dtype)?;
    unsafe {
        status::check(abi::apxinf_tts_cache_write_token(
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

pub fn embedding_gather(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let table_dims = table.shape().dims().to_vec();
    let out_dims = out.shape().dims().to_vec();
    if table_dims.len() != 2 || out_dims.len() != 2 {
        return Err(invalid("tts embedding gather expects rank-2 table/output"));
    }
    let (vocab, width) = (table_dims[0], table_dims[1]);
    let rows = out_dims[0];
    if vocab == 0 || width == 0 || rows == 0 || out_dims[1] != width {
        return Err(invalid("tts embedding gather geometry is invalid"));
    }
    let dtype = table.dtype();
    let code = dtype_code(dtype)?;
    if out.dtype() != dtype {
        return Err(invalid("tts embedding gather dtype mismatch"));
    }
    if ids.dtype() != DType::I32 || ids.numel() != rows {
        return Err(invalid(
            "tts embedding gather ids must be I32 with one entry per row",
        ));
    }
    let table_buffer = tensor_storage(ctx, table, dtype, vocab * width, "embedding table")?;
    let ids_buffer = tensor_storage(ctx, ids, DType::I32, rows, "embedding ids")?;
    let out_buffer = tensor_storage(ctx, out, dtype, rows * width, "embedding output")?;
    unsafe {
        status::check(abi::apxinf_tts_gather_rows(
            ctx.runtime(),
            table_buffer.ptr(),
            ids_buffer.ptr(),
            out_buffer.ptr(),
            rows as i64,
            width as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

pub fn copy_rows(
    ctx: &CudaContext,
    input: &Tensor,
    offset: usize,
    rows: usize,
    out: &Tensor,
) -> Result<()> {
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 || dims[1] == 0 || rows == 0 || offset + rows > dims[0] {
        return Err(invalid("tts copy rows range is invalid"));
    }
    if out.dtype() != dtype || out.numel() != rows * dims[1] {
        return Err(invalid("tts copy rows output mismatch"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, dims[0] * dims[1], "copy rows input")?;
    let out_buffer = tensor_storage(ctx, out, dtype, rows * dims[1], "copy rows output")?;
    unsafe {
        status::check(abi::apxinf_tts_copy_rows(
            ctx.runtime(),
            input_buffer.ptr(),
            out_buffer.ptr(),
            offset as i64,
            rows as i64,
            dims[1] as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

pub fn concat_columns(
    ctx: &CudaContext,
    first: &Tensor,
    second: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let first_dims = first.shape().dims().to_vec();
    let second_dims = second.shape().dims().to_vec();
    if first_dims.len() != 2 || second_dims.len() != 2 || first_dims[0] != second_dims[0] {
        return Err(invalid("tts column concat expects equal-row rank-2 inputs"));
    }
    let dtype = first.dtype();
    let code = dtype_code(dtype)?;
    if second.dtype() != dtype || out.dtype() != dtype {
        return Err(invalid("tts column concat dtype mismatch"));
    }
    let rows = first_dims[0];
    let first_cols = first_dims[1];
    let second_cols = second_dims[1];
    if rows == 0 || first_cols == 0 || second_cols == 0 {
        return Err(invalid("tts column concat geometry is invalid"));
    }
    let out_dims = out.shape().dims();
    if out_dims != [rows, first_cols + second_cols] {
        return Err(invalid("tts column concat output shape mismatch"));
    }
    let first_buffer = tensor_storage(ctx, first, dtype, rows * first_cols, "concat first")?;
    let second_buffer = tensor_storage(ctx, second, dtype, rows * second_cols, "concat second")?;
    let out_buffer = tensor_storage(ctx, out, dtype, rows * (first_cols + second_cols), "concat output")?;
    unsafe {
        status::check(abi::apxinf_tts_concat_columns(
            ctx.runtime(),
            first_buffer.ptr(),
            second_buffer.ptr(),
            out_buffer.ptr(),
            rows as i64,
            first_cols as i64,
            second_cols as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

pub fn slice_columns(
    ctx: &CudaContext,
    input: &Tensor,
    column_offset: usize,
    columns: usize,
    out: &Tensor,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 || columns == 0 || column_offset + columns > dims[1] {
        return Err(invalid("tts slice columns range is invalid"));
    }
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    if out.dtype() != dtype || out.numel() != dims[0] * columns {
        return Err(invalid("tts slice columns output mismatch"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, dims[0] * dims[1], "slice columns input")?;
    let out_buffer = tensor_storage(ctx, out, dtype, dims[0] * columns, "slice columns output")?;
    unsafe {
        status::check(abi::apxinf_tts_slice_columns(
            ctx.runtime(),
            input_buffer.ptr(),
            out_buffer.ptr(),
            dims[0] as i64,
            dims[1] as i64,
            column_offset as i64,
            columns as i64,
            code,
            ctx.stream().handle(),
        ))
    }
}

/// Bias-before-cast projection epilogue: `out_f16 = F16(projection_f32 + bias)`.
pub fn project_bias_f32_to_f16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let dims = projection.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(invalid("tts projection bias expects a non-empty matrix"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if projection.dtype() != DType::F32
        || projection.device() != Device::Cuda(ctx.device_id())
        || bias.dtype() != DType::F16
        || bias.device() != Device::Cuda(ctx.device_id())
        || bias.numel() != cols
        || out.dtype() != DType::F16
        || out.device() != Device::Cuda(ctx.device_id())
        || out.shape().dims() != [rows, cols]
    {
        return Err(invalid("tts projection bias geometry or dtype mismatch"));
    }
    let projection_buffer =
        tensor_storage(ctx, projection, DType::F32, rows * cols, "projection")?;
    let bias_buffer = tensor_storage(ctx, bias, DType::F16, cols, "projection bias")?;
    let out_buffer = tensor_storage(ctx, out, DType::F16, rows * cols, "projection output")?;
    unsafe {
        status::check(abi::apxinf_tts_project_bias_f32_to_f16(
            ctx.runtime(),
            projection_buffer.ptr(),
            bias_buffer.ptr(),
            out_buffer.ptr(),
            (rows * cols) as i64,
            cols as i64,
            ctx.stream().handle(),
        ))
    }
}

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

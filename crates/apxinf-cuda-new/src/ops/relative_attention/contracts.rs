use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::{relative_attention as abi, status};
use crate::{CudaBuffer, CudaContext};

/// Relative-position self-attention over `[batch, length, heads, head_dim]`
/// tensors with a head-shared `[2 * window + 1, head_dim]` table.
pub struct RelativeAttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub relative_key: &'a Tensor,
    pub relative_value: &'a Tensor,
    pub out: &'a mut Tensor,
    pub window: usize,
    /// Multiplies the whole logit; normally `1 / sqrt(head_dim)`.
    pub scale: f32,
}

impl<'a> RelativeAttentionArgs<'a> {
    pub fn new(
        query: &'a Tensor,
        key: &'a Tensor,
        value: &'a Tensor,
        relative_key: &'a Tensor,
        relative_value: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        let head_dim = query.shape().dims().get(3).copied().unwrap_or(1);
        Self {
            query,
            key,
            value,
            relative_key,
            relative_value,
            out,
            window: 0,
            scale: 1.0 / (head_dim as f32).sqrt(),
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

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid(
            "RelativeAttention supports F32, F16, and BF16 tensors only",
        )),
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
        return Err(invalid(format!(
            "RelativeAttention {label} device/dtype mismatch"
        )));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid(format!(
            "RelativeAttention {label} element count mismatch"
        )));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < elements * dtype.size_in_bytes() {
        return Err(invalid(format!(
            "RelativeAttention {label} storage is too small"
        )));
    }
    Ok(buffer)
}

pub(crate) fn normalize(ctx: &CudaContext, args: RelativeAttentionArgs<'_>) -> Result<Normalized> {
    let q_dims = args.query.shape().dims().to_vec();
    let k_dims = args.key.shape().dims().to_vec();
    let v_dims = args.value.shape().dims().to_vec();
    let out_dims = args.out.shape().dims().to_vec();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 || out_dims.len() != 4 {
        return Err(invalid(
            "RelativeAttention expects rank-4 query/key/value/output",
        ));
    }
    let (batch, length, heads, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    if batch == 0 || length == 0 || heads == 0 || head_dim == 0 {
        return Err(invalid("RelativeAttention geometry must be non-empty"));
    }
    if k_dims != q_dims || v_dims != q_dims || out_dims != q_dims {
        return Err(invalid(
            "RelativeAttention query/key/value/output shapes must agree",
        ));
    }
    let table_dims = args.relative_key.shape().dims().to_vec();
    let value_table_dims = args.relative_value.shape().dims().to_vec();
    if table_dims.len() != 2
        || table_dims[0] != 2 * args.window + 1
        || table_dims[1] != head_dim
        || value_table_dims != table_dims
    {
        return Err(invalid(
            "RelativeAttention tables must be [2 * window + 1, head_dim]",
        ));
    }
    if !args.scale.is_finite() {
        return Err(invalid("RelativeAttention scale must be finite"));
    }
    let dtype = args.query.dtype();
    if args.key.dtype() != dtype
        || args.value.dtype() != dtype
        || args.relative_key.dtype() != dtype
        || args.relative_value.dtype() != dtype
        || args.out.dtype() != dtype
    {
        return Err(invalid("RelativeAttention dtypes must match"));
    }
    let _ = dtype_code(dtype)?;

    let mut storage = Vec::new();
    let q_count = batch * length * heads * head_dim;
    let mut push =
        |tensor: &Tensor, count: usize, label: &str| -> Result<*const std::ffi::c_void> {
            let buffer = tensor_storage(ctx, tensor, dtype, count, label)?;
            let pointer = buffer.ptr() as *const std::ffi::c_void;
            storage.push(buffer);
            Ok(pointer)
        };
    let query = push(args.query, q_count, "query")?;
    let key = push(args.key, q_count, "key")?;
    let value = push(args.value, q_count, "value")?;
    let table_count = table_dims[0] * table_dims[1];
    let relative_key = push(args.relative_key, table_count, "relative_key")?;
    let relative_value = push(args.relative_value, table_count, "relative_value")?;
    let output_buffer = tensor_storage(ctx, args.out, dtype, q_count, "output")?;
    let output = output_buffer.ptr() as *mut std::ffi::c_void;
    let output_alignment = alignment_of(output_buffer.ptr() as usize);
    storage.push(output_buffer);
    let _ = output_alignment;

    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        dtype: dtype_code(dtype)?,
        heads_share_relative: 1,
        reserved: 0,
        batch: i64::try_from(batch).map_err(|_| invalid("batch exceeds i64"))?,
        length: i64::try_from(length).map_err(|_| invalid("length exceeds i64"))?,
        heads: i64::try_from(heads).map_err(|_| invalid("heads exceeds i64"))?,
        head_dim: i64::try_from(head_dim).map_err(|_| invalid("head_dim exceeds i64"))?,
        window: i64::try_from(args.window).map_err(|_| invalid("window exceeds i64"))?,
        scale: args.scale,
    };
    let bindings = abi::Bindings {
        query,
        key,
        value,
        relative_key,
        relative_value,
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
    let _storage = &normalized.storage;
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    unsafe {
        status::check(abi::apxinf_relative_attention_launch(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

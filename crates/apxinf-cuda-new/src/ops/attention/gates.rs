//! Attention query/gate split and output gate helpers.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{attention as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

fn alignment_of(pointer: usize) -> u32 {
    (1usize << pointer.trailing_zeros().min(8)) as u32
}

fn to_i64(value: usize, what: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| invalid(format!("{what} exceeds i64")))
}

fn gate_elements(tokens: usize, heads: usize, head_dim: usize) -> Result<usize> {
    tokens
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| invalid("Attention gate element count overflow"))
}

fn launch_gate(ctx: &CudaContext, spec: abi::GateSpec, bindings: abi::GateBindings) -> Result<()> {
    unsafe {
        status::check(abi::apxinf_attention_gate_launch(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}

/// Where the sigmoid gate values are stored.
pub enum OutputGateSource<'a> {
    /// A dense tensor with the same shape as the attention output.
    Tensor(&'a Tensor),
    /// Gate values occupy the second half of each head slot in a wider
    /// projection row: `[query, gate, ...]` per head.
    InterleavedProjection {
        projection: &'a Tensor,
        heads: usize,
        head_dim: usize,
    },
}

/// Numerics required by the checkpoint that consumes the gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputGateNumerics {
    /// Compute sigmoid and multiplication in the implementation's normal
    /// accumulator precision.
    Flexible,
    /// Round sigmoid to BF16 before multiplying, preserving the planning
    /// checkpoint's original operation order.
    RoundedSigmoid,
}

/// Sigmoid output-gate contract: `data *= sigmoid(gate)` in place.
pub struct OutputGateArgs<'a> {
    pub data: &'a Tensor,
    pub source: OutputGateSource<'a>,
    pub numerics: OutputGateNumerics,
}

impl<'a> OutputGateArgs<'a> {
    pub fn new(data: &'a Tensor, gate: &'a Tensor) -> Self {
        Self {
            data,
            source: OutputGateSource::Tensor(gate),
            numerics: OutputGateNumerics::Flexible,
        }
    }

    pub fn from_projection(
        data: &'a Tensor,
        projection: &'a Tensor,
        heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            data,
            source: OutputGateSource::InterleavedProjection {
                projection,
                heads,
                head_dim,
            },
            numerics: OutputGateNumerics::RoundedSigmoid,
        }
    }

    pub fn with_numerics(mut self, numerics: OutputGateNumerics) -> Self {
        self.numerics = numerics;
        self
    }
}

/// Split q_proj's `[tokens, heads, 2*head_dim]` output into query and gate.
pub fn split_query_and_gate(
    ctx: &CudaContext,
    fused: &Tensor,
    query: &Tensor,
    gate: &Tensor,
) -> Result<()> {
    let dims = query.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(invalid("query must be [tokens, heads, head_dim]"));
    }
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    let fused_head_dim = head_dim
        .checked_mul(2)
        .ok_or_else(|| invalid("query/gate split head dimension overflow"))?;
    let elements = gate_elements(tokens, heads, head_dim)?;
    let source_row_width = heads
        .checked_mul(fused_head_dim)
        .ok_or_else(|| invalid("query/gate split source width overflow"))?;
    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &[tokens, heads, fused_head_dim])?;
    let query_buffer = tensor_storage(ctx, query, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    launch_gate(
        ctx,
        abi::GateSpec {
            version: abi::GATE_SPEC_VERSION,
            semantic: abi::GATE_SEMANTIC_SPLIT_QUERY_GATE,
            dtype: super::contracts::dtype_code(DType::BF16)?,
            source: abi::GATE_SOURCE_PROJECTION,
            numerics: abi::GATE_NUMERICS_FLEXIBLE,
            input_alignment: alignment_of(fused_buffer.ptr() as usize),
            output_alignment: alignment_of(query_buffer.ptr() as usize),
            auxiliary_alignment: alignment_of(gate_buffer.ptr() as usize),
            tokens: to_i64(tokens, "query/gate tokens")?,
            heads: to_i64(heads, "query/gate heads")?,
            head_dim: to_i64(head_dim, "query/gate head dimension")?,
            elements: to_i64(elements, "query/gate elements")?,
            source_row_width: to_i64(source_row_width, "query/gate source width")?,
        },
        abi::GateBindings {
            input: fused_buffer.ptr(),
            output: query_buffer.ptr(),
            auxiliary: gate_buffer.ptr(),
            stream: ctx.stream().handle(),
        },
    )
}

/// Apply a sigmoid output gate in place.
pub fn apply_output_gate(ctx: &CudaContext, args: OutputGateArgs<'_>) -> Result<()> {
    let dims = args.data.shape().dims().to_vec();
    if dims.is_empty() || dims.contains(&0) {
        return Err(invalid("output gate requires a non-empty tensor"));
    }
    let data_buffer = tensor_storage(ctx, args.data, DType::BF16, &dims)?;
    let count = dims.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| invalid("output gate element count overflow"))
    })?;

    let (source_buffer, source, numerics, tokens, heads, head_dim, source_row_width) = match args
        .source
    {
        OutputGateSource::Tensor(gate) => {
            if dims != gate.shape().dims() {
                return Err(invalid("output gate requires matching shapes"));
            }
            let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
            let numerics = match args.numerics {
                OutputGateNumerics::Flexible => abi::GATE_NUMERICS_FLEXIBLE,
                OutputGateNumerics::RoundedSigmoid => abi::GATE_NUMERICS_ROUNDED_SIGMOID,
            };
            (gate_buffer, abi::GATE_SOURCE_TENSOR, numerics, 0, 0, 0, 0)
        }
        OutputGateSource::InterleavedProjection {
            projection,
            heads,
            head_dim,
        } => {
            if args.numerics != OutputGateNumerics::RoundedSigmoid {
                return Err(invalid(
                    "projection output gate requires rounded-sigmoid numerics",
                ));
            }
            let data_shape = args.data.shape().dims();
            let projection_shape = projection.shape().dims();
            let data_width = heads
                .checked_mul(head_dim)
                .ok_or_else(|| invalid("projection output gate width overflow"))?;
            let minimum_projection_width = data_width
                .checked_mul(2)
                .ok_or_else(|| invalid("projection output gate source width overflow"))?;
            if data_shape.len() != 2
                || projection_shape.len() != 2
                || data_shape[0] != projection_shape[0]
                || heads == 0
                || head_dim == 0
                || data_shape[1] != data_width
                || projection_shape[1] < minimum_projection_width
            {
                return Err(invalid("projection output gate shape mismatch"));
            }
            let projection_buffer = tensor_storage(ctx, projection, DType::BF16, projection_shape)?;
            (
                projection_buffer,
                abi::GATE_SOURCE_PROJECTION,
                abi::GATE_NUMERICS_ROUNDED_SIGMOID,
                data_shape[0],
                heads,
                head_dim,
                projection_shape[1],
            )
        }
    };

    launch_gate(
        ctx,
        abi::GateSpec {
            version: abi::GATE_SPEC_VERSION,
            semantic: abi::GATE_SEMANTIC_APPLY_OUTPUT_GATE,
            dtype: super::contracts::dtype_code(DType::BF16)?,
            source,
            numerics,
            input_alignment: alignment_of(source_buffer.ptr() as usize),
            output_alignment: alignment_of(data_buffer.ptr() as usize),
            auxiliary_alignment: 256,
            tokens: to_i64(tokens, "output gate tokens")?,
            heads: to_i64(heads, "output gate heads")?,
            head_dim: to_i64(head_dim, "output gate head dimension")?,
            elements: to_i64(count, "output gate elements")?,
            source_row_width: to_i64(source_row_width, "output gate source width")?,
        },
        abi::GateBindings {
            input: source_buffer.ptr(),
            output: data_buffer.ptr(),
            auxiliary: std::ptr::null_mut(),
            stream: ctx.stream().handle(),
        },
    )
}

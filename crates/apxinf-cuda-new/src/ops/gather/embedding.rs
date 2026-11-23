//! Plain embedding-table gather.

use apxinf_core::{DType, Result, Tensor};

use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

use super::contracts::{EmbeddingScale, GatherArgs, GatherSemantic};

/// `output[t, :] = table[ids[t], :]`.
pub fn embedding_gather(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let table_dims = table.shape().dims().to_vec();
    let output_dims = output.shape().dims().to_vec();
    if table_dims.len() != 2 || output_dims.len() != 2 {
        return Err(invalid("embedding gather expects rank-2 tensors"));
    }
    let (vocab, hidden) = (table_dims[0], table_dims[1]);
    if output_dims[1] != hidden {
        return Err(invalid("embedding output width must match the table"));
    }
    let tokens = output_dims[0];
    if !matches!(table.dtype(), DType::F32 | DType::BF16) || output.dtype() != table.dtype() {
        return Err(invalid(
            "embedding gather currently requires matching F32 or BF16 table/output",
        ));
    }
    tensor_storage(ctx, table, table.dtype(), &table_dims)?;
    let id_buffer = tensor_storage(ctx, ids, DType::I32, &[tokens])?;
    tensor_storage(ctx, output, table.dtype(), &output_dims)?;
    let mut output = output.clone();
    let mut args = GatherArgs::new(GatherSemantic::EmbeddingLookup, table, &mut output);
    args.ids = Some(&id_buffer);
    args.vocab_size = vocab;
    args.embedding_scale = EmbeddingScale::None;
    super::gather::gather(ctx, args)
}

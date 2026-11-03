//! Safe cuda-new resource and transfer seam; model computation lives in blocks.
use apxinf_core::{Result, Tensor};
pub(in crate::cuda_fastwam) use apxinf_cuda_new::{
    ops, sampling, transfers, CudaBuffer as DeviceBuffer, CudaContext as Context, ExecutionSession,
    PreparedPhase,
};

pub(in crate::cuda_fastwam) fn upload(ctx: &Context, x: &Tensor) -> Result<Tensor> {
    transfers::to_cuda(x, ctx.device_id())
}

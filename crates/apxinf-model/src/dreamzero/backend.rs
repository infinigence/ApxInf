//! Safe cuda-new contracts used directly by the family mathematics.
use apxinf_core::{DType, Result, Shape, Tensor};
pub(super) use apxinf_cuda_new::{
    capture, ops, sampling, transfers, CapturedGraph, CudaBuffer, CudaContext as Context,
    ExecutionSession,
};

#[derive(Clone)]
pub(super) struct Policies {
    pub gemm: ops::GemmPolicy,
    pub attention: ops::AttentionPolicy,
}

impl Default for Policies {
    fn default() -> Self {
        Self {
            gemm: ops::GemmPolicy {
                deterministic: true,
                algorithm: ops::GemmAlgorithmPolicy::ColdBaseline,
                // Bound per-projection retained scratch across all DiT blocks.
                workspace_limit: 16 * 1024 * 1024,
                ..Default::default()
            },
            attention: ops::AttentionPolicy {
                deterministic: true,
                // UMT5's materialized rounded attention needs 40 MiB. Bound
                // each prepared execution so joint DiT attention chooses the
                // existing linear-memory FA2 fallback on a cold recipe store.
                workspace_limit: 64 * 1024 * 1024,
                ..Default::default()
            },
        }
    }
}

pub(super) fn allocate(ctx: &Context, dims: &[usize]) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(dims.to_vec()), DType::BF16)
}

pub(super) fn linear(
    ctx: &Context,
    policy: &Policies,
    x: &Tensor,
    w: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let mut args = ops::GemmArgs::new(x, w, out);
    args.policy = policy.gemm.clone();
    ops::gemm(ctx, args)
}

pub(super) fn binary(
    ctx: &Context,
    semantic: ops::PointwiseSemantic,
    x: &Tensor,
    y: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let mut args = ops::PointwiseArgs::new(semantic, x, out);
    args.secondary = Some(y);
    ops::pointwise(ctx, args)
}

pub(super) fn staged_rms(ctx: &Context, mut args: ops::NormArgs<'_>) -> Result<()> {
    args.numerics = ops::NormNumerics::Staged;
    ops::norm(ctx, args)
}
pub(super) fn staged_l2(ctx: &Context, input: &Tensor, output: &mut Tensor, eps: f32) -> Result<()> {
    ops::norm(ctx, ops::NormArgs {
        input, output: Some(output), kind: ops::NormKind::L2,
        affine: ops::Affine::None, quantized: None, numerics: ops::NormNumerics::Staged, eps,
    })
}

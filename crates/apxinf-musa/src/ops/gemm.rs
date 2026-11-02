//! GEMM contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::require_unit_output_scale;
use crate::kernels::activation;
use crate::kernels::gemm as gemm_kernels;
use crate::kernels::not_ported;
use crate::MusaContext;

/// Identifies one immutable version of a weight tensor.
///
/// Upstream uses it as part of a prepared-weight cache key. This backend keys
/// its prepared assembly weights on the tensor address instead, so the token is
/// accepted and carried rather than consulted; a candidate that starts caching
/// transformed weights has to grow the key to include it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightVersion(u64);

impl WeightVersion {
    pub const fn new(version: u64) -> Self {
        Self(version)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// What a model wants from GEMM, declared at load.
///
/// **Carried, not consumed.** This backend decides its tactics from the session
/// `configure_tuning` builds at load (`musa_pi05/load.rs`), not from a per-call
/// value, so a field set here is accepted and ignored — the same arrangement
/// [`WeightVersion`] documents. The struct exists because upstream's typed
/// contracts carry one and the model tree fills it in.
#[derive(Clone, Debug)]
pub struct GemmPolicy {
    pub accumulation_dtype: DType,
    pub workspace_limit: usize,
    /// Measure candidates and persist the winner, instead of only reading
    /// records that already exist.
    pub online_tune: bool,
    pub allow_fallback: bool,
    pub graph_safe: bool,
    pub deterministic: bool,
    /// Where the tactic database lives; `None` uses the directory derived from
    /// the device, and `Some("")` disables disk caching.
    pub cache_dir: Option<String>,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        Self {
            accumulation_dtype: DType::F32,
            workspace_limit: 256 * 1024 * 1024,
            online_tune: true,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
            cache_dir: None,
        }
    }
}

/// Quantization contract for the operands supplied to one GEMM call.
///
/// Only [`Self::None`] is served: this backend has no FP8 path, and the W8A8
/// route is reached through [`super::w8a8`] instead, whose scale vectors are
/// FP16 with the tile ABI's pad. The other variants exist so the contract
/// matches upstream's and a caller that asks for one gets a named refusal
/// instead of a wrong answer.
#[derive(Clone, Copy)]
pub enum GemmQuantization<'a> {
    None,
    Fp8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    W8A8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    Fp8UnitScale,
}

pub struct GemmArgs<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    /// Canonical row-major output `[M, N]`. GEMM mutates this tensor.
    pub out: &'a mut Tensor,
    pub quantization: GemmQuantization<'a>,
    pub alpha: f32,
    pub output_scale: f32,
    pub policy: GemmPolicy,
    /// Explicit opt-in for candidate-specific prepared-weight caching.
    pub weight_version: Option<WeightVersion>,
}

impl<'a> GemmArgs<'a> {
    /// Create a GEMM with canonical contiguous row-major inputs:
    /// `a=[M,K]`, `b=[K,N]`, and `out=[M,N]`.
    pub fn new(a: &'a Tensor, b: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            a,
            b,
            out,
            quantization: GemmQuantization::None,
            alpha: 1.0,
            output_scale: 1.0,
            policy: GemmPolicy::default(),
            weight_version: None,
        }
    }

    /// Declare the weight immutable for the lifetime of this graph.
    pub fn with_immutable_weight(mut self, version: WeightVersion) -> Self {
        self.weight_version = Some(version);
        self
    }
}

pub struct GemmGegluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

pub fn gemm(ctx: &MusaContext, args: GemmArgs<'_>) -> Result<()> {
    require_plain(&args, "gemm")?;
    gemm_kernels::gemm_bf16_into(ctx, args.a, args.b, args.out)
}

/// `[M, K] @ [K, 2N]` followed by gated GELU, writing `[M, N]`.
pub fn gemm_geglu(ctx: &MusaContext, args: GemmGegluArgs<'_>) -> Result<()> {
    require_plain(&args.gemm, "gemm_geglu")?;
    let [rows, _inner] = args.gemm.a.shape().dims() else {
        return Err(Error::Other(
            "MUSA gemm_geglu activation must be a 2-D tensor".into(),
        ));
    };
    let [_, gate_cols] = args.gemm.b.shape().dims() else {
        return Err(Error::Other(
            "MUSA gemm_geglu weight must be a 2-D tensor".into(),
        ));
    };
    let mut gate = ctx.allocate_output(
        apxinf_core::Shape::new(vec![*rows, *gate_cols]),
        DType::BF16,
    )?;
    gemm_kernels::gemm_bf16_into(ctx, args.gemm.a, args.gemm.b, &mut gate)?;
    activation::geglu_bf16_into(ctx, &gate, args.gemm.out)
}

fn require_plain(args: &GemmArgs<'_>, operation: &str) -> Result<()> {
    match args.quantization {
        GemmQuantization::None => {}
        GemmQuantization::W8A8 { .. } => return Err(not_ported("gemm with W8A8 operands")),
        _ => return Err(not_ported("gemm with quantized operands")),
    }
    if args.alpha != 1.0 {
        return Err(Error::Other(format!(
            "MUSA {operation} applies no alpha scaling; got alpha {}",
            args.alpha
        )));
    }
    require_unit_output_scale(operation, args.output_scale)
}

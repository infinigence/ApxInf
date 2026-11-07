//! GEMM contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::{
    require_deterministic, require_f32_accumulation, require_unit_output_scale, require_workspace,
};
use crate::kernels::activation;
use crate::kernels::gemm as gemm_kernels;
use crate::kernels::{not_ported, TacticDispatch};
use crate::MusaContext;

/// What a model wants from GEMM, declared at load.
///
/// Every field is either enforced here or refused by name. `graph_safe` is the
/// exception in form rather than in effect: MUSA's routes are capture-safe
/// unconditionally, so `true` (the requirement) and `false` (a weaker request)
/// are both satisfied.
#[derive(Clone, Debug)]
pub struct GemmPolicy {
    /// MUSA GEMM accumulates in F32 — the adapter's `MUBLAS_COMPUTE_32F`
    /// contract — so any other value is refused.
    pub accumulation_dtype: DType,
    /// Bounds the bytes the operator allocates for itself: none for a plain
    /// GEMM, the gate buffer for [`gemm_geglu`]. Vendor-library internal
    /// workspaces are outside this backend's accounting.
    pub workspace_limit: usize,
    /// `false` requires an exact tactic record for this shape. A bucket
    /// extrapolation and the muBLAS default are refused; autotuning may measure
    /// the shape and write the record, which satisfies the constraint.
    pub allow_fallback: bool,
    /// Both values are accepted; see the type-level note.
    pub graph_safe: bool,
    /// Bitwise determinism is not guaranteed across the available providers, so
    /// `true` is refused rather than accepted without the guarantee.
    pub deterministic: bool,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        Self {
            accumulation_dtype: DType::F32,
            workspace_limit: 256 * 1024 * 1024,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
        }
    }
}

/// The constraints this backend carries into one GEMM call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmittedGemmPolicy {
    tactic: TacticDispatch,
}

impl GemmPolicy {
    /// Validate every field and keep only what execution must honour.
    ///
    /// `own_bytes` is the workspace this operator allocates for itself; checking
    /// it before the allocation means a refused call allocates none.
    fn admit(&self, operation: &str, own_bytes: usize) -> Result<AdmittedGemmPolicy> {
        require_f32_accumulation(operation, self.accumulation_dtype)?;
        require_deterministic(operation, self.deterministic)?;
        require_workspace(operation, own_bytes, self.workspace_limit)?;
        Ok(AdmittedGemmPolicy {
            tactic: if self.allow_fallback {
                TacticDispatch::AllowFallback
            } else {
                TacticDispatch::RequireExact
            },
        })
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
        }
    }
}

pub struct GemmGegluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

pub fn gemm(ctx: &MusaContext, args: GemmArgs<'_>) -> Result<()> {
    require_plain(&args, "gemm")?;
    // A plain GEMM writes the caller's output and allocates nothing of its own.
    let admitted = args.policy.admit("gemm", 0)?;
    gemm_kernels::gemm_bf16_into_with(ctx, args.a, args.b, args.out, admitted.tactic)
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
    // The gate is the one buffer this operator allocates for itself, so it is
    // what `workspace_limit` bounds. Computed before the allocation: a limit
    // that cannot cover it refuses the call with nothing allocated.
    let gate_bytes = rows
        .checked_mul(*gate_cols)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("MUSA gemm_geglu gate size overflow".into()))?;
    let admitted = args.gemm.policy.admit("gemm_geglu", gate_bytes)?;
    let mut gate = ctx.allocate_output(
        apxinf_core::Shape::new(vec![*rows, *gate_cols]),
        DType::BF16,
    )?;
    gemm_kernels::gemm_bf16_into_with(ctx, args.gemm.a, args.gemm.b, &mut gate, admitted.tactic)?;
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

#[cfg(test)]
mod policy_tests {
    use super::*;

    fn policy() -> GemmPolicy {
        GemmPolicy::default()
    }

    #[test]
    fn the_default_policy_is_admitted() {
        let admitted = policy().admit("gemm", 0).expect("the default policy");
        assert_eq!(admitted.tactic, TacticDispatch::AllowFallback);
    }

    #[test]
    fn a_non_f32_accumulator_is_refused() {
        let error = GemmPolicy {
            accumulation_dtype: DType::F16,
            ..policy()
        }
        .admit("gemm", 0)
        .expect_err("F32 is the only accumulation contract");
        assert!(
            error.to_string().contains("accumulation_dtype f16"),
            "{error}"
        );
        assert!(error.to_string().contains("F32"), "{error}");
    }

    #[test]
    fn deterministic_is_refused() {
        let error = GemmPolicy {
            deterministic: true,
            ..policy()
        }
        .admit("gemm", 0)
        .expect_err("there is no bitwise-determinism contract to honour");
        assert!(error.to_string().contains("deterministic"), "{error}");
    }

    #[test]
    fn a_workspace_limit_below_the_gate_is_refused() {
        let error = GemmPolicy {
            workspace_limit: 1023,
            ..policy()
        }
        .admit("gemm_geglu", 4096)
        .expect_err("the gate does not fit");
        assert!(error.to_string().contains("4096"), "{error}");
        assert!(error.to_string().contains("1023"), "{error}");
        assert!(GemmPolicy {
            workspace_limit: 4096,
            ..policy()
        }
        .admit("gemm_geglu", 4096)
        .is_ok());
    }

    #[test]
    fn a_plain_gemm_needs_no_workspace() {
        assert!(GemmPolicy {
            workspace_limit: 0,
            ..policy()
        }
        .admit("gemm", 0)
        .is_ok());
    }

    #[test]
    fn allow_fallback_false_requires_an_exact_tactic() {
        let admitted = GemmPolicy {
            allow_fallback: false,
            ..policy()
        }
        .admit("gemm", 0)
        .expect("a valid policy");
        assert_eq!(admitted.tactic, TacticDispatch::RequireExact);
    }

    #[test]
    fn graph_safe_is_accepted_either_way() {
        assert!(GemmPolicy {
            graph_safe: true,
            ..policy()
        }
        .admit("gemm", 0)
        .is_ok());
        assert!(GemmPolicy {
            graph_safe: false,
            ..policy()
        }
        .admit("gemm", 0)
        .is_ok());
    }
}

//! Statically composed GEMM backend providers.

mod cublaslt;
mod cutlass;

use apxinf_core::{Error, Result};

use crate::tuning::{Epilogue, GemmOp, GemmTuningKey, TacticBackend, TacticCandidate, TacticId};

pub(super) fn prepare(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    match tactic.backend {
        TacticBackend::Cutlass
        | TacticBackend::CutlassFp8DualGeGlu
        | TacticBackend::CutlassBf16DualGeGluM522
        | TacticBackend::CutlassBf16DualGeGluM533 => cutlass::prepare(key, tactic),
        TacticBackend::CublasLt
        | TacticBackend::CublasLtCustom
        | TacticBackend::CublasLtCustomBias
        | TacticBackend::CublasLtCustomSplitSerial
        | TacticBackend::CublasLtCustomSplitGeGluCutlass
        | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
        | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
        | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
        | TacticBackend::CublasLtCustomSplitGeGluCutlassBf16 => cublaslt::prepare(key, tactic),
        TacticBackend::Vendor if tactic.value == 0 => match key.op {
            // Native FP8 uses cuBLASLt even for the safe default. Explicitly
            // restore rank zero after candidate probing mutates its plan map.
            GemmOp::Fp8F16 => cublaslt::prepare(
                key,
                TacticId {
                    backend: TacticBackend::CublasLt,
                    value: 0,
                },
            ),
            GemmOp::Bf16 | GemmOp::W8A8 => Ok(()),
        },
        TacticBackend::Vendor => Err(Error::Other(format!(
            "invalid vendor tactic {}",
            tactic.value
        ))),
    }
}

pub(super) fn candidates(
    key: &GemmTuningKey,
    max_cublaslt_algorithms: usize,
) -> Vec<TacticCandidate> {
    let mut tactics = Vec::new();
    // The safe implementation participates in selection and is also the
    // correctness reference. Keep it first so probing cannot contaminate it.
    tactics.push(TacticCandidate {
        tactic: TacticId {
            backend: TacticBackend::Vendor,
            value: 0,
        },
    });
    if matches!(key.op, GemmOp::Bf16 | GemmOp::Fp8F16) {
        tactics.extend(
            cublaslt::candidates(max_cublaslt_algorithms)
                .into_iter()
                .map(|tactic| TacticCandidate { tactic }),
        );
    }
    if key.epilogue == Epilogue::None {
        tactics.extend(
            cutlass::candidates(key)
                .into_iter()
                .map(|tactic| TacticCandidate { tactic }),
        );
    }
    tactics
}

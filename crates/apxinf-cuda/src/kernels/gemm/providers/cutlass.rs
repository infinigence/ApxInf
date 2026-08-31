use apxinf_core::{Error, Result};

use crate::tuning::{Epilogue, GemmOp, GemmTuningKey, TacticBackend, TacticId};

pub(super) fn prepare(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    if key.epilogue != Epilogue::None {
        return Err(Error::Other(format!(
            "CUTLASS provider does not implement {:?} for {key:?}",
            key.epilogue
        )));
    }
    let valid = match tactic.backend {
        TacticBackend::Cutlass => match key.op {
            GemmOp::Fp8F16 => fp8_supported(key, tactic.value),
            GemmOp::W8A8 => w8a8_supported(key, tactic.value),
            GemmOp::Bf16 => false,
        },
        TacticBackend::CutlassFp8DualGeGlu => {
            key.op == GemmOp::Fp8F16
                && matches!(key.m, 522 | 533)
                && (key.n, key.k, tactic.value) == (32768, 2048, 0)
        }
        TacticBackend::CutlassBf16DualGeGluM522 => {
            key.op == GemmOp::Bf16 && (key.m, key.n, key.k, tactic.value) == (522, 32768, 2048, 0)
        }
        TacticBackend::CutlassBf16DualGeGluM533 => {
            key.op == GemmOp::Bf16 && (key.m, key.n, key.k, tactic.value) == (533, 32768, 2048, 0)
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "CUTLASS provider rejected {tactic:?} for {key:?}"
        )))
    }
}

pub(super) fn candidates(key: &GemmTuningKey) -> Vec<TacticId> {
    let mut candidates = Vec::new();
    if key.op == GemmOp::Fp8F16 {
        for value in 0..=7 {
            if fp8_supported(key, value) {
                candidates.push(TacticId {
                    backend: TacticBackend::Cutlass,
                    value,
                });
            }
        }
    }
    if w8a8_supported(key, 0) {
        candidates.push(TacticId {
            backend: TacticBackend::Cutlass,
            value: 0,
        });
    }
    candidates
}

fn fp8_supported(key: &GemmTuningKey, tactic: i32) -> bool {
    #[cfg(apxinf_cutlass_gemm)]
    {
        key.op == GemmOp::Fp8F16
            && (0..=7).contains(&tactic)
            && key.n >= 1024
            && key.n % 16 == 0
            && key.k % 16 == 0
    }
    #[cfg(not(apxinf_cutlass_gemm))]
    {
        let _ = (key, tactic);
        false
    }
}

fn w8a8_supported(key: &GemmTuningKey, tactic: i32) -> bool {
    #[cfg(apxinf_cutlass_int8_sm80)]
    {
        key.op == GemmOp::W8A8 && tactic == 0 && key.k % 16 == 0 && key.n % 8 == 0
    }
    #[cfg(not(apxinf_cutlass_int8_sm80))]
    {
        let _ = (key, tactic);
        false
    }
}

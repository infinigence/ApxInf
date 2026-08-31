use apxinf_core::{Error, Result};

use crate::tuning::{Epilogue, GemmOp, GemmTuningKey, TacticBackend, TacticId};

pub(super) fn prepare(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    use super::super::{bf16, fp8};

    match (key.op, tactic.backend) {
        (GemmOp::Bf16, TacticBackend::CublasLt) => {
            bf16::set_cublaslt_gemm_heuristic(key.m, key.n, key.k, tactic.value)?;
            bf16::prepare_cublaslt_gemm(key.m, key.n, key.k, false)
        }
        (GemmOp::Fp8F16, TacticBackend::CublasLt) => {
            if key.epilogue == Epilogue::None {
                fp8::set_cublaslt_gemm_heuristic(key.m, key.n, key.k, tactic.value)?;
                fp8::prepare_cublaslt_fp8_gemm(key.m, key.n, key.k)
            } else {
                fp8::set_cublaslt_fused_gemm_heuristic(
                    key.m,
                    key.n,
                    key.k,
                    key.epilogue,
                    tactic.value,
                )
            }
        }
        (GemmOp::Bf16, TacticBackend::CublasLtCustom) => {
            bf16::set_cublaslt_gemm_custom(key.m, key.n, key.k, tactic.value)?;
            bf16::prepare_cublaslt_gemm(key.m, key.n, key.k, false)
        }
        (GemmOp::Fp8F16, TacticBackend::CublasLtCustom) if key.epilogue == Epilogue::None => {
            fp8::set_cublaslt_gemm_custom(key.m, key.n, key.k, tactic.value)?;
            fp8::prepare_cublaslt_fp8_gemm(key.m, key.n, key.k)
        }
        (GemmOp::Fp8F16, TacticBackend::CublasLtCustomBias)
            if matches!(key.epilogue, Epilogue::Bias | Epilogue::BiasResidual) =>
        {
            fp8::set_cublaslt_gemm_bias_custom(key.m, key.n, key.k, key.epilogue, tactic.value)
        }
        (GemmOp::Bf16, TacticBackend::CublasLtCustomSplitSerial)
            if key.epilogue == Epilogue::None =>
        {
            bf16::set_cublaslt_gemm_split_custom(key.m, key.n, key.k, tactic.value)?;
            bf16::prepare_cublaslt_gemm(key.m, key.n, key.k, true)
        }
        (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitSerial)
        | (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlass)
        | (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto)
        | (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3)
        | (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm)
            if key.epilogue == Epilogue::None =>
        {
            fp8::set_cublaslt_gemm_split_custom(key.m, key.n, key.k, tactic.value)?;
            fp8::prepare_cublaslt_fp8_gemm_split(key.m, key.n, key.k)
        }
        (GemmOp::Bf16, TacticBackend::CublasLtCustomSplitGeGluCutlassBf16)
            if key.epilogue == Epilogue::None =>
        {
            bf16::set_cublaslt_gemm_split_custom(key.m, key.n, key.k, tactic.value)?;
            bf16::prepare_cublaslt_gemm(key.m, key.n, key.k, true)
        }
        _ => Err(Error::Other(format!(
            "cuBLASLt provider rejected {tactic:?} for {key:?}"
        ))),
    }
}

/// cuBLASLt discovers the actually supported algorithms through
/// `cublasLtMatmulAlgoGetHeuristic`; these IDs only bound the requested result
/// count passed to the native provider.
pub(super) fn candidates(max_algorithms: usize) -> Vec<TacticId> {
    (0..max_algorithms.min(64))
        .map(|value| TacticId {
            backend: TacticBackend::CublasLt,
            value: value as i32,
        })
        .collect()
}

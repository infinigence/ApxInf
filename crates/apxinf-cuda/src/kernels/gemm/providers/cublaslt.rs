use apxinf_core::{Error, Result};

use crate::tuning::{Epilogue, GemmOp, GemmTuningKey, TacticBackend, TacticId};

pub(super) fn prepare(key: &GemmTuningKey, tactic: TacticId) -> Result<()> {
    use super::super::{bf16, fp8};

    match (key.op, tactic.backend) {
        (GemmOp::Bf16, TacticBackend::CublasLt) if key.epilogue == Epilogue::None => {
            bf16::set_cublaslt_gemm_heuristic(key.m, key.n, key.k, tactic.value)?;
            bf16::prepare_cublaslt_gemm(key.m, key.n, key.k, false)
        }
        (GemmOp::Fp8F16, TacticBackend::CublasLt) => match key.epilogue {
            Epilogue::None => {
                fp8::set_cublaslt_gemm_heuristic(key.m, key.n, key.k, tactic.value)?;
                fp8::prepare_cublaslt_fp8_gemm(key.m, key.n, key.k)
            }
            Epilogue::Bias | Epilogue::BiasGelu | Epilogue::BiasResidual => {
                fp8::set_cublaslt_fused_gemm_heuristic(
                    key.m,
                    key.n,
                    key.k,
                    key.epilogue,
                    tactic.value,
                )
            }
            Epilogue::GeGlu => Err(Error::Other(format!(
                "cuBLASLt provider rejected {tactic:?} for {key:?}"
            ))),
        },
        (GemmOp::Fp8Bf16, TacticBackend::CublasLt) if key.epilogue == Epilogue::None => {
            fp8::set_cublaslt_fp8_bf16_gemm_heuristic(key.m, key.n, key.k, tactic.value)?;
            fp8::prepare_cublaslt_fp8_gemm_bf16(key.m, key.n, key.k)
        }
        (GemmOp::Bf16, TacticBackend::CublasLtCustom) if key.epilogue == Epilogue::None => {
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
            if key.epilogue == Epilogue::None =>
        {
            fp8::set_cublaslt_gemm_split_custom(key.m, key.n, key.k, tactic.value)?;
            fp8::prepare_cublaslt_fp8_gemm_split(key.m, key.n, key.k)
        }
        (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlass)
        | (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto)
        | (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3)
        | (GemmOp::Fp8F16, TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm)
            if key.epilogue == Epilogue::GeGlu =>
        {
            fp8::set_cublaslt_gemm_split_custom(key.m, key.n, key.k, tactic.value)?;
            fp8::prepare_cublaslt_fp8_gemm_split(key.m, key.n, key.k)
        }
        (GemmOp::Bf16, TacticBackend::CublasLtCustomSplitGeGluCutlassBf16)
            if key.epilogue == Epilogue::GeGlu =>
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
pub(super) fn candidates(key: &GemmTuningKey, max_algorithms: usize) -> Vec<TacticId> {
    let mut candidates = (0..max_algorithms.min(64))
        .map(|value| TacticId {
            backend: TacticBackend::CublasLt,
            value: value as i32,
        })
        .collect::<Vec<_>>();
    if key.device.sm != 110 {
        return candidates;
    }
    let custom_backend = match (key.op, key.epilogue) {
        (GemmOp::Bf16 | GemmOp::Fp8F16, Epilogue::None) => Some(TacticBackend::CublasLtCustom),
        (GemmOp::Fp8F16, Epilogue::Bias | Epilogue::BiasResidual) => {
            Some(TacticBackend::CublasLtCustomBias)
        }
        _ => None,
    };
    if let Some(backend) = custom_backend {
        candidates.extend(custom_config_ids().map(|value| TacticId { backend, value }));
    }
    if key.epilogue == Epilogue::None && matches!(key.op, GemmOp::Bf16 | GemmOp::Fp8F16) {
        candidates.extend(custom_config_ids().map(|value| TacticId {
            backend: TacticBackend::CublasLtCustomSplitSerial,
            value,
        }));
    }
    candidates
}

pub(super) fn geglu_candidates(key: &GemmTuningKey) -> Vec<TacticId> {
    if key.device.sm != 110 || key.epilogue != Epilogue::GeGlu {
        return Vec::new();
    }
    let backends: &[TacticBackend] = match (key.op, key.m, key.n, key.k) {
        (GemmOp::Fp8F16, 778, 32768, 2048) => &[
            TacticBackend::CublasLtCustomSplitGeGluCutlass,
            TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto,
            TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3,
        ],
        (GemmOp::Fp8F16, 522, 32768, 2048) => {
            &[TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm]
        }
        (GemmOp::Bf16, 789, 32768, 2048) => &[TacticBackend::CublasLtCustomSplitGeGluCutlassBf16],
        _ => &[],
    };
    backends
        .iter()
        .flat_map(|&backend| custom_config_ids().map(move |value| TacticId { backend, value }))
        .collect()
}

/// Provider-owned CUDA 13 / SM110 algorithm configurations. These are
/// physical cuBLASLt configurations, not model shapes; every exact workload
/// validates them with cublasLtMatmulAlgoCheck before benchmarking.
fn custom_config_ids() -> impl Iterator<Item = i32> {
    [
        18_377_904, 18_902_425, 18_923_959, 18_924_573, 18_924_956, 18_926_998, 18_927_004,
    ]
    .into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::{DeviceFingerprint, GemmLayout, ScaleMode, TuningDType};

    fn key(epilogue: Epilogue) -> GemmTuningKey {
        GemmTuningKey {
            op: GemmOp::Fp8F16,
            device: DeviceFingerprint {
                sm: 110,
                multiprocessor_count: 20,
            },
            m: 522,
            n: 32768,
            k: 2048,
            activation_dtype: TuningDType::F8E4M3,
            weight_dtype: TuningDType::F8E4M3,
            output_dtype: TuningDType::F8E4M3,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::PerTensor,
            epilogue,
            workspace_limit: usize::MAX,
        }
    }

    #[test]
    fn plain_fp8_candidates_include_custom_and_split_families() {
        let candidates = candidates(&key(Epilogue::None), 2);
        assert!(candidates
            .iter()
            .any(|candidate| candidate.backend == TacticBackend::CublasLtCustom));
        assert!(candidates
            .iter()
            .any(|candidate| { candidate.backend == TacticBackend::CublasLtCustomSplitSerial }));
    }

    #[test]
    fn geglu_candidates_are_shape_specific() {
        let candidates = geglu_candidates(&key(Epilogue::GeGlu));
        assert!(candidates.iter().any(|candidate| {
            candidate.backend == TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
        }));
    }

    #[test]
    fn plain_cublaslt_backends_reject_geglu_keys_before_native_prepare() {
        for backend in [
            TacticBackend::CublasLt,
            TacticBackend::CublasLtCustom,
            TacticBackend::CublasLtCustomSplitSerial,
        ] {
            let value = if backend == TacticBackend::CublasLt {
                0
            } else {
                18_377_904
            };
            assert!(prepare(&key(Epilogue::GeGlu), TacticId { backend, value }).is_err());
        }
    }
}

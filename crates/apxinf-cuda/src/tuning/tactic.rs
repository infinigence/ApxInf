//! Backend-owned tactic identities and autotune candidates.

/// Physical implementation family selected for one GEMM problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TacticBackend {
    Cutlass,
    CublasLt,
    /// Fully specified cuBLASLt algorithm configuration. Unlike a heuristic
    /// rank, this remains stable when the library reorders its candidates.
    CublasLtCustom,
    CublasLtCustomBias,
    CublasLtCustomSplitSerial,
    CublasLtCustomSplitGeGluCutlass,
    CublasLtCustomSplitGeGluCutlass2SmAuto,
    CublasLtCustomSplitGeGluCutlass2SmStage3,
    CublasLtCustomSplitGeGluCutlassM522Explicit2Sm,
    CutlassFp8DualGeGlu,
    CutlassBf16DualGeGluM522,
    CutlassBf16DualGeGluM533,
    CublasLtCustomSplitGeGluCutlassBf16,
    Vendor,
}

/// Provider-specific tactic identity. `value` is interpreted only by the
/// selected backend provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TacticId {
    pub backend: TacticBackend,
    pub value: i32,
}

/// A runnable candidate reported by a backend provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TacticCandidate {
    pub tactic: TacticId,
}

/// Decoded representation of a compact `cublaslt_custom` tactic id.
///
/// Algorithm id 66, split-K=1, reduction=none, swizzle=0, and inner-shape=0
/// are part of the backend contract. The remaining CUDA 13 configuration is
/// packed into the signed JSON-compatible tactic value as follows:
/// tile[9:0], custom[12:10], cluster[18:13], stages[24:19].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CublasLtCustomConfig {
    pub tile_id: i32,
    pub custom_option: i32,
    pub cluster_shape_id: i32,
    pub stages_id: i32,
}

pub fn decode_cublaslt_custom_tactic(value: i32) -> Option<CublasLtCustomConfig> {
    if value <= 0 || value & !0x01ff_ffff != 0 {
        return None;
    }
    let config = CublasLtCustomConfig {
        tile_id: value & 0x3ff,
        custom_option: (value >> 10) & 0x7,
        cluster_shape_id: (value >> 13) & 0x3f,
        stages_id: (value >> 19) & 0x3f,
    };
    (config.tile_id > 0 && config.stages_id > 0).then_some(config)
}

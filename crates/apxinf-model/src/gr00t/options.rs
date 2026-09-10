use std::path::PathBuf;

use crate::ModelPrecision;

use super::Gr00tConfig;

/// GR00T-specific loading policy.
///
/// This intentionally does not extend [`crate::LoadOptions`], whose explicit
/// PI0.5 configuration field is part of the existing public API.
#[derive(Clone, Debug)]
pub struct Gr00tLoadOptions {
    /// Override the checkpoint `config.json` after validating the replacement.
    pub config: Option<Gr00tConfig>,
    /// N1.7 supports BF16, calibrated FP8 on native hardware, and the existing
    /// SM80-family W8A8 backend used by Pi0.5.
    pub precision: ModelPrecision,
    /// Local Cosmos-Reason2-2B directory (or its `config.json`). GR00T's
    /// checkpoint contains the backbone weights but not the complete Qwen3-VL
    /// architecture configuration.
    pub backbone_path: Option<PathBuf>,
    /// Validated calibration JSON used by the FP8 runtime.
    ///
    /// Keep this per-load instead of using process-global environment state so
    /// multiple policies can be constructed safely in one Python process.
    pub fp8_calibration_path: Option<PathBuf>,
    /// Optional hardware-specific GEMM tactic database. The runtime installs
    /// this database into the same CUDA context that owns model execution.
    pub tuning_path: Option<PathBuf>,
}

impl Default for Gr00tLoadOptions {
    fn default() -> Self {
        Self {
            config: None,
            precision: ModelPrecision::Auto,
            backbone_path: None,
            fp8_calibration_path: None,
            tuning_path: None,
        }
    }
}

mod db;
mod engine;
mod key;
mod report;
mod session;
mod store;
mod tactic;

pub use db::{TuningDb, TuningDbHeader, TUNING_SCHEMA_V1};
pub(crate) use engine::outputs_are_close;
pub use engine::{AutoTuneConfig, AutoTuneEngine, CandidateMeasurement, TuningOutcome};
pub use key::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TuningDType,
};
pub use report::outcome_json;
pub use session::{ResolvedTactic, TacticMatch, TuningMode, TuningPaths, TuningSession};
pub use store::{GemmTuningRecord, TacticStore};
pub use tactic::{
    decode_cublaslt_custom_tactic, CublasLtCustomConfig, TacticBackend, TacticCandidate, TacticId,
};

/// Diagnostic identity of the CUDA kernel build inputs and target arch.
/// It is recorded in reports but is intentionally not part of the hardware
/// database path; unrelated source changes must not invalidate all tactics.
pub const KERNEL_BUILD_ID: &str = env!("APXINF_KERNEL_BUILD_ID");

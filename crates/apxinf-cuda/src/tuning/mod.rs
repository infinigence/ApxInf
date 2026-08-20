mod db;
mod key;
mod store;

use std::sync::OnceLock;

use apxinf_core::{Error, Result};

pub use db::{TuningDb, TuningDbHeader, TUNING_SCHEMA_V1};
pub use key::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TuningDType,
};
pub use store::{GemmTuningRecord, TacticBackend, TacticId, TacticStore};

/// Deterministic identity of the CUDA kernel build inputs and target arch.
/// Persisted tuning data must carry this exact value.
pub const KERNEL_BUILD_ID: &str = env!("APXINF_KERNEL_BUILD_ID");

static TACTIC_STORE: OnceLock<TacticStore> = OnceLock::new();

pub fn install(store: TacticStore) -> Result<()> {
    match TACTIC_STORE.set(store) {
        Ok(()) => Ok(()),
        Err(store) if TACTIC_STORE.get() == Some(&store) => Ok(()),
        Err(_) => Err(Error::Other(
            "a different CUDA tactic store is already installed".into(),
        )),
    }
}

pub fn installed() -> Option<&'static TacticStore> {
    TACTIC_STORE.get()
}

pub fn lookup_gemm(key: &GemmTuningKey) -> Option<TacticId> {
    installed()?.lookup_gemm(key)
}

pub fn lookup_gemm_exact(key: &GemmTuningKey) -> Option<TacticId> {
    installed()?.lookup_gemm_exact(key)
}

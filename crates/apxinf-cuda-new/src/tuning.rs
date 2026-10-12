//! Minimal tuning-session compatibility surface for the qwen_drive runner.
//!
//! The legacy runtime kept a mutable tactics store on the context and the
//! runner keyed its prepared plans on that session's identity and generation.
//! cuda-new tunes through per-key recipes inside each operator, so there is
//! nothing to store here — but the runner's plan-invalidation logic still
//! wants a session object with an identity and a generation counter. This
//! module provides exactly that and nothing more.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TuningMode {
    /// Reuse persisted recipes; cuda-new operators may still tune online
    /// where their policy allows it.
    #[default]
    Inference,
    /// Benchmark every legal candidate. Unused by cuda-new's direct-launch
    /// shim surface; recipe autotuning is per-operator.
    AutoTune,
}

/// Store placeholder. Legacy held GEMM tactics; cuda-new recipes live with
/// the native operators, so this carries no data.
#[derive(Clone, Copy, Debug, Default)]
pub struct TacticStore;

thread_local! {
    static AUTOTUNE_SUPPRESSED: Cell<bool> = const { Cell::new(false) };
}

/// Run `operation` with autotuning suppressed. The cuda-new operators consult
/// their own policy; this flag exists so prepared-plan capture keeps the
/// legacy calling shape.
pub fn without_autotune<T>(operation: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            AUTOTUNE_SUPPRESSED.with(|state| state.set(self.0));
        }
    }
    let _restore = AUTOTUNE_SUPPRESSED.with(|state| {
        let previous = state.get();
        state.set(true);
        Restore(previous)
    });
    operation()
}

/// Whether a `without_autotune` scope is active on this thread.
pub fn autotune_suppressed() -> bool {
    AUTOTUNE_SUPPRESSED.with(Cell::get)
}

/// Session identity + generation, for prepared-plan invalidation.
#[derive(Debug)]
pub struct TuningSession {
    mode: TuningMode,
    generation: AtomicU64,
}

impl TuningSession {
    pub fn new(mode: TuningMode) -> Self {
        Self {
            mode,
            generation: AtomicU64::new(0),
        }
    }

    pub fn inference(_store: TacticStore) -> Self {
        Self::new(TuningMode::Inference)
    }

    pub fn mode(&self) -> TuningMode {
        self.mode
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    pub fn bump_generation(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }
}

impl Default for TuningSession {
    fn default() -> Self {
        Self::new(TuningMode::Inference)
    }
}

/// Default session shared by contexts that never install one explicitly.
pub(crate) fn default_session() -> Arc<TuningSession> {
    Arc::new(TuningSession::default())
}

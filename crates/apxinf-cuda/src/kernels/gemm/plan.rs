//! One-time tactic resolution and process-local prepared GEMM plans.

use std::collections::HashMap;
use std::sync::Mutex;

use apxinf_core::{Error, Result};

use crate::context::CudaContext;
use crate::tuning::{
    GemmTuningKey, TacticBackend, TacticId, TacticMatch, TuningMode, TuningOutcome,
};

use super::providers;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanSource {
    Exact,
    Bucket,
    Default,
}

/// A tactic resolved and validated for one physical GEMM key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedGemmPlan {
    pub key: GemmTuningKey,
    pub tactic: TacticId,
    pub source: PlanSource,
    generation: u64,
}

#[derive(Debug, Default)]
pub struct GemmPlanCache {
    plans: Mutex<HashMap<GemmTuningKey, PreparedGemmPlan>>,
}

impl GemmPlanCache {
    pub fn resolve(
        &self,
        ctx: &CudaContext,
        key: &GemmTuningKey,
        default: TacticId,
    ) -> Result<PreparedGemmPlan> {
        let session = ctx.tuning();
        let generation = session.generation();
        if let Some(plan) = self
            .plans
            .lock()
            .map_err(|_| Error::Other("CUDA GEMM plan cache lock is poisoned".into()))?
            .get(key)
            .filter(|plan| plan.generation == generation)
            .cloned()
        {
            return Ok(plan);
        }

        let resolved = session.lookup_gemm(key);
        self.prepare_and_cache(key, default, resolved, generation)
    }

    /// Resolve an exact plan from the request's real operands. The tuning
    /// callback is never entered in INFERENCE mode or during graph capture.
    pub fn resolve_or_tune(
        &self,
        ctx: &CudaContext,
        key: &GemmTuningKey,
        default: TacticId,
        tune: impl FnOnce(Option<TacticId>) -> Result<TuningOutcome>,
    ) -> Result<PreparedGemmPlan> {
        let session = ctx.tuning();
        let generation = session.generation();
        if let Some(plan) = self
            .plans
            .lock()
            .map_err(|_| Error::Other("CUDA GEMM plan cache lock is poisoned".into()))?
            .get(key)
            .filter(|plan| plan.generation == generation)
            .cloned()
        {
            return Ok(plan);
        }

        let mut resolved = session.lookup_gemm(key);
        let needs_exact = !matches!(resolved, Some(value) if value.source == TacticMatch::Exact);
        if needs_exact
            && session.mode() == TuningMode::AutoTune
            && crate::workspace::may_prepare_native_resources()
            && !crate::workspace::is_preparing_workspace()
        {
            match session.tune_gemm(ctx.caps(), ctx.library_versions(), key, tune) {
                Ok(tuned) => resolved = Some(tuned),
                Err(error) => {
                    eprintln!("[apxinf] GEMM autotune failed for {key:?}: {error}; using fallback");
                    resolved = session.lookup_gemm(key);
                }
            }
        }
        self.prepare_and_cache(key, default, resolved, session.generation())
    }

    fn prepare_and_cache(
        &self,
        key: &GemmTuningKey,
        default: TacticId,
        resolved: Option<crate::tuning::ResolvedTactic>,
        generation: u64,
    ) -> Result<PreparedGemmPlan> {
        let (selected, source) = match resolved {
            Some(resolved) => (
                resolved.tactic,
                match resolved.source {
                    TacticMatch::Exact => PlanSource::Exact,
                    TacticMatch::Bucket => PlanSource::Bucket,
                },
            ),
            None => (default, PlanSource::Default),
        };

        let (tactic, source) = match providers::prepare(key, selected) {
            Ok(()) => (selected, source),
            Err(error) if selected != default => {
                eprintln!(
                    "[apxinf] rejected persisted GEMM tactic {selected:?} for {key:?}: {error}; using default"
                );
                providers::prepare(key, default)?;
                (default, PlanSource::Default)
            }
            Err(error) => return Err(error),
        };
        let plan = PreparedGemmPlan {
            key: key.clone(),
            tactic,
            source,
            generation,
        };
        self.plans
            .lock()
            .map_err(|_| Error::Other("CUDA GEMM plan cache lock is poisoned".into()))?
            .insert(key.clone(), plan.clone());
        Ok(plan)
    }

    /// Replace a rejected prepared tactic with the provider-independent safe
    /// route so subsequent calls do not retry the failing launch.
    pub fn fallback(&self, ctx: &CudaContext, key: &GemmTuningKey) -> Result<PreparedGemmPlan> {
        let tactic = TacticId {
            backend: TacticBackend::Vendor,
            value: 0,
        };
        providers::prepare(key, tactic)?;
        let plan = PreparedGemmPlan {
            key: key.clone(),
            tactic,
            source: PlanSource::Default,
            generation: ctx.tuning().generation(),
        };
        self.plans
            .lock()
            .map_err(|_| Error::Other("CUDA GEMM plan cache lock is poisoned".into()))?
            .insert(key.clone(), plan.clone());
        Ok(plan)
    }

    pub fn clear(&self) -> Result<()> {
        self.plans
            .lock()
            .map_err(|_| Error::Other("CUDA GEMM plan cache lock is poisoned".into()))?
            .clear();
        Ok(())
    }
}

pub fn default_fp8_tactic(m: usize, n: usize, k: usize) -> TacticId {
    #[cfg(apxinf_cutlass_gemm)]
    if n >= 1024 && n % 16 == 0 && k % 16 == 0 {
        let value = if m <= 16 {
            0
        } else if m <= 64 {
            1
        } else if m <= 256 {
            2
        } else {
            3
        };
        return TacticId {
            backend: TacticBackend::Cutlass,
            value,
        };
    }
    let _ = (m, n, k);
    TacticId {
        backend: TacticBackend::Vendor,
        value: 0,
    }
}

pub const fn default_bf16_tactic() -> TacticId {
    TacticId {
        backend: TacticBackend::Vendor,
        value: 0,
    }
}

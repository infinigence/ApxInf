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
        self.prepare_and_cache_with(
            key,
            default,
            resolved,
            generation,
            crate::workspace::may_prepare_native_resources(),
            providers::prepare,
        )
    }

    fn prepare_and_cache_with(
        &self,
        key: &GemmTuningKey,
        default: TacticId,
        resolved: Option<crate::tuning::ResolvedTactic>,
        generation: u64,
        may_prepare_native_resources: bool,
        mut prepare: impl FnMut(&GemmTuningKey, TacticId) -> Result<()>,
    ) -> Result<PreparedGemmPlan> {
        ensure_native_prepare_allowed(may_prepare_native_resources)?;
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

        let (tactic, source) = match prepare(key, selected) {
            Ok(()) => (selected, source),
            Err(error) if selected != default => {
                eprintln!(
                    "[apxinf] rejected persisted GEMM tactic {selected:?} for {key:?}: {error}; using default"
                );
                prepare(key, default)?;
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
        ensure_native_prepare_allowed(crate::workspace::may_prepare_native_resources())?;
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

fn ensure_native_prepare_allowed(may_prepare_native_resources: bool) -> Result<()> {
    if may_prepare_native_resources {
        Ok(())
    } else {
        Err(Error::Other(
            "CUDA GEMM plan cache miss while native resource preparation is disabled (for example during CUDA Graph capture); resolve and prepare all GEMM plans before capture".into(),
        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::{DeviceFingerprint, Epilogue, GemmLayout, GemmOp, ScaleMode, TuningDType};

    fn key() -> GemmTuningKey {
        GemmTuningKey {
            op: GemmOp::Bf16,
            device: DeviceFingerprint {
                sm: 89,
                multiprocessor_count: 128,
            },
            m: 1,
            n: 1024,
            k: 1024,
            activation_dtype: TuningDType::Bf16,
            weight_dtype: TuningDType::Bf16,
            output_dtype: TuningDType::Bf16,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::None,
            epilogue: Epilogue::None,
            workspace_limit: usize::MAX,
        }
    }

    #[test]
    fn cache_miss_during_capture_does_not_prepare_provider() {
        let cache = GemmPlanCache::default();
        let mut prepare_called = false;
        let error = cache
            .prepare_and_cache_with(&key(), default_bf16_tactic(), None, 0, false, |_, _| {
                prepare_called = true;
                Ok(())
            })
            .unwrap_err();

        assert!(!prepare_called);
        assert!(error.to_string().contains("plan cache miss"));
        assert!(error.to_string().contains("before capture"));
    }
}

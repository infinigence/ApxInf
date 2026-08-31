//! Model-independent candidate validation, measurement, and winner selection.

use apxinf_core::{Error, Result};

use super::{GemmTuningKey, GemmTuningRecord, TacticCandidate, TacticId};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateMeasurement {
    pub tactic: TacticId,
    pub milliseconds: Option<f64>,
    pub correct: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuningOutcome {
    pub winner: GemmTuningRecord,
    pub candidates: Vec<CandidateMeasurement>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoTuneConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            benchmark_iterations: 25,
        }
    }
}

/// Stateless orchestration shared by every physical operator tuner.
pub struct AutoTuneEngine {
    config: AutoTuneConfig,
}

impl AutoTuneEngine {
    pub fn new(config: AutoTuneConfig) -> Result<Self> {
        if config.benchmark_iterations == 0 {
            return Err(Error::Other(
                "autotune benchmark_iterations must be positive".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> AutoTuneConfig {
        self.config
    }

    /// Validate and benchmark candidates supplied by physical backend
    /// providers. Invalid or failed candidates are retained in the report but
    /// never participate in winner selection.
    pub fn tune(
        &self,
        key: &GemmTuningKey,
        candidates: impl IntoIterator<Item = TacticCandidate>,
        mut measure: impl FnMut(TacticCandidate, AutoTuneConfig) -> Result<CandidateMeasurement>,
    ) -> Result<TuningOutcome> {
        let mut measurements = Vec::new();
        for candidate in candidates {
            match measure(candidate, self.config) {
                Ok(measurement) => measurements.push(measurement),
                Err(_) => measurements.push(CandidateMeasurement {
                    tactic: candidate.tactic,
                    milliseconds: None,
                    correct: false,
                }),
            }
        }
        self.select(key, measurements)
    }

    /// A bucket winner is verified first with a short measurement. A valid
    /// result becomes the exact winner; otherwise all provider candidates are
    /// evaluated with the normal tuning budget.
    pub fn tune_with_preferred(
        &self,
        key: &GemmTuningKey,
        preferred: Option<TacticId>,
        candidates: impl IntoIterator<Item = TacticCandidate>,
        mut measure: impl FnMut(TacticCandidate, AutoTuneConfig) -> Result<CandidateMeasurement>,
    ) -> Result<TuningOutcome> {
        if let Some(tactic) = preferred {
            let candidate = TacticCandidate { tactic };
            let quick = AutoTuneConfig {
                warmup_iterations: 1,
                benchmark_iterations: self.config.benchmark_iterations.min(3).max(1),
            };
            if let Ok(measurement) = measure(candidate, quick) {
                if measurement.correct
                    && measurement
                        .milliseconds
                        .is_some_and(|milliseconds| milliseconds.is_finite() && milliseconds >= 0.0)
                {
                    return self.select(key, vec![measurement]);
                }
            }
        }
        self.tune(key, candidates, measure)
    }

    fn select(
        &self,
        key: &GemmTuningKey,
        measurements: Vec<CandidateMeasurement>,
    ) -> Result<TuningOutcome> {
        let winner = measurements
            .iter()
            .filter(|measurement| measurement.correct)
            .filter_map(|measurement| {
                measurement
                    .milliseconds
                    .filter(|milliseconds| milliseconds.is_finite() && *milliseconds >= 0.0)
                    .map(|milliseconds| (measurement.tactic, milliseconds))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .ok_or_else(|| Error::Other(format!("no valid tactic candidate for {key:?}")))?;
        Ok(TuningOutcome {
            winner: GemmTuningRecord {
                key: key.clone(),
                tactic: winner.0,
                implementation_version: Some(winner.0.backend.implementation_version()),
                milliseconds: Some(winner.1),
            },
            candidates: measurements,
        })
    }
}

/// Numerical guard used before a candidate may participate in selection.
pub(crate) fn outputs_are_close(
    reference: &[f32],
    candidate: &[f32],
    max_relative_l2: f64,
    min_cosine: f64,
) -> bool {
    if reference.len() != candidate.len() || reference.is_empty() {
        return false;
    }
    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut candidate_norm = 0.0f64;
    let mut error_norm = 0.0f64;
    for (&expected, &observed) in reference.iter().zip(candidate) {
        if !expected.is_finite() || !observed.is_finite() {
            return false;
        }
        let expected = f64::from(expected);
        let observed = f64::from(observed);
        dot += expected * observed;
        reference_norm += expected * expected;
        candidate_norm += observed * observed;
        let error = observed - expected;
        error_norm += error * error;
    }
    if reference_norm == 0.0 {
        return error_norm == 0.0;
    }
    let relative_l2 = (error_norm / reference_norm).sqrt();
    let cosine = if candidate_norm == 0.0 {
        0.0
    } else {
        dot / (reference_norm * candidate_norm).sqrt()
    };
    relative_l2 <= max_relative_l2 && cosine >= min_cosine
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuning::{
        DeviceFingerprint, Epilogue, GemmLayout, GemmOp, ScaleMode, TacticBackend, TuningDType,
    };

    fn key() -> GemmTuningKey {
        GemmTuningKey {
            op: GemmOp::Fp8F16,
            device: DeviceFingerprint {
                sm: 110,
                multiprocessor_count: 14,
            },
            m: 10,
            n: 1024,
            k: 2048,
            activation_dtype: TuningDType::F8E4M3,
            weight_dtype: TuningDType::F8E4M3,
            output_dtype: TuningDType::F16,
            layout: GemmLayout::RowMajor,
            scale_mode: ScaleMode::PerTensor,
            epilogue: Epilogue::None,
            workspace_limit: usize::MAX,
        }
    }

    #[test]
    fn selects_fastest_correct_candidate() {
        let candidates = [1, 2, 3].map(|value| TacticCandidate {
            tactic: TacticId {
                backend: TacticBackend::Cutlass,
                value,
            },
        });
        let outcome = AutoTuneEngine::new(AutoTuneConfig::default())
            .unwrap()
            .tune(&key(), candidates, |candidate, _| {
                Ok(CandidateMeasurement {
                    tactic: candidate.tactic,
                    milliseconds: Some(match candidate.tactic.value {
                        1 => 0.3,
                        2 => 0.1,
                        _ => 0.05,
                    }),
                    correct: candidate.tactic.value != 3,
                })
            })
            .unwrap();
        assert_eq!(outcome.winner.tactic.value, 2);
    }

    #[test]
    fn rejects_zero_benchmark_iterations() {
        assert!(AutoTuneEngine::new(AutoTuneConfig {
            warmup_iterations: 0,
            benchmark_iterations: 0,
        })
        .is_err());
    }

    #[test]
    fn preferred_candidate_uses_quick_validation() {
        let preferred = TacticId {
            backend: TacticBackend::Cutlass,
            value: 7,
        };
        let mut calls = Vec::new();
        let outcome = AutoTuneEngine::new(AutoTuneConfig::default())
            .unwrap()
            .tune_with_preferred(&key(), Some(preferred), [], |candidate, config| {
                calls.push(config);
                Ok(CandidateMeasurement {
                    tactic: candidate.tactic,
                    milliseconds: Some(0.1),
                    correct: true,
                })
            })
            .unwrap();
        assert_eq!(outcome.winner.tactic, preferred);
        assert_eq!(calls[0].warmup_iterations, 1);
        assert_eq!(calls[0].benchmark_iterations, 3);
    }
}

//! Inference observability: host-side generation metrics and profiler trace
//! ranges.
//!
//! - [`GenerationProfile`] measures timing metrics (TTFT/TPOT/TPS/latency) in
//!   process and reports numbers itself.
//! - [`trace`] emits named time-span markers consumed by an external profiler
//!   (NVTX / Nsight on CUDA); it produces no numbers of its own.

pub mod trace;

use std::time::{Duration, Instant};

/// Timing profile for a single generation run.
///
/// Captures key timing points during text generation and computes
/// standard LLM inference metrics: TTFT, TPOT, TPS, and total latency.
pub struct GenerationProfile {
    start_time: Instant,
    first_token_time: Option<Instant>,
    end_time: Option<Instant>,
    excluded_time: Duration,
    input_tokens: usize,
    output_tokens: usize,
}

impl GenerationProfile {
    /// Create a new profile, recording the start time as now.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            first_token_time: None,
            end_time: None,
            excluded_time: Duration::ZERO,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    /// Record the moment the first output token is ready (end of prefill).
    pub fn record_first_token(&mut self) {
        if self.first_token_time.is_none() {
            self.first_token_time = Some(Instant::now());
        }
    }

    /// Finalize the profile with token counts and record end time.
    pub fn finalize(&mut self, input_tokens: usize, output_tokens: usize) {
        self.end_time = Some(Instant::now());
        self.input_tokens = input_tokens;
        self.output_tokens = output_tokens;
    }

    /// Run application-side work without charging it to engine latency.
    ///
    /// Streaming callbacks commonly perform detokenization and terminal I/O;
    /// neither is part of model execution and both can otherwise inflate TPOT.
    pub(crate) fn run_unprofiled<T>(&mut self, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let output = f();
        self.excluded_time += start.elapsed();
        output
    }

    fn engine_duration(&self, start: Instant, end: Instant) -> Duration {
        end.duration_since(start).saturating_sub(self.excluded_time)
    }

    /// Time to first token (prefill duration) in milliseconds.
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_time.map(|ft| {
            (ft - self.start_time).as_nanos() as f64 / 1_000_000.0
        })
    }

    /// Time per output token (decode phase only, excluding the first token).
    /// The first token is produced by prefill; only subsequent tokens
    /// require a dedicated forward pass.
    pub fn tpot_ms(&self) -> Option<f64> {
        match (self.first_token_time, self.end_time, self.output_tokens) {
            (Some(ft), Some(et), n) if n > 1 => {
                // Decode tokens = output_tokens - 1 (first token comes "free" from prefill)
                Some(
                    self.engine_duration(ft, et).as_nanos() as f64
                        / 1_000_000.0
                        / (n - 1) as f64,
                )
            }
            _ => None,
        }
    }

    /// Generation tokens per second (decode phase only, excluding the first token).
    pub fn generation_tps(&self) -> Option<f64> {
        match (self.first_token_time, self.end_time, self.output_tokens) {
            (Some(ft), Some(et), n) if n > 1 => {
                // Decode tokens = output_tokens - 1 (first token comes "free" from prefill)
                let secs = self.engine_duration(ft, et).as_nanos() as f64 / 1_000_000_000.0;
                Some((n - 1) as f64 / secs)
            }
            _ => None,
        }
    }

    /// Total generation latency in milliseconds.
    pub fn total_latency_ms(&self) -> Option<f64> {
        self.end_time.map(|et| {
            self.engine_duration(self.start_time, et).as_nanos() as f64 / 1_000_000.0
        })
    }

    /// Number of input (prompt) tokens.
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    /// Number of output (generated) tokens.
    pub fn output_tokens(&self) -> usize {
        self.output_tokens
    }

    /// Format a human-readable summary of all metrics.
    pub fn summary(&self) -> String {
        let ttft = self.ttft_ms()
            .map(|ms| format!("{ms:.1} ms"))
            .unwrap_or_else(|| "N/A".to_string());

        let tpot = self.tpot_ms()
            .map(|ms| format!("{ms:.1} ms/token"))
            .unwrap_or_else(|| "N/A".to_string());

        let gen_tps = self.generation_tps()
            .map(|tps| format!("{tps:.2} tok/s"))
            .unwrap_or_else(|| "N/A".to_string());

        let total = self.total_latency_ms()
            .map(|ms| {
                if ms >= 1000.0 {
                    format!("{:.2} s", ms / 1000.0)
                } else {
                    format!("{ms:.1} ms")
                }
            })
            .unwrap_or_else(|| "N/A".to_string());

        format!(
            "=== Generation Profile ===\n\
             Input tokens:     {}\n\
             Output tokens:    {}\n\
             TTFT:             {}\n\
             TPOT:             {}\n\
             Generation TPS:   {}\n\
             Total latency:    {}\n\
             ========================="
            ,
            self.input_tokens,
            self.output_tokens,
            ttft,
            tpot,
            gen_tps,
            total,
        )
    }
}

impl Default for GenerationProfile {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_time_is_excluded_from_engine_metrics() {
        let start = Instant::now();
        let profile = GenerationProfile {
            start_time: start,
            first_token_time: Some(start + Duration::from_millis(10)),
            end_time: Some(start + Duration::from_millis(100)),
            excluded_time: Duration::from_millis(20),
            input_tokens: 8,
            output_tokens: 3,
        };

        assert_eq!(profile.ttft_ms(), Some(10.0));
        assert_eq!(profile.tpot_ms(), Some(35.0));
        assert_eq!(profile.total_latency_ms(), Some(80.0));
    }
}

//! Inference observability: host-side generation metrics and profiler trace
//! ranges.
//!
//! - [`GenerationProfile`] measures timing metrics (TTFT/TPOT/TPS/latency) in
//!   process and reports numbers itself.
//! - [`trace`] emits named time-span markers consumed by an external profiler
//!   (NVTX / Nsight on CUDA); it produces no numbers of its own.

pub mod trace;

use std::time::Instant;

/// Timing profile for a single generation run.
///
/// Captures key timing points during text generation and computes
/// standard LLM inference metrics: TTFT, TPOT, TPS, and total latency.
pub struct GenerationProfile {
    start_time: Instant,
    first_token_time: Option<Instant>,
    end_time: Option<Instant>,
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

    /// Time to first token (prefill duration) in milliseconds.
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_time
            .map(|ft| (ft - self.start_time).as_nanos() as f64 / 1_000_000.0)
    }

    /// Time per output token (decode phase only, excluding the first token).
    /// The first token is produced by prefill; only subsequent tokens
    /// require a dedicated forward pass.
    pub fn tpot_ms(&self) -> Option<f64> {
        match (self.first_token_time, self.end_time, self.output_tokens) {
            (Some(ft), Some(et), n) if n > 1 => {
                // Decode tokens = output_tokens - 1 (first token comes "free" from prefill)
                Some((et - ft).as_nanos() as f64 / 1_000_000.0 / (n - 1) as f64)
            }
            _ => None,
        }
    }

    /// Generation tokens per second (decode phase only, excluding the first token).
    pub fn generation_tps(&self) -> Option<f64> {
        match (self.first_token_time, self.end_time, self.output_tokens) {
            (Some(ft), Some(et), n) if n > 1 => {
                // Decode tokens = output_tokens - 1 (first token comes "free" from prefill)
                let secs = (et - ft).as_nanos() as f64 / 1_000_000_000.0;
                Some((n - 1) as f64 / secs)
            }
            _ => None,
        }
    }

    /// Total generation latency in milliseconds.
    pub fn total_latency_ms(&self) -> Option<f64> {
        self.end_time
            .map(|et| (et - self.start_time).as_nanos() as f64 / 1_000_000.0)
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
        let ttft = self
            .ttft_ms()
            .map(|ms| format!("{ms:.1} ms"))
            .unwrap_or_else(|| "N/A".to_string());

        let tpot = self
            .tpot_ms()
            .map(|ms| format!("{ms:.1} ms/token"))
            .unwrap_or_else(|| "N/A".to_string());

        let gen_tps = self
            .generation_tps()
            .map(|tps| format!("{tps:.2} tok/s"))
            .unwrap_or_else(|| "N/A".to_string());

        let total = self
            .total_latency_ms()
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
             =========================",
            self.input_tokens, self.output_tokens, ttft, tpot, gen_tps, total,
        )
    }
}

impl Default for GenerationProfile {
    fn default() -> Self {
        Self::new()
    }
}

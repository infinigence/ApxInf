//! Model-neutral decision API.

use apxinf_core::{Error, Result};

use crate::llm_trait::{ImageInput, LlmCapabilities};

/// Safe defaults for the d3 text CUDA Graph profile cache. The full public
/// Decision Index contains many distinct prompt lengths; leaving this cache
/// unbounded can exhaust a 24 GiB GPU before the run completes.
pub const D3_DEFAULT_TEXT_PROFILE_LIMIT: usize = 12;
pub const D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET: usize = 30_000;

/// One complete prompt. Video uses the same processor tensor plus `grid_thw`
/// contract as Qwen3.5 vision; callers concatenate temporal patches there.
#[derive(Clone, Copy, Debug)]
pub struct DecisionInput<'a> {
    pub token_ids: &'a [u32],
    pub image: Option<ImageInput<'a>>,
    /// Number of valid answer codes at the start of the package code table.
    pub candidate_count: usize,
}

impl<'a> DecisionInput<'a> {
    pub const fn text(token_ids: &'a [u32], candidate_count: usize) -> Self {
        Self {
            token_ids,
            image: None,
            candidate_count,
        }
    }

    pub const fn with_vision(
        token_ids: &'a [u32],
        image: ImageInput<'a>,
        candidate_count: usize,
    ) -> Self {
        Self {
            token_ids,
            image: Some(image),
            candidate_count,
        }
    }
}

/// Probability assigned to one package-defined d3 code.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionScore {
    pub code: String,
    pub token_id: u32,
    pub probability: f32,
}

/// Structured output; d3 never enters autoregressive token generation.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionOutput {
    pub selected_index: usize,
    pub scores: Vec<DecisionScore>,
}

impl DecisionOutput {
    pub fn selected(&self) -> Result<&DecisionScore> {
        self.scores.get(self.selected_index).ok_or_else(|| {
            Error::Other(format!(
                "d3 decision output: selected index {} outside {} scores",
                self.selected_index,
                self.scores.len()
            ))
        })
    }
}

/// Non-generative decision runtime. Implementations own fixed-profile CUDA
/// preparation/capture and return host-visible probabilities only at the end.
pub trait DecisionRuntime {
    fn capabilities(&self) -> LlmCapabilities;
    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput>;
    /// Package code names in index order.
    fn code_names(&self) -> &[String];
    /// Token IDs for each code.
    fn token_ids(&self) -> &[u32];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_score_is_checked() {
        let output = DecisionOutput {
            selected_index: 0,
            scores: vec![DecisionScore {
                code: "yes".into(),
                token_id: 1,
                probability: 1.0,
            }],
        };
        assert_eq!(output.selected().unwrap().code, "yes");

        let invalid = DecisionOutput {
            selected_index: 1,
            scores: output.scores,
        };
        assert!(invalid.selected().is_err());
    }
}

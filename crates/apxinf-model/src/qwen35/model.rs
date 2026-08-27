#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::{GdnDimensions, LayerType};

    fn production_schedule() -> Vec<LayerType> {
        (0..64)
            .map(|index| {
                if index % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::Gdn
                }
            })
            .collect()
    }

    #[test]
    fn request_state_allocates_exact_48_gdn_and_16_attention_layers() {
        let state = Qwen35RequestState::new(
            &production_schedule(),
            GdnDimensions::new(4, 1, 1, 1, 1).unwrap(),
            1,
            1,
            8,
        )
        .unwrap();
        assert_eq!(state.gdn_count(), 48);
        assert_eq!(state.attention_count(), 16);
        assert_eq!(state.position(), 0);
    }

    #[test]
    fn executor_runs_prefill_then_incremental_single_token_decode() {
        let mut executor = Qwen35ReferenceExecutor::tiny(production_schedule(), 16, 8).unwrap();
        let mut calls = Vec::new();
        let output = executor
            .generate(
                &[1, 2],
                3,
                true,
                || false,
                |tokens, position, _state| {
                    let step = calls.len();
                    calls.push((tokens.to_vec(), position));
                    let mut logits = vec![0.0; 16];
                    logits[step + 5] = 1.0;
                    Ok(logits)
                },
            )
            .unwrap();
        assert_eq!(calls, vec![(vec![1, 2], 0), (vec![5], 2), (vec![6], 3)]);
        assert_eq!(output.tokens, vec![5, 6, 7]);
        assert_eq!(output.stop_reason, StopReason::Budget);
    }

    #[test]
    fn executor_honors_both_eos_ids_and_ignore_eos() {
        let mut executor =
            Qwen35ReferenceExecutor::tiny(production_schedule(), 248_320, 8).unwrap();
        let eos = executor
            .generate(
                &[1],
                3,
                false,
                || false,
                |_tokens, _position, _state| {
                    let mut logits = vec![0.0; 248_320];
                    logits[248_046] = 1.0;
                    Ok(logits)
                },
            )
            .unwrap();
        assert_eq!(eos.tokens, vec![248_046]);
        assert_eq!(eos.stop_reason, StopReason::Eos);

        let ignored = executor
            .generate(
                &[1],
                2,
                true,
                || false,
                |_tokens, _position, _state| {
                    let mut logits = vec![0.0; 248_320];
                    logits[248_044] = 1.0;
                    Ok(logits)
                },
            )
            .unwrap();
        assert_eq!(ignored.tokens, vec![248_044, 248_044]);
        assert_eq!(ignored.stop_reason, StopReason::Budget);
    }

    #[test]
    fn executor_rejects_budget_nonfinite_logits_and_cancellation() {
        let mut executor = Qwen35ReferenceExecutor::tiny(production_schedule(), 8, 3).unwrap();
        assert!(matches!(
            executor.generate(
                &[1, 2],
                2,
                true,
                || false,
                |_tokens, _position, _state| Ok(vec![0.0; 8])
            ),
            Err(ExecutorError::Budget { .. })
        ));
        assert!(matches!(
            executor.generate(
                &[1],
                1,
                true,
                || false,
                |_tokens, _position, _state| Ok(vec![f32::NAN; 8])
            ),
            Err(ExecutorError::NonFiniteLogits)
        ));
        assert!(matches!(
            executor.generate(
                &[1],
                1,
                true,
                || true,
                |_tokens, _position, _state| Ok(vec![0.0; 8])
            ),
            Err(ExecutorError::Cancelled)
        ));
        assert!(!executor.has_active_state());
    }

    #[test]
    fn greedy_argmax_accepts_model_vocab_boundary_and_rejects_wrong_width() {
        assert_eq!(greedy_argmax(&[0.0, 1.0, 1.0], 3).unwrap(), 2);
        assert_eq!(greedy_argmax(&[0.0, 21.375, 21.25], 3).unwrap(), 1);
        assert_eq!(greedy_argmax(&[0.0, 1.0, 0.875_001], 3).unwrap(), 1);
        assert_eq!(greedy_argmax(&[0.0, 1.0, 0.875], 3).unwrap(), 1);
        assert!(greedy_argmax(&[0.0, 1.0], 3).is_err());
    }
}
use std::fmt;

use super::attention::{AttentionError, FullAttentionKv};
use super::config::LayerType;
use super::gdn::{GdnDimensions, GdnState};

const EOS_IDS: [u32; 2] = [248_046, 248_044];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorError {
    Schedule {
        expected: usize,
        got: usize,
    },
    Budget {
        prompt_tokens: usize,
        max_new_tokens: usize,
        max_model_len: usize,
    },
    Vocabulary {
        token_id: u32,
        vocab_size: usize,
    },
    NonFiniteLogits,
    WrongLogitWidth {
        expected: usize,
        got: usize,
    },
    Cancelled,
    Callback(String),
    Attention(AttentionError),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ExecutorError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    Budget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    pub stop_reason: StopReason,
}

pub struct Qwen35RequestState {
    layer_types: Vec<LayerType>,
    gdn: Vec<Option<GdnState>>,
    attention: Vec<Option<FullAttentionKv>>,
    position: usize,
}

impl Qwen35RequestState {
    pub fn new(
        layer_types: &[LayerType],
        gdn_dimensions: GdnDimensions,
        n_kv_heads: usize,
        head_dim: usize,
        max_model_len: usize,
    ) -> Result<Self, ExecutorError> {
        if layer_types.len() != 64 || max_model_len == 0 {
            return Err(ExecutorError::Schedule {
                expected: 64,
                got: layer_types.len(),
            });
        }
        let mut gdn = Vec::with_capacity(layer_types.len());
        let mut attention = Vec::with_capacity(layer_types.len());
        for layer_type in layer_types {
            match layer_type {
                LayerType::Gdn => {
                    gdn.push(Some(GdnState::new(gdn_dimensions).map_err(|_| {
                        ExecutorError::Schedule {
                            expected: 64,
                            got: layer_types.len(),
                        }
                    })?));
                    attention.push(None);
                }
                LayerType::FullAttention => {
                    gdn.push(None);
                    attention.push(Some(
                        FullAttentionKv::new(n_kv_heads, head_dim, max_model_len).map_err(
                            |_| ExecutorError::Schedule {
                                expected: 64,
                                got: layer_types.len(),
                            },
                        )?,
                    ));
                }
            }
        }
        Ok(Self {
            layer_types: layer_types.to_vec(),
            gdn,
            attention,
            position: 0,
        })
    }

    pub fn gdn_count(&self) -> usize {
        self.gdn.iter().filter(|state| state.is_some()).count()
    }
    pub fn attention_count(&self) -> usize {
        self.attention
            .iter()
            .filter(|state| state.is_some())
            .count()
    }
    pub const fn position(&self) -> usize {
        self.position
    }
    pub fn advance(&mut self, amount: usize) {
        self.position += amount;
    }
}

impl Drop for Qwen35RequestState {
    fn drop(&mut self) {
        for state in &mut self.gdn {
            *state = None;
        }
        for state in &mut self.attention {
            *state = None;
        }
        self.position = 0;
    }
}

pub struct Qwen35ReferenceExecutor {
    layer_types: Vec<LayerType>,
    vocab_size: usize,
    max_model_len: usize,
    gdn_dimensions: GdnDimensions,
    n_kv_heads: usize,
    head_dim: usize,
    active_state: Option<Qwen35RequestState>,
}

impl Qwen35ReferenceExecutor {
    pub fn tiny(
        layer_types: Vec<LayerType>,
        vocab_size: usize,
        max_model_len: usize,
    ) -> Result<Self, ExecutorError> {
        if layer_types.len() != 64 || vocab_size == 0 || max_model_len == 0 {
            return Err(ExecutorError::Schedule {
                expected: 64,
                got: layer_types.len(),
            });
        }
        Ok(Self {
            layer_types,
            vocab_size,
            max_model_len,
            gdn_dimensions: GdnDimensions::new(4, 1, 1, 1, 1).map_err(|_| {
                ExecutorError::Schedule {
                    expected: 64,
                    got: 0,
                }
            })?,
            n_kv_heads: 1,
            head_dim: 1,
            active_state: None,
        })
    }

    pub fn generate<F, C>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
        mut cancelled: C,
        mut forward: F,
    ) -> Result<GenerationOutput, ExecutorError>
    where
        F: FnMut(&[u32], usize, &mut Qwen35RequestState) -> Result<Vec<f32>, ExecutorError>,
        C: FnMut() -> bool,
    {
        if prompt_tokens.is_empty() || prompt_tokens.len() + max_new_tokens > self.max_model_len {
            return Err(ExecutorError::Budget {
                prompt_tokens: prompt_tokens.len(),
                max_new_tokens,
                max_model_len: self.max_model_len,
            });
        }
        for &token in prompt_tokens {
            if token as usize >= self.vocab_size {
                return Err(ExecutorError::Vocabulary {
                    token_id: token,
                    vocab_size: self.vocab_size,
                });
            }
        }
        if max_new_tokens == 0 {
            return Err(ExecutorError::Budget {
                prompt_tokens: prompt_tokens.len(),
                max_new_tokens,
                max_model_len: self.max_model_len,
            });
        }
        let mut state = Qwen35RequestState::new(
            &self.layer_types,
            self.gdn_dimensions,
            self.n_kv_heads,
            self.head_dim,
            self.max_model_len,
        )?;
        let mut next_input = prompt_tokens.to_vec();
        let mut position = 0usize;
        let mut output = Vec::with_capacity(max_new_tokens);
        let mut stop_reason = StopReason::Budget;
        for step in 0..max_new_tokens {
            if cancelled() {
                self.active_state = None;
                return Err(ExecutorError::Cancelled);
            }
            let logits = forward(&next_input, position, &mut state)?;
            let token = greedy_argmax(&logits, self.vocab_size)?;
            output.push(token);
            let consumed = next_input.len();
            position += consumed;
            state.advance(consumed);
            next_input = vec![token];
            if !ignore_eos && EOS_IDS.contains(&token) {
                stop_reason = StopReason::Eos;
                break;
            }
            if step + 1 == max_new_tokens {
                break;
            }
        }
        self.active_state = Some(state);
        let result = GenerationOutput {
            tokens: output,
            stop_reason,
        };
        self.active_state = None;
        Ok(result)
    }

    pub fn has_active_state(&self) -> bool {
        self.active_state.is_some()
    }
}

pub fn greedy_argmax(logits: &[f32], vocab_size: usize) -> Result<u32, ExecutorError> {
    if logits.len() != vocab_size {
        return Err(ExecutorError::WrongLogitWidth {
            expected: vocab_size,
            got: logits.len(),
        });
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(ExecutorError::NonFiniteLogits);
    }
    // Keep the highest token id when logits are exactly equal. This is the
    // deterministic tie rule used by the checkpoint trajectory artifacts;
    // near-ties must not be treated as ties because temperature=0 is exact
    // greedy decoding.
    let maximum_index = logits
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.partial_cmp(right)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(index, _)| index)
        .ok_or(ExecutorError::WrongLogitWidth {
            expected: vocab_size,
            got: 0,
        })?;
    Ok(maximum_index as u32)
}

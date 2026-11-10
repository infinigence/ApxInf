use thiserror::Error;

use super::config::MODEL_VOCAB_SIZE;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    #[error("input_ids must not be empty")]
    EmptyPrompt,
    #[error("token id {token_id} is outside model vocabulary [0, {vocab_size})")]
    TokenOutOfRange { token_id: u32, vocab_size: usize },
    #[error("prompt tokens {prompt_tokens} plus max_new_tokens {max_new_tokens} exceed max_model_len {max_model_len}")]
    TotalBudgetExceeded {
        prompt_tokens: usize,
        max_new_tokens: usize,
        max_model_len: usize,
    },
}

pub fn validate_input_ids(input_ids: &[u32]) -> Result<(), AdmissionError> {
    validate_input_ids_with_vocab(input_ids, MODEL_VOCAB_SIZE)
}

pub fn validate_input_ids_with_vocab(
    input_ids: &[u32],
    vocab_size: usize,
) -> Result<(), AdmissionError> {
    if input_ids.is_empty() {
        return Err(AdmissionError::EmptyPrompt);
    }
    for &token_id in input_ids {
        if (token_id as usize) >= vocab_size {
            return Err(AdmissionError::TokenOutOfRange {
                token_id,
                vocab_size,
            });
        }
    }
    Ok(())
}

pub fn validate_total_budget(
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_model_len: usize,
) -> Result<(), AdmissionError> {
    let total = prompt_tokens.saturating_add(max_new_tokens);
    if total > max_model_len {
        return Err(AdmissionError::TotalBudgetExceeded {
            prompt_tokens,
            max_new_tokens,
            max_model_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::{IMAGE_TOKEN_ID, MODEL_VOCAB_SIZE};

    #[test]
    fn model_vocab_accepts_image_token_beyond_tokenizer_vocab() {
        assert!(validate_input_ids(&[IMAGE_TOKEN_ID]).is_ok());
        assert_eq!(MODEL_VOCAB_SIZE, 248_320);
    }

    #[test]
    fn model_vocab_rejects_upper_bound_and_u32_max() {
        assert!(matches!(
            validate_input_ids(&[MODEL_VOCAB_SIZE as u32]),
            Err(AdmissionError::TokenOutOfRange { .. })
        ));
        assert!(matches!(
            validate_input_ids(&[u32::MAX]),
            Err(AdmissionError::TokenOutOfRange { .. })
        ));
    }

    #[test]
    fn empty_prompt_is_rejected() {
        assert!(matches!(
            validate_input_ids(&[]),
            Err(AdmissionError::EmptyPrompt)
        ));
    }

    #[test]
    fn total_budget_includes_prompt_and_new_tokens() {
        assert!(validate_total_budget(8, 16, 24).is_ok());
        assert!(matches!(
            validate_total_budget(8, 17, 24),
            Err(AdmissionError::TotalBudgetExceeded { .. })
        ));
    }
}

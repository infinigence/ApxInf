//! Native tokenizer adapters for Hugging Face `tokenizer.json` and, behind the
//! `sentencepiece` feature, SentencePiece `.model` checkpoints.

use std::path::Path;

use apxinf_core::{Error, Result};
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

#[cfg(feature = "sentencepiece")]
use sentencepiece::SentencePieceProcessor;

/// Chat message for template rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }
}

/// Configuration loaded from tokenizer_config.json.
#[derive(Debug, Deserialize, Default)]
struct TokenizerConfig {
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

/// Wrapper around HuggingFace tokenizer with chat template support.
pub struct Tokenizer {
    inner: HfTokenizer,
    config: TokenizerConfig,
    chat_template: Option<String>,
}

impl Tokenizer {
    /// Load tokenizer from tokenizer.json, optionally loading tokenizer_config.json
    /// from the same directory for chat template and special tokens.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let inner = HfTokenizer::from_file(path_ref)
            .map_err(|e| Error::Other(format!("tokenizer load: {e}")))?;

        // Try to load tokenizer_config.json from same directory
        let config_path = path_ref
            .parent()
            .map(|p| p.join("tokenizer_config.json"))
            .unwrap_or_else(|| path_ref.with_extension("config.json"));

        let config = if config_path.exists() {
            std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            TokenizerConfig::default()
        };

        // Store chat template separately (we'll create env on demand)
        let chat_template = config.chat_template.clone();

        Ok(Self {
            inner,
            config,
            chat_template,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Other(format!("tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Add ordinary model tokens and return the number newly inserted.
    pub fn add_tokens(&mut self, tokens: &[String]) -> usize {
        let tokens = tokens
            .iter()
            .cloned()
            .map(|token| AddedToken::from(token, false))
            .collect::<Vec<_>>();
        self.inner.add_tokens(&tokens)
    }

    /// Resolve one token to its vocabulary ID.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| Error::Other(format!("tokenizer decode: {e}")))
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the EOS token ID.
    /// First checks tokenizer_config.json, then falls back to vocabulary search.
    pub fn eos_token_id(&self) -> Option<u32> {
        // Prefer explicit ID from config
        if let Some(id) = self.config.eos_token_id {
            return Some(id);
        }

        // Try to find by token name from config
        if let Some(token) = &self.config.eos_token {
            if let Some(&id) = self.inner.get_vocab(true).get(token) {
                return Some(id);
            }
        }

        // Fallback: search vocabulary for common EOS tokens
        self.inner
            .get_vocab(true)
            .iter()
            .find(|(token, _)| {
                *token == "</s>" || *token == "<|eot_id|>" || *token == "<|end_of_text|>"
            })
            .map(|(_, &id)| id)
    }

    /// Get the BOS (beginning-of-sequence) token ID.
    /// First checks tokenizer_config.json, then falls back to vocabulary search.
    pub fn bos_token_id(&self) -> Option<u32> {
        // Prefer explicit ID from config
        if let Some(id) = self.config.bos_token_id {
            return Some(id);
        }

        // Try to find by token name from config
        if let Some(token) = &self.config.bos_token {
            if let Some(&id) = self.inner.get_vocab(true).get(token) {
                return Some(id);
            }
        }

        // Fallback: search vocabulary for common BOS tokens
        self.inner
            .get_vocab(true)
            .iter()
            .find(|(token, _)| *token == "<s>" || *token == "<|begin_of_text|>")
            .map(|(_, &id)| id)
    }

    /// Check if a chat template is available.
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Apply chat template to messages, returning formatted prompt string.
    ///
    /// Requires tokenizer_config.json with `chat_template` field.
    /// Uses minijinja to render the Jinja2 template.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        let template_str = self.chat_template.as_ref()
            .ok_or_else(|| Error::Other("no chat template available (missing tokenizer_config.json with chat_template field)".to_string()))?;

        // Create environment and template on demand
        let mut env = Environment::new();
        env.add_template("chat", template_str)
            .map_err(|e| Error::Other(format!("template error: {e}")))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| Error::Other(format!("template error: {e}")))?;

        // Build template context
        let bos = self
            .config
            .bos_token
            .clone()
            .or_else(|| {
                self.inner
                    .get_vocab(true)
                    .iter()
                    .find(|(t, _)| *t == "<s>" || *t == "<|begin_of_text|>")
                    .map(|(t, _)| t.clone())
            })
            .unwrap_or_default();

        let eos = self
            .config
            .eos_token
            .clone()
            .or_else(|| {
                self.inner
                    .get_vocab(true)
                    .iter()
                    .find(|(t, _)| *t == "</s>" || *t == "<|eot_id|>" || *t == "<|end_of_text|>")
                    .map(|(t, _)| t.clone())
            })
            .unwrap_or_default();

        // Create context as serde Value (map)
        let context = serde_json::json!({
            "messages": messages,
            "bos_token": bos,
            "eos_token": eos,
            "add_generation_prompt": true,
        });

        let result = tmpl
            .render(context)
            .map_err(|e| Error::Other(format!("template render error: {e}")))?;

        // Jinja2 in Python strips whitespace around control blocks, but minijinja doesn't.
        // Normalize by collapsing all consecutive newlines to single newlines.
        let mut normalized = String::new();
        let mut prev_was_newline = false;
        for c in result.trim().chars() {
            if c == '\n' {
                if !prev_was_newline {
                    normalized.push('\n');
                    prev_was_newline = true;
                }
            } else {
                normalized.push(c);
                prev_was_newline = false;
            }
        }

        // Ensure trailing newline (matching PyTorch behavior)
        if !normalized.ends_with('\n') {
            normalized.push('\n');
        }

        Ok(normalized)
    }

    /// Encode messages using chat template.
    /// Convenience method that applies template and encodes the result.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let prompt = self.apply_chat_template(messages)?;
        println!("Formatted prompt:\n{}", prompt);
        self.encode(&prompt)
    }
}

/// Native SentencePiece tokenizer for checkpoints that carry a `.model` file.
///
/// This adapter deliberately exposes only deterministic inference. Training and
/// subword sampling remain outside the ApxInf runtime.
#[cfg(feature = "sentencepiece")]
pub struct SentencePieceTokenizer {
    inner: SentencePieceProcessor,
}

#[cfg(feature = "sentencepiece")]
impl SentencePieceTokenizer {
    /// Load a native SentencePiece protobuf model.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let inner = SentencePieceProcessor::open(path_ref)
            .map_err(|error| Error::Other(format!("sentencepiece load: {error}")))?;
        Ok(Self { inner })
    }

    /// Encode text to IDs, optionally prefixing the model-declared BOS token.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let pieces = self
            .inner
            .encode(text)
            .map_err(|error| Error::Other(format!("sentencepiece encode: {error}")))?;
        let mut ids = Vec::with_capacity(pieces.len() + usize::from(add_bos));
        if add_bos {
            let bos = self.inner.bos_id().ok_or_else(|| {
                Error::Other("sentencepiece model does not declare a BOS token".to_string())
            })?;
            ids.push(bos);
        }
        ids.extend(pieces.into_iter().map(|piece| piece.id));
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn test_chat_message_constructors() {
        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");
        assert_eq!(user.content, "Hello");

        let assistant = ChatMessage::assistant("Hi there");
        assert_eq!(assistant.role, "assistant");

        let system = ChatMessage::system("You are helpful");
        assert_eq!(system.role, "system");
    }

    #[test]
    fn added_tokens_are_encoded_and_resolved() {
        let vocab = HashMap::from([("[UNK]".to_string(), 0), ("<|image_pad|>".to_string(), 1)]);
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer {
            inner: HfTokenizer::new(model),
            config: TokenizerConfig::default(),
            chat_template: None,
        };

        assert_eq!(
            tokenizer.add_tokens(&["<|propri|>".to_string(), "<|action|>".to_string()]),
            2
        );
        assert_eq!(tokenizer.token_to_id("<|image_pad|>"), Some(1));
        assert_eq!(tokenizer.token_to_id("<|propri|>"), Some(2));
        assert_eq!(tokenizer.token_to_id("<|action|>"), Some(3));
        assert_eq!(tokenizer.encode("<|action|>").unwrap(), vec![3]);
    }
}

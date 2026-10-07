//! Native tokenizer adapters for Hugging Face `tokenizer.json` and, behind the
//! `sentencepiece` feature, SentencePiece `.model` checkpoints.

use std::path::Path;

use apxinf_core::{Error, Result};
use minijinja::Environment;
use serde::{Deserialize, Serialize};
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};

/// Transformers overrides Jinja's HTML-oriented `tojson` with Python
/// json.dumps: Unicode is preserved by default, HTML is not escaped, object
/// insertion order is retained, and default separators include spaces.
fn hf_tojson(
    value: minijinja::Value,
    arguments: minijinja::value::Rest<minijinja::Value>,
) -> std::result::Result<String, minijinja::Error> {
    use minijinja::{
        value::{from_args, Kwargs, Value},
        ErrorKind,
    };
    let invalid =
        |message: &str| minijinja::Error::new(ErrorKind::InvalidOperation, message.to_owned());
    let (args, kwargs) = from_args::<(&[Value], Kwargs)>(&arguments)?;
    if args.len() > 4 {
        return Err(invalid("tojson accepts at most four options"));
    }
    let get = |index: usize, name: &str| -> std::result::Result<Option<Value>, minijinja::Error> {
        let keyword: Option<Value> = kwargs.get(name)?;
        if args.get(index).is_some() && keyword.is_some() {
            return Err(invalid("duplicate tojson option"));
        }
        Ok(args.get(index).cloned().or(keyword))
    };
    let ascii = get(0, "ensure_ascii")?.map_or(false, |v| v.is_true());
    let indent = match get(1, "indent")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => Some(if let Some(s) = v.as_str() {
            s.to_owned()
        } else if v.kind() == minijinja::value::ValueKind::Bool {
            if v.is_true() { " " } else { "" }.to_owned()
        } else {
            let n = i64::try_from(v)
                .map_err(|_| invalid("tojson indent must be an integer, string, or none"))?;
            " ".repeat(usize::try_from(n.max(0)).map_err(|_| invalid("indent exceeds usize"))?)
        }),
    };
    let separators = match get(2, "separators")? {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => {
            let items: Vec<String> = serde_json::from_value(
                serde_json::to_value(v).map_err(|e| invalid(&e.to_string()))?,
            )
            .map_err(|_| invalid("tojson separators must be two strings"))?;
            if items.len() != 2 {
                return Err(invalid("tojson separators must be two strings"));
            }
            Some((items[0].clone(), items[1].clone()))
        }
    };
    let sort_keys = get(3, "sort_keys")?.map_or(false, |v| v.is_true());
    kwargs.assert_all_used()?;
    let (item_separator, key_separator) = separators.unwrap_or_else(|| {
        (
            if indent.is_some() { "," } else { ", " }.to_owned(),
            ": ".to_owned(),
        )
    });
    let value = serde_json::to_value(value).map_err(|e| invalid(&e.to_string()))?;
    fn string(value: &str, ascii: bool) -> String {
        let escaped = serde_json::to_string(value).expect("strings serialize to JSON");
        if !ascii {
            return escaped;
        }
        let mut out = String::new();
        for c in escaped.chars() {
            if c <= '\u{7e}' {
                out.push(c);
            } else {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    use std::fmt::Write;
                    write!(&mut out, "\\u{unit:04x}").unwrap();
                }
            }
        }
        out
    }
    fn render(
        value: &serde_json::Value,
        ascii: bool,
        indent: Option<&str>,
        item: &str,
        key: &str,
        sort: bool,
        depth: usize,
    ) -> String {
        use serde_json::Value as J;
        let (open, close, entries): (&str, &str, Vec<String>) = match value {
            J::Array(values) => (
                "[",
                "]",
                values
                    .iter()
                    .map(|v| render(v, ascii, indent, item, key, sort, depth + 1))
                    .collect(),
            ),
            J::Object(values) => {
                let mut keys = values.keys().collect::<Vec<_>>();
                if sort {
                    keys.sort();
                }
                (
                    "{",
                    "}",
                    keys.into_iter()
                        .map(|k| {
                            format!(
                                "{}{}{}",
                                string(k, ascii),
                                key,
                                render(&values[k], ascii, indent, item, key, sort, depth + 1)
                            )
                        })
                        .collect(),
                )
            }
            J::String(s) => return string(s, ascii),
            J::Number(n) if n.is_f64() => {
                // Python's repr uses scientific notation below 1e-4 or at
                // 1e16, and pads exponent magnitudes to at least two digits.
                let raw = n.to_string();
                let sign = if raw.starts_with('-') { "-" } else { "" };
                let unsigned = raw.trim_start_matches('-');
                let (mantissa, exponent) = unsigned
                    .split_once(['e', 'E'])
                    .map_or((unsigned, 0), |(m, e)| (m, e.parse::<i32>().unwrap()));
                let point = mantissa.find('.').unwrap_or(mantissa.len());
                let digits = mantissa.replace('.', "");
                if let Some(first) = digits.find(|c| c != '0') {
                    let exponent = exponent + point as i32 - first as i32 - 1;
                    if exponent < -4 || exponent >= 16 {
                        let digits = digits[first..].trim_end_matches('0');
                        let fraction = if digits.len() > 1 {
                            format!(".{}", &digits[1..])
                        } else {
                            String::new()
                        };
                        return format!(
                            "{sign}{}{fraction}e{}{:02}",
                            &digits[..1],
                            if exponent < 0 { '-' } else { '+' },
                            exponent.unsigned_abs()
                        );
                    }
                }
                return raw;
            }
            _ => return value.to_string(),
        };
        if entries.is_empty() {
            return format!("{open}{close}");
        }
        match indent {
            None => format!("{open}{}{close}", entries.join(item)),
            Some(unit) => {
                let pad = unit.repeat(depth + 1);
                format!(
                    "{open}\n{pad}{}\n{}{close}",
                    entries.join(&format!("{item}\n{pad}")),
                    unit.repeat(depth)
                )
            }
        }
    }
    Ok(render(
        &value,
        ascii,
        indent.as_deref(),
        &item_separator,
        &key_separator,
        sort_keys,
        0,
    ))
}

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

        // Store chat template separately (we'll create env on demand).
        // Newer HF checkpoints ship the template as a standalone
        // chat_template.jinja next to tokenizer.json instead of a field in
        // tokenizer_config.json; accept both, preferring the config field.
        let chat_template = config.chat_template.clone().or_else(|| {
            path_ref
                .parent()
                .map(|parent| parent.join("chat_template.jinja"))
                .filter(|candidate| candidate.exists())
                .and_then(|candidate| std::fs::read_to_string(candidate).ok())
        });

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
        self.decode_with_options(tokens, true)
    }

    /// Decode with explicit special-token handling. Set `skip_special_tokens`
    /// to false when protocol markers such as tool-call XML are meaningful.
    pub fn decode_with_options(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special_tokens)
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
        let result = self.render_chat_template(messages, &serde_json::Map::new(), false)?;
        // Retain historic normalization for legacy callers. Explicit template
        // options use the exact rendered bytes instead.
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

    /// Render the checkpoint template exactly, preserving all whitespace.
    /// Options such as `enable_thinking` and `tools` are application inputs;
    /// they cannot replace messages or reserved generation/template fields.
    pub fn apply_chat_template_with_options(
        &self,
        messages: &[ChatMessage],
        options: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<String> {
        self.render_chat_template(messages, options, true)
    }

    fn render_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &serde_json::Map<String, serde_json::Value>,
        hf_whitespace: bool,
    ) -> Result<String> {
        let template_str = self.chat_template.as_ref()
            .ok_or_else(|| Error::Other("no chat template available (missing tokenizer_config.json with chat_template field)".to_string()))?;

        // Create environment and template on demand
        let mut env = Environment::new();
        if hf_whitespace {
            env.set_trim_blocks(true);
            env.set_lstrip_blocks(true);
        }
        env.add_filter("tojson", hf_tojson);
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        // HF chat templates are written for Jinja2 and freely use Python
        // methods (str.startswith etc.); enable minijinja's pycompat shims.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
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
        let mut context = serde_json::json!({
            "messages": messages,
            "bos_token": bos,
            "eos_token": eos,
            "add_generation_prompt": true,
        });

        for (key, value) in options {
            if [
                "messages",
                "bos_token",
                "eos_token",
                "add_generation_prompt",
            ]
            .contains(&key.as_str())
            {
                return Err(Error::Other(format!(
                    "reserved chat template option: {key}"
                )));
            }
            context[key] = value.clone();
        }

        let result = tmpl
            .render(context)
            .map_err(|e| Error::Other(format!("template render error: {e}")))?;

        Ok(result)
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

    fn render_json_filter(template: &str) -> String {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson);
        let data: serde_json::Value = serde_json::from_str(
            r#"{"z":"中文<&'😀","a":{"b":true,"a":null},"v":[1,2.0,0.00001,1e20]}"#,
        )
        .unwrap();
        env.render_str(template, serde_json::json!({"data":data}))
            .unwrap()
    }

    #[test]
    fn hf_json_unicode_order_and_spacing_match_python_defaults() {
        assert_eq!(
            render_json_filter("{{ data | tojson }}"),
            r#"{"z": "中文<&'😀", "a": {"b": true, "a": null}, "v": [1, 2.0, 1e-05, 1e+20]}"#
        );
        assert_eq!(
            render_json_filter("{{ data.z | tojson(ensure_ascii=True) }}"),
            r#""\u4e2d\u6587<&'\ud83d\ude00""#
        );
        assert_eq!(
            render_json_filter("{{ data.a | tojson(False, None, [',', ':'], True) }}"),
            r#"{"a":null,"b":true}"#
        );
        assert_eq!(
            render_json_filter("{{ data.a | tojson(indent=2, sort_keys=True) }}"),
            "{\n  \"a\": null,\n  \"b\": true\n}"
        );
        assert_eq!(
            render_json_filter("{{ data.a | tojson(indent='\t', separators=[',', ':']) }}"),
            "{\n\t\"b\":true,\n\t\"a\":null\n}"
        );
        assert_eq!(
            render_json_filter("{{ data.a | tojson(indent=-1) }}"),
            "{\n\"b\": true,\n\"a\": null\n}"
        );
    }

    #[test]
    fn hf_json_rejects_unknown_or_duplicate_options() {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson);
        for source in [
            "{{ {}|tojson(unsupported=True) }}",
            "{{ {}|tojson(True, ensure_ascii=False) }}",
            "{{ {}|tojson(separators=[':']) }}",
        ] {
            assert!(env.render_str(source, ()).is_err());
        }
    }

    #[test]
    fn explicit_template_uses_hf_block_whitespace_without_changing_legacy() {
        let tokenizer = Tokenizer {
            inner: HfTokenizer::new(WordLevel::default()),
            config: TokenizerConfig::default(),
            chat_template: Some("before\n   {% if true %}\nafter\n   {% endif %}\nend".into()),
        };
        assert_eq!(
            tokenizer
                .apply_chat_template_with_options(&[], &serde_json::Map::new())
                .unwrap(),
            "before\nafter\nend"
        );
        assert_eq!(
            tokenizer.apply_chat_template(&[]).unwrap(),
            "before\n   \nafter\n   \nend\n"
        );
    }

    #[test]
    fn explicit_template_options_preserve_exact_prompt_bytes() {
        let tokenizer = Tokenizer {
            inner: HfTokenizer::new(WordLevel::default()),
            config: TokenizerConfig::default(),
            chat_template: Some("{{ messages[0].content }}\n\n{% if enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}".into()),
        };
        let messages = [ChatMessage::user("hello")];
        let mut options = serde_json::Map::new();
        options.insert("enable_thinking".into(), false.into());
        assert_eq!(
            tokenizer
                .apply_chat_template_with_options(&messages, &options)
                .unwrap(),
            "hello\n\n<think>\n\n</think>\n\n"
        );
        assert_eq!(tokenizer.apply_chat_template(&messages).unwrap(), "hello\n");
        options.insert("messages".into(), serde_json::json!([]));
        assert!(tokenizer
            .apply_chat_template_with_options(&messages, &options)
            .is_err());
    }

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

    #[test]
    fn explicit_decode_keeps_tool_protocol_special_tokens() {
        let mut tokenizer = Tokenizer {
            inner: HfTokenizer::new(WordLevel::default()),
            config: TokenizerConfig::default(),
            chat_template: None,
        };
        tokenizer
            .inner
            .add_special_tokens(&[AddedToken::from("<function>", true)]);
        let id = tokenizer.token_to_id("<function>").unwrap();
        assert_eq!(tokenizer.decode(&[id]).unwrap(), "");
        assert_eq!(tokenizer.decode_with_options(&[id], true).unwrap(), "");
        assert_eq!(
            tokenizer.decode_with_options(&[id], false).unwrap(),
            "<function>"
        );
    }
}

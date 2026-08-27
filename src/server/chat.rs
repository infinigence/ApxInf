//! `/v1/chat/completions` request preparation for the multimodal contract.
//!
//! Scope is exactly the frozen image contract: one user message whose content
//! is one `data:image/png;base64` image_url part followed by one text part,
//! `temperature = 0`, `stream = false`,
//! `chat_template_kwargs.enable_thinking = false`. Anything else is a 400.
//!
//! The chat template is rendered manually for this restricted input instead
//! of through a Jinja engine. For a single user message with one image part,
//! one text part, `enable_thinking=false` and `add_generation_prompt=true`,
//! the checkpoint's `chat_template.jinja` reduces to the exact string
//! produced below; the offline oracle verifies the resulting `input_ids`
//! against `Qwen3VLProcessor.apply_chat_template` byte for byte.

use apxinf_model::qwen35::IMAGE_TOKEN_ID;
use apxinf_model::MultimodalPayload;
use apxinf_tokenizer::Tokenizer;

use crate::server::image;

/// Per-request generation budget cap, over the contract's 32.
const MAX_COMPLETION_CAP: usize = 1024;

pub struct ChatPreprocessor {
    tokenizer: Tokenizer,
    max_model_len: usize,
}

pub struct PreparedChat {
    pub input_ids: Vec<u32>,
    pub multimodal: MultimodalPayload,
    pub max_new_tokens: usize,
}

#[derive(Debug)]
pub struct ChatInvalid(pub String);

impl ChatPreprocessor {
    pub fn new(tokenizer: Tokenizer, max_model_len: usize) -> Self {
        Self {
            tokenizer,
            max_model_len,
        }
    }

    pub fn from_model_dir(
        model_dir: &std::path::Path,
        max_model_len: usize,
    ) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|error| format!("chat tokenizer load: {error}"))?;
        Ok(Self::new(tokenizer, max_model_len))
    }

    /// Render the chat template for the restricted single-image request.
    fn render_prompt(text: &str) -> String {
        let content = format!("<|vision_start|><|image_pad|><|vision_end|>{text}");
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            content.trim()
        )
    }

    pub fn decode(&self, output_ids: &[u32]) -> Result<String, String> {
        self.tokenizer
            .decode(output_ids)
            .map_err(|error| format!("chat decode: {error}"))
    }

    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedChat, ChatInvalid> {
        let invalid = |message: &str| ChatInvalid(message.to_string());
        let value: serde_json::Value = serde_json::from_slice(raw)
            .map_err(|error| ChatInvalid(format!("request body is not valid JSON: {error}")))?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid("request body must be a JSON object"))?;

        if let Some(stream) = object.get("stream") {
            if stream.as_bool() != Some(false) {
                return Err(invalid("stream must be false for chat completions"));
            }
        }
        if let Some(temperature) = object.get("temperature") {
            let t = temperature
                .as_f64()
                .ok_or_else(|| invalid("temperature must be a number"))?;
            if t != 0.0 {
                return Err(invalid("only temperature 0 is supported"));
            }
        }
        match object
            .get("chat_template_kwargs")
            .and_then(|kwargs| kwargs.get("enable_thinking"))
            .and_then(serde_json::Value::as_bool)
        {
            Some(false) => {}
            _ => {
                return Err(invalid(
                    "chat_template_kwargs.enable_thinking must be false",
                ))
            }
        }
        let max_new_tokens = match object
            .get("max_completion_tokens")
            .or_else(|| object.get("max_tokens"))
        {
            None => 32,
            Some(value) => {
                let budget = value
                    .as_u64()
                    .ok_or_else(|| invalid("max_completion_tokens must be a positive integer"))?
                    as usize;
                if budget == 0 || budget > MAX_COMPLETION_CAP {
                    return Err(ChatInvalid(format!(
                        "max_completion_tokens must be in [1, {MAX_COMPLETION_CAP}]"
                    )));
                }
                budget
            }
        };

        let messages = object
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid("messages must be an array"))?;
        if messages.len() != 1 {
            return Err(invalid(
                "exactly one user message is supported for image chat completions",
            ));
        }
        let message = &messages[0];
        if message.get("role").and_then(serde_json::Value::as_str) != Some("user") {
            return Err(invalid("the message role must be user"));
        }
        let parts = message
            .get("content")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| invalid("message content must be an array of parts"))?;
        if parts.len() != 2 {
            return Err(invalid(
                "message content must be one image_url part followed by one text part",
            ));
        }
        let image_part = &parts[0];
        if image_part.get("type").and_then(serde_json::Value::as_str) != Some("image_url") {
            return Err(invalid("the first content part must be an image_url part"));
        }
        let url = match image_part.get("image_url") {
            Some(serde_json::Value::String(url)) => url.as_str(),
            Some(serde_json::Value::Object(image_url)) => image_url
                .get("url")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("image_url.url must be a string"))?,
            _ => return Err(invalid("image_url must be a string or an object with url")),
        };
        let text_part = &parts[1];
        if text_part.get("type").and_then(serde_json::Value::as_str) != Some("text") {
            return Err(invalid("the second content part must be a text part"));
        }
        let text = text_part
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid("text part must contain a string"))?;

        let decoded =
            image::decode_png_data_url(url).map_err(|error| ChatInvalid(error.to_string()))?;
        let payload = image::preprocess_to_payload(&decoded)
            .map_err(|error| ChatInvalid(error.to_string()))?;
        let merge_squared = (apxinf_model::qwen35::vision::VISION_MERGE
            * apxinf_model::qwen35::vision::VISION_MERGE) as u32;
        let image_tokens =
            (payload.grid[0] * payload.grid[1] * payload.grid[2] / merge_squared) as usize;

        let prompt = Self::render_prompt(text);
        let base_ids = self
            .tokenizer
            .encode(&prompt)
            .map_err(|error| ChatInvalid(format!("prompt tokenization failed: {error}")))?;
        let pad_positions: Vec<usize> = base_ids
            .iter()
            .enumerate()
            .filter(|(_, id)| **id == IMAGE_TOKEN_ID)
            .map(|(index, _)| index)
            .collect();
        if pad_positions.len() != 1 {
            return Err(invalid(
                "the rendered prompt must contain exactly one image placeholder",
            ));
        }
        let mut input_ids = Vec::with_capacity(base_ids.len() + image_tokens - 1);
        input_ids.extend_from_slice(&base_ids[..pad_positions[0]]);
        input_ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(image_tokens));
        input_ids.extend_from_slice(&base_ids[pad_positions[0] + 1..]);

        if input_ids.len() + max_new_tokens > self.max_model_len {
            return Err(ChatInvalid(format!(
                "prompt ({} tokens) plus budget ({max_new_tokens}) exceeds max_model_len {}",
                input_ids.len(),
                self.max_model_len
            )));
        }

        Ok(PreparedChat {
            input_ids,
            multimodal: payload,
            max_new_tokens,
        })
    }
}

/// OpenAI-style chat completion response body.
pub fn chat_completion_json(
    request_id: &str,
    content: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &str,
) -> Vec<u8> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    serde_json::to_vec(&serde_json::json!({
        "id": format!("chatcmpl-{request_id}"),
        "object": "chat.completion",
        "created": created,
        "model": "Qwen3.8-27B-AWQ-INT4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }))
    .expect("JSON serialization is infallible")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The service's own prompt construction (manual template render +
    /// tokenize + image-pad expansion) must reproduce
    /// `Qwen3VLProcessor.apply_chat_template(..., enable_thinking=False)`
    /// token for token. Golden files are produced by the offline oracle
    /// generator; the test skips when they are absent.
    #[test]
    fn prepared_input_ids_bit_match_hf_processor_golden() {
        let e2e_dir = std::env::var("APXINF_VISION_E2E_DIR")
            .unwrap_or_else(|_| "/tmp/apxinf-vision-e2e".to_string());
        let e2e = std::path::Path::new(&e2e_dir);
        let model_dir = std::path::Path::new("/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4");
        if !e2e.join("cases.json").is_file() || !model_dir.join("tokenizer.json").is_file() {
            eprintln!("skipping: e2e golden or tokenizer not present");
            return;
        }
        let preprocessor = ChatPreprocessor::from_model_dir(model_dir, 32_768).unwrap();
        let categories = [
            "seven_segment_ocr",
            "spatial_color",
            "bar_arithmetic",
            "object_count",
        ];
        for category in categories {
            let request = std::fs::read(e2e.join(format!("request-{category}.json"))).unwrap();
            let golden: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(e2e.join(format!("input_ids-{category}.json"))).unwrap(),
            )
            .unwrap();
            let golden_ids: Vec<u32> = golden["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let prepared = preprocessor.prepare(&request).unwrap();
            assert_eq!(
                prepared.input_ids, golden_ids,
                "{category}: prepared input_ids differ from the HF processor golden"
            );
            assert_eq!(prepared.multimodal.grid, [1, 28, 28], "{category} grid");
            assert_eq!(prepared.max_new_tokens, 32, "{category} budget");
        }
    }

    #[test]
    fn prepare_rejects_out_of_scope_requests() {
        let model_dir = std::path::Path::new("/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4");
        if !model_dir.join("tokenizer.json").is_file() {
            eprintln!("skipping: tokenizer not present");
            return;
        }
        let preprocessor = ChatPreprocessor::from_model_dir(model_dir, 32_768).unwrap();
        let tiny_png = {
            let mut encoded = Vec::new();
            let mut encoder = png::Encoder::new(&mut encoded, 448, 448);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&vec![127u8; 448 * 448 * 3])
                .unwrap();
            drop(writer);
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(&encoded)
        };
        let valid = serde_json::json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{tiny_png}")}},
                {"type": "text", "text": "hi"},
            ]}],
            "temperature": 0.0,
            "max_completion_tokens": 32,
            "stream": false,
            "chat_template_kwargs": {"enable_thinking": false},
        });
        assert!(preprocessor
            .prepare(&serde_json::to_vec(&valid).unwrap())
            .is_ok());

        let mutate = |mutator: &dyn Fn(&mut serde_json::Value)| {
            let mut value = valid.clone();
            mutator(&mut value);
            preprocessor.prepare(&serde_json::to_vec(&value).unwrap())
        };
        // stream=true, nonzero temperature, thinking on, missing image, and
        // swapped part order are all out of contract scope.
        assert!(mutate(&|v| v["stream"] = serde_json::json!(true)).is_err());
        assert!(mutate(&|v| v["temperature"] = serde_json::json!(0.7)).is_err());
        assert!(
            mutate(&|v| v["chat_template_kwargs"]["enable_thinking"] = serde_json::json!(true))
                .is_err()
        );
        assert!(mutate(&|v| {
            v["messages"][0]["content"] = serde_json::json!([{"type": "text", "text": "hi"}]);
        })
        .is_err());
        assert!(mutate(&|v| {
            let parts = v["messages"][0]["content"].as_array().unwrap().clone();
            v["messages"][0]["content"] = serde_json::json!([parts[1], parts[0]]);
        })
        .is_err());
    }

    #[test]
    fn rendered_prompt_matches_the_frozen_template_reduction() {
        let prompt = ChatPreprocessor::render_prompt("How many circles?");
        assert_eq!(
            prompt,
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>How many circles?\
             <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        // Trailing whitespace in the text is trimmed, matching Jinja's |trim.
        let trimmed = ChatPreprocessor::render_prompt("count \n");
        assert!(trimmed.contains("<|vision_end|>count<|im_end|>"));
    }
}

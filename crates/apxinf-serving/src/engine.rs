use std::collections::{BTreeSet, VecDeque};
use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_model::{
    AutoModel, GenerationOptions, ImageInput, LlmInput, LoadOptions, LoadedModel, SamplingMode,
};
use apxinf_tokenizer::{ChatTemplateMessage, ChatToolCall, ChatToolCallFunction, Tokenizer};
use serde_json::{Map, Value};

use crate::error::ApiError;
use crate::protocol::{ChatCompletionMessage, ToolDefinition};
use crate::tool_calling::ToolChoice;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct EngineRequest {
    pub messages: Vec<ChatCompletionMessage>,
    pub images: Vec<apxinf_model::vision_preprocess::DecodedImage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub chat_template_kwargs: Map<String, Value>,
    pub enable_thinking: bool,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_k: Option<i64>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// Stop at a model-native tool terminator when the caller permits one call.
    pub stop_after_tool_call: bool,
}

#[derive(Clone, Debug)]
pub struct EngineOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub reused_prompt_tokens: usize,
    pub completion_tokens: usize,
    pub hit_length: bool,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub generation_tps: Option<f64>,
    pub total_latency_ms: Option<f64>,
}

pub trait ChatEngine {
    fn vision_contract(&self) -> Option<&apxinf_model::vision_preprocess::VisionContract> {
        None
    }

    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError>;

    /// Generate a completion while publishing decoded text as it becomes
    /// stable. Returning `false` from the callback means the HTTP consumer is
    /// gone; implementations should stop publishing and may cancel inference
    /// when their runtime supports cooperative cancellation.
    ///
    /// The default keeps third-party/test engines source compatible. It emits
    /// their complete result once, while native ApxInf engines override this
    /// method to forward model token callbacks during generation.
    fn complete_stream(
        &mut self,
        request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        let output = self.complete(request)?;
        if !output.text.is_empty() {
            let _ = on_text(&output.text);
        }
        Ok(output)
    }
}

pub struct ApxInfEngineConfig {
    pub model_path: PathBuf,
    pub device: Device,
    pub load_options: LoadOptions,
    pub default_max_tokens: usize,
    pub max_input_tokens: usize,
}

pub struct ApxInfChatEngine {
    tokenizer: Tokenizer,
    model: LoadedModel,
    default_max_tokens: usize,
    max_input_tokens: usize,
    /// Cache token IDs by structured request and template options. This cache
    /// only avoids repeated rendering/tokenization; it retains no model KV state.
    prompt_token_cache: VecDeque<(String, Vec<u32>)>,
}

const PROMPT_TOKEN_CACHE_ENTRIES: usize = 4;

impl ApxInfChatEngine {
    pub(crate) fn load(config: ApxInfEngineConfig) -> Result<Self, ApiError> {
        let tokenizer_path = config.model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            ApiError::unavailable(format!(
                "failed to load tokenizer {}: {error}",
                tokenizer_path.display()
            ))
        })?;
        if !tokenizer.has_chat_template() {
            return Err(ApiError::unavailable(
                "the checkpoint does not provide a chat template",
            ));
        }
        let model = AutoModel::load_model(config.device, &config.model_path, &config.load_options)
            .map_err(|error| ApiError::unavailable(format!("failed to load model: {error}")))?;
        model.text_capabilities().map_err(|error| {
            ApiError::unavailable(format!("model is not a text model: {error}"))
        })?;
        if model.sequence_capacity().is_none() {
            return Err(ApiError::unavailable(
                "model does not declare its runtime sequence capacity",
            ));
        }
        if model
            .text_capabilities()
            .map_err(|e| ApiError::unavailable(e.to_string()))?
            .image
            && model.vision_contract().is_none()
        {
            return Err(ApiError::unavailable(
                "image-capable model has no validated checkpoint vision contract",
            ));
        }
        eprintln!(
            "model limits: sequence_capacity={:?} vision={:?}",
            model.sequence_capacity(),
            model.vision_contract()
        );
        Ok(Self {
            tokenizer,
            model,
            default_max_tokens: config.default_max_tokens,
            max_input_tokens: config.max_input_tokens,
            prompt_token_cache: VecDeque::new(),
        })
    }
}

impl ChatEngine for ApxInfChatEngine {
    fn vision_contract(&self) -> Option<&apxinf_model::vision_preprocess::VisionContract> {
        self.model.vision_contract()
    }

    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        self.complete_stream(request, &mut |_| true)
    }

    fn complete_stream(
        &mut self,
        request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        if !request.images.is_empty()
            && !self
                .model
                .text_capabilities()
                .map_err(|e| ApiError::internal(e.to_string()))?
                .image
        {
            return Err(ApiError::invalid(
                "the loaded model does not support image input",
                Some("messages".into()),
            ));
        }
        let mut messages = to_template_messages(&request.messages)?;
        apply_tool_choice_instruction(&mut messages, &request.tool_choice);
        let mut options = request.chat_template_kwargs;
        for reserved in [
            "messages",
            "bos_token",
            "eos_token",
            "add_generation_prompt",
            "tools",
        ] {
            if options.contains_key(reserved) {
                return Err(ApiError::invalid(
                    format!("reserved chat_template_kwargs key: {reserved}"),
                    Some("chat_template_kwargs".to_owned()),
                ));
            }
        }
        options.insert("enable_thinking".into(), request.enable_thinking.into());
        if !request.enable_thinking {
            // Preserve Qwen's disabled-thinking control span when rendering
            // the assistant turn back into conversation history.
            options
                .entry("preserve_thinking")
                .or_insert(Value::Bool(true));
        }
        let rendered_tools = match &request.tool_choice {
            ToolChoice::None => Vec::new(),
            ToolChoice::Function(name) => request
                .tools
                .iter()
                .filter(|tool| tool.function.name == *name)
                .cloned()
                .collect(),
            ToolChoice::Auto | ToolChoice::Required => request.tools.clone(),
        };
        if !rendered_tools.is_empty() {
            options.insert(
                "tools".into(),
                serde_json::to_value(&rendered_tools)
                    .map_err(|error| ApiError::internal(error.to_string()))?,
            );
        }
        let max_tokens = request
            .max_tokens
            .max(1)
            .min(self.default_max_tokens.max(1));
        let capacity = self
            .model
            .sequence_capacity()
            .ok_or_else(|| ApiError::unavailable("model sequence capacity is unknown"))?;
        let max_prompt = capacity
            .checked_sub(max_tokens)
            .ok_or_else(|| {
                ApiError::invalid(
                    format!("max_tokens={max_tokens} exceeds model KV capacity={capacity}"),
                    Some("max_tokens".into()),
                )
            })?
            .min(self.max_input_tokens);
        let prompt_key = serde_json::to_string(&(&messages, &options))
            .map_err(|error| ApiError::internal(error.to_string()))?;
        let rendered_tokens = if request.images.is_empty() {
            if let Some(index) = self
                .prompt_token_cache
                .iter()
                .position(|(key, _)| key == &prompt_key)
            {
                let entry = self
                    .prompt_token_cache
                    .remove(index)
                    .expect("located cache entry");
                let tokens = entry.1.clone();
                self.prompt_token_cache.push_front(entry);
                tokens
            } else {
                let prompt = self
                    .tokenizer
                    .apply_chat_template_messages_with_options(&messages, &options)
                    .map_err(|e| ApiError::invalid(e.to_string(), Some("messages".into())))?;
                let tokens = self
                    .tokenizer
                    .encode(&prompt)
                    .map_err(|e| ApiError::invalid(e.to_string(), Some("messages".into())))?;
                if self.prompt_token_cache.len() >= PROMPT_TOKEN_CACHE_ENTRIES {
                    self.prompt_token_cache.pop_back();
                }
                self.prompt_token_cache
                    .push_front((prompt_key, tokens.clone()));
                tokens
            }
        } else {
            let prompt = self
                .tokenizer
                .apply_chat_template_messages_with_options(&messages, &options)
                .map_err(|e| ApiError::invalid(e.to_string(), Some("messages".into())))?;
            self.tokenizer
                .encode(&prompt)
                .map_err(|e| ApiError::invalid(e.to_string(), Some("messages".into())))?
        };
        let prepared_image = if request.images.is_empty() {
            None
        } else {
            Some(
                self.model
                    .prepare_images(&rendered_tokens, &request.images, max_prompt)
                    .map_err(|e| ApiError::invalid(e.to_string(), Some("messages".into())))?,
            )
        };
        let prompt_tokens = prepared_image
            .as_ref()
            .map(|p| p.tokens.clone())
            .unwrap_or(rendered_tokens);
        if prompt_tokens
            .len()
            .checked_add(max_tokens)
            .is_none_or(|n| n > capacity)
        {
            return Err(ApiError::invalid(
                format!(
                    "prompt={} + max_tokens={max_tokens} exceeds model KV capacity={capacity}",
                    prompt_tokens.len()
                ),
                Some("max_tokens".into()),
            ));
        }
        if prompt_tokens.is_empty() {
            return Err(ApiError::invalid(
                "chat template produced an empty prompt",
                Some("messages".to_owned()),
            ));
        }
        if prompt_tokens.len() > self.max_input_tokens {
            return Err(ApiError::invalid(
                format!(
                    "prompt has {} tokens, exceeding the configured limit of {}",
                    prompt_tokens.len(),
                    self.max_input_tokens
                ),
                Some("messages".to_owned()),
            ));
        }
        let default_eos = self
            .model
            .generation_defaults()
            .map_err(|error| ApiError::internal(error.to_string()))?
            .eos_token_ids
            .clone();
        let mut eos_token_ids = default_eos
            .clone()
            .or_else(|| self.tokenizer.eos_token_id().map(|id| vec![id]));
        let mut generation_eos_token_ids = eos_token_ids.clone();
        if request.stop_after_tool_call && !request.tools.is_empty() {
            // Stop a single-tool response at its native closing token when
            // the tokenizer represents the delimiter as one exact token.
            // Other tokenizers retain normal EOS handling.
            if let Ok(ids) = self.tokenizer.encode("</tool_call>") {
                if let [tool_end] = ids.as_slice() {
                    // An unknown-token fallback is not a tool terminator.
                    if self
                        .tokenizer
                        .decode_with_options(&ids, false)
                        .ok()
                        .as_deref()
                        == Some("</tool_call>")
                    {
                        let stop_ids = generation_eos_token_ids.get_or_insert_with(Vec::new);
                        if !stop_ids.contains(tool_end) {
                            stop_ids.push(*tool_end);
                        }
                        // Preserve the delimiter in both streamed and final text:
                        // the structured-tool parser needs the closing tag.
                        if let Some(hidden_eos) = eos_token_ids.as_mut() {
                            hidden_eos.retain(|id| id != tool_end);
                        }
                    }
                }
            }
        }
        let generation = GenerationOptions {
            max_new_tokens: Some(max_tokens),
            eos_token_ids: generation_eos_token_ids,
            sampling_mode: request.temperature.map(|temperature| {
                if temperature == 0.0 {
                    SamplingMode::Greedy
                } else {
                    SamplingMode::Random
                }
            }),
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            repetition_penalty: None,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            seed: request.seed,
            return_logprob: Some(false),
        };
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let mut consumer_connected = true;
        let input = match prepared_image.as_ref() {
            Some(image) => {
                LlmInput::with_image(&prompt_tokens, ImageInput::new(&image.pixels, &image.grid))
            }
            None => LlmInput::text(&prompt_tokens),
        };
        let output = self
            .model
            .generate_streaming_with_options(input, &generation, |token| {
                if !consumer_connected
                    || decode_error.is_some()
                    || eos_token_ids
                        .as_ref()
                        .is_some_and(|eos| eos.contains(&token.token_id))
                {
                    return;
                }
                match decoder.push(token.token_id) {
                    Ok(Some(delta)) => consumer_connected = on_text(&delta),
                    Ok(None) => {}
                    Err(error) => decode_error = Some(error),
                }
            })
            .map_err(|error| ApiError::internal(format!("generation failed: {error}")))?;
        if let Some(error) = decode_error {
            return Err(ApiError::internal(format!(
                "incremental decode failed: {error}"
            )));
        }
        let mut output_ids = output.token_ids();
        let stopped_on_eos = output_ids.last().is_some_and(|token| {
            eos_token_ids
                .as_ref()
                .is_some_and(|eos| eos.contains(token))
        });
        if stopped_on_eos {
            output_ids.pop();
        }
        let text = self
            .tokenizer
            .decode_with_options(&output_ids, false)
            .map_err(|error| ApiError::internal(format!("decode failed: {error}")))?;
        if consumer_connected {
            if let Some(delta) = decoder.finish(&text).map_err(ApiError::internal)? {
                let _ = on_text(&delta);
            }
        }
        let completion_tokens = output.profile.output_tokens();
        Ok(EngineOutput {
            text,
            prompt_tokens: output.profile.input_tokens(),
            reused_prompt_tokens: 0,
            completion_tokens,
            hit_length: !stopped_on_eos && completion_tokens >= max_tokens,
            ttft_ms: output.profile.ttft_ms(),
            tpot_ms: output.profile.tpot_ms(),
            generation_tps: output.profile.generation_tps(),
            total_latency_ms: output.profile.total_latency_ms(),
        })
    }
}

/// Cumulative decoding is required because byte-fallback tokens may not be
/// valid UTF-8 in isolation. We publish only the prefix that was unchanged by
/// the following token, then flush the final suffix when generation ends. This
/// usually adds a one-token stabilization delay. Byte-fallback sequences can
/// need more tokens: defer replacement characters until final decoding so a
/// later byte never repairs text that has already been published.
struct IncrementalTextDecoder<'a> {
    tokenizer: &'a Tokenizer,
    token_ids: Vec<u32>,
    previous: String,
    emitted_bytes: usize,
}

impl<'a> IncrementalTextDecoder<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            previous: String::new(),
            emitted_bytes: 0,
        }
    }

    fn push(&mut self, token_id: u32) -> Result<Option<String>, String> {
        self.token_ids.push(token_id);
        let current = self
            .tokenizer
            .decode_with_options(&self.token_ids, false)
            .map_err(|error| error.to_string())?;
        let mut stable_end = common_prefix_bytes(&self.previous, &current);
        // Two incomplete prefixes of a three/four-byte code point can both
        // decode to the same replacement character. Equality alone does not
        // make that character stable. A genuine replacement in final output
        // is still preserved by finish().
        if let Some(replacement) = self.previous[..stable_end].find('\u{fffd}') {
            stable_end = replacement;
        }
        if stable_end < self.emitted_bytes {
            return Err("tokenizer changed text that was already streamed".into());
        }
        let delta = self.previous[self.emitted_bytes..stable_end].to_owned();
        self.emitted_bytes = stable_end;
        self.previous = current;
        Ok((!delta.is_empty()).then_some(delta))
    }

    fn finish(&mut self, text: &str) -> Result<Option<String>, String> {
        if !text.is_char_boundary(self.emitted_bytes)
            || !text
                .get(..self.emitted_bytes)
                .is_some_and(|prefix| self.previous.starts_with(prefix))
        {
            return Err("final decode changed text that was already streamed".into());
        }
        let delta = text[self.emitted_bytes..].to_owned();
        self.emitted_bytes = text.len();
        Ok((!delta.is_empty()).then_some(delta))
    }
}

fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.char_indices()
        .zip(right.chars())
        .take_while(|((_, left), right)| left == right)
        .last()
        .map_or(0, |((offset, character), _)| offset + character.len_utf8())
}

fn to_template_messages(
    messages: &[ChatCompletionMessage],
) -> Result<Vec<ChatTemplateMessage>, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid(
            "messages must not be empty",
            Some("messages".to_owned()),
        ));
    }
    let mut seen_calls = BTreeSet::new();
    let mut output = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let param = || Some(format!("messages[{index}]"));
        match message.role.as_str() {
            "system" | "user" => {
                if message.tool_calls.is_some() || message.tool_call_id.is_some() {
                    return Err(ApiError::invalid(
                        "system/user messages cannot carry tool call fields",
                        param(),
                    ));
                }
                let content = match message.content.as_ref() {
                    Some(Value::Array(parts)) if message.role == "user" => {
                        Value::Array(parts.clone())
                    }
                    _ => Value::String(text_content(message.content.as_ref(), false, index)?),
                };
                let mut turn = ChatTemplateMessage::text(&message.role, "");
                turn.content = Some(content);
                output.push(turn);
            }
            "assistant" => {
                if message.tool_call_id.is_some() {
                    return Err(ApiError::invalid(
                        "assistant messages cannot carry tool_call_id",
                        param(),
                    ));
                }
                let content = optional_text_content(message.content.as_ref(), index)?;
                let mut calls = Vec::new();
                for call in message.tool_calls.as_deref().unwrap_or_default() {
                    if call.kind != "function" || call.id.is_empty() {
                        return Err(ApiError::invalid(
                            "assistant tool calls require a nonempty id and type 'function'",
                            param(),
                        ));
                    }
                    let arguments: Value =
                        serde_json::from_str(&call.function.arguments).map_err(|error| {
                            ApiError::invalid(
                                format!("assistant tool arguments are not JSON: {error}"),
                                param(),
                            )
                        })?;
                    let arguments = arguments.as_object().cloned().ok_or_else(|| {
                        ApiError::invalid(
                            "assistant tool arguments must encode a JSON object",
                            param(),
                        )
                    })?;
                    if !seen_calls.insert(call.id.clone()) {
                        return Err(ApiError::invalid(
                            format!("duplicate tool call id '{}'", call.id),
                            param(),
                        ));
                    }
                    calls.push(ChatToolCall {
                        id: Some(call.id.clone()),
                        kind: Some("function".into()),
                        function: ChatToolCallFunction {
                            name: call.function.name.clone(),
                            arguments,
                        },
                    });
                }
                output.push(if calls.is_empty() {
                    ChatTemplateMessage {
                        role: "assistant".into(),
                        content: content.map(Value::String),
                        reasoning_content: message.reasoning_content.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                    }
                } else {
                    ChatTemplateMessage::assistant_tool_calls(
                        content,
                        message.reasoning_content.clone(),
                        calls,
                    )
                });
            }
            "tool" => {
                let id = message.tool_call_id.as_deref().ok_or_else(|| {
                    ApiError::invalid("tool messages require tool_call_id", param())
                })?;
                if !seen_calls.contains(id) {
                    return Err(ApiError::invalid(
                        format!("dangling tool_call_id '{id}'"),
                        param(),
                    ));
                }
                if message.tool_calls.is_some() {
                    return Err(ApiError::invalid(
                        "tool messages cannot carry tool_calls",
                        param(),
                    ));
                }
                output.push(ChatTemplateMessage::tool(
                    text_content(message.content.as_ref(), true, index)?,
                    id,
                ));
            }
            other => {
                return Err(ApiError::invalid(
                    format!("unsupported message role '{other}'"),
                    Some(format!("messages[{index}].role")),
                ));
            }
        }
    }
    Ok(output)
}

fn apply_tool_choice_instruction(messages: &mut Vec<ChatTemplateMessage>, choice: &ToolChoice) {
    let instruction = match choice {
        ToolChoice::Required => Some(
            "You must call at least one of the provided tools. Do not answer without a tool call."
                .to_owned(),
        ),
        ToolChoice::Function(name) => Some(format!(
                "You must call the function '{name}'. Do not call another function and do not answer without that tool call."
            )),
        ToolChoice::None | ToolChoice::Auto => None,
    };
    let Some(instruction) = instruction else {
        return;
    };
    if let Some(system) = messages
        .first_mut()
        .filter(|message| message.role == "system")
    {
        let existing = system
            .content
            .as_ref()
            .and_then(Value::as_str)
            .unwrap_or("");
        system.content = Some(Value::String(if existing.is_empty() {
            instruction
        } else {
            format!("{existing}\n\n{instruction}")
        }));
    } else {
        messages.insert(0, ChatTemplateMessage::text("system", instruction));
    }
}

fn optional_text_content(
    content: Option<&Value>,
    index: usize,
) -> Result<Option<String>, ApiError> {
    match content {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(ApiError::invalid(
            "assistant message content must be a string",
            Some(format!("messages[{index}].content")),
        )),
    }
}

fn text_content(
    content: Option<&Value>,
    allow_json: bool,
    index: usize,
) -> Result<String, ApiError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(value @ (Value::Object(_) | Value::Array(_))) if allow_json => {
            serde_json::to_string(value).map_err(|error| ApiError::internal(error.to_string()))
        }
        _ => Err(ApiError::invalid(
            "message content must be text",
            Some(format!("messages[{index}].content")),
        )),
    }
}

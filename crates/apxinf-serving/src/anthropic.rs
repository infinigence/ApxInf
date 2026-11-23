use axum::{
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::error::ApiError;
use crate::protocol::{
    ChatCompletionMessage, FunctionCall, FunctionDefinition, ToolCall, ToolDefinition, Usage,
};

#[derive(Clone, Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub max_tokens: Option<usize>,
    pub messages: Vec<MessagesMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub tools: Vec<MessagesToolDefinition>,
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub thinking: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MessagesMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MessagesToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "empty_schema")]
    pub input_schema: Value,
}

fn empty_schema() -> Value {
    json!({"type":"object","properties":{}})
}

#[derive(Debug)]
pub struct ConvertedMessagesRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatCompletionMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: bool,
    pub stream: bool,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub enable_thinking: bool,
}

impl MessagesRequest {
    pub fn into_openai(self) -> Result<ConvertedMessagesRequest, ApiError> {
        let max_tokens = self.max_tokens.ok_or_else(|| {
            ApiError::invalid("max_tokens is required", Some("max_tokens".to_owned()))
        })?;
        if self
            .stop_sequences
            .as_ref()
            .is_some_and(|sequences| !sequences.is_empty())
        {
            return Err(ApiError::invalid(
                "stop_sequences are not implemented",
                Some("stop_sequences".to_owned()),
            ));
        }

        let mut messages = Vec::new();
        if let Some(system) = self.system.as_ref() {
            let text = text_blocks(system, "system")?;
            if !text.is_empty() {
                messages.push(text_message("system", text));
            }
        }
        for (index, message) in self.messages.iter().enumerate() {
            convert_message(message, index, &mut messages)?;
        }

        let tools = self
            .tools
            .into_iter()
            .map(|tool| ToolDefinition {
                kind: "function".into(),
                function: FunctionDefinition {
                    name: tool.name,
                    description: tool.description,
                    parameters: tool.input_schema,
                    strict: None,
                },
            })
            .collect();
        let (tool_choice, parallel_tool_calls) = convert_tool_choice(self.tool_choice)?;
        let enable_thinking = self
            .thinking
            .as_ref()
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            .is_some_and(|kind| kind == "enabled");

        Ok(ConvertedMessagesRequest {
            model: self.model,
            messages,
            tools,
            tool_choice,
            parallel_tool_calls,
            stream: self.stream,
            max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            enable_thinking,
        })
    }
}

fn convert_message(
    message: &MessagesMessage,
    message_index: usize,
    output: &mut Vec<ChatCompletionMessage>,
) -> Result<(), ApiError> {
    match message.role.as_str() {
        "user" => convert_user_content(&message.content, message_index, output),
        "assistant" => convert_assistant_content(&message.content, message_index, output),
        other => Err(ApiError::invalid(
            format!("unsupported message role '{other}'"),
            Some(format!("messages[{message_index}].role")),
        )),
    }
}

fn convert_user_content(
    content: &Value,
    message_index: usize,
    output: &mut Vec<ChatCompletionMessage>,
) -> Result<(), ApiError> {
    if let Some(text) = content.as_str() {
        output.push(text_message("user", text.to_owned()));
        return Ok(());
    }
    let blocks = content.as_array().ok_or_else(|| {
        ApiError::invalid(
            "user content must be text or an array of content blocks",
            Some(format!("messages[{message_index}].content")),
        )
    })?;
    let mut pending_text = Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => pending_text.push(serde_json::json!({"type":"text", "text":block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        block_error(message_index, block_index, "text must be a string")
                    })?
                    .to_owned()})),
            Some("image") => {
                let source = block.get("source").ok_or_else(|| {
                    block_error(message_index, block_index, "image needs a source")
                })?;
                if source.get("type").and_then(Value::as_str) != Some("base64") {
                    return Err(block_error(
                        message_index,
                        block_index,
                        "only base64 image sources are supported",
                    ));
                }
                let media_type = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        block_error(message_index, block_index, "image source needs media_type")
                    })?;
                let data = source.get("data").and_then(Value::as_str).ok_or_else(|| {
                    block_error(message_index, block_index, "image source needs base64 data")
                })?;
                pending_text.push(serde_json::json!({"type":"image_url","image_url":{"url":format!("data:{media_type};base64,{data}")}}));
            }
            Some("tool_result") => {
                flush_user_text(&mut pending_text, output);
                let tool_call_id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| {
                        block_error(message_index, block_index, "tool_use_id must be a string")
                    })?;
                let result = tool_result_text(block.get("content"), message_index, block_index)?;
                output.push(ChatCompletionMessage {
                    role: "tool".into(),
                    content: Some(Value::String(result)),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: Some(tool_call_id.to_owned()),
                });
            }
            Some(other) => {
                return Err(block_error(
                    message_index,
                    block_index,
                    format!("unsupported user content block type '{other}'"),
                ));
            }
            None => {
                return Err(block_error(
                    message_index,
                    block_index,
                    "content block needs type",
                ))
            }
        }
    }
    flush_user_text(&mut pending_text, output);
    Ok(())
}

fn convert_assistant_content(
    content: &Value,
    message_index: usize,
    output: &mut Vec<ChatCompletionMessage>,
) -> Result<(), ApiError> {
    if let Some(text) = content.as_str() {
        output.push(text_message("assistant", text.to_owned()));
        return Ok(());
    }
    let blocks = content.as_array().ok_or_else(|| {
        ApiError::invalid(
            "assistant content must be text or an array of content blocks",
            Some(format!("messages[{message_index}].content")),
        )
    })?;
    let mut text = Vec::new();
    let mut tool_calls = Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push(
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        block_error(message_index, block_index, "text must be a string")
                    })?
                    .to_owned(),
            ),
            Some("tool_use") => {
                let id = required_block_string(block, "id", message_index, block_index)?;
                let name = required_block_string(block, "name", message_index, block_index)?;
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                if !input.is_object() {
                    return Err(block_error(
                        message_index,
                        block_index,
                        "tool_use input must be an object",
                    ));
                }
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".into(),
                    function: FunctionCall {
                        name,
                        arguments: serde_json::to_string(&input)
                            .map_err(|error| ApiError::internal(error.to_string()))?,
                    },
                });
            }
            Some(other) => {
                return Err(block_error(
                    message_index,
                    block_index,
                    format!("unsupported assistant content block type '{other}'"),
                ));
            }
            None => {
                return Err(block_error(
                    message_index,
                    block_index,
                    "content block needs type",
                ))
            }
        }
    }
    output.push(ChatCompletionMessage {
        role: "assistant".into(),
        content: (!text.is_empty()).then(|| Value::String(text.join("\n"))),
        reasoning_content: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id: None,
    });
    Ok(())
}

fn text_message(role: &str, text: String) -> ChatCompletionMessage {
    ChatCompletionMessage {
        role: role.into(),
        content: Some(Value::String(text)),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn flush_user_text(text: &mut Vec<Value>, output: &mut Vec<ChatCompletionMessage>) {
    if !text.is_empty() {
        let blocks = std::mem::take(text);
        let content = if blocks.iter().all(|block| block["type"] == "text") {
            Value::String(
                blocks
                    .iter()
                    .map(|block| block["text"].as_str().unwrap())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        } else {
            Value::Array(blocks)
        };
        let mut message = text_message("user", String::new());
        message.content = Some(content);
        output.push(message);
    }
}

fn text_blocks(value: &Value, param: &str) -> Result<String, ApiError> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_owned());
    }
    let blocks = value.as_array().ok_or_else(|| {
        ApiError::invalid(
            "system must be text or an array of text blocks",
            Some(param.to_owned()),
        )
    })?;
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            if block.get("type").and_then(Value::as_str) != Some("text") {
                return Err(ApiError::invalid(
                    "system content blocks must have type 'text'",
                    Some(format!("{param}[{index}]")),
                ));
            }
            block
                .get("text")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| {
                    ApiError::invalid(
                        "system text must be a string",
                        Some(format!("{param}[{index}].text")),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|parts| parts.join("\n"))
}

fn tool_result_text(
    content: Option<&Value>,
    message_index: usize,
    block_index: usize,
) -> Result<String, ApiError> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(_)) => text_blocks(
            content.expect("matched Some"),
            &format!("messages[{message_index}].content[{block_index}].content"),
        ),
        Some(value) => {
            serde_json::to_string(value).map_err(|error| ApiError::internal(error.to_string()))
        }
    }
}

fn required_block_string(
    block: &Value,
    field: &str,
    message_index: usize,
    block_index: usize,
) -> Result<String, ApiError> {
    block
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            block_error(
                message_index,
                block_index,
                format!("{field} must be a nonempty string"),
            )
        })
}

fn block_error(message_index: usize, block_index: usize, message: impl Into<String>) -> ApiError {
    ApiError::invalid(
        message,
        Some(format!("messages[{message_index}].content[{block_index}]")),
    )
}

fn convert_tool_choice(raw: Option<Value>) -> Result<(Option<Value>, bool), ApiError> {
    let Some(raw) = raw else {
        return Ok((None, true));
    };
    let object = raw.as_object().ok_or_else(|| {
        ApiError::invalid(
            "tool_choice must be an object",
            Some("tool_choice".to_owned()),
        )
    })?;
    let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        ApiError::invalid(
            "tool_choice.type must be a string",
            Some("tool_choice".to_owned()),
        )
    })?;
    let parallel = !object
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let choice = match kind {
        "auto" => json!("auto"),
        "any" => json!("required"),
        "none" => json!("none"),
        "tool" => {
            let name = object.get("name").and_then(Value::as_str).ok_or_else(|| {
                ApiError::invalid(
                    "tool_choice.name must be a string for type 'tool'",
                    Some("tool_choice".to_owned()),
                )
            })?;
            json!({"type":"function","function":{"name":name}})
        }
        other => {
            return Err(ApiError::invalid(
                format!("unsupported tool_choice.type '{other}'"),
                Some("tool_choice".to_owned()),
            ));
        }
    };
    Ok((Some(choice), parallel))
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub content: Vec<Value>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: MessagesUsage,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MessagesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cache_read_input_tokens: usize,
}

pub fn response_content(
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
) -> Result<Vec<Value>, ApiError> {
    let mut blocks = Vec::new();
    if let Some(content) = content.filter(|content| !content.is_empty()) {
        blocks.push(json!({"type":"text","text":content}));
    }
    for call in tool_calls {
        let input: Value = serde_json::from_str(&call.function.arguments)
            .map_err(|error| ApiError::internal(format!("tool arguments are not JSON: {error}")))?;
        blocks.push(json!({
            "type":"tool_use",
            "id":call.id,
            "name":call.function.name,
            "input":input,
        }));
    }
    Ok(blocks)
}

pub fn stop_reason(openai_reason: &str) -> &'static str {
    match openai_reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

impl From<Usage> for MessagesUsage {
    fn from(usage: Usage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            cache_read_input_tokens: usage
                .prompt_tokens_details
                .map_or(0, |details| details.cached_tokens),
        }
    }
}

pub struct MessagesApiError(pub ApiError);

impl From<ApiError> for MessagesApiError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

impl IntoResponse for MessagesApiError {
    fn into_response(self) -> Response {
        let error = self.0;
        (
            error.status,
            Json(json!({
                "type":"error",
                "error":{"type":error.error_type,"message":error.message},
            })),
        )
            .into_response()
    }
}

pub fn empty_options() -> Map<String, Value> {
    Map::new()
}

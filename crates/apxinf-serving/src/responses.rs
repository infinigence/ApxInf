//! OpenAI Responses API wire types and their conversion to the internal request.
//!
//! Codex CLI 0.172.0 dropped `wire_api = "chat"`, so its only provider wire API
//! is the Responses API. These types parse the request body Codex actually
//! sends and turn it into the same `ChatCompletionMessage`/`ToolDefinition`
//! pair the chat and Messages protocols hand the engine.

use std::collections::BTreeSet;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::protocol::{
    ChatCompletionMessage, FunctionCall, FunctionDefinition, ToolCall, ToolDefinition, Usage,
};

fn default_parallel_tool_calls() -> bool {
    true
}

fn default_parameters() -> Value {
    json!({"type":"object","properties":{}})
}

/// A `POST /v1/responses` body. Unknown fields are ignored, never rejected.
#[derive(Clone, Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub stream: bool,
    pub input: ResponsesInput,
    #[serde(default)]
    pub tools: Vec<ResponsesTool>,
    pub tool_choice: Option<Value>,
    #[serde(default = "default_parallel_tool_calls")]
    pub parallel_tool_calls: bool,
    /// Accepted and ignored: `{effort, summary}`.
    #[allow(dead_code)]
    pub reasoning: Option<Value>,
    #[allow(dead_code)]
    pub store: Option<bool>,
    #[allow(dead_code)]
    pub include: Option<Vec<String>>,
    #[allow(dead_code)]
    pub prompt_cache_key: Option<String>,
    #[allow(dead_code)]
    pub client_metadata: Option<Value>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Rejected: this server keeps no response store to resume from.
    pub previous_response_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: MessageContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: FunctionCallOutputContent,
    },
    /// Skipped: this server streams no encrypted reasoning back to the client.
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default)]
        #[allow(dead_code)]
        id: Option<String>,
    },
    /// Any other item `type` is skipped rather than rejected.
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(default)]
        image_url: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum FunctionCallOutputContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesTool {
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default = "default_parameters")]
        parameters: Value,
        #[serde(default)]
        strict: Option<bool>,
    },
    #[serde(rename = "namespace")]
    Namespace {
        #[serde(default)]
        tools: Vec<ResponsesTool>,
    },
    /// `web_search` and every other non-function tool is dropped.
    #[serde(other)]
    Other,
}

#[derive(Debug)]
pub struct ConvertedResponsesRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatCompletionMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: bool,
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub enable_thinking: bool,
}

impl ResponsesRequest {
    pub fn into_chat_completion(self) -> Result<ConvertedResponsesRequest, ApiError> {
        if self.previous_response_id.is_some() {
            return Err(ApiError::invalid(
                "previous_response_id is not supported; resend the full conversation",
                Some("previous_response_id".to_owned()),
            ));
        }
        let mut messages = Vec::new();
        if let Some(instructions) = self.instructions.filter(|text| !text.is_empty()) {
            messages.push(text_message("system", instructions));
        }
        match self.input {
            ResponsesInput::Text(text) => messages.push(text_message("user", text)),
            ResponsesInput::Items(items) => {
                for (index, item) in items.into_iter().enumerate() {
                    convert_item(item, index, &mut messages)?;
                }
            }
        }
        Ok(ConvertedResponsesRequest {
            model: self.model,
            messages,
            tools: convert_tools(self.tools),
            tool_choice: self.tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            stream: self.stream,
            max_tokens: self.max_output_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            enable_thinking: false,
        })
    }
}

fn convert_item(
    item: ResponsesInputItem,
    index: usize,
    output: &mut Vec<ChatCompletionMessage>,
) -> Result<(), ApiError> {
    match item {
        ResponsesInputItem::Message { role, content } => {
            let role = match role.as_str() {
                "developer" | "system" => "system",
                "user" => "user",
                "assistant" => "assistant",
                other => {
                    return Err(ApiError::invalid(
                        format!("unsupported message role '{other}'"),
                        Some(format!("input[{index}].role")),
                    ))
                }
            };
            output.push(ChatCompletionMessage {
                role: role.to_owned(),
                content: Some(message_content_value(content)),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            });
        }
        ResponsesInputItem::FunctionCall {
            call_id,
            name,
            arguments,
        } => output.push(ChatCompletionMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: call_id,
                kind: "function".into(),
                function: FunctionCall { name, arguments },
            }]),
            tool_call_id: None,
        }),
        ResponsesInputItem::FunctionCallOutput {
            call_id,
            output: body,
        } => {
            output.push(ChatCompletionMessage {
                role: "tool".into(),
                content: Some(Value::String(function_output_text(body))),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: Some(call_id),
            });
        }
        ResponsesInputItem::Reasoning { .. } | ResponsesInputItem::Other => {}
    }
    Ok(())
}

/// Engine message content expects a plain string for every role but a user
/// message whose blocks carry an image, which stays an array of blocks.
fn message_content_value(content: MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => Value::String(text),
        MessageContent::Parts(parts) => {
            let mut blocks = Vec::new();
            for part in parts {
                match part {
                    ContentPart::InputText { text } | ContentPart::OutputText { text } => {
                        blocks.push(json!({"type":"text","text":text}));
                    }
                    ContentPart::InputImage { image_url } => {
                        if let Some(image_url) = image_url {
                            blocks.push(json!({"type":"image_url","image_url":{"url":image_url}}));
                        }
                    }
                    ContentPart::Other => {}
                }
            }
            if blocks.iter().all(|block| block["type"] == "text") {
                Value::String(join_text_blocks(&blocks))
            } else {
                Value::Array(blocks)
            }
        }
    }
}

fn function_output_text(output: FunctionCallOutputContent) -> String {
    match output {
        FunctionCallOutputContent::Text(text) => text,
        FunctionCallOutputContent::Parts(parts) => {
            let blocks = parts
                .into_iter()
                .filter_map(|part| match part {
                    ContentPart::InputText { text } | ContentPart::OutputText { text } => {
                        Some(json!({"type":"text","text":text}))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            join_text_blocks(&blocks)
        }
    }
}

fn join_text_blocks(blocks: &[Value]) -> String {
    blocks
        .iter()
        .filter_map(|block| block["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn convert_tools(tools: Vec<ResponsesTool>) -> Vec<ToolDefinition> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for tool in tools {
        match tool {
            ResponsesTool::Function {
                name,
                description,
                parameters,
                strict,
            } => push_function(&mut out, &mut seen, name, description, parameters, strict),
            ResponsesTool::Namespace { tools } => {
                for nested in tools {
                    if let ResponsesTool::Function {
                        name,
                        description,
                        parameters,
                        strict,
                    } = nested
                    {
                        push_function(&mut out, &mut seen, name, description, parameters, strict);
                    }
                }
            }
            ResponsesTool::Other => {}
        }
    }
    out
}

fn push_function(
    out: &mut Vec<ToolDefinition>,
    seen: &mut BTreeSet<String>,
    name: String,
    description: Option<String>,
    parameters: Value,
    strict: Option<bool>,
) {
    if !seen.insert(name.clone()) {
        eprintln!("dropped duplicate tool name: {name}");
        return;
    }
    out.push(ToolDefinition {
        kind: "function".into(),
        function: FunctionDefinition {
            name,
            description,
            parameters,
            strict,
        },
    });
}

/// The `tools` array echoed back inside every response object.
pub fn response_tools(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            let mut value = json!({
                "type": "function",
                "name": tool.function.name,
                "parameters": tool.function.parameters,
            });
            if let Some(description) = &tool.function.description {
                value["description"] = json!(description);
            }
            if let Some(strict) = tool.function.strict {
                value["strict"] = json!(strict);
            }
            value
        })
        .collect()
}

/// The response object carried by `response.created`, `response.in_progress`
/// and every terminal event. Callers overwrite `status`, `output` and `usage`.
pub fn responses_response_base(
    id: &str,
    created: u64,
    model: &str,
    tool_choice: &Value,
    tools: &[Value],
    parallel_tool_calls: bool,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "model": model,
        "output": [],
        "parallel_tool_calls": parallel_tool_calls,
        "tool_choice": tool_choice,
        "tools": tools,
        "usage": Value::Null,
    })
}

pub fn responses_usage(usage: &Usage) -> Value {
    json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.prompt_tokens_details.map_or(0, |details| details.cached_tokens),
        },
        "output_tokens_details": {"reasoning_tokens": 0},
    })
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

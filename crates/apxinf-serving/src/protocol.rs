use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: Option<Value>,
    #[serde(default = "default_parallel_tool_calls")]
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<i64>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub n: Option<usize>,
    pub stop: Option<Value>,
    pub logprobs: Option<bool>,
    pub response_format: Option<Value>,
    pub enable_thinking: Option<bool>,
    /// Reserved for the later speculative-decoding runtime integration.
    /// Requests setting this to true are rejected.
    #[serde(default)]
    pub disable_spec_decode: bool,
    #[serde(default)]
    pub chat_template_kwargs: Map<String, Value>,
}

fn default_parallel_tool_calls() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// OpenAI represents arguments as a JSON object encoded into a string.
    pub arguments: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default = "default_parameters")]
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

fn default_parameters() -> Value {
    serde_json::json!({"type":"object","properties":{}})
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

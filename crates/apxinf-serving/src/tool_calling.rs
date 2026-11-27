//! Strict request validation and model output/tool-call parsing.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::error::ApiError;
use crate::protocol::{FunctionCall, FunctionDefinition, ToolCall, ToolDefinition};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolFormat {
    #[default]
    Auto,
    Qwen3Xml,
    Inkling2Json,
}

impl ToolFormat {
    pub fn from_cli_name(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "qwen3" | "qwen3_xml" | "qwen3-xml" => Ok(Self::Qwen3Xml),
            "inkling2" | "inkling2_json" | "inkling2-json" => Ok(Self::Inkling2Json),
            other => Err(format!(
                "unsupported tool format '{other}'; use auto, qwen3, or inkling2"
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(String),
}

#[derive(Clone, Debug)]
pub struct ToolRegistry {
    definitions: BTreeMap<String, FunctionDefinition>,
}

#[derive(Debug, PartialEq)]
pub struct ParsedAssistant {
    pub reasoning_content: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl ToolRegistry {
    pub fn validate(tools: &[ToolDefinition]) -> Result<Self, ApiError> {
        let mut definitions = BTreeMap::new();
        for (index, tool) in tools.iter().enumerate() {
            if tool.kind != "function" {
                return Err(ApiError::invalid(
                    format!("tools[{index}].type must be 'function'"),
                    Some(format!("tools[{index}].type")),
                ));
            }
            let name = tool.function.name.trim();
            if !valid_identifier(name) {
                return Err(ApiError::invalid(
                    format!("tools[{index}].function.name is invalid"),
                    Some(format!("tools[{index}].function.name")),
                ));
            }
            validate_root_schema(&tool.function.parameters).map_err(|message| {
                ApiError::invalid(
                    format!("tools[{index}].function.parameters: {message}"),
                    Some(format!("tools[{index}].function.parameters")),
                )
            })?;
            if definitions
                .insert(name.to_owned(), tool.function.clone())
                .is_some()
            {
                return Err(ApiError::invalid(
                    format!("duplicate tool name: {name}"),
                    Some("tools".to_owned()),
                ));
            }
        }
        Ok(Self { definitions })
    }

    pub fn names(&self) -> BTreeSet<&str> {
        self.definitions.keys().map(String::as_str).collect()
    }

    fn get(&self, name: &str) -> Option<&FunctionDefinition> {
        self.definitions.get(name)
    }
}

pub fn validate_tool_choice(
    raw: Option<&Value>,
    registry: &ToolRegistry,
) -> Result<ToolChoice, ApiError> {
    let names = registry.names();
    match raw {
        None => Ok(if names.is_empty() {
            ToolChoice::None
        } else {
            ToolChoice::Auto
        }),
        Some(Value::String(value)) if value == "none" => Ok(ToolChoice::None),
        Some(Value::String(value)) if value == "auto" => {
            require_tools(&names, "auto")?;
            Ok(ToolChoice::Auto)
        }
        Some(Value::String(value)) if value == "required" => {
            require_tools(&names, "required")?;
            Ok(ToolChoice::Required)
        }
        Some(Value::String(value)) if names.contains(value.as_str()) => {
            Ok(ToolChoice::Function(value.clone()))
        }
        Some(Value::Object(object)) => {
            if object.get("type").and_then(Value::as_str) != Some("function") {
                return Err(ApiError::invalid(
                    "tool_choice.type must be 'function'",
                    Some("tool_choice".to_owned()),
                ));
            }
            let name = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiError::invalid(
                        "tool_choice.function.name must be a string",
                        Some("tool_choice".to_owned()),
                    )
                })?;
            if !names.contains(name) {
                return Err(ApiError::invalid(
                    format!("unknown forced tool name: {name}"),
                    Some("tool_choice".to_owned()),
                ));
            }
            Ok(ToolChoice::Function(name.to_owned()))
        }
        _ => Err(ApiError::invalid(
            "tool_choice must be 'none', 'auto', 'required', a tool name, or a function choice",
            Some("tool_choice".to_owned()),
        )),
    }
}

fn require_tools(names: &BTreeSet<&str>, choice: &str) -> Result<(), ApiError> {
    if names.is_empty() {
        Err(ApiError::invalid(
            format!("tool_choice '{choice}' requires at least one tool"),
            Some("tool_choice".to_owned()),
        ))
    } else {
        Ok(())
    }
}

pub fn parse_assistant_output(
    text: &str,
    format: ToolFormat,
    registry: &ToolRegistry,
    choice: &ToolChoice,
) -> Result<ParsedAssistant, ApiError> {
    match format {
        ToolFormat::Auto => {
            if looks_like_inkling2(text) {
                parse_inkling2_json(text, registry, choice)
            } else {
                parse_qwen3_xml(text, registry, choice)
            }
        }
        ToolFormat::Qwen3Xml => parse_qwen3_xml(text, registry, choice),
        ToolFormat::Inkling2Json => parse_inkling2_json(text, registry, choice),
    }
}

pub fn looks_like_inkling2(text: &str) -> bool {
    [
        "<|message_model|>",
        "<|content_thinking|>",
        "<|content_text|>",
        "<|content_invoke_tool_json|>",
        "<|content_model_end_sampling|>",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

/// Parse the XML-like protocol emitted by Qwen3 provider templates.
///
/// Natural-language content may precede the first call. Nothing except
/// whitespace may follow the final call, which prevents accidental execution of
/// malformed or prompt-injected suffixes.
pub fn parse_qwen3_xml(
    text: &str,
    registry: &ToolRegistry,
    choice: &ToolChoice,
) -> Result<ParsedAssistant, ApiError> {
    let first = text.find("<tool_call>");
    if first.is_none() {
        if contains_tool_marker(text) {
            return Err(protocol_error("malformed Qwen tool-call marker"));
        }
        if matches!(choice, ToolChoice::Required | ToolChoice::Function(_)) {
            return Err(protocol_error(
                "the selected tool_choice did not produce a tool call",
            ));
        }
        return Ok(ParsedAssistant {
            reasoning_content: None,
            content: nonempty(text),
            tool_calls: Vec::new(),
        });
    }
    if matches!(choice, ToolChoice::None) {
        return Err(protocol_error(
            "the model emitted a tool call while tool_choice was 'none'",
        ));
    }

    let first = first.expect("checked above");
    // Formatting before a tool-only response is not a text block. Preserve
    // every byte when the prefix actually contains assistant text.
    let prefix = (!text[..first].trim().is_empty()).then(|| text[..first].to_owned());
    let mut cursor = Cursor::new(&text[first..]);
    let mut calls = Vec::new();
    while cursor.remaining().trim_start().starts_with("<tool_call>") {
        cursor.skip_ws();
        cursor.expect("<tool_call>")?;
        cursor.skip_ws();
        cursor.expect("<function=")?;
        let name = cursor.take_until(">")?.trim().to_owned();
        if !valid_identifier(&name) {
            return Err(protocol_error("invalid function name in tool call"));
        }
        let definition = match registry.get(&name) {
            Some(definition) => Some(definition),
            None => {
                return Err(protocol_error(format!(
                    "model called unknown function '{name}'"
                )))
            }
        };
        if let ToolChoice::Function(required) = choice {
            if required != &name {
                return Err(protocol_error(format!(
                    "model called '{name}' but tool_choice requires '{required}'"
                )));
            }
        }

        let mut arguments = Map::new();
        loop {
            cursor.skip_ws();
            if cursor.remaining().starts_with("</function>") {
                break;
            }
            cursor.expect("<parameter=")?;
            let parameter = cursor.take_until(">")?.trim().to_owned();
            if !valid_identifier(&parameter) {
                return Err(protocol_error("invalid parameter name in tool call"));
            }
            let raw = cursor.take_until("</parameter>")?.trim();
            let declared = definition.and_then(|d| parameter_type(&d.parameters, &parameter));
            let value = if declared == Some("string") {
                // Qwen XML carries every value as text; keep it a string when the
                // schema says so (e.g. radius=5000 for a string-typed parameter).
                Value::String(raw.to_owned())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
            };
            if arguments.insert(parameter.clone(), value).is_some() {
                return Err(protocol_error(format!(
                    "duplicate parameter '{parameter}' in call to '{name}'"
                )));
            }
        }
        cursor.expect("</function>")?;
        cursor.skip_ws();
        cursor.expect("</tool_call>")?;
        if let Some(definition) = definition {
            validate_arguments(&name, &arguments, &definition.parameters)
                .map_err(protocol_error)?;
        }
        calls.push(ToolCall {
            id: String::new(),
            kind: "function".into(),
            function: FunctionCall {
                name,
                arguments: serde_json::to_string(&arguments)
                    .map_err(|error| protocol_error(error.to_string()))?,
            },
        });
        cursor.skip_ws();
    }
    if !cursor.remaining().is_empty() {
        return Err(protocol_error(
            "only whitespace may follow the final Qwen tool call",
        ));
    }
    if calls.is_empty() {
        return Err(protocol_error("no complete Qwen tool call was found"));
    }

    Ok(ParsedAssistant {
        reasoning_content: None,
        content: prefix,
        tool_calls: calls,
    })
}

/// Parse Inkling2's structured message protocol:
/// `<|message_model|><|content_thinking|>...<|end_message|>`
/// `<|message_model|><|content_text|>...<|end_message|>`
/// `<|message_model|>{fn}<|content_invoke_tool_json|>{"name":"...","args":{...}}<|end_message|>`
/// optionally followed by `<|content_model_end_sampling|>` / `<|endoftext|>`.
pub fn parse_inkling2_json(
    text: &str,
    registry: &ToolRegistry,
    choice: &ToolChoice,
) -> Result<ParsedAssistant, ApiError> {
    const MSG: &str = "<|message_model|>";
    const END_MSG: &str = "<|end_message|>";
    const THINKING: &str = "<|content_thinking|>";
    const TEXT: &str = "<|content_text|>";
    const TOOL_JSON: &str = "<|content_invoke_tool_json|>";
    const END_SAMPLING: &str = "<|content_model_end_sampling|>";
    const ENDOFTEXT: &str = "<|endoftext|>";

    if !looks_like_inkling2(text) {
        if matches!(choice, ToolChoice::Required | ToolChoice::Function(_)) {
            return Err(protocol_error(
                "the selected tool_choice did not produce a tool call",
            ));
        }
        return Ok(ParsedAssistant {
            reasoning_content: None,
            content: nonempty(text),
            tool_calls: Vec::new(),
        });
    }

    let mut cursor = Cursor::new(text);
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut calls = Vec::new();

    loop {
        cursor.skip_ws();
        let rem = cursor.remaining();
        if rem.is_empty() {
            break;
        }
        if rem.starts_with(END_SAMPLING) {
            cursor.expect(END_SAMPLING)?;
            continue;
        }
        if rem.starts_with(ENDOFTEXT) {
            cursor.expect(ENDOFTEXT)?;
            continue;
        }
        if !rem.starts_with(MSG) {
            return Err(protocol_error(
                "unexpected trailing content in Inkling2 output",
            ));
        }
        cursor.expect(MSG)?;
        // Graceful truncation: when the model hits max_tokens before emitting
        // <|end_message|>, accept the remaining text as the block body instead
        // of erroring. This lets callers see hit_length=true with partial
        // reasoning/content rather than a 500.
        let body = match cursor.take_until(END_MSG) {
            Ok(b) => b,
            Err(_) if cursor.remaining().is_empty() => "",
            Err(_) => {
                let b = cursor.remaining();
                cursor.offset += b.len();
                b
            }
        };
        if let Some(thought) = body.strip_prefix(THINKING) {
            reasoning.push_str(thought);
        } else if let Some(text_part) = body.strip_prefix(TEXT) {
            content.push_str(text_part);
        } else if let Some(pos) = body.find(TOOL_JSON) {
            if matches!(choice, ToolChoice::None) {
                return Err(protocol_error(
                    "the model emitted a tool call while tool_choice was 'none'",
                ));
            }
            let header_name = body[..pos].trim();
            if !valid_identifier(header_name) {
                return Err(protocol_error("invalid function name in Inkling2 tool call"));
            }
            let raw_json = body[pos + TOOL_JSON.len()..].trim();
            let value: Value = serde_json::from_str(raw_json)
                .map_err(|error| protocol_error(format!("invalid Inkling2 tool JSON: {error}")))?;
            let object = value
                .as_object()
                .ok_or_else(|| protocol_error("Inkling2 tool call must be a JSON object"))?;
            if object.keys().any(|key| key != "name" && key != "args") {
                return Err(protocol_error(
                    "Inkling2 tool call contains an unknown field",
                ));
            }
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| protocol_error("Inkling2 tool call is missing 'name'"))?
                .to_owned();
            if name != header_name {
                return Err(protocol_error(format!(
                    "Inkling2 tool header '{header_name}' does not match JSON name '{name}'"
                )));
            }
            let definition = registry.get(&name).ok_or_else(|| {
                protocol_error(format!("model called unknown function '{name}'"))
            })?;
            if let ToolChoice::Function(required) = choice {
                if required != &name {
                    return Err(protocol_error(format!(
                        "model called '{name}' but tool_choice requires '{required}'"
                    )));
                }
            }
            let arguments = match object.get("args") {
                None => Map::new(),
                Some(Value::Object(map)) => map.clone(),
                Some(_) => {
                    return Err(protocol_error(
                        "Inkling2 tool call 'args' must be a JSON object",
                    ))
                }
            };
            validate_arguments(&name, &arguments, &definition.parameters)
                .map_err(protocol_error)?;
            calls.push(ToolCall {
                id: String::new(),
                kind: "function".into(),
                function: FunctionCall {
                    name,
                    arguments: serde_json::to_string(&arguments)
                        .map_err(|error| protocol_error(error.to_string()))?,
                },
            });
        } else {
            return Err(protocol_error(
                "Inkling2 message block has unrecognized content tag",
            ));
        }
    }

    if calls.is_empty() && matches!(choice, ToolChoice::Required | ToolChoice::Function(_)) {
        return Err(protocol_error(
            "the selected tool_choice did not produce a tool call",
        ));
    }

    Ok(ParsedAssistant {
        reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
        content: (!content.is_empty()).then_some(content),
        tool_calls: calls,
    })
}

fn parameter_type<'s>(schema: &'s Value, parameter: &str) -> Option<&'s str> {
    schema
        .get("properties")?
        .get(parameter)?
        .get("type")?
        .as_str()
}

fn protocol_error(message: impl Into<String>) -> ApiError {
    ApiError::internal(format!("tool protocol error: {}", message.into()))
}

fn contains_tool_marker(text: &str) -> bool {
    [
        "<tool_call",
        "</tool_call",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn nonempty(text: &str) -> Option<String> {
    (!text.is_empty()).then(|| text.to_owned())
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
}

fn validate_root_schema(schema: &Value) -> Result<(), String> {
    let object = schema.as_object().ok_or("schema must be an object")?;
    if object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("object")
        != "object"
    {
        return Err("top-level schema type must be 'object'".into());
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or("schema.properties must be an object")?;
        for (name, property) in properties {
            if !valid_identifier(name) {
                return Err(format!("invalid property name '{name}'"));
            }
            if !property.is_object() {
                return Err(format!("property '{name}' schema must be an object"));
            }
        }
    }
    if let Some(required) = object.get("required") {
        let required = required
            .as_array()
            .ok_or("schema.required must be an array")?;
        if required.iter().any(|name| !name.is_string()) {
            return Err("schema.required entries must be strings".into());
        }
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for name in required {
            let name = name.as_str().expect("validated above");
            if !properties.contains_key(name) {
                return Err(format!("required property '{name}' is not declared"));
            }
        }
    }
    Ok(())
}

fn validate_arguments(
    name: &str,
    arguments: &Map<String, Value>,
    schema: &Value,
) -> Result<(), String> {
    let schema = schema.as_object().ok_or("tool schema must be an object")?;
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in required {
        let required = required
            .as_str()
            .ok_or("tool schema required entries must be strings")?;
        if !arguments.contains_key(required) {
            return Err(format!(
                "call to '{name}' is missing required parameter '{required}'"
            ));
        }
    }
    let reject_extra = schema.get("additionalProperties").and_then(Value::as_bool) == Some(false);
    for (parameter, value) in arguments {
        let Some(property_schema) = properties.get(parameter) else {
            if reject_extra {
                return Err(format!(
                    "call to '{name}' has unknown parameter '{parameter}'"
                ));
            }
            continue;
        };
        validate_value(value, property_schema)
            .map_err(|reason| format!("parameter '{parameter}' in call to '{name}' {reason}"))?;
    }
    Ok(())
}

fn validate_value(value: &Value, schema: &Value) -> Result<(), String> {
    let schema = schema
        .as_object()
        .ok_or_else(|| "has a non-object schema".to_owned())?;
    if let Some(kind) = schema.get("type").and_then(Value::as_str) {
        let valid = match kind {
            "string" => value.is_string(),
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "object" => value.is_object(),
            "array" => value.is_array(),
            "null" => value.is_null(),
            _ => return Err(format!("uses unsupported schema type '{kind}'")),
        };
        if !valid {
            return Err(format!("must have type '{kind}'"));
        }
        match (kind, value) {
            ("object", Value::Object(object)) => validate_object_value(object, schema)?,
            ("array", Value::Array(items)) => {
                if let Some(item_schema) = schema.get("items") {
                    for (index, item) in items.iter().enumerate() {
                        validate_value(item, item_schema)
                            .map_err(|reason| format!("array item {index} {reason}"))?;
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(choices) = schema.get("enum").and_then(Value::as_array) {
        if !choices.contains(value) {
            return Err("is not one of the allowed enum values".into());
        }
    }
    Ok(())
}

fn validate_object_value(
    object: &Map<String, Value>,
    schema: &Map<String, Value>,
) -> Result<(), String> {
    let properties = schema
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for required in schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let required = required
            .as_str()
            .ok_or_else(|| "object required entries must be strings".to_owned())?;
        if !object.contains_key(required) {
            return Err(format!("is missing required property '{required}'"));
        }
    }
    let reject_extra = schema.get("additionalProperties").and_then(Value::as_bool) == Some(false);
    for (name, value) in object {
        match properties.get(name) {
            Some(property_schema) => validate_value(value, property_schema)
                .map_err(|reason| format!("property '{name}' {reason}"))?,
            None if reject_extra => return Err(format!("has unknown property '{name}'")),
            None => {}
        }
    }
    Ok(())
}

struct Cursor<'a> {
    source: &'a str,
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(source: &'a str) -> Self {
        Self { source, offset: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.source[self.offset..]
    }

    fn skip_ws(&mut self) {
        let bytes = self.remaining().as_bytes();
        let count = bytes
            .iter()
            .take_while(|byte| byte.is_ascii_whitespace())
            .count();
        self.offset += count;
    }

    fn expect(&mut self, expected: &str) -> Result<(), ApiError> {
        if self.remaining().starts_with(expected) {
            self.offset += expected.len();
            Ok(())
        } else {
            Err(protocol_error(format!("expected '{expected}'")))
        }
    }

    fn take_until(&mut self, delimiter: &str) -> Result<&'a str, ApiError> {
        let end = self
            .remaining()
            .find(delimiter)
            .ok_or_else(|| protocol_error(format!("missing '{delimiter}'")))?;
        let value = &self.remaining()[..end];
        self.offset += end + delimiter.len();
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> ToolRegistry {
        ToolRegistry::validate(&[ToolDefinition {
            kind: "function".into(),
            function: FunctionDefinition {
                name: "weather.get".into(),
                description: None,
                parameters: serde_json::json!({
                    "type":"object",
                    "properties": {
                        "city":{"type":"string"},
                        "days":{"type":"integer"}
                    },
                    "required":["city"],
                    "additionalProperties":false
                }),
                strict: Some(true),
            },
        }])
        .unwrap()
    }

    #[test]
    fn parses_qwen_xml_and_validates_typed_arguments() {
        let parsed = parse_qwen3_xml(
            "checking\n<tool_call>\n<function=weather.get>\n<parameter=city>\n上海\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
            &registry(),
            &ToolChoice::Auto,
        )
        .unwrap();
        assert_eq!(parsed.content.as_deref(), Some("checking\n"));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "weather.get");
        assert_eq!(
            serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).unwrap(),
            serde_json::json!({"city":"上海","days":2})
        );
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_trailing_data() {
        for output in [
            "<tool_call><function=weather.get></function></tool_call>",
            "<tool_call><function=unknown><parameter=city>x</parameter></function></tool_call>",
            "<tool_call><function=weather.get><parameter=city>x</parameter><parameter=city>y</parameter></function></tool_call>",
            "<tool_call><function=weather.get><parameter=city>x</parameter></function></tool_call> trailing",
            "<tool_call><function=weather.get><parameter=city>x</parameter><parameter=extra>y</parameter></function></tool_call>",
        ] {
            assert!(
                parse_qwen3_xml(output, &registry(), &ToolChoice::Auto).is_err(),
                "accepted {output}"
            );
        }
    }

    #[test]
    fn decodes_string_arguments_without_weakening_schema_validation() {
        for raw in [
            "10001",
            "true",
            "null",
            "1.5",
            "[1,2]",
            "{\"x\":1}",
            "\"quoted\"",
        ] {
            let output = format!(
                "<tool_call><function=weather.get><parameter=city>{raw}</parameter></function></tool_call>"
            );
            let parsed = parse_qwen3_xml(&output, &registry(), &ToolChoice::Auto).unwrap();
            let arguments: Value =
                serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
            assert_eq!(arguments["city"], Value::String(raw.into()));
        }
        assert!(parse_qwen3_xml(
            "<tool_call><function=weather.get><parameter=city>10001</parameter><parameter=days>no</parameter></function></tool_call>",
            &registry(),
            &ToolChoice::Auto,
        ).is_err());
    }

    #[test]
    fn keeps_plain_text_whitespace_and_ignores_only_tool_prefix_formatting() {
        for text in ["    print(1)\n", "\t\n  ", "\n中文 \t"] {
            let parsed = parse_qwen3_xml(text, &registry(), &ToolChoice::None).unwrap();
            assert_eq!(parsed.content.as_deref(), Some(text));
        }
        let call = "<tool_call><function=weather.get><parameter=city>上海</parameter></function></tool_call>";
        let parsed =
            parse_qwen3_xml(&format!(" \t\n{call}"), &registry(), &ToolChoice::Auto).unwrap();
        assert!(parsed.content.is_none());
        let parsed = parse_qwen3_xml(
            &format!(" \tchecking\n{call}"),
            &registry(),
            &ToolChoice::Auto,
        )
        .unwrap();
        assert_eq!(parsed.content.as_deref(), Some(" \tchecking\n"));
    }

    #[test]
    fn parses_multiple_calls_without_a_greedy_cross_call_match() {
        let call = "<tool_call><function=weather.get><parameter=city>上海</parameter></function></tool_call>";
        let output = format!("{call}\n{call}");
        let parsed = parse_qwen3_xml(&output, &registry(), &ToolChoice::Required).unwrap();
        assert_eq!(parsed.tool_calls.len(), 2);
        assert!(parsed.content.is_none());
    }

    #[test]
    fn rejects_partial_protocol_markers_even_when_no_call_opener_exists() {
        for output in [
            "plain </tool_call>",
            "<tool_call typo>",
            "<function=weather.get>",
            "<parameter=city>上海</parameter>",
        ] {
            assert!(
                parse_qwen3_xml(output, &registry(), &ToolChoice::Auto).is_err(),
                "accepted {output}"
            );
        }
    }

    #[test]
    fn enforces_none_required_and_forced_choices() {
        assert!(parse_qwen3_xml("plain answer", &registry(), &ToolChoice::Required).is_err());
        let call =
            "<tool_call><function=weather.get><parameter=city>x</parameter></function></tool_call>";
        assert!(parse_qwen3_xml(call, &registry(), &ToolChoice::None).is_err());
        assert!(parse_qwen3_xml(
            call,
            &registry(),
            &ToolChoice::Function("other.tool".into())
        )
        .is_err());
    }

    #[test]
    fn parses_inkling2_thinking_text_and_tool_call() {
        let output = concat!(
            "<|message_model|><|content_thinking|>Need weather.<|end_message|>",
            "<|message_model|><|content_text|>Checking now.<|end_message|>",
            "<|message_model|>weather.get<|content_invoke_tool_json|>",
            "{\"name\":\"weather.get\",\"args\":{\"city\":\"上海\",\"days\":2}}",
            "<|end_message|><|content_model_end_sampling|>"
        );
        let parsed =
            parse_assistant_output(output, ToolFormat::Auto, &registry(), &ToolChoice::Auto)
                .unwrap();
        assert_eq!(parsed.reasoning_content.as_deref(), Some("Need weather."));
        assert_eq!(parsed.content.as_deref(), Some("Checking now."));
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "weather.get");
        assert_eq!(
            serde_json::from_str::<Value>(&parsed.tool_calls[0].function.arguments).unwrap(),
            serde_json::json!({"city": "上海", "days": 2})
        );
    }

    #[test]
    fn rejects_invalid_inkling2_outputs() {
        for bad in [
            "<|message_model|>weather.get<|content_invoke_tool_json|>{\"name\":\"other\",\"args\":{\"city\":\"x\"}}<|end_message|>",
            "<|message_model|>weather.get<|content_invoke_tool_json|>{\"name\":\"weather.get\",\"args\":{\"city\":\"x\"},\"extra\":1}<|end_message|>",
            "<|message_model|><|content_text|>ok<|end_message|> trailing",
        ] {
            assert!(
                parse_inkling2_json(bad, &registry(), &ToolChoice::Auto).is_err(),
                "accepted {bad}"
            );
        }
    }

    #[test]
    fn accepts_a_message_body_truncated_before_its_end_marker() {
        // Hitting max_tokens mid-message is ordinary for a reasoning model; the
        // caller must see the partial body with the truncation, not a protocol
        // error. See `hit_length` in the completion response.
        let truncated = "<|message_model|><|content_text|>unclosed";
        let parsed = parse_inkling2_json(truncated, &registry(), &ToolChoice::Auto).unwrap();
        assert_eq!(parsed.content.as_deref(), Some("unclosed"));
        assert!(parsed.tool_calls.is_empty());

        let truncated_tool = "<|message_model|>weather.get<|content_invoke_tool_json|>{\"name\":\"weather.get\",\"args\":{\"city\":\"x\"}";
        assert!(parse_inkling2_json(truncated_tool, &registry(), &ToolChoice::Auto).is_err());
    }
}

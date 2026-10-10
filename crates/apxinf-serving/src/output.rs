//! Incremental output parsing for the declared Qwen text profile.
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text(String),
    Tool {
        id: String,
        name: String,
        input: Value,
    },
}

/// Hold only a suffix that can still become a stop sequence.
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    passed_bytes: usize,
    pub matched: Option<String>,
    pub cutoff: usize,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            passed_bytes: 0,
            matched: None,
            cutoff: 0,
        }
    }
    pub fn push(&mut self, text: &str, last: bool) -> String {
        if self.matched.is_some() {
            return String::new();
        }
        self.pending.push_str(text);
        let found = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s).map(|p| (p, s.clone())))
            .min_by_key(|v| v.0);
        if let Some((offset, stop)) = found {
            let result = self.pending[..offset].to_owned();
            self.cutoff = self.passed_bytes + offset;
            self.matched = Some(stop);
            self.pending.clear();
            return result;
        }
        let keep = if last {
            0
        } else {
            suffix_prefix_len(
                &self.pending,
                &self.stops.iter().map(String::as_str).collect::<Vec<_>>(),
            )
        };
        let split = self.pending.len() - keep;
        let result = self.pending[..split].to_owned();
        self.pending.drain(..split);
        self.passed_bytes += result.len();
        result
    }
}

fn suffix_prefix_len(value: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| {
            (1..marker.len())
                .filter(|&n| marker.is_char_boundary(n))
                .filter(|&n| value.ends_with(&marker[..n]))
        })
        .max()
        .unwrap_or(0)
}

#[derive(PartialEq)]
enum Mode {
    Text,
    Thinking,
    Tool,
}

pub struct OutputParser {
    pending: String,
    mode: Mode,
    tools: Vec<Value>,
    pub tool_count: usize,
}

impl OutputParser {
    pub fn new(tools: Vec<Value>) -> Self {
        Self {
            pending: String::new(),
            mode: Mode::Text,
            tools,
            tool_count: 0,
        }
    }

    pub fn push(&mut self, text: &str, last: bool) -> Result<Vec<Part>, String> {
        self.pending.push_str(text);
        if self.pending.len() > 262_144 {
            return Err("The output block exceeds the byte limit.".into());
        }
        let mut parts = Vec::new();
        loop {
            match self.mode {
                Mode::Text => {
                    let markers = ["<tool_call>", "<think>"];
                    let found = markers
                        .iter()
                        .filter_map(|m| self.pending.find(m).map(|i| (i, *m)))
                        .min_by_key(|v| v.0);
                    if let Some((i, marker)) = found {
                        if i > 0 {
                            parts.push(Part::Text(self.pending[..i].to_owned()));
                        }
                        self.pending.drain(..i + marker.len());
                        self.mode = if marker == markers[0] {
                            Mode::Tool
                        } else {
                            Mode::Thinking
                        };
                    } else {
                        let keep = if last {
                            0
                        } else {
                            suffix_prefix_len(&self.pending, &markers)
                        };
                        let split = self.pending.len() - keep;
                        if split > 0 {
                            parts.push(Part::Text(self.pending[..split].to_owned()));
                            self.pending.drain(..split);
                        }
                        break;
                    }
                }
                Mode::Thinking => {
                    if let Some(i) = self.pending.find("</think>") {
                        self.pending.drain(..i + "</think>".len());
                        self.mode = Mode::Text;
                    } else {
                        let keep = suffix_prefix_len(&self.pending, &["</think>"]);
                        self.pending.drain(..self.pending.len() - keep);
                        break;
                    }
                }
                Mode::Tool => {
                    if let Some(i) = self.pending.find("</tool_call>") {
                        let (name, input) = parse_tool(&self.pending[..i], &self.tools)?;
                        self.pending.drain(..i + "</tool_call>".len());
                        parts.push(Part::Tool {
                            id: format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                            name,
                            input,
                        });
                        self.tool_count += 1;
                        self.mode = Mode::Text;
                    } else {
                        break;
                    }
                }
            }
        }
        if last && self.mode != Mode::Text {
            return Err("The model output contains an incomplete structured block.".into());
        }
        Ok(parts)
    }
}

fn parse_tool(body: &str, tools: &[Value]) -> Result<(String, Value), String> {
    let body = body.trim();
    let (name, input) = if body.starts_with('{') {
        let value = crate::contracts::parse_document(body.as_bytes())
            .map_err(|_| "The model emitted invalid tool JSON.")?;
        let object = value
            .as_object()
            .ok_or("The tool call must be an object.")?;
        if object.keys().any(|k| k != "name" && k != "arguments") {
            return Err("The tool call contains an unknown field.".into());
        }
        let name = value["name"]
            .as_str()
            .ok_or("The tool call has no name.")?
            .to_owned();
        (name, value["arguments"].clone())
    } else {
        let rest = body
            .strip_prefix("<function=")
            .ok_or("The tool call has no function marker.")?;
        let end = rest.find('>').ok_or("The function marker has no end.")?;
        let name = rest[..end].to_owned();
        let schema = find_tool(tools, &name)?;
        let mut rest = rest[end + 1..].trim();
        let mut args = Map::new();
        while !rest.starts_with("</function>") {
            rest = rest
                .strip_prefix("<parameter=")
                .ok_or("The tool parameter marker is invalid.")?;
            let end = rest.find('>').ok_or("The parameter marker has no end.")?;
            let key = rest[..end].to_owned();
            rest = &rest[end + 1..];
            let end = rest
                .find("</parameter>")
                .ok_or("The parameter value has no end.")?;
            let raw = rest[..end].strip_prefix('\n').unwrap_or(&rest[..end]);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            let kind = schema["parameters"]["properties"][&key]["type"].as_str();
            let value = if kind == Some("string") {
                Value::String(raw.to_owned())
            } else {
                crate::contracts::parse_document(raw.as_bytes()).map_err(|error| {
                    format!("The tool parameter {key} contains invalid JSON: {error}")
                })?
            };
            if args.insert(key, value).is_some() {
                return Err("The tool call repeats a parameter.".into());
            }
            rest = rest[end + "</parameter>".len()..].trim();
        }
        if rest != "</function>" {
            return Err("The tool call contains trailing data.".into());
        }
        (name, Value::Object(args))
    };
    let schema = find_tool(tools, &name)?;
    if !input.is_object() {
        return Err("Tool arguments must be an object.".into());
    }
    validate_arguments(&input, &schema["parameters"])?;
    Ok((name, input))
}

fn find_tool<'a>(tools: &'a [Value], name: &str) -> Result<&'a Value, String> {
    tools
        .iter()
        .map(|t| &t["function"])
        .find(|t| t["name"].as_str() == Some(name))
        .ok_or_else(|| "The model selected an undeclared tool.".into())
}

fn validate_arguments(value: &Value, schema: &Value) -> Result<(), String> {
    if let Some(k) = schema["type"].as_str() {
        let valid = match k {
            "string" => value.is_string(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "object" => value.is_object(),
            "array" => value.is_array(),
            "null" => value.is_null(),
            _ => true,
        };
        if !valid {
            return Err(format!("A tool argument does not have type {k}."));
        }
    }
    if let Some(variants) = schema["enum"].as_array() {
        if !variants.contains(value) {
            return Err("A tool argument is outside its enum.".into());
        }
    }
    if let Some(required) = schema["required"].as_array() {
        for key in required {
            if let Some(key) = key.as_str() {
                if value.get(key).is_none() {
                    return Err(format!("The tool call omits required argument {key}."));
                }
            }
        }
    }
    if let Some(object) = value.as_object() {
        for (key, member) in object {
            if let Some(child) = schema["properties"].get(key) {
                validate_arguments(member, child)?;
            } else if schema["additionalProperties"] == false {
                return Err("The tool call contains an undeclared argument.".into());
            }
        }
    }
    if let Some(array) = value.as_array() {
        for item in array {
            validate_arguments(item, &schema["items"])?;
        }
    }
    Ok(())
}

pub fn anthropic_blocks(parts: &[Part]) -> Vec<Value> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            Part::Text(text) => {
                if let Some(Value::Object(last)) = blocks.last_mut() {
                    if last.get("type") == Some(&json!("text")) {
                        if let Some(Value::String(previous)) = last.get_mut("text") {
                            previous.push_str(text);
                            continue;
                        }
                    }
                }
                blocks.push(json!({"type":"text","text":text}));
            }
            Part::Tool { id, name, input } => {
                blocks.push(json!({"type":"tool_use","id":id,"name":name,"input":input}))
            }
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_unicode_stop_never_escapes() {
        let mut s = StopMatcher::new(vec!["停下".into()]);
        assert_eq!(s.push("前停", false), "前");
        assert_eq!(s.push("下后", false), "");
        assert_eq!(s.cutoff, 3);
        assert_eq!(s.push("ignored", true), "");
    }
    #[test]
    fn fragmented_xml_preserves_string_and_typed_arguments() {
        let tools = vec![
            json!({"function":{"name":"write","parameters":{"type":"object","required":["text"],"properties":{"text":{"type":"string"},"count":{"type":"integer"}}}}}),
        ];
        let text = "hello<tool_call>\n<function=write>\n<parameter=text>\n001\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>";
        let mut p = OutputParser::new(tools);
        let mut parts = Vec::new();
        for c in text.chars() {
            parts.extend(p.push(&c.to_string(), false).unwrap());
        }
        parts.extend(p.push("", true).unwrap());
        assert_eq!(anthropic_blocks(&parts)[0]["text"], "hello");
        assert_eq!(
            anthropic_blocks(&parts)[1]["input"],
            json!({"text":"001","count":2})
        );
    }
    #[test]
    fn invalid_tool_does_not_become_a_successful_call() {
        let mut p = OutputParser::new(vec![]);
        assert!(p
            .push(
                "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>",
                true
            )
            .is_err());
        let mut p = OutputParser::new(vec![]);
        assert!(p.push("<tool_call>{", true).is_err());
    }

    fn typed_parameter_tools() -> Vec<Value> {
        vec![json!({"function":{"name":"configure","parameters":{
            "type":"object","properties":{
                "options":{"type":"object"},
                "weights":{"type":"array","items":{"type":"number"}},
                "retries":{"type":"integer"},
                "ratio":{"type":"number"},
                "enabled":{"type":"boolean"},
                "empty":{"type":"null"},
                "label":{"type":"string"}
            }
        }}})]
    }

    #[test]
    fn tool_objects_reject_integer_literals_that_would_be_rounded() {
        for output in [
            "<tool_call><function=configure><parameter=options>{\"revision\":18446744073709551617}</parameter></function></tool_call>",
            "<tool_call>{\"name\":\"configure\",\"arguments\":{\"options\":{\"revision\":-9223372036854775809}}}</tool_call>",
        ] {
            let mut parser = OutputParser::new(typed_parameter_tools());
            let result = parser.push(output, true);
            assert!(result.is_err(), "Unsafe integer became a tool call: {result:?}");
            assert_eq!(parser.tool_count, 0);
        }
    }

    #[test]
    fn xml_object_parameter_rejects_duplicate_json_keys() {
        for raw in [
            r#"{"mode":1,"mode":2}"#,
            r#"{"nested":{"mode":1,"\u006dode":2}}"#,
        ] {
            let mut parser = OutputParser::new(typed_parameter_tools());
            let output = format!(
                "<tool_call><function=configure><parameter=options>{raw}</parameter></function></tool_call>"
            );
            let error = parser.push(&output, true).unwrap_err();
            assert!(error.contains("duplicate"), "{error}");
            assert_eq!(parser.tool_count, 0);
        }
    }

    #[test]
    fn xml_parameters_preserve_valid_json_types_and_literal_strings() {
        let mut parser = OutputParser::new(typed_parameter_tools());
        let output = concat!(
            "<tool_call><function=configure>",
            "<parameter=options>{\"mode\":2,\"nested\":{\"value\":3}}</parameter>",
            "<parameter=weights>[0.25,1,2.5]</parameter>",
            "<parameter=retries>2</parameter>",
            "<parameter=ratio>0.5</parameter>",
            "<parameter=enabled>false</parameter>",
            "<parameter=empty>null</parameter>",
            "<parameter=label>{\"mode\":1,\"mode\":2}</parameter>",
            "</function></tool_call>"
        );
        let parts = parser.push(output, true).unwrap();
        assert_eq!(
            anthropic_blocks(&parts)[0]["input"],
            json!({
                "options":{"mode":2,"nested":{"value":3}},
                "weights":[0.25,1,2.5],"retries":2,"ratio":0.5,
                "enabled":false,"empty":null,"label":"{\"mode\":1,\"mode\":2}"
            })
        );
        assert_eq!(parser.tool_count, 1);
    }

    #[test]
    fn xml_typed_parameters_propagate_parse_and_type_errors() {
        for (key, raw, expected) in [
            ("options", "{broken}", "invalid JSON"),
            ("weights", "[1,]", "invalid JSON"),
            ("retries", "two", "invalid JSON"),
            ("ratio", "NaN", "invalid JSON"),
            ("enabled", "True", "invalid JSON"),
            ("empty", "nil", "invalid JSON"),
            ("options", "[]", "type object"),
            ("weights", "{}", "type array"),
            ("retries", "1.5", "type integer"),
            ("ratio", "\"0.5\"", "type number"),
            ("enabled", "\"false\"", "type boolean"),
            ("empty", "false", "type null"),
        ] {
            let mut parser = OutputParser::new(typed_parameter_tools());
            let output = format!(
                "<tool_call><function=configure><parameter={key}>{raw}</parameter></function></tool_call>"
            );
            let error = parser.push(&output, true).unwrap_err();
            assert!(error.contains(expected), "{key}={raw}: {error}");
            assert_eq!(parser.tool_count, 0);
        }
    }
}

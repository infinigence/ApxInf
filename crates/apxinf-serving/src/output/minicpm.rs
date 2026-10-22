use serde_json::{Map, Value};

use super::{find_tool, validate_arguments};

pub(super) enum Opening {
    Complete,
    Invalid,
    Incomplete,
}

#[derive(Default)]
pub(super) struct OpeningBoundary {
    scanned: usize,
}

impl OpeningBoundary {
    pub(super) fn check(&mut self, pending: &str) -> Opening {
        while self.scanned < pending.len() {
            let remaining = &pending[self.scanned..];
            let character = remaining.chars().next().unwrap();
            if character == '"' {
                return if remaining.len() == 1 {
                    Opening::Incomplete
                } else if remaining.starts_with("\">") && self.scanned > 0 {
                    Opening::Complete
                } else {
                    Opening::Invalid
                };
            }
            if !name_character(character) {
                return Opening::Invalid;
            }
            self.scanned += character.len_utf8();
        }
        Opening::Incomplete
    }
}

#[derive(Default)]
pub(super) struct ToolBoundary {
    scanned: usize,
    in_cdata: bool,
}

impl ToolBoundary {
    pub(super) fn find_end(&mut self, pending: &str) -> Option<usize> {
        const CDATA_OPEN: &str = "<![CDATA[";
        const CDATA_CLOSE: &str = "]]>";
        const FUNCTION_CLOSE: &str = "</function>";

        while self.scanned < pending.len() {
            let remaining = &pending[self.scanned..];
            if self.in_cdata {
                if remaining.starts_with(CDATA_CLOSE) {
                    self.scanned += CDATA_CLOSE.len();
                    self.in_cdata = false;
                } else if CDATA_CLOSE.starts_with(remaining) {
                    return None;
                } else {
                    self.scanned += remaining.chars().next().unwrap().len_utf8();
                }
            } else if remaining.starts_with(CDATA_OPEN) {
                self.scanned += CDATA_OPEN.len();
                self.in_cdata = true;
            } else if CDATA_OPEN.starts_with(remaining) {
                return None;
            } else if remaining.starts_with(FUNCTION_CLOSE) {
                return Some(self.scanned);
            } else if FUNCTION_CLOSE.starts_with(remaining) {
                return None;
            } else {
                self.scanned += remaining.chars().next().unwrap().len_utf8();
            }
        }
        None
    }
}

pub(super) fn parse_tool(body: &str, tools: &[Value]) -> Result<(String, Value), String> {
    let name_end = body.find("\">").ok_or("The function opener has no end.")?;
    let name = &body[..name_end];
    if !valid_name(name) {
        return Err("The function name is invalid.".into());
    }
    let schema = find_tool(tools, name)?;
    let mut rest = &body[name_end + 2..];
    let mut args = Map::new();

    loop {
        rest = rest.trim_start_matches(char::is_whitespace);
        if rest.is_empty() {
            break;
        }
        rest = rest
            .strip_prefix("<param name=\"")
            .ok_or("The function contains an invalid parameter tag.")?;
        let key_end = rest.find("\">").ok_or("The parameter opener has no end.")?;
        let key = &rest[..key_end];
        if !valid_name(key) {
            return Err("The parameter name is invalid.".into());
        }
        let property = schema["parameters"]["properties"]
            .get(key)
            .ok_or("The tool call contains an undeclared argument.")?;
        rest = &rest[key_end + 2..];
        let (raw, remaining) = parse_value(rest)?;
        let value = if property["type"] == "string" {
            Value::String(raw)
        } else {
            crate::contracts::parse_document(raw.as_bytes()).map_err(|error| {
                format!("The tool parameter {key} contains invalid JSON: {error}")
            })?
        };
        if args.insert(key.to_owned(), value).is_some() {
            return Err("The tool call repeats a parameter.".into());
        }
        rest = remaining;
    }

    let input = Value::Object(args);
    validate_arguments(&input, &schema["parameters"])?;
    Ok((name.to_owned(), input))
}

fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.chars().all(name_character)
}

fn name_character(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | '-' | '.')
}

fn parse_value(mut rest: &str) -> Result<(String, &str), String> {
    let mut value = String::new();
    loop {
        if let Some(remaining) = rest.strip_prefix("</param>") {
            return Ok((value, remaining));
        }
        if let Some(cdata) = rest.strip_prefix("<![CDATA[") {
            let end = cdata
                .find("]]>")
                .ok_or("The parameter CDATA section has no end.")?;
            let literal = &cdata[..end];
            if !literal.chars().all(xml_character) {
                return Err("The parameter contains an invalid XML character.".into());
            }
            value.push_str(literal);
            rest = &cdata[end + 3..];
            continue;
        }
        if let Some(entity) = rest.strip_prefix('&') {
            let end = entity.find(';').ok_or("The parameter entity has no end.")?;
            value.push(decode_entity(&entity[..end])?);
            rest = &entity[end + 1..];
            continue;
        }
        if rest.starts_with('<') || rest.starts_with("]]>") {
            return Err("The parameter contains invalid XML text.".into());
        }
        let character = rest
            .chars()
            .next()
            .ok_or("The parameter has no closing tag.")?;
        if !xml_character(character) {
            return Err("The parameter contains an invalid XML character.".into());
        }
        value.push(character);
        rest = &rest[character.len_utf8()..];
    }
}

fn decode_entity(entity: &str) -> Result<char, String> {
    let character = match entity {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        _ => {
            let number = if let Some(hex) = entity.strip_prefix("#x") {
                u32::from_str_radix(hex, 16)
            } else if let Some(decimal) = entity.strip_prefix('#') {
                decimal.parse::<u32>()
            } else {
                return Err("The parameter contains an unknown XML entity.".into());
            }
            .map_err(|_| "The parameter contains an invalid XML character reference.")?;
            char::from_u32(number)
                .ok_or("The parameter contains an invalid XML character reference.")?
        }
    };
    if !xml_character(character) {
        return Err("The parameter contains an invalid XML character reference.".into());
    }
    Ok(character)
}

fn xml_character(character: char) -> bool {
    matches!(character as u32, 0x9 | 0xA | 0xD | 0x20..=0xD7FF | 0xE000..=0xFFFD | 0x10000..=0x10FFFF)
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::super::{anthropic_blocks, OutputParser, OutputProfile, Part};

    fn tools() -> Vec<Value> {
        vec![json!({"function":{"name":"Read","parameters":{
            "type":"object","required":["file_path"],"additionalProperties":false,
            "properties":{
                "file_path":{"type":"string"},
                "source":{"type":"string"},
                "count":{"type":"integer"},
                "settings":{"type":"object","additionalProperties":false,
                    "properties":{"enabled":{"type":"boolean"}}}
            }
        }}})]
    }

    fn parse(output: &str, split: bool) -> Result<(Vec<Part>, usize), String> {
        let mut parser = OutputParser::with_profile(tools(), OutputProfile::MiniCpm5);
        let mut parts = Vec::new();
        if split {
            for character in output.chars() {
                parts.extend(parser.push(&character.to_string(), false)?);
            }
            parts.extend(parser.push("", true)?);
        } else {
            parts.extend(parser.push(output, true)?);
        }
        Ok((parts, parser.tool_count))
    }

    #[test]
    fn bare_call_survives_every_character_boundary() {
        let output = "前文<function name=\"Read\"><param name=\"file_path\">/workspace/brief.txt</param></function>后文";
        for split in [false, true] {
            let (parts, count) = parse(output, split).unwrap();
            let blocks = anthropic_blocks(&parts);
            assert_eq!(blocks.len(), 3);
            assert_eq!(blocks[0]["text"], "前文");
            assert_eq!(blocks[1]["name"], "Read");
            assert_eq!(
                blocks[1]["input"],
                json!({"file_path":"/workspace/brief.txt"})
            );
            assert_eq!(blocks[2]["text"], "后文");
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn cdata_keeps_code_svg_and_closing_tags() {
        let output = concat!(
            "<function name=\"Read\">",
            "<param name=\"file_path\">x</param>",
            "<param name=\"source\"><![CDATA[<svg><text>中</text></svg>\n",
            "</param></function> &amp; raw]]></param>",
            "</function>"
        );
        for split in [false, true] {
            let (parts, count) = parse(output, split).unwrap();
            assert_eq!(
                anthropic_blocks(&parts)[0]["input"]["source"],
                "<svg><text>中</text></svg>\n</param></function> &amp; raw"
            );
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn entities_whitespace_and_typed_json_keep_their_meaning() {
        let output = concat!(
            "<function name=\"Read\">\n",
            "<param name=\"file_path\">  x&amp;y&lt;z&gt;&quot;&apos;&#35;&#x4E2D;  </param>\n",
            "<param name=\"count\"> 2 </param>",
            "<param name=\"settings\">{\"enabled\":true}</param>",
            "</function>"
        );
        for split in [false, true] {
            let (parts, count) = parse(output, split).unwrap();
            assert_eq!(
                anthropic_blocks(&parts)[0]["input"],
                json!({"file_path":"  x&y<z>\"'#中  ","count":2,"settings":{"enabled":true}})
            );
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn multiple_calls_and_thinking_keep_text_in_order() {
        let output = concat!(
            "A<think>private</think>B",
            "<function name=\"Read\"><param name=\"file_path\">first</param></function>",
            "C",
            "<function name=\"Read\"><param name=\"file_path\">second</param></function>",
            "D"
        );
        for split in [false, true] {
            let (parts, count) = parse(output, split).unwrap();
            let blocks = anthropic_blocks(&parts);
            assert_eq!(blocks.len(), 5);
            assert_eq!(blocks[0]["text"], "AB");
            assert_eq!(blocks[1]["input"]["file_path"], "first");
            assert_eq!(blocks[2]["text"], "C");
            assert_eq!(blocks[3]["input"]["file_path"], "second");
            assert_eq!(blocks[4]["text"], "D");
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn function_like_prose_roundtrips_as_text() {
        let output = concat!(
            "Use <function=Read>, <function name ='Read'>, or <function name='Read'>. ",
            "The example <function name=\"Read\" > has a space before >. ",
            "The phrase is <function name=\"unfinished"
        );
        for split in [false, true] {
            let (parts, count) = parse(output, split).unwrap();
            assert_eq!(
                anthropic_blocks(&parts),
                vec![json!({"type":"text","text":output})]
            );
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn invalid_calls_never_count_as_tools() {
        let invalid = [
            "<function name=\"Other\"></function>",
            "<function name=\"Read\"><param name=\"other\">x</param></function>",
            "<function name=\"Read\"><param name=\"file_path\">x</param><param name=\"file_path\">y</param></function>",
            "<function name=\"Read\"><param name=\"file_path\">x</param><wrong/></function>",
            "<function name=\"Read\"><param name=\"file_path\">&unknown;</param></function>",
            "<function name=\"Read\"><param name=\"file_path\">&#0;</param></function>",
            "<function name=\"Read\"><param name=\"file_path\"><![CDATA[\u{1}]]></param></function>",
            "<function name=\"Read\"><param name=\"file_path\"><svg/></param></function>",
            "<function name=\"Read\"><param name=\"count\">2</param></function>",
            "<function name=\"Read\"><param name=\"file_path\">x</param><param name=\"count\">2.5</param></function>",
            "<function name=\"Read\"><param name=\"file_path\">x</param><param name=\"settings\">{\"enabled\":true,\"enabled\":false}</param></function>",
            "<function name=\"Read\"><param name=\"file_path\"><![CDATA[</function>]]></param>",
            "<function name=\"Read\"><param name=\"file_path\">x</param>",
        ];
        for output in invalid {
            for split in [false, true] {
                assert!(parse(output, split).is_err(), "Accepted {output}");
            }
        }
    }
}

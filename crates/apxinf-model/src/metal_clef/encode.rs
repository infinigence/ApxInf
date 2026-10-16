//! Clef request validation and the joint-schema prompt encoding.
//!
//! Mirrors `encode_record` of the clef-flash MLX reference (text path,
//! `truncate=False`): every piece is tokenized separately, without added
//! special tokens, and question/option spans index the concatenated IDs.
use super::config::MAX_INPUT_TOKENS;
use crate::d3::{DecisionQuestion, DecisionSchema};
use apxinf_core::{Error, Result};
use apxinf_tokenizer::Tokenizer;
use serde_json::Value;

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
Each answer must be exactly one of that field's allowed options.";
const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Noul = 0,
    Choice = 1,
    Score = 2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedQuestion {
    pub id: String,
    pub kind: Kind,
    pub span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    pub input_ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
}

fn invalid(message: String) -> Error {
    Error::Other(format!("Clef request: {message}"))
}

/// Options in reference order: noul `true, false`; choice by sorted ID;
/// score by level index.
fn options(question: &DecisionQuestion) -> Result<(Kind, Vec<(String, Option<Value>)>)> {
    let id = &question.id;
    match (question.kind.as_str(), &question.criteria) {
        ("noul", None) => Ok((
            Kind::Noul,
            vec![
                ("true".into(), Some(NOUL_TRUE.into())),
                ("false".into(), Some(NOUL_FALSE.into())),
            ],
        )),
        ("noul", Some(Value::Object(criteria)))
            if criteria.keys().all(|k| k == "true" || k == "false") =>
        {
            let describe = |key: &str, default: &str| match criteria.get(key) {
                Some(Value::Null) => None,
                Some(value) => Some(value.clone()),
                None => Some(default.into()),
            };
            Ok((
                Kind::Noul,
                vec![
                    ("true".into(), describe("true", NOUL_TRUE)),
                    ("false".into(), describe("false", NOUL_FALSE)),
                ],
            ))
        }
        ("noul", _) => Err(invalid(format!("invalid noul criteria: {id}"))),
        ("choice", Some(Value::Object(criteria)))
            if !criteria.is_empty() && criteria.keys().all(|k| !k.is_empty()) =>
        {
            let mut options: Vec<_> = criteria
                .iter()
                .map(|(k, v)| (k.clone(), (!v.is_null()).then(|| v.clone())))
                .collect();
            options.sort_by(|a, b| a.0.cmp(&b.0));
            Ok((Kind::Choice, options))
        }
        ("choice", _) => Err(invalid(format!(
            "choice requires nonempty string option IDs: {id}"
        ))),
        ("score", Some(Value::Array(levels))) if !levels.is_empty() => Ok((
            Kind::Score,
            levels
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), (!v.is_null()).then(|| v.clone())))
                .collect(),
        )),
        ("score", _) => Err(invalid(format!(
            "score requires a nonempty criteria list: {id}"
        ))),
        _ => Err(invalid(format!("unsupported question type: {id}"))),
    }
}

/// Python truthiness of a JSON value, as in `instructions or question_id`.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `float.__repr__`: shortest round-trip digits, positional when the
/// decimal exponent is in [-4, 16), otherwise `d.ddde+XX`.
fn python_float(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(invalid("non-finite number".into()));
    }
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i32 = exponent.parse().unwrap();
    let point = exponent + 1;
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let body = if digits == "0" {
        "0.0".to_owned()
    } else if point <= -4 || point > 16 {
        let tail = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits.clone()
        };
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{tail}e{sign}{:02}", exponent.abs())
    } else if point <= 0 {
        format!("0.{}{}", "0".repeat((-point) as usize), digits)
    } else if point as usize >= digits.len() {
        format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    };
    Ok(format!("{sign}{body}"))
}

fn json_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) if n.is_f64() => out.push_str(&python_float(n.as_f64().unwrap())?),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => json_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json_string(key, out);
                out.push(':');
                json(item, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Reference `render`: strings verbatim, other values as compact sorted JSON
/// (`json.dumps(..., ensure_ascii=False, separators=(",", ":"), sort_keys=True)`).
pub fn render(value: &Value) -> Result<String> {
    if let Value::String(s) = value {
        return Ok(s.clone());
    }
    let mut out = String::new();
    json(value, &mut out)?;
    Ok(out)
}

pub fn encode(tokenizer: &Tokenizer, schema: &DecisionSchema) -> Result<Encoded> {
    let tokens = |text: &str| tokenizer.encode_with_special_tokens(text, false);
    let mut schema_ids = tokens("\n\nSCHEMA FIELDS:\n")?;
    let mut questions = Vec::with_capacity(schema.questions.len());
    for (qi, question) in schema.questions.iter().enumerate() {
        if question.id.is_empty() {
            return Err(invalid("question IDs must be nonempty strings".into()));
        }
        let (kind, options) = options(question)?;
        schema_ids.extend(tokens(&format!(
            "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
            qi + 1,
            question.id,
            question.kind
        ))?);
        let start = schema_ids.len();
        let instructions = match &question.instructions {
            Some(value) if truthy(value) => render(value)?,
            _ => question.id.clone(),
        };
        schema_ids.extend(tokens(&instructions)?);
        let span = (start, schema_ids.len());
        schema_ids.extend(tokens("\nALLOWED OPTIONS:\n")?);
        let mut option_spans = Vec::with_capacity(options.len());
        let mut option_ids = Vec::with_capacity(options.len());
        for (oi, (option_id, description)) in options.into_iter().enumerate() {
            schema_ids.extend(tokens(&format!("OPTION {}: ", oi + 1))?);
            let start = schema_ids.len();
            let mut semantics = serde_json::Map::new();
            semantics.insert("option_id".into(), Value::String(option_id.clone()));
            if let Some(description) = description {
                semantics.insert("description".into(), description);
            }
            schema_ids.extend(tokens(&render(&Value::Object(semantics))?)?);
            option_spans.push((start, schema_ids.len()));
            option_ids.push(option_id);
            schema_ids.extend(tokens("\n")?);
        }
        schema_ids.extend(tokens("END FIELD\n")?);
        questions.push(EncodedQuestion {
            id: question.id.clone(),
            kind,
            span,
            option_spans,
            option_ids,
        });
    }
    let prefix = tokens(&format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ))?;
    let suffix = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let state = tokens(&render(&schema.state)?)?;
    let total = prefix.len() + state.len() + schema_ids.len() + suffix.len();
    if total > MAX_INPUT_TOKENS {
        return Err(Error::Other(format!(
            "Clef request needs {total} tokens; maximum is {MAX_INPUT_TOKENS}"
        )));
    }
    let offset = prefix.len() + state.len();
    for question in &mut questions {
        question.span = (question.span.0 + offset, question.span.1 + offset);
        for span in &mut question.option_spans {
            *span = (span.0 + offset, span.1 + offset);
        }
    }
    let mut input_ids = prefix;
    input_ids.extend(state);
    input_ids.extend(schema_ids);
    input_ids.extend(suffix);
    Ok(Encoded {
        input_ids,
        questions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_matches_python_json_dumps() {
        assert_eq!(render(&json!("raw text")).unwrap(), "raw text");
        assert_eq!(
            render(&json!({"b": [1, 2.5, null, true], "a": {"z": "é\"\n\t\u{1f}", "y": false}}))
                .unwrap(),
            r#"{"a":{"y":false,"z":"é\"\n\t\u001f"},"b":[1,2.5,null,true]}"#
        );
        for (value, python) in [
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (123456.789, "123456.789"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (2.5e300, "2.5e+300"),
            (-3.25, "-3.25"),
        ] {
            assert_eq!(python_float(value).unwrap(), python, "{value}");
        }
        assert_eq!(render(&json!(7)).unwrap(), "7");
        assert_eq!(render(&json!(-7)).unwrap(), "-7");
    }

    fn question(kind: &str, criteria: Option<Value>) -> DecisionQuestion {
        DecisionQuestion {
            id: "q".into(),
            kind: kind.into(),
            instructions: None,
            criteria,
        }
    }

    #[test]
    fn options_follow_reference_order_and_validation() {
        let (kind, noul) = options(&question("noul", None)).unwrap();
        assert_eq!(kind, Kind::Noul);
        assert_eq!(noul[0], ("true".into(), Some(json!(NOUL_TRUE))));
        assert_eq!(noul[1].0, "false");
        let (_, custom) = options(&question(
            "noul",
            Some(json!({"false": "No", "true": null})),
        ))
        .unwrap();
        assert_eq!(
            custom,
            [("true".into(), None), ("false".into(), Some(json!("No")))]
        );
        let (kind, choice) = options(&question(
            "choice",
            Some(json!({"zeta": "Z", "alpha": {"x": 1}})),
        ))
        .unwrap();
        assert_eq!(kind, Kind::Choice);
        assert_eq!(choice[0].0, "alpha");
        assert_eq!(choice[1], ("zeta".into(), Some(json!("Z"))));
        let (kind, score) =
            options(&question("score", Some(json!(["low", "mid", "high"])))).unwrap();
        assert_eq!(kind, Kind::Score);
        let ids: Vec<_> = score.iter().map(|o| o.0.as_str()).collect();
        assert_eq!(ids, ["0", "1", "2"]);
        for (kind, criteria) in [
            ("noul", Some(json!({"maybe": "x"}))),
            ("noul", Some(json!(["true"]))),
            ("choice", None),
            ("choice", Some(json!({}))),
            ("choice", Some(json!({"": "empty"}))),
            ("score", Some(json!([]))),
            ("score", Some(json!({"0": "x"}))),
            ("rank", None),
        ] {
            assert!(
                options(&question(kind, criteria.clone())).is_err(),
                "{kind} {criteria:?}"
            );
        }
    }

    /// Run with `--ignored --exact metal_clef::encode::tests::recorded_reference_encoding`.
    /// APXINF_CLEF_CHECKPOINT names the checkpoint directory (for tokenizer.json)
    /// and APXINF_CLEF_ENCODING_CASES a JSONL file whose rows hold `request` (the
    /// JSON request text), the reference `input_ids` and per-question `id`,
    /// `kind`, `span`, `option_spans` and `option_ids`.
    #[test]
    #[ignore = "requires the Clef checkpoint tokenizer and recorded reference encodings"]
    fn recorded_reference_encoding() {
        #[derive(serde::Deserialize)]
        struct Question {
            id: String,
            kind: usize,
            span: (usize, usize),
            option_spans: Vec<(usize, usize)>,
            option_ids: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            id: String,
            request: String,
            input_ids: Vec<u32>,
            questions: Vec<Question>,
        }
        let path = |name| std::env::var_os(name).unwrap_or_else(|| panic!("set {name}"));
        let checkpoint = std::path::PathBuf::from(path("APXINF_CLEF_CHECKPOINT"));
        let tokenizer = Tokenizer::from_file(checkpoint.join("tokenizer.json")).unwrap();
        let cases = std::fs::read_to_string(path("APXINF_CLEF_ENCODING_CASES")).unwrap();
        let mut count = 0;
        for line in cases.lines() {
            let case: Case = serde_json::from_str(line).unwrap();
            let schema = DecisionSchema::from_json(&case.request).unwrap();
            let encoded = encode(&tokenizer, &schema).unwrap();
            assert_eq!(encoded.input_ids, case.input_ids, "{}", case.id);
            assert_eq!(encoded.questions.len(), case.questions.len(), "{}", case.id);
            for (actual, expected) in encoded.questions.iter().zip(&case.questions) {
                assert_eq!(actual.id, expected.id, "{}", case.id);
                assert_eq!(actual.kind as usize, expected.kind, "{}", case.id);
                assert_eq!(actual.span, expected.span, "{}", case.id);
                assert_eq!(actual.option_spans, expected.option_spans, "{}", case.id);
                assert_eq!(actual.option_ids, expected.option_ids, "{}", case.id);
            }
            count += 1;
        }
        assert!(count > 0, "no recorded cases");
        println!("{count} recorded requests encoded identically");
    }

    #[test]
    fn python_truthiness_of_instructions() {
        for falsy in [
            json!(null),
            json!(""),
            json!(0),
            json!(0.0),
            json!(false),
            json!([]),
            json!({}),
        ] {
            assert!(!truthy(&falsy), "{falsy}");
        }
        for value in [
            json!("0"),
            json!(1),
            json!(-0.5),
            json!(true),
            json!([0]),
            json!({"a": 0}),
        ] {
            assert!(truthy(&value), "{value}");
        }
    }
}

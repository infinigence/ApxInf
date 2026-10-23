//! Typed text decisions, exact encodings, and probability projection.
use crate::contracts::{self, ContractError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

type Result<T> = std::result::Result<T, ContractError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Noul {
        instructions: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<BTreeMap<String, String>>,
    },
    Choice {
        instructions: String,
        criteria: BTreeMap<String, String>,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

impl Question {
    pub fn option_ids(&self) -> Vec<String> {
        match self {
            Self::Noul { .. } => vec!["true".into(), "false".into()],
            Self::Choice { criteria, .. } => criteria.keys().cloned().collect(),
            Self::Score { criteria, .. } => (0..criteria.len()).map(|n| n.to_string()).collect(),
        }
    }
    fn type_code(&self) -> u64 {
        match self {
            Self::Noul { .. } => 0,
            Self::Choice { .. } => 1,
            Self::Score { .. } => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub state: String,
    pub questions: BTreeMap<String, Question>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRequest {
    pub model: String,
    pub record: Record,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestionEncoding {
    pub question_id: String,
    pub question_type: u64,
    pub question_span: [usize; 2],
    pub option_spans: Vec<[usize; 2]>,
    pub option_ids: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub questions: Vec<QuestionEncoding>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Probability {
    pub question_id: String,
    pub option_ids: Vec<String>,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionResult {
    pub probabilities: Vec<Probability>,
    pub input_tokens: u64,
}

fn check(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(ContractError(message.into()))
    }
}
fn bounded_text(value: &str, min: usize, max: usize) -> Result<()> {
    check(
        (min..=max).contains(&value.len()),
        "decision text length outside allowed range",
    )
}
fn identifier(value: &str) -> Result<()> {
    check(
        (1..=64).contains(&value.len())
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
        "invalid decision identifier",
    )
}

pub fn parse_decision_request(bytes: &[u8]) -> Result<DecisionRequest> {
    check(bytes.len() <= 131072, "decision request exceeds byte limit")?;
    let value = contracts::parse_document(bytes)?;
    let object = value
        .as_object()
        .ok_or_else(|| ContractError("expected decision request object".into()))?;
    check(
        object.len() == 3
            && ["model", "state", "questions"]
                .iter()
                .all(|key| object.contains_key(*key)),
        "invalid decision request fields",
    )?;
    let model = value["model"]
        .as_str()
        .ok_or_else(|| ContractError("expected model string".into()))?
        .to_owned();
    bounded_text(&model, 1, 128)?;
    let record =
        validate_record(&json!({"state": value["state"], "questions": value["questions"]}))?;
    Ok(DecisionRequest { model, record })
}

pub fn validate_record(value: &Value) -> Result<Record> {
    let record: Record =
        serde_json::from_value(value.clone()).map_err(|e| ContractError(e.to_string()))?;
    bounded_text(&record.state, 0, 65536)?;
    check(
        (1..=16).contains(&record.questions.len()),
        "invalid decision question count",
    )?;
    for (id, question) in &record.questions {
        identifier(id)?;
        let instructions = match question {
            Question::Noul {
                instructions,
                criteria,
            } => {
                // Serde accepts null as None; the wire contract permits only absence.
                if value["questions"][id].get("criteria").is_some() {
                    check(criteria.is_some(), "null noul criteria")?;
                }
                if let Some(criteria) = criteria {
                    check(
                        criteria.len() == 2
                            && criteria.contains_key("true")
                            && criteria.contains_key("false"),
                        "invalid noul criteria",
                    )?;
                    for text in criteria.values() {
                        bounded_text(text, 1, 2048)?;
                    }
                }
                instructions
            }
            Question::Choice {
                instructions,
                criteria,
            } => {
                check((2..=16).contains(&criteria.len()), "invalid choice count")?;
                for (option, text) in criteria {
                    identifier(option)?;
                    bounded_text(text, 1, 2048)?;
                }
                instructions
            }
            Question::Score {
                instructions,
                criteria,
            } => {
                check((2..=16).contains(&criteria.len()), "invalid score count")?;
                for text in criteria {
                    bounded_text(text, 1, 2048)?;
                }
                instructions
            }
        };
        bounded_text(instructions, 1, 4096)?;
    }
    Ok(record)
}

pub fn validate_encoding(
    value: &Value,
    record: &Record,
    max_context: usize,
    vocab_size: u64,
) -> Result<Encoding> {
    let encoding: Encoding =
        serde_json::from_value(value.clone()).map_err(|e| ContractError(e.to_string()))?;
    check(
        !encoding.input_ids.is_empty()
            && encoding.input_ids.len() <= max_context
            && encoding
                .input_ids
                .iter()
                .all(|id| u64::from(*id) < vocab_size),
        "invalid decision input IDs",
    )?;
    check(
        encoding.questions.len() == record.questions.len(),
        "decision metadata cardinality mismatch",
    )?;
    let mut previous = 0;
    for (entry, (id, question)) in encoding.questions.iter().zip(&record.questions) {
        check(
            entry.question_id == *id
                && entry.question_type == question.type_code()
                && entry.option_ids == question.option_ids()
                && entry.option_spans.len() == entry.option_ids.len(),
            "decision metadata does not match record",
        )?;
        for span in std::iter::once(&entry.question_span).chain(&entry.option_spans) {
            check(
                span[0] >= previous && span[0] < span[1] && span[1] <= encoding.input_ids.len(),
                "invalid or overlapping decision span",
            )?;
            previous = span[1];
        }
    }
    Ok(encoding)
}

pub fn validate_probabilities(value: &Value, record: &Record) -> Result<Vec<Probability>> {
    let values: Vec<Probability> =
        serde_json::from_value(value.clone()).map_err(|e| ContractError(e.to_string()))?;
    check(
        values.len() == record.questions.len(),
        "decision probability cardinality mismatch",
    )?;
    for (entry, (id, question)) in values.iter().zip(&record.questions) {
        check(
            entry.question_id == *id
                && entry.option_ids == question.option_ids()
                && entry.values.len() == entry.option_ids.len(),
            "decision probability IDs mismatch",
        )?;
        check(
            entry
                .values
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                && (entry.values.iter().sum::<f64>() - 1.0).abs() <= 0.00001,
            "invalid decision probability vector",
        )?;
    }
    Ok(values)
}

pub fn project_response(model: &str, record: &Record, result: &DecisionResult) -> Value {
    let mut answers = BTreeMap::new();
    for (probability, (id, question)) in result.probabilities.iter().zip(&record.questions) {
        let answer = match question {
            Question::Noul { .. } => json!({"type":"noul", "noul":probability.values[0]}),
            Question::Choice { .. } => {
                let selected = probability
                    .values
                    .iter()
                    .enumerate()
                    .fold(0, |best, (index, p)| {
                        if *p > probability.values[best] {
                            index
                        } else {
                            best
                        }
                    });
                let probabilities: BTreeMap<_, _> = probability
                    .option_ids
                    .iter()
                    .zip(&probability.values)
                    .collect();
                json!({"type":"choice", "choice":probability.option_ids[selected], "confidence":probability.values[selected], "probabilities":probabilities})
            }
            Question::Score { criteria, .. } => {
                let score: f64 = probability
                    .values
                    .iter()
                    .enumerate()
                    .map(|(index, p)| index as f64 * p)
                    .sum();
                let probabilities: BTreeMap<_, _> = probability
                    .option_ids
                    .iter()
                    .zip(&probability.values)
                    .collect();
                let legend: BTreeMap<_, _> = probability.option_ids.iter().zip(criteria).collect();
                json!({"type":"score", "score":score,"confidence":probability.values.iter().copied().fold(0.0, f64::max),"legend":legend,"probabilities":probabilities})
            }
        };
        answers.insert(id, answer);
    }
    json!({"model":model,"answers":answers,"usage":{"input_tokens":result.input_tokens,"output_tokens":0}})
}

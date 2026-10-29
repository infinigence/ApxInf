//! Model-neutral decision API, shared by decision families (d3, metal_clef).

use std::fmt;

use apxinf_core::{Error, Result};
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::Deserialize;

use crate::llm_trait::{ImageInput, LlmCapabilities};

/// One complete prompt. Video uses the same processor tensor plus `grid_thw`
/// contract as Qwen3.5 vision; callers concatenate temporal patches there.
#[derive(Clone, Copy, Debug)]
pub struct DecisionInput<'a> {
    pub token_ids: &'a [u32],
    pub image: Option<ImageInput<'a>>,
    /// Number of valid answer codes at the start of the package code table.
    pub candidate_count: usize,
}

impl<'a> DecisionInput<'a> {
    pub const fn text(token_ids: &'a [u32], candidate_count: usize) -> Self {
        Self {
            token_ids,
            image: None,
            candidate_count,
        }
    }

    pub const fn with_vision(
        token_ids: &'a [u32],
        image: ImageInput<'a>,
        candidate_count: usize,
    ) -> Self {
        Self {
            token_ids,
            image: Some(image),
            candidate_count,
        }
    }
}

/// A state and its questions, in request order. The family defines the
/// meaning of `kind`, `instructions` and `criteria`, and derives each
/// question's options from them.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionSchema {
    pub state: serde_json::Value,
    pub questions: Vec<DecisionQuestion>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionQuestion {
    pub id: String,
    pub kind: String,
    pub instructions: Option<serde_json::Value>,
    pub criteria: Option<serde_json::Value>,
}

impl DecisionSchema {
    /// Parses `{"state": ..., "questions": {"<id>": {"type", "instructions"?,
    /// "criteria"?}, ...}}`. Questions keep their document order; duplicate
    /// question IDs and unknown fields are rejected.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| Error::Other(format!("decision request: {e}")))
    }
}

impl<'de> Deserialize<'de> for DecisionSchema {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Body {
            state: serde_json::Value,
            questions: Questions,
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Question {
            #[serde(rename = "type")]
            kind: String,
            instructions: Option<serde_json::Value>,
            criteria: Option<serde_json::Value>,
        }
        struct Questions(Vec<DecisionQuestion>);
        impl<'de> Deserialize<'de> for Questions {
            fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
                struct Ordered;
                impl<'de> Visitor<'de> for Ordered {
                    type Value = Questions;
                    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        f.write_str("a non-empty object of named questions")
                    }
                    fn visit_map<A: MapAccess<'de>>(
                        self,
                        mut map: A,
                    ) -> std::result::Result<Questions, A::Error> {
                        let mut out: Vec<DecisionQuestion> = Vec::new();
                        while let Some((id, q)) = map.next_entry::<String, Question>()? {
                            if out.iter().any(|existing| existing.id == id) {
                                return Err(de::Error::custom(format!(
                                    "duplicate question `{id}`"
                                )));
                            }
                            out.push(DecisionQuestion {
                                id,
                                kind: q.kind,
                                instructions: q.instructions,
                                criteria: q.criteria,
                            });
                        }
                        if out.is_empty() {
                            return Err(de::Error::custom("at least one question is required"));
                        }
                        Ok(Questions(out))
                    }
                }
                d.deserialize_map(Ordered)
            }
        }
        let body = Body::deserialize(deserializer)?;
        Ok(Self {
            state: body.state,
            questions: body.questions.0,
        })
    }
}

/// Probability assigned to one package-defined d3 code.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionScore {
    pub code: String,
    pub token_id: u32,
    pub probability: f32,
}

/// Structured output; d3 never enters autoregressive token generation.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionOutput {
    pub selected_index: usize,
    pub scores: Vec<DecisionScore>,
}

impl DecisionOutput {
    pub fn selected(&self) -> Result<&DecisionScore> {
        self.scores.get(self.selected_index).ok_or_else(|| {
            Error::Other(format!(
                "d3 decision output: selected index {} outside {} scores",
                self.selected_index,
                self.scores.len()
            ))
        })
    }
}

/// Probability assigned to a named schema option, which has no package token ID.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemaDecisionScore {
    pub code: String,
    pub probability: f32,
}

/// Scores of one schema question, in option order.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemaDecisionAnswer {
    pub question_id: String,
    pub selected_index: usize,
    pub scores: Vec<SchemaDecisionScore>,
}

impl SchemaDecisionAnswer {
    pub fn selected(&self) -> Result<&SchemaDecisionScore> {
        self.scores.get(self.selected_index).ok_or_else(|| {
            Error::Other(format!(
                "decision output: selected index {} outside {} scores",
                self.selected_index,
                self.scores.len()
            ))
        })
    }
}

/// Schema answers follow request order. Scores follow each question's option order.
#[derive(Clone, Debug, PartialEq)]
pub struct SchemaDecisionOutput {
    pub answers: Vec<SchemaDecisionAnswer>,
}

/// Non-generative decision runtime. Implementations own their preparation
/// and return host-visible probabilities only at the end.
pub trait DecisionRuntime {
    fn capabilities(&self) -> LlmCapabilities;
    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput>;
    /// Package code names in index order; empty without a package code table.
    fn code_names(&self) -> &[String];
    /// Token IDs for each package code; empty without a package code table.
    fn token_ids(&self) -> &[u32];

    /// Score a named schema when the runtime supports this request form.
    fn decide_schema(&mut self, _request: &DecisionSchema) -> Result<SchemaDecisionOutput> {
        Err(Error::Other(
            "schema decisions are unsupported by this runtime".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_score_is_checked() {
        let output = DecisionOutput {
            selected_index: 0,
            scores: vec![DecisionScore {
                code: "yes".into(),
                token_id: 1,
                probability: 1.0,
            }],
        };
        assert_eq!(output.selected().unwrap().code, "yes");

        let invalid = DecisionOutput {
            selected_index: 1,
            ..output
        };
        assert!(invalid.selected().is_err());
    }

    #[test]
    fn schema_keeps_question_order_and_rejects_ambiguous_requests() {
        let schema = DecisionSchema::from_json(
            r#"{"state": {"b": 1, "a": "x"}, "questions": {
                "zeta": {"type": "noul", "instructions": "Is it open?"},
                "alpha": {"type": "choice", "criteria": {"y": "Yes", "n": "No"}},
                "mid": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]}}}"#,
        )
        .unwrap();
        let ids: Vec<_> = schema.questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["zeta", "alpha", "mid"]);
        assert_eq!(schema.questions[0].kind, "noul");
        assert_eq!(schema.questions[0].criteria, None);
        assert_eq!(schema.questions[1].instructions, None);
        assert_eq!(
            schema.questions[2].criteria,
            Some(serde_json::json!(["low", "high"]))
        );
        assert_eq!(schema.state, serde_json::json!({"a": "x", "b": 1}));

        for invalid in [
            r#"{"state": "s", "questions": {}}"#,
            r#"{"state": "s"}"#,
            r#"{"questions": {"q": {"type": "noul"}}}"#,
            r#"{"state": "s", "questions": {"q": {"type": "noul"}}, "extra": 1}"#,
            r#"{"state": "s", "questions": {"q": {"type": "noul", "label": 1}}}"#,
            r#"{"state": "s", "questions": {"q": {"instructions": "no type"}}}"#,
            r#"{"state": "s", "questions": {"q": {"type": "noul"}, "q": {"type": "noul"}}}"#,
            r#"{"state": "s", "questions": [{"type": "noul"}]}"#,
        ] {
            assert!(DecisionSchema::from_json(invalid).is_err(), "{invalid}");
        }
    }
}

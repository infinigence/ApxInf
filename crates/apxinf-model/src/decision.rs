//! Model-neutral decision API, shared by decision families (d3, metal_clef).

use std::{collections::HashSet, fmt};

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
    /// Checks the question-list invariants for parsed and directly constructed requests.
    pub fn validate(&self) -> Result<()> {
        let invalid = |message: String| Error::Other(format!("decision request: {message}"));
        if self.questions.is_empty() {
            return Err(invalid("at least one question is required".into()));
        }
        let mut ids = HashSet::with_capacity(self.questions.len());
        for question in &self.questions {
            if question.id.is_empty() {
                return Err(invalid("question IDs must be nonempty strings".into()));
            }
            if !ids.insert(question.id.as_str()) {
                return Err(invalid(format!("duplicate question `{}`", question.id)));
            }
        }
        Ok(())
    }

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
                            out.push(DecisionQuestion {
                                id,
                                kind: q.kind,
                                instructions: q.instructions,
                                criteria: q.criteria,
                            });
                        }
                        Ok(Questions(out))
                    }
                }
                d.deserialize_map(Ordered)
            }
        }
        let body = Body::deserialize(deserializer)?;
        let schema = Self {
            state: body.state,
            questions: body.questions.0,
        };
        schema.validate().map_err(de::Error::custom)?;
        Ok(schema)
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

/// Encoded-prompt decisions: the caller tokenizes a complete prompt and the
/// runtime scores the leading entries of its package code table (d3).
pub trait CodeDecisionRuntime {
    fn capabilities(&self) -> LlmCapabilities;
    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput>;
    /// Package code names in index order.
    fn code_names(&self) -> &[String];
    /// Token IDs for each package code.
    fn token_ids(&self) -> &[u32];
}

/// Named-schema decisions: the runtime encodes the state and typed questions
/// itself and scores each question's own options (Clef).
pub trait SchemaDecisionRuntime {
    fn capabilities(&self) -> LlmCapabilities;
    fn decide_schema(&mut self, request: &DecisionSchema) -> Result<SchemaDecisionOutput>;
}

/// The d3 name for [`CodeDecisionRuntime`], kept for existing callers.
pub use self::CodeDecisionRuntime as DecisionRuntime;

/// A loaded non-generative decision model. The variant is the request
/// protocol the family supports, fixed at load time.
pub enum DecisionModel {
    Code(Box<dyn CodeDecisionRuntime>),
    Schema(Box<dyn SchemaDecisionRuntime>),
}

impl DecisionModel {
    pub fn capabilities(&self) -> LlmCapabilities {
        match self {
            Self::Code(model) => model.capabilities(),
            Self::Schema(model) => model.capabilities(),
        }
    }

    pub fn code_mut(&mut self) -> Result<&mut dyn CodeDecisionRuntime> {
        match self {
            Self::Code(model) => Ok(&mut **model),
            Self::Schema(_) => Err(Error::Other(
                "decision model takes schema requests, not encoded prompts".into(),
            )),
        }
    }

    pub fn code(&self) -> Result<&dyn CodeDecisionRuntime> {
        match self {
            Self::Code(model) => Ok(&**model),
            Self::Schema(_) => Err(Error::Other(
                "decision model takes schema requests and has no package code table".into(),
            )),
        }
    }

    pub fn schema_mut(&mut self) -> Result<&mut dyn SchemaDecisionRuntime> {
        match self {
            Self::Schema(model) => Ok(&mut **model),
            Self::Code(_) => Err(Error::Other(
                "decision model takes encoded prompts, not schema requests".into(),
            )),
        }
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
    fn direct_and_json_schemas_share_question_invariants() {
        let question = |id: &str| DecisionQuestion {
            id: id.into(),
            kind: "noul".into(),
            instructions: None,
            criteria: None,
        };
        for (questions, json, message) in [
            (
                vec![],
                r#"{"state":null,"questions":{}}"#,
                "at least one question is required",
            ),
            (
                vec![question("")],
                r#"{"state":null,"questions":{"":{"type":"noul"}}}"#,
                "question IDs must be nonempty strings",
            ),
            (
                vec![question("q"), question("q")],
                r#"{"state":null,"questions":{"q":{"type":"noul"},"q":{"type":"noul"}}}"#,
                "duplicate question `q`",
            ),
        ] {
            let direct = DecisionSchema {
                state: serde_json::Value::Null,
                questions,
            };
            assert!(direct.validate().unwrap_err().to_string().contains(message));
            assert!(DecisionSchema::from_json(json)
                .unwrap_err()
                .to_string()
                .contains(message));
            assert!(serde_json::from_str::<DecisionSchema>(json).is_err());
        }

        let direct = DecisionSchema {
            state: serde_json::Value::Null,
            questions: vec![question("zeta"), question("alpha")],
        };
        direct.validate().unwrap();
        assert_eq!(
            direct,
            DecisionSchema::from_json(
                r#"{"state":null,"questions":{"zeta":{"type":"noul"},"alpha":{"type":"noul"}}}"#
            )
            .unwrap()
        );
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

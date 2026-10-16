//! Model-neutral decision API, shared by decision families (d3, metal_clef).

use std::fmt;

use apxinf_core::{Error, Result};
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde::Deserialize;

use crate::llm_trait::{ImageInput, LlmCapabilities};

/// One complete request. A runtime accepts the forms it documents and returns
/// an error for the others.
#[derive(Clone, Copy, Debug)]
pub enum DecisionInput<'a> {
    /// A prompt the caller already encoded, scored over the package code
    /// table. Video uses the same processor tensor plus `grid_thw` contract as
    /// Qwen3.5 vision; callers concatenate temporal patches there.
    Encoded {
        token_ids: &'a [u32],
        image: Option<ImageInput<'a>>,
        /// Number of valid answer codes at the start of the package code table.
        candidate_count: usize,
    },
    /// A state with named, typed questions that the family encodes itself.
    Schema(&'a DecisionSchema),
}

impl<'a> DecisionInput<'a> {
    pub const fn text(token_ids: &'a [u32], candidate_count: usize) -> Self {
        Self::Encoded {
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
        Self::Encoded {
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

/// Probability assigned to one option. Package code tables (d3) also report
/// the code's token; options of a schema question have none.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionScore {
    pub code: String,
    pub token_id: Option<u32>,
    pub probability: f32,
}

/// Scores of one question, in option order.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionAnswer {
    /// The schema question ID; `None` for an encoded prompt.
    pub question_id: Option<String>,
    pub selected_index: usize,
    pub scores: Vec<DecisionScore>,
}

impl DecisionAnswer {
    pub fn selected(&self) -> Result<&DecisionScore> {
        self.scores.get(self.selected_index).ok_or_else(|| {
            Error::Other(format!(
                "decision output: selected index {} outside {} scores",
                self.selected_index,
                self.scores.len()
            ))
        })
    }
}

/// Structured output; decision runtimes never enter autoregressive token
/// generation. Answers follow request order; an encoded prompt has one.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionOutput {
    pub answers: Vec<DecisionAnswer>,
}

impl DecisionOutput {
    /// The answer of a single-question request, such as an encoded prompt.
    pub fn into_single(self) -> Result<DecisionAnswer> {
        let count = self.answers.len();
        let mut answers = self.answers.into_iter();
        match (answers.next(), answers.next()) {
            (Some(answer), None) => Ok(answer),
            _ => Err(Error::Other(format!(
                "decision output has {count} answers, expected one"
            ))),
        }
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_score_is_checked() {
        let answer = DecisionAnswer {
            question_id: None,
            selected_index: 0,
            scores: vec![DecisionScore {
                code: "yes".into(),
                token_id: Some(1),
                probability: 1.0,
            }],
        };
        assert_eq!(answer.selected().unwrap().code, "yes");

        let invalid = DecisionAnswer {
            selected_index: 1,
            ..answer.clone()
        };
        assert!(invalid.selected().is_err());

        let output = DecisionOutput {
            answers: vec![answer.clone()],
        };
        assert_eq!(output.into_single().unwrap(), answer);
        let two = DecisionOutput {
            answers: vec![answer.clone(), answer],
        };
        assert!(two.into_single().is_err());
        assert!(DecisionOutput { answers: vec![] }.into_single().is_err());
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

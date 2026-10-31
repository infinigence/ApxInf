use std::cell::Cell;
use std::rc::Rc;

use apxinf_core::Result;
use apxinf_model::d3::{
    DecisionInput as D3Input, DecisionOutput as D3Output, DecisionRuntime as D3Runtime,
    DecisionScore as D3Score,
};
use apxinf_model::{
    DecisionInput, DecisionModel, DecisionOutput, DecisionSchema, DecisionScore, LlmCapabilities,
    LoadedModel, SchemaDecisionAnswer, SchemaDecisionOutput, SchemaDecisionRuntime,
    SchemaDecisionScore,
};

struct LegacyRuntime {
    calls: Rc<Cell<usize>>,
    codes: Vec<String>,
    ids: [u32; 2],
}

impl D3Runtime for LegacyRuntime {
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn decide(&mut self, input: D3Input<'_>) -> Result<D3Output> {
        assert_eq!(input.token_ids, [17, 42]);
        assert!(input.image.is_none());
        assert_eq!(input.candidate_count, 2);
        self.calls.set(self.calls.get() + 1);
        Ok(D3Output {
            selected_index: 1,
            scores: vec![
                D3Score {
                    code: self.codes[0].clone(),
                    token_id: self.ids[0],
                    probability: 0.25,
                },
                D3Score {
                    code: self.codes[1].clone(),
                    token_id: self.ids[1],
                    probability: 0.75,
                },
            ],
        })
    }

    fn code_names(&self) -> &[String] {
        &self.codes
    }

    fn token_ids(&self) -> &[u32] {
        &self.ids
    }
}

fn kind(model: &LoadedModel) -> &'static str {
    match model {
        LoadedModel::Text { .. } => "text",
        LoadedModel::Vla(_) => "vla",
        LoadedModel::Decision(_) => "decision",
    }
}

fn request() -> DecisionSchema {
    DecisionSchema::from_json(
        r#"{"state":"open","questions":{"zeta":{"type":"noul"},"alpha":{"type":"choice"}}}"#,
    )
    .unwrap()
}

#[test]
fn old_imports_literals_methods_and_custom_implementors_remain_compatible() {
    let calls = Rc::new(Cell::new(0));
    let mut model = LoadedModel::Decision(DecisionModel::Code(Box::new(LegacyRuntime {
        calls: calls.clone(),
        codes: vec!["A".into(), "B".into()],
        ids: [101, 102],
    })));
    assert_eq!(kind(&model), "decision");
    let input: D3Input<'_> = DecisionInput {
        token_ids: &[17, 42],
        image: None,
        candidate_count: 2,
    };
    let output: DecisionOutput = model.decide(input).unwrap();
    let selected_token: u32 = output.selected().unwrap().token_id;
    assert_eq!(selected_token, 102);
    assert_eq!(output.selected_index, 1);
    let scores: Vec<DecisionScore> = output.scores;
    let restored: D3Output = DecisionOutput {
        selected_index: 1,
        scores,
    };
    let replay = model
        .decision_mut()
        .unwrap()
        .decide(DecisionInput::text(&[17, 42], 2))
        .unwrap();
    assert_eq!(replay, restored);
    assert_eq!(model.decision_code_names().unwrap(), ["A", "B"]);
    assert_eq!(model.decision_token_ids().unwrap(), [101, 102]);
    assert_eq!(calls.get(), 2);

    let error = model.decide_schema(&request()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "decision model takes encoded prompts, not schema requests"
    );
    assert_eq!(calls.get(), 2);
    let invalid = DecisionOutput {
        selected_index: 2,
        scores: restored.scores,
    };
    assert_eq!(
        invalid.selected().unwrap_err().to_string(),
        "d3 decision output: selected index 2 outside 2 scores"
    );
}

struct SchemaRuntime;

impl SchemaDecisionRuntime for SchemaRuntime {
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn decide_schema(&mut self, request: &DecisionSchema) -> Result<SchemaDecisionOutput> {
        Ok(SchemaDecisionOutput {
            answers: request
                .questions
                .iter()
                .map(|question| SchemaDecisionAnswer {
                    question_id: question.id.clone(),
                    selected_index: 1,
                    scores: vec![
                        SchemaDecisionScore {
                            code: "stay".into(),
                            probability: 0.25,
                        },
                        SchemaDecisionScore {
                            code: "go".into(),
                            probability: 0.75,
                        },
                    ],
                })
                .collect(),
        })
    }
}

#[test]
fn schema_models_take_only_schema_requests_and_keep_ordered_scores() {
    let mut model = LoadedModel::Decision(DecisionModel::Schema(Box::new(SchemaRuntime)));
    let output = model.decide_schema(&request()).unwrap();
    assert_eq!(
        output
            .answers
            .iter()
            .map(|answer| answer.question_id.as_str())
            .collect::<Vec<_>>(),
        ["zeta", "alpha"]
    );
    for answer in &output.answers {
        assert_eq!(answer.selected_index, 1);
        assert_eq!(answer.scores[0].code, "stay");
        assert_eq!(answer.scores[0].probability, 0.25);
        assert_eq!(answer.selected().unwrap().code, "go");
        assert_eq!(answer.selected().unwrap().probability, 0.75);
    }
    let invalid = SchemaDecisionAnswer {
        selected_index: 2,
        ..output.answers[0].clone()
    };
    assert!(invalid.selected().is_err());
    assert_eq!(
        model
            .decide(DecisionInput::text(&[17, 42], 2))
            .unwrap_err()
            .to_string(),
        "decision model takes schema requests, not encoded prompts"
    );
    assert_eq!(
        model.decision_code_names().unwrap_err().to_string(),
        "decision model takes schema requests and has no package code table"
    );
    assert_eq!(
        model.decision_capabilities().unwrap(),
        LlmCapabilities::TEXT_ONLY
    );
}

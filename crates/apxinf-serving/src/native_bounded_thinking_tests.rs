use super::*;
use apxinf_core::{RngKey, TokenSamplingParams};
use apxinf_model::{
    generate_streaming_controlled_with_sampler, GenerationBoundary, GenerationOptions,
    GenerationRequest, LlmInput, LlmTrait, SamplingMode,
};
use serde_json::json;

struct ToyModel {
    consumed: Vec<(Vec<u32>, u32)>,
    eos_at: Option<usize>,
}

impl LlmTrait for ToyModel {
    fn load(
        _: apxinf_loader::ModelConfig,
        _: std::collections::HashMap<String, Tensor>,
        _: apxinf_core::Device,
    ) -> apxinf_core::Result<Self> {
        Err(Error::Other("The test model has no checkpoint".into()))
    }

    fn forward(&mut self, ids: &[u32], position: u32) -> apxinf_core::Result<Tensor> {
        self.consumed.push((ids.to_vec(), position));
        let index = self.consumed.len() - 1;
        Ok(row(if self.eos_at == Some(index) || ids == [9] {
            7
        } else {
            1
        }))
    }

    fn backend(&self) -> &dyn apxinf_core::Backend {
        &CpuBackend
    }

    fn reset(&mut self) {
        self.consumed.clear();
    }

    fn vocab_size(&self) -> usize {
        10
    }
}

fn tokenizer_document() -> Value {
    json!({
        "version":"1.0",
        "added_tokens":[{"id":9,"content":"</think>","single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":false}],
        "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
        "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##",
            "max_input_chars_per_word":100,
            "vocab":{"[UNK]":0,"x":1,"<":2,"##/":3,"##think":4,"##>":5,
                "y":6,"[EOS]":7,"z":8,"</think>":9}}
    })
}

fn load_tokenizer(document: &Value) -> Tokenizer {
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../devlocal/serving-minicpm/bounded-worker/tokenizers")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("tokenizer.json");
    std::fs::write(&path, serde_json::to_vec(document).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(&path).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    tokenizer
}

fn sampler(tokenizer: &Tokenizer, capacity: usize) -> BoundedThinkingSampler<'_> {
    BoundedThinkingSampler::new(
        TokenSamplingSpec {
            vocab_size: tokenizer.vocab_size(),
            max_sequence_len: capacity,
        },
        tokenizer,
        &[7],
    )
    .unwrap()
}

fn random(temperature: f32, top_p: f32) -> TokenSamplingParams {
    TokenSamplingParams {
        selection: TokenSelection::Random {
            temperature,
            top_k: None,
            top_p,
        },
        ..Default::default()
    }
}

fn begin(sampler: &mut dyn TokenSampler, params: &TokenSamplingParams) {
    sampler
        .begin(TokenSamplingInit {
            prompt_token_ids: &[9, 1],
            params,
            rng: RngKey::new(42, 17, 0),
        })
        .unwrap();
}

fn row(token: u32) -> Tensor {
    let mut values = vec![-f32::MAX; 10];
    values[token as usize] = 0.0;
    Tensor::from_f32(vec![10], &values).unwrap()
}

fn select(sampler: &mut dyn TokenSampler, row: &Tensor) -> u32 {
    sampler
        .sample(NextTokenLogits::last(row, 10).unwrap())
        .unwrap()
        .token_id
}

#[test]
fn close_uses_position_512_for_both_modes_and_resets_per_request() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    for params in [TokenSamplingParams::greedy(), random(1.0, 0.95)] {
        let mut sampler = sampler(&tokenizer, 516);
        for _ in 0..2 {
            begin(&mut sampler, &params);
            let actual = (0..514)
                .map(|_| select(&mut sampler, &row(1)))
                .collect::<Vec<_>>();
            assert!(actual[..512].iter().all(|id| *id == 1));
            assert_eq!(&actual[511..514], &[1, 9, 1]);
            assert!(sampler
                .sample(NextTokenLogits::last(&row(1), 10).unwrap())
                .is_err());
        }
    }
}

#[test]
fn natural_multitoken_close_and_later_literal_closes_never_reopen() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    assert_eq!(
        tokenizer.decode_with_options(&[2, 3, 4, 5], false).unwrap(),
        CLOSE_TEXT
    );
    for params in [TokenSamplingParams::greedy(), random(2.0, 1.0)] {
        let mut sampler = sampler(&tokenizer, 1100);
        begin(&mut sampler, &params);
        for token in [2, 3, 4, 5] {
            assert_eq!(select(&mut sampler, &row(token)), token);
        }
        for index in 4..1050 {
            let token = if index == 500 { 9 } else { 1 };
            assert_eq!(select(&mut sampler, &row(token)), token, "index {index}");
        }
    }
}

#[test]
fn incomplete_close_at_boundary_does_not_postpone_forcing() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    let mut sampler = sampler(&tokenizer, 520);
    begin(&mut sampler, &TokenSamplingParams::greedy());
    for _ in 0..510 {
        assert_eq!(select(&mut sampler, &row(1)), 1);
    }
    assert_eq!(select(&mut sampler, &row(2)), 2);
    assert_eq!(select(&mut sampler, &row(3)), 3);
    assert_eq!(select(&mut sampler, &row(4)), 9);
    assert_eq!(select(&mut sampler, &row(6)), 6);
}

#[test]
fn eos_is_not_part_of_the_observed_decoder_prefix() {
    let mut document = tokenizer_document();
    document["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("[EOS]");
    document["model"]["vocab"]["##/think>"] = 7.into();
    let tokenizer = load_tokenizer(&document);
    assert_eq!(
        tokenizer.decode_with_options(&[2, 7], false).unwrap(),
        CLOSE_TEXT
    );
    let mut sampler = sampler(&tokenizer, 520);
    begin(&mut sampler, &TokenSamplingParams::greedy());
    assert_eq!(select(&mut sampler, &row(2)), 2);
    assert_eq!(select(&mut sampler, &row(7)), 7);
    for _ in 2..512 {
        assert_eq!(select(&mut sampler, &row(1)), 1);
    }
    assert_eq!(select(&mut sampler, &row(1)), 9);
}

#[test]
fn forced_selection_preserves_full_cpu_random_sequence_and_capacity() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    let mut bounded = sampler(&tokenizer, 602);
    let mut ordinary = CpuBackend.create_token_sampler(bounded.spec()).unwrap();
    let params = random(0.8, 1.0);
    for _ in 0..2 {
        begin(&mut bounded, &params);
        begin(ordinary.as_mut(), &params);
        let alternatives = Tensor::from_f32(
            vec![10],
            &[-20., 0., -20., -20., -20., -20., 0., -20., 0., -20.],
        )
        .unwrap();
        let mut output = Vec::new();
        let constrained = row(9);
        for index in 0..600 {
            let expected_row = if index == 512 {
                &constrained
            } else {
                &alternatives
            };
            let expected = select(ordinary.as_mut(), expected_row);
            let actual = select(&mut bounded, &alternatives);
            assert_eq!(actual, expected, "selection {index}");
            output.push(actual);
        }
        assert_eq!(output[512], 9);
        assert!(output[513..].windows(2).any(|pair| pair[0] != pair[1]));
        let logits = NextTokenLogits::last(&alternatives, 10).unwrap();
        assert!(bounded.sample(logits).is_err());
        assert!(ordinary.sample(logits).is_err());
    }
}

#[test]
fn constrained_row_accepts_positive_f32_extremes_within_serving_bounds() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    for temperature in [f32::from_bits(1), f32::MIN_POSITIVE, 1.0, 2.0] {
        for top_p in [f32::from_bits(1), 0.95, 1.0] {
            let mut sampler = sampler(&tokenizer, 515);
            begin(&mut sampler, &random(temperature, top_p));
            for _ in 0..512 {
                assert_eq!(select(&mut sampler, &row(1)), 1);
            }
            assert_eq!(select(&mut sampler, &row(1)), 9);
        }
    }
}

#[test]
fn unsupported_sampling_is_rejected_before_selection() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    let mut invalid = Vec::new();
    for temperature in [
        0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::from_bits(2.0_f32.to_bits() + 1),
    ] {
        invalid.push(random(temperature, 1.0));
    }
    for top_p in [
        0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::from_bits(1.0_f32.to_bits() + 1),
    ] {
        invalid.push(random(1.0, top_p));
    }
    invalid.push(TokenSamplingParams {
        selection: TokenSelection::Random {
            temperature: 1.0,
            top_p: 1.0,
            top_k: Some(1),
        },
        ..Default::default()
    });
    invalid.push(TokenSamplingParams {
        return_logprob: true,
        ..Default::default()
    });
    for penalties in [
        TokenPenalties {
            repetition: 1.1,
            ..Default::default()
        },
        TokenPenalties {
            frequency: 0.1,
            ..Default::default()
        },
        TokenPenalties {
            presence: 0.1,
            ..Default::default()
        },
    ] {
        invalid.push(TokenSamplingParams {
            penalties,
            ..Default::default()
        });
    }
    for params in invalid {
        let mut sampler = sampler(&tokenizer, 520);
        assert!(sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[],
                params: &params,
                rng: RngKey::default()
            })
            .is_err());
        assert!(sampler
            .sample(NextTokenLogits::last(&row(1), 10).unwrap())
            .is_err());
    }
}

#[test]
fn tokenizer_qualification_rejects_declaration_encoding_decoding_and_eos_mismatches() {
    let document = tokenizer_document();
    let tokenizer = load_tokenizer(&document);
    qualify_tokenizer(&tokenizer, &document, &[7]).unwrap();
    for (field, value) in [
        ("id", json!(8)),
        ("content", json!("wrong")),
        ("special", json!(true)),
        ("special", Value::Null),
    ] {
        let mut changed = document.clone();
        changed["added_tokens"][0][field] = value;
        assert!(qualify_tokenizer(&tokenizer, &changed, &[7]).is_err());
    }
    let mut duplicate = document.clone();
    duplicate["added_tokens"]
        .as_array_mut()
        .unwrap()
        .push(document["added_tokens"][0].clone());
    assert!(qualify_tokenizer(&tokenizer, &duplicate, &[7]).is_err());
    assert!(qualify_tokenizer(&tokenizer, &document, &[9]).is_err());
    let mut wrong_encoding = document.clone();
    wrong_encoding["added_tokens"] = json!([]);
    wrong_encoding["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("</think>");
    let smaller = load_tokenizer(&wrong_encoding);
    assert!(qualify_tokenizer(&smaller, &document, &[7]).is_err());
    wrong_encoding["model"]["vocab"]["other"] = 9.into();
    let wrong = load_tokenizer(&wrong_encoding);
    assert!(qualify_tokenizer(&wrong, &document, &[7]).is_err());
    let mut wrong_decoding = document.clone();
    wrong_decoding["decoder"] =
        json!({"type":"Replace", "pattern":{"String":"</think>"}, "content":"closed"});
    let wrong = load_tokenizer(&wrong_decoding);
    assert_eq!(wrong.encode(CLOSE_TEXT).unwrap(), [9]);
    assert!(qualify_tokenizer(&wrong, &document, &[7]).is_err());
}

#[test]
fn generation_accounts_for_close_and_eos_and_consumes_close_on_the_next_forward() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    for random_mode in [false, true] {
        for (allowance, expected_count) in [(0, 0), (512, 512), (513, 513), (514, 514), (600, 514)]
        {
            let mut model = ToyModel {
                consumed: Vec::new(),
                eos_at: None,
            };
            let mut sampler = sampler(&tokenizer, 610);
            let mut options = GenerationOptions::greedy(allowance, Some(7));
            if random_mode {
                options.sampling_mode = Some(SamplingMode::Random);
                options.temperature = Some(1.0);
                options.top_p = Some(0.95);
                options.seed = Some(42);
            }
            let mut emitted = Vec::new();
            let output = generate_streaming_controlled_with_sampler(
                &mut model,
                GenerationRequest {
                    input: LlmInput::text(&[1, 6]),
                    options: &options,
                },
                &mut sampler,
                |boundary| {
                    if let GenerationBoundary::Token(token) = boundary {
                        emitted.push(token.token_id);
                    }
                    Ok(true)
                },
            )
            .unwrap();
            assert_eq!(emitted.len(), expected_count);
            assert_eq!(output.token_ids(), emitted);
            assert_eq!(output.profile.output_tokens(), expected_count);
            if allowance <= 512 {
                assert!(emitted.iter().all(|id| *id == 1));
            }
            if allowance >= 513 {
                assert_eq!(emitted[512], 9);
            }
            if allowance >= 514 {
                assert_eq!(emitted[513], 7);
                assert_eq!(model.consumed.last(), Some(&(vec![9], 514)));
            } else {
                assert!(model.consumed.iter().all(|(ids, _)| ids != &[9]));
            }
        }
    }
}

#[test]
fn eos_and_callback_refusal_stop_before_any_delayed_close() {
    let tokenizer = load_tokenizer(&tokenizer_document());
    for (eos_at, refuse_at, expected_count) in [
        (Some(511), None, 512),
        (None, Some(512), 512),
        (None, Some(513), 513),
        (None, Some(0), 0),
    ] {
        let mut model = ToyModel {
            consumed: Vec::new(),
            eos_at,
        };
        let mut sampler = sampler(&tokenizer, 610);
        let options = GenerationOptions::greedy(600, Some(7));
        let mut emitted = Vec::new();
        let output = generate_streaming_controlled_with_sampler(
            &mut model,
            GenerationRequest {
                input: LlmInput::text(&[1, 6]),
                options: &options,
            },
            &mut sampler,
            |boundary| {
                if let GenerationBoundary::Token(token) = boundary {
                    emitted.push(token.token_id);
                }
                Ok(refuse_at != Some(emitted.len()))
            },
        )
        .unwrap();
        assert_eq!(emitted.len(), expected_count);
        assert_eq!(output.profile.output_tokens(), expected_count);
        assert_eq!(output.token_ids(), emitted);
        assert!(model.consumed.iter().all(|(ids, _)| ids != &[9]));
        if eos_at.is_some() {
            assert_eq!(emitted.last(), Some(&7));
        }
        if expected_count <= 512 {
            assert!(!emitted.contains(&9));
        }
    }
}

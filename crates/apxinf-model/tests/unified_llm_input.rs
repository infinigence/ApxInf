use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{
    Backend, CpuBackend, Device, Error, NextTokenLogits, Result, RngKey, Tensor, TokenSample,
    TokenSampler, TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec,
};
use apxinf_loader::ModelConfig;
use apxinf_model::{
    generate_streaming_controlled_with_sampler, register, Action, AutoModel, GenerationBoundary,
    GenerationOptions, GenerationRequest, ImageInput, InferenceSpec, LlmCapabilities, LlmInput,
    LlmTrait, LoadOptions, LoadedModel, PreparedInference, SamplingMode, VlaContract, VlaRequest,
    VlaRuntime,
};

#[derive(Default)]
struct TextOnlyModel {
    forward_calls: Vec<(Vec<u32>, u32)>,
    prewarm_calls: Vec<(usize, usize)>,
}

impl TextOnlyModel {
    fn logits(seq_len: usize, token: u32) -> Result<Tensor> {
        let vocab_size = 4;
        let mut values = vec![0.0; seq_len * vocab_size];
        values[(seq_len - 1) * vocab_size + token as usize] = 1.0;
        Tensor::from_f32(vec![seq_len, vocab_size], &values)
    }
}

impl LlmTrait for TextOnlyModel {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self::default())
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.forward_calls.push((token_ids.to_vec(), start_pos));
        let token = match self.forward_calls.len() {
            1 => 2,
            2 => 3,
            _ => 1,
        };
        Self::logits(token_ids.len(), token)
    }

    fn backend(&self) -> &dyn Backend {
        static BACKEND: CpuBackend = CpuBackend;
        &BACKEND
    }

    fn reset(&mut self) {
        self.forward_calls.clear();
    }

    fn prewarm_decode(&mut self, prompt_len: usize, max_new_tokens: usize) {
        self.prewarm_calls.push((prompt_len, max_new_tokens));
    }

    fn vocab_size(&self) -> usize {
        4
    }
}

struct SuppliedSampler {
    spec: TokenSamplingSpec,
    tokens: Vec<u32>,
    sample_calls: usize,
    initializations: Vec<(Vec<u32>, TokenSamplingParams, RngKey)>,
    fail_begin: bool,
}

impl SuppliedSampler {
    fn new(capacity: usize, tokens: &[u32]) -> Self {
        Self {
            spec: TokenSamplingSpec {
                vocab_size: 4,
                max_sequence_len: capacity,
            },
            tokens: tokens.to_vec(),
            sample_calls: 0,
            initializations: Vec::new(),
            fail_begin: false,
        }
    }
}

impl TokenSampler for SuppliedSampler {
    fn spec(&self) -> TokenSamplingSpec {
        self.spec
    }

    fn begin(&mut self, init: TokenSamplingInit<'_>) -> Result<()> {
        self.initializations.push((
            init.prompt_token_ids.to_vec(),
            init.params.clone(),
            init.rng,
        ));
        self.sample_calls = 0;
        if self.fail_begin {
            return Err(Error::Other(
                "supplied sampler initialization failed".into(),
            ));
        }
        Ok(())
    }

    fn sample(&mut self, logits: NextTokenLogits<'_>) -> Result<TokenSample> {
        assert_eq!(logits.vocab_size(), self.spec.vocab_size);
        assert_eq!(logits.row_index() + 1, logits.tensor().shape().dims()[0]);
        let token_id = self.tokens[self.sample_calls];
        self.sample_calls += 1;
        Ok(TokenSample {
            token_id,
            logprob: Some(-0.5),
        })
    }
}

#[test]
fn supplied_sampler_controls_prefill_and_decode_tokens() {
    let mut model = TextOnlyModel::default();
    let options = GenerationOptions::greedy(3, None);
    let mut sampler = SuppliedSampler::new(8, &[1, 0, 2]);
    let mut streamed = Vec::new();
    let output = generate_streaming_controlled_with_sampler(
        &mut model as &mut dyn LlmTrait,
        GenerationRequest {
            input: LlmInput::text(&[0, 1]),
            options: &options,
        },
        &mut sampler,
        |boundary| {
            if let GenerationBoundary::Token(token) = boundary {
                streamed.push(token);
            }
            Ok(true)
        },
    )
    .unwrap();

    assert_eq!(output.token_ids(), vec![1, 0, 2]);
    assert_eq!(streamed, output.tokens);
    assert!(streamed.iter().all(|token| token.logprob == Some(-0.5)));
    assert_eq!(sampler.sample_calls, 3);
    assert_eq!(
        model.forward_calls,
        vec![(vec![0, 1], 0), (vec![1], 2), (vec![0], 3)]
    );
}

#[test]
fn supplied_sampler_begins_each_request_with_prompt_resolved_params_and_seed() {
    let mut model = TextOnlyModel::default();
    let mut sampler = SuppliedSampler::new(8, &[0]);
    let options = GenerationOptions {
        max_new_tokens: Some(1),
        temperature: Some(0.7),
        top_p: Some(0.8),
        repetition_penalty: Some(1.2),
        seed: Some(41),
        ..GenerationOptions::default()
    };
    let second = GenerationOptions {
        seed: Some(97),
        ..options.clone()
    };
    for (prompt, settings) in [(&[3, 1, 3][..], &options), (&[2][..], &second)] {
        generate_streaming_controlled_with_sampler(
            &mut model,
            GenerationRequest {
                input: LlmInput::text(prompt),
                options: settings,
            },
            &mut sampler,
            |_| Ok(true),
        )
        .unwrap();
    }

    assert_eq!(sampler.initializations.len(), 2);
    assert_eq!(sampler.initializations[0].0, vec![3, 1, 3]);
    assert_eq!(sampler.initializations[1].0, vec![2]);
    assert_eq!(sampler.initializations[0].2, RngKey::new(41, 0, 0));
    assert_eq!(sampler.initializations[1].2, RngKey::new(97, 0, 0));
    let params = &sampler.initializations[0].1;
    assert_eq!(
        params.selection,
        apxinf_core::TokenSelection::Random {
            temperature: 0.7,
            top_k: None,
            top_p: 0.8,
        }
    );
    assert_eq!(
        params.penalties,
        apxinf_core::TokenPenalties {
            repetition: 1.2,
            frequency: 0.0,
            presence: 0.0,
        }
    );
    assert!(!params.return_logprob);
    assert_eq!(sampler.initializations[1].1, *params);
}

#[test]
fn supplied_sampler_rejects_bad_specs_before_model_mutation() {
    for (vocab_size, capacity) in [(3, 8), (4, 3), (0, 8), (4, 0)] {
        let mut model = TextOnlyModel::default();
        model.forward_calls.push((vec![3], 19));
        let options = GenerationOptions::greedy(2, None);
        let mut sampler = SuppliedSampler::new(capacity, &[]);
        sampler.spec.vocab_size = vocab_size;
        let result = generate_streaming_controlled_with_sampler(
            &mut model,
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            &mut sampler,
            |_| panic!("invalid sampler invoked boundary callback"),
        );
        assert!(result.is_err());
        assert_eq!(model.forward_calls, vec![(vec![3], 19)]);
        assert!(model.prewarm_calls.is_empty());
        assert!(sampler.initializations.is_empty());
    }
}

#[test]
fn supplied_sampler_zero_output_checks_spec_without_initialization_or_model_mutation() {
    let mut model = TextOnlyModel::default();
    model.forward_calls.push((vec![3], 19));
    let options = GenerationOptions::greedy(0, None);
    let mut sampler = SuppliedSampler::new(2, &[]);
    for capacity in [2, 1] {
        sampler.spec.max_sequence_len = capacity;
        let result = generate_streaming_controlled_with_sampler(
            &mut model,
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            &mut sampler,
            |_| panic!("zero-output request invoked boundary callback"),
        );
        if capacity == 2 {
            assert!(result.unwrap().tokens.is_empty());
        } else {
            assert!(result.is_err());
        }
    }
    assert_eq!(model.forward_calls, vec![(vec![3], 19)]);
    assert!(model.prewarm_calls.is_empty());
    assert!(sampler.initializations.is_empty());
}

#[test]
fn supplied_sampler_prefill_cancel_does_not_sample() {
    for cancel_at_start in [true, false] {
        let mut model = TextOnlyModel::default();
        let options = GenerationOptions::greedy(3, None);
        let mut sampler = SuppliedSampler::new(5, &[]);
        let mut boundaries = Vec::new();
        let output = generate_streaming_controlled_with_sampler(
            &mut model,
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            &mut sampler,
            |boundary| {
                boundaries.push(boundary);
                Ok(!cancel_at_start && matches!(boundary, GenerationBoundary::PrefillStart))
            },
        )
        .unwrap();
        assert!(output.tokens.is_empty());
        assert_eq!(sampler.initializations.len(), 1);
        assert_eq!(sampler.sample_calls, 0);
        assert_eq!(boundaries[0], GenerationBoundary::PrefillStart);
        if cancel_at_start {
            assert_eq!(boundaries.len(), 1);
            assert!(model.forward_calls.is_empty());
        } else {
            assert_eq!(
                boundaries,
                vec![
                    GenerationBoundary::PrefillStart,
                    GenerationBoundary::PrefillProgress { consumed_tokens: 2 }
                ]
            );
            assert_eq!(model.forward_calls, vec![(vec![0, 1], 0)]);
        }
    }
}

#[test]
fn supplied_sampler_eos_and_token_stop_preserve_the_last_emitted_token() {
    for stop_on_token in [true, false] {
        let mut model = TextOnlyModel::default();
        let options = GenerationOptions {
            eos_token_ids: Some(vec![0, 3]),
            ..GenerationOptions::greedy(4, None)
        };
        let mut sampler = SuppliedSampler::new(6, &[1, 3]);
        let mut streamed = Vec::new();
        let output = generate_streaming_controlled_with_sampler(
            &mut model,
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            &mut sampler,
            |boundary| {
                if let GenerationBoundary::Token(token) = boundary {
                    streamed.push(token.token_id);
                    return Ok(!stop_on_token);
                }
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(
            output.token_ids(),
            if stop_on_token { vec![1] } else { vec![1, 3] }
        );
        assert_eq!(output.token_ids(), streamed);
        assert_eq!(model.forward_calls.len(), streamed.len());
        assert_eq!(sampler.sample_calls, streamed.len());
    }
}

#[test]
fn supplied_sampler_begin_error_precedes_model_mutation() {
    let mut model = TextOnlyModel::default();
    model.forward_calls.push((vec![3], 19));
    let options = GenerationOptions::greedy(1, None);
    let mut sampler = SuppliedSampler::new(3, &[]);
    sampler.fail_begin = true;
    let error = generate_streaming_controlled_with_sampler(
        &mut model,
        GenerationRequest {
            input: LlmInput::text(&[0, 1]),
            options: &options,
        },
        &mut sampler,
        |_| panic!("failed initialization invoked boundary callback"),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("initialization failed"));
    assert_eq!(model.forward_calls, vec![(vec![3], 19)]);
    assert!(model.prewarm_calls.is_empty());
}

fn load_generation_config_test_model(
    _path: &Path,
    _device: Device,
    _backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    Ok(LoadedModel::text(Box::new(TextOnlyModel::default())))
}

#[test]
fn controlled_generation_stops_after_prefill_without_sampling_a_token() {
    let mut model = TextOnlyModel::default();
    let options = GenerationOptions::greedy(4, Some(1));
    let mut boundaries = Vec::new();
    let output = model
        .generate_streaming_controlled_dyn(
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            &mut |boundary| {
                boundaries.push(boundary);
                Ok(matches!(
                    boundary,
                    apxinf_model::GenerationBoundary::PrefillStart
                ))
            },
        )
        .unwrap();
    assert!(output.tokens.is_empty());
    assert_eq!(
        boundaries,
        vec![
            apxinf_model::GenerationBoundary::PrefillStart,
            apxinf_model::GenerationBoundary::PrefillProgress { consumed_tokens: 2 },
        ]
    );
    assert_eq!(model.forward_calls, vec![(vec![0, 1], 0)]);
}

#[test]
fn controlled_generation_stops_before_prompt_execution() {
    let mut model = TextOnlyModel::default();
    let options = GenerationOptions::greedy(4, Some(1));
    let mut boundaries = Vec::new();
    let output = model
        .generate_streaming_controlled_dyn(
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            &mut |boundary| {
                boundaries.push(boundary);
                Ok(false)
            },
        )
        .unwrap();
    assert_eq!(
        boundaries,
        vec![apxinf_model::GenerationBoundary::PrefillStart]
    );
    assert!(output.tokens.is_empty());
    assert!(model.forward_calls.is_empty());
}

#[test]
fn controlled_generation_preserves_tokens_and_stops_before_another_decode() {
    let options = GenerationOptions::greedy(4, Some(1));
    let mut model = TextOnlyModel::default();
    let request = GenerationRequest {
        input: LlmInput::text(&[0]),
        options: &options,
    };
    let baseline = model
        .generate_streaming_with_options(request, |_| {})
        .unwrap()
        .token_ids();
    let controlled = model
        .generate_streaming_controlled_dyn(request, &mut |_| Ok(true))
        .unwrap()
        .token_ids();
    assert_eq!(baseline, controlled);
    let stopped = model
        .generate_streaming_controlled_dyn(request, &mut |boundary| {
            Ok(!matches!(
                boundary,
                apxinf_model::GenerationBoundary::Token(_)
            ))
        })
        .unwrap()
        .token_ids();
    assert_eq!(stopped, vec![2]);
    assert_eq!(model.forward_calls.len(), 1);
}

struct DummyVlaModel;

impl VlaRuntime for DummyVlaModel {
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [1, 1],
            patch_shape: [1, 1],
            max_token_len: 1,
            num_views: 1,
            image_size: 1,
            patch_size: 1,
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, _request: &VlaRequest<'_>) -> Result<Action> {
        unreachable!("generation-config routing test does not run VLA inference")
    }

    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        unreachable!("generation-config routing test does not prepare VLA inference")
    }

    fn infer_host_f32(&self, _request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        unreachable!("generation-config routing test does not run VLA inference")
    }
}

fn load_generation_config_test_vla(
    _path: &Path,
    _device: Device,
    _backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    Ok(LoadedModel::Vla(Box::new(DummyVlaModel)))
}

#[derive(Default)]
struct VisionModel {
    saw_image_prefill: bool,
    decode_calls: Vec<(Vec<u32>, u32)>,
}

impl LlmTrait for VisionModel {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self::default())
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities { image: true }
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        self.saw_image_prefill = input.image.is_some();
        TextOnlyModel::logits(input.token_ids.len(), 2)
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.decode_calls.push((token_ids.to_vec(), start_pos));
        TextOnlyModel::logits(token_ids.len(), 3)
    }

    fn backend(&self) -> &dyn Backend {
        static BACKEND: CpuBackend = CpuBackend;
        &BACKEND
    }

    fn reset(&mut self) {
        self.saw_image_prefill = false;
        self.decode_calls.clear();
    }

    fn vocab_size(&self) -> usize {
        4
    }
}

#[test]
fn text_generation_keeps_the_existing_prefill_and_decode_path() {
    let mut model = TextOnlyModel::default();
    let mut streamed = Vec::new();

    let (generated, _) = model
        .generate_streaming(
            LlmInput::text(&[0, 1]),
            3,
            |token| streamed.push(token),
            None,
        )
        .unwrap();

    assert_eq!(generated, vec![2, 3, 1]);
    assert_eq!(streamed, generated);
    assert_eq!(model.prewarm_calls, vec![(2, 3)]);
    assert_eq!(
        model.forward_calls,
        vec![(vec![0, 1], 0), (vec![2], 2), (vec![3], 3)]
    );
}

#[test]
fn text_only_model_rejects_an_image_before_forward() {
    let pixels = Tensor::from_f32(vec![1, 4], &[0.0; 4]).unwrap();
    let grid = [[1, 2, 2]];
    let mut model = TextOnlyModel::default();

    let error = match model.generate_streaming(
        LlmInput::with_image(&[0, 1], ImageInput::new(&pixels, &grid)),
        1,
        |_| {},
        None,
    ) {
        Ok(_) => panic!("text-only model accepted image input"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("does not support image input"));
    assert!(model.forward_calls.is_empty());
    assert!(model.prewarm_calls.is_empty());
}

#[test]
fn image_is_consumed_once_at_prefill_and_not_in_the_decode_loop() {
    let pixels = Tensor::from_f32(vec![1, 4], &[0.0; 4]).unwrap();
    let grid = [[1, 2, 2]];
    let mut model = VisionModel::default();

    let (generated, _) = model
        .generate_streaming(
            LlmInput::with_image(&[0, 1, 2], ImageInput::new(&pixels, &grid)),
            2,
            |_| {},
            None,
        )
        .unwrap();

    assert_eq!(generated, vec![2, 3]);
    assert!(model.saw_image_prefill);
    assert_eq!(model.decode_calls, vec![(vec![2], 3)]);
}

#[test]
fn loaded_model_uses_the_same_generation_interface() {
    let mut model = LoadedModel::text(Box::new(TextOnlyModel::default()));

    assert_eq!(
        model.text_capabilities().unwrap(),
        LlmCapabilities::default()
    );
    let (generated, _) = model
        .generate_streaming(LlmInput::text(&[1, 2]), 2, |_| {}, None)
        .unwrap();

    assert_eq!(generated, vec![2, 3]);
}

#[test]
fn options_api_exposes_sampling_and_logprob_without_changing_model_hooks() {
    let mut model = TextOnlyModel::default();
    let options = GenerationOptions {
        max_new_tokens: Some(2),
        eos_token_ids: Some(Vec::new()),
        sampling_mode: Some(SamplingMode::Random),
        temperature: Some(0.8),
        top_k: Some(1),
        top_p: Some(1.0),
        seed: Some(7),
        return_logprob: Some(true),
        ..GenerationOptions::default()
    };
    let mut streamed = Vec::new();
    let output = model
        .generate_streaming_with_options(
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            |token| streamed.push(token),
        )
        .unwrap();

    assert_eq!(output.token_ids(), vec![2, 3]);
    assert_eq!(streamed, output.tokens);
    assert!(output.tokens.iter().all(|token| token.logprob == Some(0.0)));
}

#[test]
fn zero_token_generation_does_not_run_the_model() {
    let mut model = TextOnlyModel::default();
    let options = GenerationOptions::greedy(0, None);
    let output = model
        .generate_streaming_with_options(
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            |_| panic!("zero-token request invoked callback"),
        )
        .unwrap();

    assert!(output.tokens.is_empty());
    assert!(model.forward_calls.is_empty());
    assert!(model.prewarm_calls.is_empty());
}

#[test]
fn any_configured_eos_token_stops_before_another_decode() {
    let mut model = TextOnlyModel::default();
    let options = GenerationOptions {
        max_new_tokens: Some(8),
        eos_token_ids: Some(vec![0, 3]),
        ..GenerationOptions::greedy(8, None)
    };
    let mut streamed = Vec::new();
    let output = model
        .generate_streaming_with_options(
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            |token| streamed.push(token.token_id),
        )
        .unwrap();

    assert_eq!(output.token_ids(), vec![2, 3]);
    assert_eq!(streamed, vec![2, 3]);
    assert_eq!(model.forward_calls, vec![(vec![0, 1], 0), (vec![2], 2)]);
}

#[test]
fn invalid_sampling_options_fail_before_model_work() {
    let mut model = TextOnlyModel::default();
    let options = GenerationOptions {
        max_new_tokens: Some(1),
        eos_token_ids: Some(Vec::new()),
        sampling_mode: Some(SamplingMode::Random),
        temperature: Some(-0.1),
        ..GenerationOptions::default()
    };
    let error = model
        .generate_streaming_with_options(
            GenerationRequest {
                input: LlmInput::text(&[0, 1]),
                options: &options,
            },
            |_| {},
        )
        .err()
        .expect("negative sampling temperature should fail");

    assert!(error.to_string().contains("temperature"));
    assert!(model.forward_calls.is_empty());
    assert!(model.prewarm_calls.is_empty());
}

#[test]
fn auto_model_detects_the_registry_name_from_hugging_face_config() {
    let dir =
        std::env::temp_dir().join(format!("apxinf-unified-input-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_vl"}"#).unwrap();

    let detected = AutoModel::detect_model_name(&dir).unwrap();

    assert_eq!(detected, "qwen3_vl");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn auto_model_loads_generation_config_and_request_values_override_it() {
    let dir = std::env::temp_dir().join(format!(
        "apxinf-generation-defaults-test-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"generation_config_test_model"}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        r#"{
            "max_new_tokens": 3,
            "do_sample": true,
            "temperature": 0.7,
            "top_k": 1,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "eos_token_id": [2, 3]
        }"#,
    )
    .unwrap();
    register(
        "generation_config_test_model",
        load_generation_config_test_model,
    );

    let mut model = AutoModel::load_model(Device::Cpu, &dir, &LoadOptions::default()).unwrap();
    let defaults = model.generation_defaults().unwrap();
    assert_eq!(defaults.max_new_tokens, Some(3));
    assert_eq!(defaults.sampling_mode, Some(SamplingMode::Random));
    assert_eq!(defaults.temperature, Some(0.7));
    assert_eq!(defaults.eos_token_ids, Some(vec![2, 3]));

    let deployment = LoadOptions {
        generation_overrides: GenerationOptions {
            max_new_tokens: Some(2),
            temperature: Some(0.0),
            ..GenerationOptions::default()
        },
        ..LoadOptions::default()
    };
    let deployed = AutoModel::load_model(Device::Cpu, &dir, &deployment).unwrap();
    let deployed_defaults = deployed.generation_defaults().unwrap();
    assert_eq!(deployed_defaults.max_new_tokens, Some(2));
    assert_eq!(deployed_defaults.sampling_mode, Some(SamplingMode::Greedy));

    let request = GenerationOptions {
        max_new_tokens: Some(1),
        temperature: Some(0.0),
        eos_token_ids: Some(Vec::new()),
        ..GenerationOptions::default()
    };
    let output = model
        .generate_streaming_with_options(LlmInput::text(&[0, 1]), &request, |_| {})
        .unwrap();
    assert_eq!(output.token_ids(), vec![2]);

    std::fs::write(dir.join("generation_config.json"), "not valid json").unwrap();
    let malformed = AutoModel::load_model(Device::Cpu, &dir, &LoadOptions::default())
        .err()
        .expect("malformed text generation config should fail model load");
    assert!(malformed.to_string().contains("generation_config.json"));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn vla_loads_do_not_read_generation_config() {
    let dir =
        std::env::temp_dir().join(format!("apxinf-generation-vla-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"generation_config_test_vla"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("generation_config.json"), "not valid json").unwrap();
    register(
        "generation_config_test_vla",
        load_generation_config_test_vla,
    );

    let model = AutoModel::load_model(Device::Cpu, &dir, &LoadOptions::default()).unwrap();
    assert!(model.generation_defaults().is_err());

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn load_model_unifies_detected_and_explicit_model_selection() {
    let dir = std::env::temp_dir().join(format!("apxinf-unified-load-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"missing_auto_model"}"#,
    )
    .unwrap();

    let detected_error = AutoModel::load_model(Device::Cpu, &dir, &LoadOptions::default())
        .err()
        .expect("an unregistered detected model should fail");
    assert!(detected_error.to_string().contains("missing_auto_model"));

    let options = LoadOptions {
        model_name: Some("missing_override_model".to_owned()),
        ..LoadOptions::default()
    };
    let override_error = AutoModel::load_model(Device::Cpu, &dir, &options)
        .err()
        .expect("an unregistered override model should fail");
    assert!(override_error
        .to_string()
        .contains("missing_override_model"));

    std::fs::remove_dir_all(dir).unwrap();
}


struct OverrideSensitiveModel(TextOnlyModel);
impl LlmTrait for OverrideSensitiveModel {
    fn load(_: ModelConfig, _: HashMap<String, Tensor>, _: Device) -> Result<Self> { unreachable!() }
    fn forward(&mut self, ids: &[u32], position: u32) -> Result<Tensor> { self.0.forward(ids, position) }
    fn backend(&self) -> &dyn Backend { self.0.backend() }
    fn reset(&mut self) { self.0.reset(); }
    fn vocab_size(&self) -> usize { 4 }
    fn token_sequence<'a>(&'a mut self, _: Tensor, _: Box<dyn TokenSampler>, _: &TokenSamplingParams, _: u32)
        -> Result<Box<dyn apxinf_model::llm_trait::TokenSequence + 'a>> {
        Err(Error::Contract("family sequence selected"))
    }
}

#[test]
fn caller_sampler_is_not_bypassed_by_a_family_greedy_sequence() {
    let mut model = OverrideSensitiveModel(TextOnlyModel::default());
    let options = GenerationOptions::greedy(2, None);
    let input = LlmInput::text(&[0, 1]);
    let mut sampler = SuppliedSampler::new(4, &[0, 1]);
    let result = generate_streaming_controlled_with_sampler(&mut model,
        GenerationRequest { input, options: &options }, &mut sampler, |_| Ok(true)).unwrap();
    assert_eq!(result.token_ids(), vec![0, 1]);
    let error = model.generate_streaming_with_options(GenerationRequest { input, options: &options }, |_| {}).err().unwrap();
    assert!(error.to_string().contains("family sequence selected"));
}

#[test]
fn callback_error_finishes_the_family_sequence_and_preserves_the_error() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Producer(Arc<AtomicBool>);
    struct Sequence(Arc<AtomicBool>);
    impl apxinf_model::llm_trait::TokenSequence for Sequence {
        fn next(&mut self, _: bool) -> Result<apxinf_model::GeneratedToken> {
            Ok(apxinf_model::GeneratedToken { token_id: 2, logprob: None })
        }
        fn finish(self: Box<Self>, _: usize) -> Result<()> {
            self.0.store(true, Ordering::Release);
            Ok(())
        }
    }
    impl LlmTrait for Producer {
        fn load(_: ModelConfig, _: HashMap<String, Tensor>, _: Device) -> Result<Self> { unreachable!() }
        fn forward(&mut self, ids: &[u32], _: u32) -> Result<Tensor> { TextOnlyModel::logits(ids.len(), 2) }
        fn backend(&self) -> &dyn Backend { &CpuBackend }
        fn reset(&mut self) {}
        fn vocab_size(&self) -> usize { 4 }
        fn token_sequence<'a>(&'a mut self, _: Tensor, _: Box<dyn TokenSampler>, _: &TokenSamplingParams, _: u32)
            -> Result<Box<dyn apxinf_model::llm_trait::TokenSequence + 'a>> {
            Ok(Box::new(Sequence(self.0.clone())))
        }
    }
    let finished = Arc::new(AtomicBool::new(false));
    let mut model = Producer(finished.clone());
    let options = GenerationOptions::greedy(3, None);
    let failure = model.generate_streaming_controlled_dyn(GenerationRequest {
        input: LlmInput::text(&[0, 1]), options: &options,
    }, &mut |event| {
        assert!(!finished.load(Ordering::Acquire));
        if matches!(event, GenerationBoundary::Token(_)) { Err(Error::Contract("consumer rejected output")) }
        else { Ok(true) }
    }).err().unwrap();
    assert!(failure.to_string().contains("consumer rejected output"));
    assert!(finished.load(Ordering::Acquire));
}

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, CpuBackend, Device, Result, Tensor};
use apxinf_loader::ModelConfig;
use apxinf_model::{
    register, Action, AutoModel, GenerationOptions, GenerationRequest, ImageInput, InferenceSpec,
    LlmCapabilities, LlmInput, LlmTrait, LoadOptions, LoadedModel, PreparedInference, SamplingMode,
    VlaRequest, VlaRuntime,
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

fn load_generation_config_test_model(
    _path: &Path,
    _device: Device,
    _backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    Ok(LoadedModel::text(Box::new(TextOnlyModel::default())))
}

struct DummyVlaModel;

impl VlaRuntime for DummyVlaModel {
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

use std::collections::HashMap;

use apxinf_core::{Device, Result, Tensor};
use apxinf_loader::ModelConfig;
use apxinf_model::{
    AutoModel, ImageInput, LlmCapabilities, LlmInput, LlmTrait, LoadOptions, LoadedModel,
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
            LlmInput::text(&[7, 8]),
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
        vec![(vec![7, 8], 0), (vec![2], 2), (vec![3], 3)]
    );
}

#[test]
fn text_only_model_rejects_an_image_before_forward() {
    let pixels = Tensor::from_f32(vec![1, 4], &[0.0; 4]).unwrap();
    let grid = [[1, 2, 2]];
    let mut model = TextOnlyModel::default();

    let error = match model.generate_streaming(
        LlmInput::with_image(&[7, 8], ImageInput::new(&pixels, &grid)),
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
            LlmInput::with_image(&[10, 11, 12], ImageInput::new(&pixels, &grid)),
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
    let mut model = LoadedModel::Text(Box::new(TextOnlyModel::default()));

    assert_eq!(
        model.text_capabilities().unwrap(),
        LlmCapabilities::default()
    );
    let (generated, _) = model
        .generate_streaming(LlmInput::text(&[5, 6]), 2, |_| {}, None)
        .unwrap();

    assert_eq!(generated, vec![2, 3]);
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

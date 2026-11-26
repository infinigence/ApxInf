//! Offline, CPU-only replay of checkpoint processor outputs. No reference
//! framework, network, model weights or accelerator is required by these tests.
use apxinf_core::DType;
use apxinf_model::vision_preprocess::{DecodedImage, VisionContract};
use apxinf_tokenizer::{ChatTemplateMessage, Tokenizer};
use serde_json::{json, Value};
use std::{io::Read, path::PathBuf};
fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3vl-preprocess")
}
fn inflate(name: &str) -> Vec<u8> {
    let input = std::fs::File::open(fixtures().join(name)).unwrap();
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(input)
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}
fn tokenizer() -> Tokenizer {
    Tokenizer::from_json_assets(
        &inflate("tokenizer.json.gz"),
        &std::fs::read(fixtures().join("tokenizer_config.json")).unwrap(),
        None,
    )
    .unwrap()
}
fn manifest() -> Value {
    serde_json::from_slice(&std::fs::read(fixtures().join("manifest.json")).unwrap()).unwrap()
}
fn tokens(tokenizer: &Tokenizer, messages: Value, options: &Value) -> Vec<u32> {
    let messages: Vec<ChatTemplateMessage> = serde_json::from_value(messages).unwrap();
    let text = tokenizer
        .apply_chat_template_messages_with_options(&messages, options.as_object().unwrap())
        .unwrap();
    tokenizer.encode(&text).unwrap()
}
fn rgb(name: &str, width: usize, height: usize) -> DecodedImage {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            if name == "scale-down" {
                rgb.extend([
                    ((x / 1024 * 40) % 256) as u8,
                    ((y / 1024 * 50) % 256) as u8,
                    120,
                ]);
            } else {
                rgb.extend([
                    ((x * 17 + y * 3) % 256) as u8,
                    ((x * 5 + y * 11) % 256) as u8,
                    ((x * 7 + y * 13) % 256) as u8,
                ]);
            }
        }
    }
    DecodedImage { rgb, width, height }
}
#[test]
fn checkpoint_pixels_grids_and_tokens_are_elementwise_exact() {
    let tokenizer = tokenizer();
    let manifest = manifest();
    let contract = VisionContract::from_dir(&fixtures(), "qwen3vl", 1, DType::F32).unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let width = case["width"].as_u64().unwrap() as usize;
        let height = case["height"].as_u64().unwrap() as usize;
        let initial = tokens(
            &tokenizer,
            json!([{"role":"user","content":[{"type":"text","text":"Describe."},{"type":"image"}]}]),
            &json!({"enable_thinking":false}),
        );
        let prepared = contract
            .prepare(&initial, &[rgb(name, width, height)], usize::MAX)
            .unwrap();
        assert_eq!(json!(prepared.grid), case["grid"], "{name}");
        assert_eq!(json!(prepared.tokens), case["tokens"], "{name}");
        assert_eq!(
            json!(prepared.pixels.shape().dims()),
            case["shape"],
            "{name}"
        );
        let expected = inflate(&format!("{name}.f32.gz"));
        let values = prepared.pixels.as_f32().unwrap();
        assert_eq!(values.len() * 4, expected.len());
        let mut max_abs = 0f32;
        for (i, (actual, raw)) in values.iter().zip(expected.chunks_exact(4)).enumerate() {
            let reference = f32::from_le_bytes(raw.try_into().unwrap());
            assert!(actual.is_finite() && reference.is_finite());
            let delta = (actual - reference).abs();
            max_abs = max_abs.max(delta);
            assert_eq!(
                delta, 0.,
                "{name}: pixel {i}, native={actual}, reference={reference}"
            );
        }
        assert_eq!(max_abs, 0.);
    }
}
#[test]
fn structured_tool_history_uses_one_template_and_preserves_text_tokens() {
    let tokenizer = tokenizer();
    let manifest = manifest();
    let tool = &manifest["tool_history"];
    let contract = VisionContract::from_dir(&fixtures(), "qwen3vl", 1, DType::F32).unwrap();
    let initial = tokens(&tokenizer, tool["messages"].clone(), &tool["options"]);
    let prepared = contract
        .prepare(
            &initial,
            &[DecodedImage {
                rgb: [255, 0, 0].repeat(40 * 24),
                width: 40,
                height: 24,
            }],
            usize::MAX,
        )
        .unwrap();
    assert_eq!(json!(prepared.tokens), tool["tokens"]);
    let mut text = tool["messages"].clone();
    text[4]["content"] = json!("Compare.");
    let text_tokens = tokens(&tokenizer, text, &tool["options"]);
    assert_eq!(json!(text_tokens), tool["text_tokens"]);
    let config: Value =
        serde_json::from_slice(&std::fs::read(fixtures().join("config.json")).unwrap()).unwrap();
    let start = config["vision_start_token_id"].as_u64().unwrap() as u32;
    let end = config["vision_end_token_id"].as_u64().unwrap() as u32;
    let mut stripped = Vec::new();
    let mut in_image = false;
    for token in prepared.tokens {
        if token == start {
            in_image = true;
        } else if token == end {
            in_image = false;
        } else if !in_image {
            stripped.push(token);
        }
    }
    assert_eq!(stripped, text_tokens);
}
#[test]
fn checkpoint_contract_failures_name_file_and_field() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        fixtures().join("config.json"),
        dir.path().join("config.json"),
    )
    .unwrap();
    let missing = VisionContract::from_dir(dir.path(), "qwen3vl", 1, DType::BF16)
        .unwrap_err()
        .to_string();
    assert!(missing.contains("preprocessor_config.json"));
    let original: Value = serde_json::from_slice(
        &std::fs::read(fixtures().join("preprocessor_config.json")).unwrap(),
    )
    .unwrap();
    for (field, value) in [
        ("patch_size", json!(3)),
        ("do_normalize", json!(false)),
        ("image_std", json!([0., 0., 0.])),
        ("rescale_factor", json!(-1.)),
    ] {
        let mut invalid = original.clone();
        invalid[field] = value;
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            invalid.to_string(),
        )
        .unwrap();
        let error = VisionContract::from_dir(dir.path(), "qwen3vl", 1, DType::BF16)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("preprocessor_config.json") && error.contains(field),
            "{error}"
        );
    }
}
#[test]
fn image_budget_and_placeholder_errors_are_rejected_before_pixel_work() {
    let c = VisionContract::from_dir(&fixtures(), "qwen3vl", 1, DType::F32).unwrap();
    let image = DecodedImage {
        rgb: vec![],
        width: 512,
        height: 320,
    };
    let error = c
        .prepare(&[c.image_token_id], &[image.clone(), image.clone()], 4096)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("max_images=1"));
    let error = c
        .prepare(&[c.image_token_id], &[image], 1)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("prompt limit"));
}

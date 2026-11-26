use super::*;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

static FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

/// A real, tiny CPU Llama checkpoint. Zero projections make greedy output
/// deterministic, so these checks exercise loading/generation/detokenization
/// without downloading weights or replacing the native model with a mock.
struct Checkpoint(PathBuf);

impl Checkpoint {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../devlocal/serving-contracts")
            .join(format!(
                "{}-{}",
                std::process::id(),
                FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&path).unwrap();
        let config = json!({
            "model_type":"llama", "hidden_size":8, "intermediate_size":16,
            "num_hidden_layers":1, "num_attention_heads":2,
            "num_key_value_heads":1, "vocab_size":8,
            "max_position_embeddings":128, "rms_norm_eps":0.00001
        });
        std::fs::write(path.join("config.json"), config.to_string()).unwrap();
        let mut header = Map::new();
        header.insert("__metadata__".into(), json!({"config":config.to_string()}));
        let mut data = Vec::new();
        for (name, shape, value) in [
            ("model.embed_tokens.weight", vec![8, 8], 0.0_f32),
            ("model.norm.weight", vec![8], 1.0),
            ("lm_head.weight", vec![8, 8], 0.0),
            ("model.layers.0.input_layernorm.weight", vec![8], 1.0),
            (
                "model.layers.0.post_attention_layernorm.weight",
                vec![8],
                1.0,
            ),
            ("model.layers.0.self_attn.q_proj.weight", vec![8, 8], 0.0),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 8], 0.0),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 8], 0.0),
            ("model.layers.0.self_attn.o_proj.weight", vec![8, 8], 0.0),
            ("model.layers.0.mlp.gate_proj.weight", vec![16, 8], 0.0),
            ("model.layers.0.mlp.up_proj.weight", vec![16, 8], 0.0),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 16], 0.0),
        ] {
            let start = data.len();
            for _ in 0..shape.iter().product::<usize>() {
                data.extend(value.to_le_bytes());
            }
            header.insert(
                name.into(),
                json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        std::fs::write(path.join("model.safetensors"), bytes).unwrap();
        std::fs::write(
            path.join("tokenizer.json"),
            json!({
                "version":"1.0", "truncation":null, "padding":null,
                "added_tokens":[], "normalizer":null,
                "pre_tokenizer":{"type":"WhitespaceSplit"},
                "post_processor":null, "decoder":null,
                "model":{"type":"WordLevel","unk_token":"[UNK]","vocab":{
                    "hello":0,"world":1,"[UNK]":2,"user":3,
                    "assistant":4,"system":5,"tool":6,"[EOS]":7
                }}
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            path.join("tokenizer_config.json"),
            json!({
                "eos_token":"[EOS]",
                "chat_template":"{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}assistant"
            })
            .to_string(),
        )
        .unwrap();
        Self(path)
    }

    fn engine(&self, max_input_tokens: usize) -> ApxInfChatEngine {
        ApxInfChatEngine::load(ApxInfEngineConfig {
            model_path: self.0.clone(),
            device: Device::Cpu,
            load_options: LoadOptions::default(),
            default_max_tokens: 4,
            max_input_tokens,
        })
        .unwrap()
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn request() -> EngineRequest {
    EngineRequest {
        images: vec![],
        messages: vec![ChatCompletionMessage {
            role: "user".into(),
            content: Some(json!("world")),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: vec![],
        tool_choice: ToolChoice::None,
        chat_template_kwargs: Map::new(),
        enable_thinking: false,
        max_tokens: 4,
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: Some(0),
        stop_after_tool_call: false,
    }
}

#[test]
fn native_cpu_engine_streams_exact_text_and_resets_between_requests() {
    let checkpoint = Checkpoint::new();
    let mut engine = checkpoint.engine(32);
    let plain = engine.complete(request()).unwrap();
    assert_eq!(plain.text, "hello hello hello hello");
    assert_eq!(plain.completion_tokens, 4);
    assert!(plain.hit_length);
    assert_eq!(plain.reused_prompt_tokens, 0);
    let mut streamed = String::new();
    let output = engine
        .complete_stream(request(), &mut |delta| {
            streamed.push_str(delta);
            true
        })
        .unwrap();
    assert_eq!(streamed, plain.text);
    assert_eq!(output.text, plain.text);
    assert_eq!(output.prompt_tokens, plain.prompt_tokens);
    // A warm host token cache must not report model KV reuse.
    assert_eq!(output.reused_prompt_tokens, 0);
    assert!(output.total_latency_ms.is_some_and(|ms| ms.is_finite()));
}

#[test]
fn native_text_model_rejects_images_and_remains_usable() {
    let checkpoint = Checkpoint::new();
    let mut engine = checkpoint.engine(32);
    let mut image_request = request();
    image_request
        .images
        .push(apxinf_model::vision_preprocess::DecodedImage {
            rgb: vec![255, 0, 0],
            width: 1,
            height: 1,
        });
    let error = engine.complete(image_request).unwrap_err();
    assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
    assert!(error.message.contains("does not support image"));
    assert_eq!(
        engine.complete(request()).unwrap().text,
        "hello hello hello hello"
    );
}

#[test]
fn native_engine_rejects_over_limit_prompts_and_reserved_template_tools() {
    let checkpoint = Checkpoint::new();
    let error = checkpoint.engine(1).complete(request()).unwrap_err();
    assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(error.param.as_deref(), Some("messages"));

    let mut request = request();
    request
        .chat_template_kwargs
        .insert("tools".into(), json!([]));
    let error = checkpoint.engine(32).complete(request).unwrap_err();
    assert_eq!(error.param.as_deref(), Some("chat_template_kwargs"));
}

#[test]
fn native_engine_rejects_missing_chat_template_at_startup() {
    let checkpoint = Checkpoint::new();
    std::fs::remove_file(checkpoint.0.join("tokenizer_config.json")).unwrap();
    let result = ApxInfChatEngine::load(ApxInfEngineConfig {
        model_path: checkpoint.0.clone(),
        device: Device::Cpu,
        load_options: LoadOptions::default(),
        default_max_tokens: 4,
        max_input_tokens: 32,
    });
    match result {
        Ok(_) => panic!("missing chat template must fail loading"),
        Err(error) => assert!(error.message.contains("chat template")),
    }
}

#[test]
fn incremental_decode_holds_incomplete_multibyte_fallback_tokens() {
    let checkpoint = Checkpoint::new();
    let path = checkpoint.0.join("tokenizer.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["model"]["vocab"] = json!({
        "<0xE4>":0,"<0xBD>":1,"<0xA0>":2,"!":3,"[UNK]":4
    });
    config["decoder"] = json!({"type":"ByteFallback"});
    std::fs::write(&path, config.to_string()).unwrap();
    let tokenizer = Tokenizer::from_file(path).unwrap();
    let mut decoder = IncrementalTextDecoder::new(&tokenizer);
    assert_eq!(decoder.push(0).unwrap(), None);
    assert_eq!(decoder.push(1).unwrap(), None);
    assert_eq!(decoder.push(2).unwrap(), None);
    assert_eq!(decoder.push(3).unwrap(), Some("你".into()));
    assert_eq!(decoder.finish("你!").unwrap(), Some("!".into()));
}

#[test]
fn kv_capacity_rejects_overflow_and_next_request_still_works() {
    let checkpoint = Checkpoint::new();
    let mut engine = checkpoint.engine(4096);
    let mut over = request();
    // Fixture template adds user/assistant markers: 123 + 2 + 4 > 128.
    over.messages[0].content = Some(json!(vec!["world"; 123].join(" ")));
    let error = engine.complete(over).unwrap_err();
    assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
    assert!(error.message.contains("KV capacity=128"));
    let mut exact = request();
    exact.messages[0].content = Some(json!(vec!["world"; 122].join(" ")));
    assert_eq!(engine.complete(exact).unwrap().prompt_tokens, 124);
    assert_eq!(engine.complete(request()).unwrap().completion_tokens, 4);
}

#[test]
fn native_pixels_contract_expansion_and_second_turn_alignment() {
    use apxinf_core::{Backend, DType, Tensor};
    use apxinf_model::vision_preprocess::{DecodedImage, VisionContract};
    use apxinf_model::{LlmCapabilities, LlmTrait};
    use std::{collections::HashMap, sync::Arc};
    struct Probe {
        inner: Box<dyn LlmTrait>,
        contract: VisionContract,
        seen: Arc<AtomicUsize>,
    }
    impl LlmTrait for Probe {
        fn load(
            _: apxinf_loader::ModelConfig,
            _: HashMap<String, Tensor>,
            _: Device,
        ) -> apxinf_core::Result<Self> {
            unreachable!()
        }
        fn forward(&mut self, t: &[u32], p: u32) -> apxinf_core::Result<Tensor> {
            self.inner.forward(t, p)
        }
        fn backend(&self) -> &dyn Backend {
            self.inner.backend()
        }
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
        fn reset(&mut self) {
            self.inner.reset()
        }
        fn sequence_capacity(&self) -> Option<usize> {
            self.inner.sequence_capacity()
        }
        fn capabilities(&self) -> LlmCapabilities {
            LlmCapabilities::VISION
        }
        fn vision_contract(&self) -> Option<&VisionContract> {
            Some(&self.contract)
        }
        fn prefill(&mut self, input: LlmInput<'_>) -> apxinf_core::Result<Tensor> {
            if let Some(image) = input.image {
                assert_eq!(input.token_ids.iter().filter(|&&t| t == 7).count(), 2);
                assert_eq!(image.grid_thw, [[1, 1, 2]]);
                assert_eq!(image.pixel_values.dtype(), DType::F32);
                assert_eq!(image.pixel_values.to_f32_vec()?, [1., 2., 3., 4., 5., 6.]);
                self.seen.fetch_add(1, Ordering::Relaxed);
            }
            self.inner.prefill(LlmInput::text(input.token_ids))
        }
    }
    let checkpoint = Checkpoint::new();
    let config = checkpoint.0.join("tokenizer_config.json");
    let mut value: Value = serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
    value["chat_template"]=json!("{% for m in messages %}{{ m.role }} {% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{% if p.type == 'image' %}[EOS]{% else %}{{ p.text }}{% endif %} {% endfor %}{% endif %} {% endfor %}assistant");
    std::fs::write(config, value.to_string()).unwrap();
    let mut engine = checkpoint.engine(128);
    let seen = Arc::new(AtomicUsize::new(0));
    let LoadedModel::Text {
        model,
        generation_defaults,
    } = engine.model
    else {
        unreachable!()
    };
    engine.model = LoadedModel::Text {
        model: Box::new(Probe {
            inner: model,
            seen: seen.clone(),
            contract: VisionContract {
                model: "probe".into(),
                max_images: 1,
                min_pixels: 1,
                max_pixels: 2,
                patch_size: 1,
                temporal_patch_size: 1,
                spatial_merge_size: 1,
                image_token_id: 7,
                pixel_dtype: DType::F32,
                image_mean: [0.; 3],
                image_std: [1.; 3],
                rescale_factor: 1.,
            },
        }),
        generation_defaults,
    };
    let mut r = request();
    r.tool_choice = ToolChoice::Required;
    r.messages.push(ChatCompletionMessage {
        role: "assistant".into(),
        content: Some(json!("hello")),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    });
    r.messages.push(ChatCompletionMessage {
        role: "user".into(),
        content: Some(
            json!([{"type":"text","text":"world"},{"type":"image"},{"type":"text","text":"hello"}]),
        ),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    });
    r.images = vec![DecodedImage {
        rgb: vec![1, 2, 3, 4, 5, 6],
        width: 2,
        height: 1,
    }];
    let mut turns = to_template_messages(&r.messages).unwrap();
    apply_tool_choice_instruction(&mut turns, &r.tool_choice);
    assert_eq!(turns[3].content.as_ref().unwrap()[1]["type"], "image");
    let plain = engine.complete(r.clone()).unwrap();
    let mut streamed = String::new();
    engine
        .complete_stream(r.clone(), &mut |d| {
            streamed.push_str(d);
            true
        })
        .unwrap();
    assert_eq!(plain.text, streamed);
    assert_eq!(seen.load(Ordering::Relaxed), 2);
    assert!(engine.prompt_token_cache.is_empty());
    r.images.push(r.images[0].clone());
    let e = engine.complete(r).unwrap_err();
    assert!(e.message.contains("max_images=1"));
    assert!(engine.complete(request()).is_ok());
}

#[tokio::test]
async fn overflowing_http_request_does_not_revoke_worker_readiness() {
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;
    let checkpoint = Checkpoint::new();
    let path = checkpoint.0.clone();
    let worker = crate::WorkerClient::spawn(2, move || {
        ApxInfChatEngine::load(ApxInfEngineConfig {
            model_path: path,
            device: Device::Cpu,
            load_options: LoadOptions::default(),
            default_max_tokens: 50,
            max_input_tokens: 4096,
        })
        .map(|m| Box::new(m) as Box<dyn ChatEngine>)
    });
    for _ in 0..100 {
        if worker.ready() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(worker.ready());
    let app = crate::router(
        crate::ApiConfig {
            served_model_name: "tiny".into(),
            max_output_tokens: 50,
        },
        worker.clone(),
    );
    let request = |text: String, max_tokens: usize| {
        Request::post("/v1/chat/completions").header("content-type","application/json").body(Body::from(json!({"model":"tiny","messages":[{"role":"user","content":text}],"max_tokens":max_tokens}).to_string())).unwrap()
    };
    let bad = app
        .clone()
        .oneshot(request(vec!["world"; 102].join(" "), 50))
        .await
        .unwrap();
    assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(bad.into_body(), 4096).await.unwrap();
    assert!(String::from_utf8_lossy(&bytes).contains("KV capacity=128"));
    assert!(worker.ready());
    let good = app.oneshot(request("world".into(), 1)).await.unwrap();
    assert_eq!(good.status(), StatusCode::OK);
    assert!(worker.ready());
}

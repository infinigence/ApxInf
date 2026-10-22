//! Opt-in checkpoint regressions against externally captured official tensors.
//! Fixture files and run logs live outside the maintained source tree.
use super::{clip, vae};
use crate::dreamzero::backend;
use crate::dreamzero::{
    backend::{transfers, Context, Policies},
    weights,
};
use apxinf_core::Tensor;
use std::{collections::HashMap, path::PathBuf};

fn inputs() -> (HashMap<String, Tensor>, HashMap<String, Tensor>) {
    let shard = PathBuf::from(
        std::env::var("APXINF_DREAMZERO_COMPONENT_SHARD")
            .expect("set the verified official shard containing CLIP/VAE"),
    );
    let fixture = PathBuf::from(
        std::env::var("APXINF_DREAMZERO_COMPONENT_FIXTURE")
            .expect("set the pinned official component fixture safetensors"),
    );
    (
        apxinf_loader::safetensors::load_native(&shard).unwrap().0,
        apxinf_loader::safetensors::load_native(&fixture).unwrap().0,
    )
}
fn host(ctx: &Context, x: &Tensor) -> Vec<f32> {
    ctx.synchronize().unwrap();
    transfers::to_cpu(x).unwrap().to_f32_vec().unwrap()
}
fn compare(name: &str, actual: &[f32], expected: &Tensor) {
    compare_with_limit(name, actual, expected, 0.01)
}
fn compare_with_limit(name: &str, actual: &[f32], expected: &Tensor, limit: f64) {
    let expected = expected.to_f32_vec().unwrap();
    assert_eq!(actual.len(), expected.len(), "{name}");
    assert!(actual.iter().all(|x| x.is_finite()), "{name} nonfinite");
    let mut squared = 0f64;
    let mut energy = 0f64;
    let mut maximum = 0f64;
    for (&a, &b) in actual.iter().zip(&expected) {
        let d = (a as f64 - b as f64).abs();
        squared += d * d;
        energy += (b as f64) * (b as f64);
        maximum = maximum.max(d);
    }
    let relative = (squared / energy.max(1e-30)).sqrt();
    eprintln!(
        "{}",
        serde_json::json!({"checkpoint":name,"relative_l2":relative,
        "rmse":(squared/actual.len() as f64).sqrt(),"max_abs":maximum})
    );
    assert!(
        relative <= limit,
        "{name} relative L2 {relative} exceeds {limit} tolerance"
    );
}

#[test]
#[ignore = "requires official DROID shard and independent CLIP captures"]
fn checkpoint_clip_eager_capture_changed_input_and_lifetime() {
    let (mut map, fixture) = inputs();
    map.retain(|key, _| key.starts_with("action_head.image_encoder."));
    let ctx =
        Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
            .unwrap();
    let weights = weights::clip::ClipWeights::from_map(&ctx, &mut map).unwrap();
    assert!(map.is_empty());
    let mut w = clip::ClipWorkspace::new(&ctx).unwrap();
    let p = Policies::default();
    transfers::copy_cpu_to_cuda(&fixture["clip.input"], &w.patches).unwrap();
    let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
    session
        .prepare(|| clip::encode_patches(&ctx, &p, &weights, &mut w))
        .unwrap();
    let first = host(&ctx, &w.output);
    let graph = session
        .run(|| backend::capture(&ctx, || clip::encode_patches(&ctx, &p, &weights, &mut w)))
        .unwrap();
    for _ in 0..3 {
        graph.replay().unwrap();
        assert_eq!(host(&ctx, &w.output), first);
    }
    transfers::copy_cpu_to_cuda(&fixture["clip.changed.input"], &w.patches).unwrap();
    session
        .run(|| clip::encode_patches(&ctx, &p, &weights, &mut w))
        .unwrap();
    let changed = host(&ctx, &w.output);
    assert_ne!(changed, first);
    let output = w.output.clone();
    drop(w);
    drop(weights);
    drop(session);
    graph.replay().unwrap();
    assert_eq!(host(&ctx, &output), changed);
    eprintln!("CLIP eager/captured, changed-input and retained-resource replay passed");
    compare_with_limit(
        "clip.changed",
        &changed,
        &fixture["clip.changed.output"],
        0.01,
    );
    compare_with_limit("clip", &first, &fixture["clip.output"], 0.01);
}

#[test]
#[ignore = "requires official DROID shard and independent temporal VAE captures"]
fn checkpoint_vae_temporal_profiles_eager_capture_and_lifetime() {
    let (mut map, fixture) = inputs();
    map.retain(|key, _| key.starts_with("action_head.vae."));
    let ctx =
        Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
            .unwrap();
    let weights = weights::vae::VaeWeights::from_map(&ctx, &mut map).unwrap();
    drop(map); // decoder is validated by the complete checkpoint loader.
    let p = Policies::default();
    for frames in [1, 9, 33] {
        let name = format!("vae{frames}");
        let w = vae::VaeWorkspace::new(&ctx, frames, 352, 640, frames == 33).unwrap();
        transfers::copy_cpu_to_cuda(&fixture[&format!("{name}.input")], &w.input).unwrap();
        let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
        session
            .prepare(|| vae::encode(&ctx, &p, &weights, &w))
            .unwrap();
        let first = host(&ctx, &w.output);
        compare(&name, &first, &fixture[&format!("{name}.output")]);
        let graph = session
            .run(|| backend::capture(&ctx, || vae::encode(&ctx, &p, &weights, &w)))
            .unwrap();
        graph.replay().unwrap();
        assert_eq!(host(&ctx, &w.output), first, "{name} captured parity");
        transfers::copy_cpu_to_cuda(&fixture[&format!("{name}.changed.input")], &w.input).unwrap();
        session.run(|| vae::encode(&ctx, &p, &weights, &w)).unwrap();
        let changed = host(&ctx, &w.output);
        compare(
            &format!("{name}.changed"),
            &changed,
            &fixture[&format!("{name}.changed.output")],
        );
        assert_ne!(first, changed);
        let output = w.output.clone();
        drop(w);
        drop(session);
        graph.replay().unwrap();
        assert_eq!(
            host(&ctx, &output),
            changed,
            "{name} retained graph storage"
        );
    }
}

#[test]
#[ignore = "requires official CLIP intermediate captures"]
fn checkpoint_clip_layer_diagnostics() {
    let (mut map, fixture) = inputs();
    let path = std::env::var("APXINF_DREAMZERO_CLIP_LAYERS").unwrap();
    let layers = apxinf_loader::safetensors::load_native(std::path::Path::new(&path))
        .unwrap()
        .0;
    map.retain(|key, _| key.starts_with("action_head.image_encoder."));
    let ctx =
        Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
            .unwrap();
    let weights = weights::clip::ClipWeights::from_map(&ctx, &mut map).unwrap();
    let mut w = clip::ClipWorkspace::new(&ctx).unwrap();
    let p = Policies::default();
    transfers::copy_cpu_to_cuda(&fixture["clip.input"], &w.patches).unwrap();
    let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
    session
        .prepare(|| clip::encode_prefix(&ctx, &p, &weights, &mut w))
        .unwrap();
    compare("clip.pre_norm", &host(&ctx, &w.hidden), &layers["pre_norm"]);
    for (i, b) in weights.blocks.iter().enumerate() {
        let input = if i == 0 {
            &layers["pre_norm"]
        } else {
            &layers[&format!("transformer.{}", i - 1)]
        };
        transfers::copy_cpu_to_cuda(&input.reshape(vec![257, 1280]).unwrap(), &w.hidden).unwrap();
        session
            .prepare(|| clip::encode_block(&ctx, &p, b, &mut w))
            .unwrap();
        if i == 0 {
            compare(
                "clip.block0.up",
                &host(&ctx, &w.up),
                &layers["transformer.0.mlp.0"],
            );
            compare(
                "clip.block0.activation",
                &host(&ctx, &w.activated),
                &layers["transformer.0.mlp.1"],
            );
        }
        compare_with_limit(
            &format!("clip.block{i}"),
            &host(&ctx, &w.hidden),
            &layers[&format!("transformer.{i}")],
            0.005,
        );
    }
}

#[test]
#[ignore = "requires official DROID text shards and independently captured prompts"]
fn checkpoint_text_both_prompts_eager_capture() {
    let root = PathBuf::from(std::env::var("APXINF_DREAMZERO_CHECKPOINT").unwrap());
    let path = PathBuf::from(std::env::var("APXINF_DREAMZERO_TEXT_FIXTURE").unwrap());
    let fixture = apxinf_loader::safetensors::load_native(&path).unwrap().0;
    let mut map = HashMap::new();
    for i in 1..=3 {
        let mut shard = apxinf_loader::safetensors::load_native(
            &root.join(format!("model-{i:05}-of-00010.safetensors")),
        )
        .unwrap()
        .0;
        shard.retain(|key, _| key.starts_with("action_head.text_encoder."));
        map.extend(shard);
    }
    let ctx =
        Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
            .unwrap();
    let weights = weights::TextWeights::from_map(&ctx, &mut map).unwrap();
    assert!(map.is_empty());
    let mut w = super::text::TextWorkspace::new(&ctx).unwrap();
    let p = Policies::default();
    let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
    let bind = |name: &str, w: &super::text::TextWorkspace| {
        let ids = fixture[&format!("{name}.ids")]
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let mask = fixture[&format!("{name}.mask")]
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|&x| {
                if x == 1. {
                    half::bf16::ZERO
                } else {
                    half::bf16::MIN
                }
            })
            .collect::<Vec<_>>();
        transfers::copy_cpu_to_cuda(&Tensor::from_i32(vec![512], &ids).unwrap(), &w.ids).unwrap();
        transfers::copy_cpu_to_cuda(&Tensor::from_bf16(vec![1, 512], &mask).unwrap(), &w.mask)
            .unwrap();
    };
    bind("positive", &w);
    session
        .prepare(|| super::text::encode(&ctx, &p, &weights, &mut w))
        .unwrap();
    let positive = host(&ctx, &w.output);
    for name in ["positive", "negative"] {
        bind(name, &w);
        super::super::backend::ops::embedding_gather(&ctx, &weights.embedding, &w.ids, &w.hidden)
            .unwrap();
        for (i, block) in weights.blocks.iter().enumerate() {
            if i > 0 {
                transfers::copy_cpu_to_cuda(
                    &fixture[&format!("{name}.block{}", i - 1)]
                        .reshape(vec![512, 4096])
                        .unwrap(),
                    &w.hidden,
                )
                .unwrap();
            }
            session
                .prepare(|| super::text::encode_block(&ctx, &p, block, &mut w))
                .unwrap();
            compare_with_limit(
                &format!("text.{name}.block{i}.isolated"),
                &host(&ctx, &w.hidden),
                &fixture[&format!("{name}.block{i}")],
                0.005,
            );
        }
    }
    bind("positive", &w);
    session
        .prepare(|| super::text::encode(&ctx, &p, &weights, &mut w))
        .unwrap();
    let graph = session
        .run(|| backend::capture(&ctx, || super::text::encode(&ctx, &p, &weights, &mut w)))
        .unwrap();
    graph.replay().unwrap();
    assert_eq!(positive, host(&ctx, &w.output));
    bind("negative", &w);
    session
        .run(|| super::text::encode(&ctx, &p, &weights, &mut w))
        .unwrap();
    let negative = host(&ctx, &w.output);
    let output = w.output.clone();
    drop(w);
    drop(weights);
    drop(session);
    graph.replay().unwrap();
    assert_eq!(negative, host(&ctx, &output));
    eprintln!("UMT5 eager/captured, changed-prompt and retained-resource replay passed");
    compare_with_limit(
        "text.negative",
        &negative,
        &fixture["negative.output"],
        0.01,
    );
    compare_with_limit(
        "text.positive",
        &positive,
        &fixture["positive.output"],
        0.01,
    );
    for (name, actual) in [("positive", &positive), ("negative", &negative)] {
        let count = fixture[&format!("{name}.mask")]
            .to_f32_vec()
            .unwrap()
            .iter()
            .filter(|&&x| x == 1.)
            .count();
        let expected = fixture[&format!("{name}.output")].to_f32_vec().unwrap();
        compare_with_limit(
            &format!("text.{name}.valid"),
            &actual[..count * 4096],
            &Tensor::from_f32(vec![count, 4096], &expected[..count * 4096]).unwrap(),
            0.01,
        );
    }
}

#[test]
#[ignore = "requires official observed/joint DiT block captures and complete checkpoint"]
fn checkpoint_dit_observed_and_joint_blocks_eager_capture() {
    use super::dit_block::{block, BlockWorkspace};
    let root = PathBuf::from(std::env::var("APXINF_DREAMZERO_CHECKPOINT").unwrap());
    let fixtures = PathBuf::from(std::env::var("APXINF_DREAMZERO_BLOCK_FIXTURES").unwrap());
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let ctx =
        Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
            .unwrap();
    for layer in [0, 19, 39] {
        let prefix = format!("action_head.model.blocks.{layer}.");
        let shards: std::collections::BTreeSet<_> = index["weight_map"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(_, v)| v.as_str().unwrap())
            .collect();
        let mut map = HashMap::new();
        for shard in shards {
            let mut data = apxinf_loader::safetensors::load_native(&root.join(shard))
                .unwrap()
                .0;
            data.retain(|k, _| k.starts_with(&prefix));
            map.extend(data);
        }
        let weights = weights::dit::Block::from_map(&ctx, &mut map, layer).unwrap();
        assert!(map.is_empty());
        for call in 0..4 {
            let name = format!("dit.block{layer}.{call}");
            let input = apxinf_loader::safetensors::load_native(
                &fixtures.join(format!("{name}.safetensors")),
            )
            .unwrap()
            .0;
            let joint = call >= 2;
            let n = if joint { 1760 } else { 880 };
            let h = if joint { 880 } else { 0 };
            let mut work = BlockWorkspace::new(&ctx, n, if joint { 25 } else { 0 }, h).unwrap();
            transfers::copy_cpu_to_cuda(&input["hidden"], &work.hidden).unwrap();
            let time = transfers::to_cuda(&input["time"], 0).unwrap();
            let text = transfers::to_cuda(&input["text"], 0).unwrap();
            let image = transfers::to_cuda(&input["image"], 0).unwrap();
            let history = if joint {
                Some((
                    transfers::to_cuda(&input["k"], 0).unwrap(),
                    transfers::to_cuda(&input["v"], 0).unwrap(),
                ))
            } else {
                None
            };
            let next = if joint {
                None
            } else {
                Some((
                    super::super::backend::allocate(&ctx, &[880, 5120]).unwrap(),
                    super::super::backend::allocate(&ctx, &[880, 5120]).unwrap(),
                ))
            };
            let bytes = std::fs::read(fixtures.join(format!("{name}.rotary.f64"))).unwrap();
            assert_eq!(bytes.len() % 8, 0);
            let coefficients: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let generated = super::dit::rotary_coefficients(
                if joint { 2 } else { 1 },
                if joint { 1 } else { 0 },
                joint,
            );
            assert_eq!(generated.len(), coefficients.len());
            for (actual, expected) in generated.iter().zip(&coefficients) {
                assert!(
                    (actual - expected).abs() <= 1e-12,
                    "native rotary coefficient mismatch"
                );
            }
            let rotary = super::super::backend::ops::ComplexRotaryTable::new(
                &ctx,
                n + if joint { 25 } else { 0 },
                128,
                &generated,
            )
            .unwrap();
            let policies = Policies::default();
            let invoke = |work: &mut BlockWorkspace| {
                block(
                    &ctx,
                    &policies,
                    &weights,
                    work,
                    &time,
                    &text,
                    &image,
                    &rotary,
                    history.as_ref().map(|x| &x.0),
                    history.as_ref().map(|x| &x.1),
                    next.as_ref().map(|x| (&x.0, &x.1)),
                )
            };
            let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
            session.prepare(|| invoke(&mut work)).unwrap();
            let actual = host(&ctx, &work.hidden);
            compare_with_limit(&name, &actual, &input["output"], 0.005);
            if let Some((k, v)) = &next {
                let expected = apxinf_loader::safetensors::load_native(
                    &fixtures.join(format!("dit.block{layer}.{}.safetensors", call + 2)),
                )
                .unwrap()
                .0;
                compare_with_limit(
                    &format!("{name}.committed-k"),
                    &host(&ctx, k),
                    &expected["k"],
                    0.005,
                );
                compare_with_limit(
                    &format!("{name}.committed-v"),
                    &host(&ctx, v),
                    &expected["v"],
                    0.005,
                );
            }
            // Block residual writes in place, so rebind its original input.
            transfers::copy_cpu_to_cuda(&input["hidden"], &work.hidden).unwrap();
            let graph = session
                .run(|| backend::capture(&ctx, || invoke(&mut work)))
                .unwrap();
            graph.replay().unwrap();
            assert_eq!(actual, host(&ctx, &work.hidden));
            transfers::copy_cpu_to_cuda(&input["hidden"], &work.hidden).unwrap();
            let output = work.hidden.clone();
            drop(work);
            drop(session);
            drop(rotary);
            drop(history);
            drop(next);
            drop(time);
            drop(text);
            drop(image);
            graph.replay().unwrap();
            assert_eq!(actual, host(&ctx, &output));
        }
    }
}

#[test]
#[ignore = "requires official DROID component fixture including initial 33-frame conditioning"]
fn checkpoint_initial_vision_boundary_and_graph() {
    let (mut map, fixture) = inputs();
    map.retain(|k, _| {
        k.starts_with("action_head.image_encoder.") || k.starts_with("action_head.vae.")
    });
    let ctx =
        Context::new_with_autotune(0, std::env::var_os("APXINF_DREAMZERO_AUTOTUNE").is_some())
            .unwrap();
    let clip = weights::clip::ClipWeights::from_map(&ctx, &mut map).unwrap();
    let vae = weights::vae::VaeWeights::from_map(&ctx, &mut map).unwrap();
    drop(map);
    let mut work = super::vision::VisionWorkspace::new(&ctx, 1, true).unwrap();
    let p = Policies::default();
    let bind = |name: &str, work: &super::vision::VisionWorkspace| {
        let data = fixture[name].as_bf16().unwrap();
        let first = Tensor::from_bf16(vec![352 * 640, 3], &data[..352 * 640 * 3]).unwrap();
        transfers::copy_cpu_to_cuda(&first, &work.images).unwrap();
    };
    bind("vae33.input", &work);
    let session = backend::ExecutionSession::with_capacity(256, 0).unwrap();
    session
        .prepare(|| work.forward(&ctx, &p, &clip, &vae))
        .unwrap();
    assert_eq!(
        host(&ctx, &work.clip.patches),
        fixture["clip.input"].to_f32_vec().unwrap(),
        "checkpoint-specific CLIP preprocessing"
    );
    compare_with_limit(
        "vision.clip",
        &host(&ctx, &work.clip.output),
        &fixture["clip.output"],
        0.01,
    );
    compare(
        "vision.vae33",
        &host(&ctx, &work.vae.output),
        &fixture["vae33.output"],
    );
    let observed = host(&ctx, &work.observed);
    let condition = host(&ctx, &work.conditioning);
    let graph = session
        .run(|| backend::capture(&ctx, || work.forward(&ctx, &p, &clip, &vae)))
        .unwrap();
    graph.replay().unwrap();
    assert_eq!(observed, host(&ctx, &work.observed));
    assert_eq!(condition, host(&ctx, &work.conditioning));
    bind("vae33.changed.input", &work);
    session.run(|| work.forward(&ctx, &p, &clip, &vae)).unwrap();
    assert_eq!(
        host(&ctx, &work.clip.patches),
        fixture["clip.changed.input"].to_f32_vec().unwrap()
    );
    compare_with_limit(
        "vision.changed.clip",
        &host(&ctx, &work.clip.output),
        &fixture["clip.changed.output"],
        0.01,
    );
    compare(
        "vision.changed.vae33",
        &host(&ctx, &work.vae.output),
        &fixture["vae33.changed.output"],
    );
    let changed = host(&ctx, &work.conditioning);
    assert_ne!(changed, condition);
    let output = work.conditioning.clone();
    drop(work);
    drop(session);
    drop(clip);
    drop(vae);
    graph.replay().unwrap();
    assert_eq!(changed, host(&ctx, &output));
}

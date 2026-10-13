//! Loading the real Qwen3.8-27B-NVFP4 checkpoint.
//!
//! This is an integration test against a 20 GiB on-disk checkpoint, so it is
//! `#[ignore]`d by default and run explicitly:
//!
//! ```text
//! APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//!   cargo test -p apxinf-loader --test qwen38_nvfp4 -- --ignored --nocapture
//! ```
//!
//! It checks the properties the model port depends on, which unit tests over
//! synthetic files cannot: that packed FP4 weights load at all, that the
//! stored shape relates to the logical one as `[N, K/2]`, and that each
//! quantized weight is accompanied by the scales its GEMM contract needs.

use std::collections::HashMap;
use std::path::PathBuf;

use apxinf_core::{DType, Tensor};

fn checkpoint_path() -> Option<PathBuf> {
    std::env::var_os("APXINF_QWEN38_CHECKPOINT").map(PathBuf::from)
}

fn load() -> HashMap<String, Tensor> {
    let path = checkpoint_path().expect("set APXINF_QWEN38_CHECKPOINT");
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(&path)
        .expect("checkpoint failed to load");
    tensors
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint"]
fn loads_every_tensor_with_the_expected_dtype_mix() {
    let tensors = load();
    assert_eq!(tensors.len(), 2194, "tensor count changed");

    let mut counts: HashMap<DType, usize> = HashMap::new();
    let mut bytes: HashMap<DType, usize> = HashMap::new();
    for tensor in tensors.values() {
        *counts.entry(tensor.dtype()).or_default() += 1;
        *bytes.entry(tensor.dtype()).or_default() +=
            tensor.shape().numel() * tensor.dtype().size_in_bytes();
    }
    for (dtype, count) in &counts {
        println!(
            "{dtype}: {count} tensors, {:.3} GiB",
            bytes[dtype] as f64 / (1u64 << 30) as f64
        );
    }

    // The mix is what makes this checkpoint a mixed-precision port rather than
    // a pure-FP4 one; a regression here means the dtype mapping drifted.
    assert_eq!(counts.get(&DType::E2M1Pair), Some(&193), "packed FP4 count");
    assert_eq!(counts.get(&DType::F8E4M3), Some(&401), "FP8 count");
    assert_eq!(counts.get(&DType::BF16), Some(&798), "BF16 count");
    assert_eq!(counts.get(&DType::F32), Some(&802), "scalar scale count");
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint"]
fn nvfp4_weights_carry_matching_packed_shapes_and_scales() {
    let tensors = load();
    let hidden = 5120usize;
    let intermediate = 17408usize;
    let block = 16usize;

    for (name, out_features, in_features) in [
        ("model.language_model.layers.0.mlp.gate_proj", intermediate, hidden),
        ("model.language_model.layers.0.mlp.up_proj", intermediate, hidden),
        ("model.language_model.layers.0.mlp.down_proj", hidden, intermediate),
        ("lm_head", 248320, hidden),
    ] {
        let weight = tensors
            .get(&format!("{name}.weight"))
            .unwrap_or_else(|| panic!("{name}.weight missing"));
        assert_eq!(weight.dtype(), DType::E2M1Pair, "{name} dtype");
        assert_eq!(
            weight.shape().dims(),
            &[out_features, in_features / 2],
            "{name} is stored [N, K/2]"
        );

        let scale = tensors
            .get(&format!("{name}.weight_scale"))
            .unwrap_or_else(|| panic!("{name}.weight_scale missing"));
        assert_eq!(scale.dtype(), DType::F8E4M3, "{name} block scale dtype");
        assert_eq!(
            scale.shape().dims(),
            &[out_features, in_features / block],
            "{name} has one scale per {block} elements along K"
        );

        // Both per-tensor scales must exist: their product is the alpha the
        // NVFP4 GEMM contract expects.
        for suffix in ["weight_scale_2", "input_scale"] {
            let scalar = tensors
                .get(&format!("{name}.{suffix}"))
                .unwrap_or_else(|| panic!("{name}.{suffix} missing"));
            assert_eq!(scalar.dtype(), DType::F32, "{name}.{suffix} dtype");
            assert_eq!(scalar.shape().numel(), 1, "{name}.{suffix} is per-tensor");
        }
    }
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint"]
fn fp8_projections_carry_scalar_scales_not_vectors() {
    let tensors = load();
    // The existing Fp8 GEMM arm expects [M] and [N] scale vectors. This
    // checkpoint has scalars instead, which is why the port folds them into
    // alpha rather than binding scale tensors.
    for name in [
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.out_proj",
    ] {
        let weight = tensors
            .get(&format!("{name}.weight"))
            .unwrap_or_else(|| panic!("{name}.weight missing"));
        assert_eq!(weight.dtype(), DType::F8E4M3, "{name} dtype");
        let scale = tensors
            .get(&format!("{name}.weight_scale"))
            .unwrap_or_else(|| panic!("{name}.weight_scale missing"));
        assert_eq!(scale.shape().numel(), 1, "{name}.weight_scale is per-tensor");
        assert!(
            !tensors.contains_key(&format!("{name}.weight_scale_2")),
            "{name} is FP8 and must not carry a second-level scale"
        );
    }
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint"]
fn gate_and_up_share_the_scales_that_make_fusion_lossless() {
    let tensors = load();
    let scalar = |name: &str| -> f32 {
        let tensor = tensors.get(name).unwrap_or_else(|| panic!("{name} missing"));
        tensor.to_f32_vec().unwrap()[0]
    };
    // Fusing gate and up into one [2N, K] GEMM is only exact when both
    // per-tensor scales match, because they become a single alpha.
    for layer in 0..64 {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        assert_eq!(
            scalar(&format!("{prefix}.gate_proj.input_scale")),
            scalar(&format!("{prefix}.up_proj.input_scale")),
            "layer {layer} input_scale"
        );
        assert_eq!(
            scalar(&format!("{prefix}.gate_proj.weight_scale_2")),
            scalar(&format!("{prefix}.up_proj.weight_scale_2")),
            "layer {layer} weight_scale_2"
        );
    }
}

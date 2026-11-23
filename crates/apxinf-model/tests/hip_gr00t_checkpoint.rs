//! Load a real GR00T N1.7 checkpoint through the HIP port's host-side reader.
//!
//! Needs no GPU: this is the sharded SafeTensors interpretation, the Cosmos
//! backbone config and the typed weight split, all on the host. Gated on the
//! checkpoint path so the suite still runs without one.
//!
//!   APXINF_GR00T_CHECKPOINT=/models/GR00T-N1.7-LIBERO/libero_10 \
//!     cargo test -p apxinf-model --features hip --test hip_gr00t_checkpoint

#![cfg(feature = "hip")]

use std::path::PathBuf;

use apxinf_model::hip_gr00t::weights::Gr00tWeights;
use apxinf_model::hip_gr00t::config::Gr00tConfig;

fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("APXINF_GR00T_CHECKPOINT")?);
    path.is_dir().then_some(path)
}

#[test]
fn reads_the_released_libero_checkpoint_into_typed_weights() {
    let Some(root) = checkpoint() else {
        eprintln!("APXINF_GR00T_CHECKPOINT not set; skipping");
        return;
    };
    let config = Gr00tConfig::from_json_file(&root.join("config.json")).unwrap();
    // The released LIBERO checkpoint truncates Qwen3-VL to 16 layers and runs
    // a 32-block DiT -- both differ from the constructor defaults, so reading
    // them from the checkpoint rather than the defaults is load-bearing.
    assert_eq!(config.select_layer, 16, "select_layer comes from config.json");
    assert_eq!(
        config.diffusion.num_layers, 32,
        "DiT depth comes from config.json"
    );
    assert_eq!(config.max_action_dim, 132);
    assert_eq!(config.action_horizon, 40);

    let backbone = root.join("assets/cosmos");
    assert!(
        backbone.join("config.json").is_file(),
        "prepare the Cosmos assets first: {}",
        backbone.display()
    );

    let (qwen, weights) = Gr00tWeights::from_safetensors(&config, &backbone, &root).unwrap();

    // Backbone geometry the fixture and the NVIDIA reference were produced at.
    assert_eq!(qwen.text.hidden_size, 2048);
    assert_eq!(qwen.text.n_layers, config.select_layer, "truncated in place");
    assert_eq!(qwen.text.n_heads, 16);
    assert_eq!(qwen.text.n_kv_heads, 8);
    assert_eq!(qwen.text.head_dim, 128);
    assert_eq!(qwen.text.mrope_section, [24, 20, 20]);
    assert!(qwen.text.mrope_interleaved);
    assert_eq!(qwen.vision.depth, 24);
    assert_eq!(qwen.vision.hidden_size, 1024);
    assert_eq!(qwen.vision.out_hidden_size, 2048);
    assert_eq!(qwen.vision.spatial_merge_size, 2);

    // Every layer materialized, and the projections are the [in, out] the
    // row-major GEMM wants rather than the checkpoint's [out, in].
    assert_eq!(weights.backbone_text.layers.len(), 16);
    let layer = &weights.backbone_text.layers[0];
    assert_eq!(layer.wq.shape().dims(), &[2048, 16 * 128]);
    assert_eq!(layer.wk.shape().dims(), &[2048, 8 * 128]);
    assert_eq!(weights.backbone_vision.blocks.len(), 24);

    // `from_map` refuses a checkpoint with unconsumed tensors, so reaching
    // here means every tensor found an owner -- the port's main guard against
    // a silently incomplete weight mapping.
}

#[test]
fn uploads_every_weight_the_bf16_path_reads_to_the_device() {
    let Some(root) = checkpoint() else {
        eprintln!("APXINF_GR00T_CHECKPOINT not set; skipping");
        return;
    };
    let Ok(ctx) = apxinf_hip::HipContext::new(0) else {
        eprintln!("no hip:0 device; skipping");
        return;
    };
    let config = Gr00tConfig::from_json_file(&root.join("config.json")).unwrap();
    let backbone = root.join("assets/cosmos");
    let (qwen, host) = Gr00tWeights::from_safetensors(&config, &backbone, &root).unwrap();

    // `libero_sim` is embodiment 2 in the released checkpoint's
    // embodiment_id.json, which is what the fixture records.
    let (device, embodiment) =
        apxinf_model::hip_gr00t::weights::HipGr00tWeights::from_checkpoint(
            &host, &config, &qwen, 2, &ctx,
        )
        .unwrap();

    assert_eq!(device.text_layers.len(), config.select_layer);
    assert_eq!(device.dit_blocks.len(), config.diffusion.num_layers);
    assert_eq!(device.vision_blocks.len(), qwen.vision.depth);
    assert_eq!(device.deepstack_mergers.len(), 3);

    // Everything the inference path reads now lives on the device, not the host.
    let on_device = |tensor: &apxinf_core::Tensor| {
        assert_eq!(
            tensor.device(),
            apxinf_core::Device::Hip(0),
            "weight stayed on the host"
        );
        assert_eq!(tensor.dtype(), apxinf_core::DType::BF16);
    };
    on_device(&device.token_embedding);
    on_device(&device.text_output_norm);
    on_device(&device.text_layers[0].q.weight);
    on_device(&device.vision_blocks[0].qkv.weight);
    on_device(&device.dit_blocks[0].adaptive_norm.weight);
    on_device(&device.output_projection.weight);
    on_device(&embodiment.state_encoder_in.weight);
    on_device(&embodiment.action_decoder_out.weight);

    // The per-embodiment adapters are sliced out of [embodiment, in, out], so
    // the device copy must be rank 2 -- a rank-3 upload would mean the whole
    // category table was shipped and the wrong rows would be read.
    assert_eq!(embodiment.state_encoder_in.weight.shape().dims().len(), 2);
    assert_eq!(embodiment.action_encoder_out.weight.shape().dims().len(), 2);

    // Out-of-range embodiments are refused rather than silently clamped.
    assert!(
        apxinf_model::hip_gr00t::weights::HipGr00tWeights::from_checkpoint(
            &host,
            &config,
            &qwen,
            config.max_num_embodiments,
            &ctx,
        )
        .is_err(),
        "embodiment id past the table"
    );
}

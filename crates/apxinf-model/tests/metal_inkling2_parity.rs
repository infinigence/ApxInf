#![cfg(feature = "mlx")]

use apxinf_core::Device;
use apxinf_mlx::MlxBackend;
use apxinf_model::metal_inkling2::{Config, MetalInkling2, Variant};
use apxinf_model::{GenerationOptions, GenerationRequest, LlmInput, LlmTrait};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn config_and_schedule_parsing() {
    let modelopt_json = r#"{
        "model_type": "inkling2_loop",
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION"
        },
        "text_config": {
            "model_type": "inkling2_loop",
            "hidden_size": 2304,
            "num_hidden_layers": 24,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 200192,
            "intermediate_size": 896,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "n_layers_per_attn_res_block": 6,
            "loop_config": {
                "core_start": 6,
                "core_end": 18,
                "num_passes": 3,
                "gate_rank": 64
            }
        }
    }"#;

    let cfg = Config::from_json(modelopt_json).expect("parse ModelOpt config");
    assert_eq!(cfg.text_config.hidden_size, 2304);
    assert_eq!(cfg.text_config.num_hidden_layers, 24);
    assert!(!cfg.is_mlx_quantized());

    let sched = cfg.layer_schedule();
    assert_eq!(sched.len(), 48);
    assert_eq!(
        &sched[0..6],
        &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    );
    assert_eq!(sched[6], (6, 0));
    assert_eq!(sched[17], (17, 0));
    assert_eq!(sched[18], (6, 1));
    assert_eq!(sched[29], (17, 1));
    assert_eq!(sched[30], (6, 2));
    assert_eq!(sched[41], (17, 2));
    assert_eq!(sched[42], (18, 0));
    assert_eq!(sched[47], (23, 0));

    assert_eq!(Variant::parse(None).unwrap(), Variant::Nvfp4Mixed);
    assert_eq!(Variant::parse(Some("auto")).unwrap(), Variant::Nvfp4Mixed);
    assert_eq!(
        Variant::parse(Some("nvfp4-mixed")).unwrap(),
        Variant::Nvfp4Mixed
    );
    assert_eq!(
        Variant::parse(Some("oq4e-mtp")).unwrap(),
        Variant::Oq4eMtp
    );
    assert!(Variant::parse(Some("bogus")).is_err());
}

#[derive(Deserialize)]
struct Stage4Ref {
    short: ShortRef,
    e1k: E1kRef,
}

#[derive(Deserialize)]
struct TapRef {
    shape: Vec<i32>,
    head8: Vec<f32>,
    l2: f32,
}

#[derive(Deserialize)]
struct ShortRef {
    prompt_ids: Vec<u32>,
    forced_prefix: Vec<u32>,
    prefill_last_top16_ids: Vec<u32>,
    prefill_last_top16_vals: Vec<f32>,
    tap_stats: BTreeMap<String, TapRef>,
    tokens_mtp_off: Vec<u32>,
    tokens_mtp_on: Vec<u32>,
}

#[derive(Deserialize)]
struct E1kRef {
    id: String,
    prompt_len: usize,
    prompt_ids: Vec<u32>,
    tokens_mtp_off: Vec<u32>,
    tokens_mtp_on: Vec<u32>,
    mtp_off_ttft_s: f64,
    mtp_off_tpot_ms: f64,
    mtp_on_ttft_s: f64,
    mtp_on_tpot_ms: f64,
    mtp_on_med_cycle_ms: f64,
    mtp_on_tpc: f64,
}

fn require_env_path(key: &str) -> PathBuf {
    let val = std::env::var(key).unwrap_or_else(|_| {
        panic!("set {key} before running this ignored test");
    });
    PathBuf::from(val)
}

#[test]
#[ignore = "requires local Inkling-2-Mini checkpoint and Stage-4 reference JSON"]
fn stage4_parity_and_perf() {
    let model_dir = require_env_path("INKLING2_MODEL_DIR");
    let ref_path = require_env_path("INKLING2_STAGE4_REF");
    let stage4: Stage4Ref =
        serde_json::from_str(&std::fs::read_to_string(&ref_path).expect("read stage4 ref"))
            .expect("parse stage4 ref");

    let backend = MlxBackend::new(0).expect("init MlxBackend");
    let t0 = Instant::now();
    let mut model =
        MetalInkling2::load(&model_dir, Device::Metal(0), backend.clone(), None).expect("load");
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "Loaded Inkling-2-Mini in {load_s:.2}s | variant={:?} | has_draft={} | nax={}",
        model.variant(),
        model.has_draft(),
        backend.stream().metal_caps().is_some_and(|c| c.supports_nax()),
    );
    assert_eq!(model.variant(), Variant::Nvfp4Mixed);
    assert!(model.has_draft(), "expected mtp.safetensors draft loaded");

    // 1. Tap-by-tap + top-16 logits parity on short prompt.
    let taps = [0usize, 5, 6, 17, 18, 29, 30, 41, 42, 47, 48];
    let (logits, captured) = model
        .prefill_with_taps(&stage4.short.prompt_ids, &taps)
        .expect("prefill_with_taps");
    for (visit, arr) in &captured {
        let shape = arr.shape();
        let ref_tap = &stage4.short.tap_stats[&visit.to_string()];
        assert_eq!(
            shape,
            ref_tap.shape.iter().map(|&x| x as usize).collect::<Vec<_>>(),
            "visit {visit} tap shape mismatch"
        );
        let last_row = arr
            .slice(&[0, shape[1] - 1, 0], &[1, shape[1], shape[2]])
            .expect("slice last row")
            .reshape(&[shape[2]])
            .expect("reshape last row")
            .cast(apxinf_mlx::MlxDType::F32)
            .expect("cast f32");
        let vals = last_row.to_f32_vec().expect("to_f32_vec");
        let l2 = vals.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let max_head8 = vals[..8]
            .iter()
            .zip(ref_tap.head8.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let l2_rel = (l2 - ref_tap.l2).abs() / ref_tap.l2.max(1e-6);
        eprintln!(
            "tap visit={visit:2}: l2={l2:.4} (ref {:.4}, rel={l2_rel:.2e}) max_head8_abs={max_head8:.4e}",
            ref_tap.l2
        );
        assert!(
            l2_rel < 2e-2,
            "visit {visit} l2 diverged: got {l2} vs ref {}",
            ref_tap.l2
        );
        assert!(
            max_head8 < 0.25,
            "visit {visit} head8 diverged: max_abs={max_head8} got={:?} ref={:?}",
            &vals[..8],
            &ref_tap.head8
        );
    }

    let logits_f32 = logits
        .reshape(&[logits.shape()[1]])
        .expect("reshape logits")
        .cast(apxinf_mlx::MlxDType::F32)
        .expect("cast logits")
        .to_f32_vec()
        .expect("logits to_f32_vec");
    let mut ranked: Vec<(usize, f32)> = logits_f32.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top16_ids: Vec<u32> = ranked.iter().take(16).map(|(i, _)| *i as u32).collect();
    let top16_vals: Vec<f32> = ranked.iter().take(16).map(|(_, v)| *v).collect();
    let top1_diff = (top16_vals[0] - stage4.short.prefill_last_top16_vals[0]).abs();
    eprintln!(
        "short prefill top-1: id={} val={:.4} (ref id={} val={:.4}, diff={top1_diff:.4e})",
        top16_ids[0],
        top16_vals[0],
        stage4.short.prefill_last_top16_ids[0],
        stage4.short.prefill_last_top16_vals[0],
    );
    assert_eq!(top16_ids[0], stage4.short.prefill_last_top16_ids[0]);
    assert!(top1_diff < 0.25, "top-1 logit diff too large: {top1_diff}");

    // 2. Short 32-token generation parity: MTP-off vs MTP-on vs Stage-4 ref.
    let short_off = model
        .generate_greedy_benchmark(
            &stage4.short.prompt_ids,
            32,
            &stage4.short.forced_prefix,
            false,
        )
        .expect("short mtp-off");
    let short_on = model
        .generate_greedy_benchmark(
            &stage4.short.prompt_ids,
            32,
            &stage4.short.forced_prefix,
            true,
        )
        .expect("short mtp-on");
    let short_mtp = short_on.mtp.as_ref().expect("mtp stats");
    eprintln!(
        "short MTP-off: ttft={:.1}ms tpot={:.2}ms | MTP-on: ttft={:.1}ms tpot={:.2}ms med_cycle={:.2}ms tpc={:.3} fallbacks={}",
        short_off.ttft_s * 1000.0,
        short_off.tpot_ms,
        short_on.ttft_s * 1000.0,
        short_on.tpot_ms,
        short_mtp.median_cycle_ms().unwrap_or(0.0),
        short_mtp.tokens_per_cycle().unwrap_or(0.0),
        short_mtp.screen_fallbacks,
    );
    assert_eq!(
        short_off.tokens, stage4.short.tokens_mtp_off,
        "short MTP-off token mismatch vs Stage-4 ref"
    );
    assert_eq!(
        short_on.tokens, stage4.short.tokens_mtp_on,
        "short MTP-on token mismatch vs Stage-4 ref"
    );
    assert_eq!(short_off.tokens, short_on.tokens);
    assert_eq!(short_mtp.screen_fallbacks, 0);

    // 3. e1k warmup + 64-token parity and perf.
    let _warm = model
        .generate_greedy_benchmark(
            &stage4.e1k.prompt_ids,
            16,
            &stage4.short.forced_prefix,
            true,
        )
        .expect("e1k warmup");

    let c_before_off = backend.stream().counters();
    let e1k_off = model
        .generate_greedy_benchmark(
            &stage4.e1k.prompt_ids,
            64,
            &stage4.short.forced_prefix,
            false,
        )
        .expect("e1k mtp-off");
    let c_after_off = backend.stream().counters();

    let c_before_on = backend.stream().counters();
    let e1k_on = model
        .generate_greedy_benchmark(
            &stage4.e1k.prompt_ids,
            64,
            &stage4.short.forced_prefix,
            true,
        )
        .expect("e1k mtp-on");
    let c_after_on = backend.stream().counters();
    let e1k_mtp = e1k_on.mtp.as_ref().expect("e1k mtp stats");

    eprintln!(
        "e1k ({}, len={}) MTP-off: ttft={:.1}ms (ref {:.1}ms) tpot={:.3}ms (ref {:.3}ms) peak={:.2}GB traces_delta={}",
        stage4.e1k.id,
        stage4.e1k.prompt_len,
        e1k_off.ttft_s * 1000.0,
        stage4.e1k.mtp_off_ttft_s * 1000.0,
        e1k_off.tpot_ms,
        stage4.e1k.mtp_off_tpot_ms,
        e1k_off.peak_memory_gb,
        c_after_off.trace_callbacks - c_before_off.trace_callbacks,
    );
    eprintln!(
        "e1k ({}, len={}) MTP-on:  ttft={:.1}ms (ref {:.1}ms) tpot={:.3}ms (ref {:.3}ms) med_cycle={:.3}ms (ref {:.3}ms) tpc={:.3} (ref {:.3}) fallbacks={} peak={:.2}GB traces_delta={}",
        stage4.e1k.id,
        stage4.e1k.prompt_len,
        e1k_on.ttft_s * 1000.0,
        stage4.e1k.mtp_on_ttft_s * 1000.0,
        e1k_on.tpot_ms,
        stage4.e1k.mtp_on_tpot_ms,
        e1k_mtp.median_cycle_ms().unwrap_or(0.0),
        stage4.e1k.mtp_on_med_cycle_ms,
        e1k_mtp.tokens_per_cycle().unwrap_or(0.0),
        stage4.e1k.mtp_on_tpc,
        e1k_mtp.screen_fallbacks,
        e1k_on.peak_memory_gb,
        c_after_on.trace_callbacks - c_before_on.trace_callbacks,
    );
    assert_eq!(
        e1k_off.tokens, stage4.e1k.tokens_mtp_off,
        "e1k MTP-off token mismatch vs Stage-4 ref"
    );
    assert_eq!(
        e1k_on.tokens, stage4.e1k.tokens_mtp_on,
        "e1k MTP-on token mismatch vs Stage-4 ref"
    );
    assert_eq!(e1k_off.tokens, e1k_on.tokens);
    assert_eq!(e1k_mtp.screen_fallbacks, 0);

    // Optional extended decode length benchmark (e.g. INKLING2_BENCH_GEN=256).
    if let Ok(raw) = std::env::var("INKLING2_BENCH_GEN") {
        if let Ok(gen_len) = raw.parse::<usize>() {
            if gen_len > 64 {
                let run_off = model
                    .generate_greedy_benchmark(
                        &stage4.e1k.prompt_ids,
                        gen_len,
                        &stage4.short.forced_prefix,
                        false,
                    )
                    .expect("e1k ext off");
                let run_on = model
                    .generate_greedy_benchmark(
                        &stage4.e1k.prompt_ids,
                        gen_len,
                        &stage4.short.forced_prefix,
                        true,
                    )
                    .expect("e1k ext on");
                let m = run_on.mtp.as_ref().unwrap();
                eprintln!(
                    "e1k o{gen_len}: MTP-off ttft={:.1}ms tpot={:.3}ms | MTP-on ttft={:.1}ms tpot={:.3}ms med_cycle={:.3}ms tpc={:.3} fallbacks={} exact={}",
                    run_off.ttft_s * 1000.0,
                    run_off.tpot_ms,
                    run_on.ttft_s * 1000.0,
                    run_on.tpot_ms,
                    m.median_cycle_ms().unwrap_or(0.0),
                    m.tokens_per_cycle().unwrap_or(0.0),
                    m.screen_fallbacks,
                    run_off.tokens == run_on.tokens,
                );
                assert_eq!(run_off.tokens, run_on.tokens);
            }
        }
    }

    // 4. LlmTrait streaming parity (MTP-off vs MTP-on, greedy, no forced prefix).
    let opts = GenerationOptions::greedy(20, None);
    let req = GenerationRequest {
        input: LlmInput::text(&stage4.short.prompt_ids),
        options: &opts,
    };
    model.set_mtp_enabled(false);
    let mut llm_off = Vec::new();
    model
        .generate_streaming_with_options(req, |tok| llm_off.push(tok.token_id))
        .expect("llm streaming mtp-off");

    model.set_mtp_enabled(true);
    let mut llm_on = Vec::new();
    model
        .generate_streaming_with_options(req, |tok| llm_on.push(tok.token_id))
        .expect("llm streaming mtp-on");
    eprintln!(
        "LlmTrait streaming 20 tok: exact_match={}",
        llm_off == llm_on
    );
    assert_eq!(llm_off.len(), 20);
    assert_eq!(llm_off, llm_on, "LlmTrait MTP-on diverged from MTP-off");
}

#[test]
#[ignore = "requires local Inkling-2-Mini-oQ4e-mtp checkpoint"]
fn oq4e_mtp_checkpoint_parity() {
    let model_dir = require_env_path("INKLING2_OQ4E_DIR");
    let backend = MlxBackend::new(0).expect("init MlxBackend");
    let t0 = Instant::now();
    let mut model =
        MetalInkling2::load(&model_dir, Device::Metal(0), backend, None).expect("load oq4e-mtp");
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!(
        "Loaded Inkling-2-Mini-oQ4e-mtp in {load_s:.2}s | variant={:?} | has_draft={}",
        model.variant(),
        model.has_draft()
    );
    assert_eq!(model.variant(), Variant::Oq4eMtp);
    assert!(model.has_draft());

    let prompt: Vec<u32> = vec![
        200002, 200004, 1735, 149914, 149878, 149911, 4123, 171159, 13, 200010, 200000, 200004,
        86228, 149899, 152703, 25, 198256, 147818, 145972, 17, 62, 147107, 62, 146056, 200010,
    ];
    let forced = [200010u32, 200001, 200004];
    let off = model
        .generate_greedy_benchmark(&prompt, 32, &forced, false)
        .expect("oq4e off");
    let on = model
        .generate_greedy_benchmark(&prompt, 32, &forced, true)
        .expect("oq4e on");
    let mtp = on.mtp.as_ref().expect("mtp stats");
    eprintln!(
        "oQ4e-mtp 32 tok: MTP-off tpot={:.2}ms | MTP-on tpot={:.2}ms med_cycle={:.2}ms tpc={:.3} fallbacks={} exact={}",
        off.tpot_ms,
        on.tpot_ms,
        mtp.median_cycle_ms().unwrap_or(0.0),
        mtp.tokens_per_cycle().unwrap_or(0.0),
        mtp.screen_fallbacks,
        off.tokens == on.tokens,
    );
    assert_eq!(off.tokens, on.tokens, "oQ4e-mtp MTP-on != MTP-off");
    assert_eq!(mtp.screen_fallbacks, 0);
}

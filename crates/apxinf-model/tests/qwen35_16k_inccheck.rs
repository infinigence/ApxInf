#![cfg(feature = "cuda")]

use std::path::Path;

use apxinf_model::qwen35::gpu::CudaQwen35;

fn gen_prompt(l: usize) -> Vec<u32> {
    let seed: [u32; 8] = [248045, 8678, 198, 24342, 286, 4879, 369, 716];
    let fill: [u32; 16] = [264, 4879, 310, 716, 13, 279, 430, 369, 220, 274, 1236, 70410, 91, 11, 8, 317];
    let mut v = Vec::with_capacity(l);
    for i in 0..l {
        v.push(if i < 8 { seed[i] } else { fill[(i - 8) % 16] });
    }
    v
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    best
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        d += (a[i] as f64) * (b[i] as f64);
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

#[test]
#[ignore]
fn qwen35_16k_incremental_decode_matches_prefill() {
    let model_dir = std::env::var("Q35_MODEL_DIR")
        .expect("set Q35_MODEL_DIR to the Qwen3.8-27B-AWQ-INT4 checkpoint");
    std::env::set_var("Q35_NOGRAPH", "1");

    let mut model = CudaQwen35::load(Path::new(&model_dir)).expect("load CUDA Qwen35");
    let prompt = gen_prompt(16_384);
    let last = prompt[16_383];

    let _ = model.prefill_logits(&prompt[..16_383]).expect("prefill L-1");
    let inc_logits = model.decode_logits(last).expect("decode last token");
    let full_logits = model.prefill_logits(&prompt).expect("full prefill");

    let inc_argmax = argmax(&inc_logits);
    let full_argmax = argmax(&full_logits);
    let cos = cosine(&full_logits, &inc_logits);
    let maxdiff = full_logits
        .iter()
        .zip(&inc_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert_eq!(
        full_argmax,
        inc_argmax,
        "16K decode/prefill argmax mismatch: full={full_argmax} inc={inc_argmax} cos={cos:.6} maxdiff={maxdiff:.4}"
    );
    assert!(
        cos > 0.999,
        "16K decode/prefill cosine too low: cos={cos:.6} maxdiff={maxdiff:.4} full_argmax={full_argmax} inc_argmax={inc_argmax}"
    );
}

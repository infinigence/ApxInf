use std::path::Path;
use std::time::Instant;

use apxinf_model::qwen35::gpu::CudaQwen35;

fn dump(name: &str, logits: &[f32]) {
    let mut buf = Vec::with_capacity(logits.len() * 4);
    for v in logits {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(name, &buf).expect("write logits");
}

fn top(logits: &[f32]) -> (usize, Vec<(usize, f32)>) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    (order[0], order[..5].iter().map(|&i| (i, logits[i])).collect())
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: qwen35_gpu_smoke <model_dir>");
    eprintln!("[gpu] loading model from {dir}");
    let t0 = Instant::now();
    let mut m = CudaQwen35::load(Path::new(&dir)).expect("load");
    eprintln!("[gpu] loaded in {:.1?}", t0.elapsed());

    let ids = [248045u32, 8678, 198, 24342, 286, 4879, 369, 716];
    let t1 = Instant::now();
    let logits = m.prefill_logits(&ids).expect("prefill");
    eprintln!("[gpu] 8-token prefill in {:.2?}", t1.elapsed());
    dump("/tmp/gpu_logits.bin", &logits);
    let (tok1, top5) = top(&logits);
    eprintln!("[gpu] top5: {top5:?}");
    eprintln!("[gpu] argmax: {tok1}");

    // incremental decode: prefill(8) + decode(tok1) must match prefill([.., tok1])
    let t2 = Instant::now();
    let l9 = m.decode_logits(tok1 as u32).expect("decode");
    eprintln!("[gpu] 1-token decode in {:.2?}", t2.elapsed());
    dump("/tmp/gpu_logits9_dec.bin", &l9);
    let (tok2, top5b) = top(&l9);
    eprintln!("[gpu] after decode: argmax {tok2}, top5 {top5b:?}");

    if std::env::var_os("Q35_FULL9").is_some() {
        let mut ids9 = ids.to_vec();
        ids9.push(tok1 as u32);
        let t3 = Instant::now();
        let l9f = m.prefill_logits(&ids9).expect("prefill9");
        eprintln!("[gpu] 9-token prefill in {:.2?}", t3.elapsed());
        dump("/tmp/gpu_logits9_full.bin", &l9f);
    }

    if std::env::var_os("Q35_GEN").is_some() {
        let mut seq = ids.to_vec();
        let mut cur = tok1 as u32;
        let mut tok_s = 0.0f64;
        let steps = std::env::var("Q35_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
        for _ in 0..steps {
            seq.push(cur);
            let ts = Instant::now();
            let lg = m.decode_logits(cur).expect("decode");
            tok_s += ts.elapsed().as_secs_f64();
            let (nx, _) = top(&lg);
            cur = nx as u32;
        }
        eprintln!("[gpu] generated {steps} tokens, avg {:.3}s/token", tok_s / steps as f64);
        eprintln!("[gpu] seq: {seq:?} next={cur}");
    }
}

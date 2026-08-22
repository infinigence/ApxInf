//! GPU long-context prefill dump + incremental decode stability driver.
//! Env: Q35_DEV=<gpu>  Q35_PF_L=<prefill len>  Q35_DEC_STEPS=<n>
//! Prefill dump: /tmp/gpu_logits_L<L>.bin
//! Decode stats printed every 100 steps; generated sequence dumped at end.

use std::path::Path;
use std::time::Instant;

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

fn dump(name: &str, logits: &[f32]) {
    let mut b: Vec<u8> = Vec::with_capacity(logits.len() * 4);
    for v in logits {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(name, &b).expect("write logits");
}

fn top(logits: &[f32]) -> (usize, Vec<(usize, f32)>) {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    (order[0], order[..5].iter().map(|&i| (i, logits[i])).collect())
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

fn main() {
    let dir = std::env::args().nth(1).expect("usage: qwen35_gpu_long <model_dir>");
    let l: usize = std::env::var("Q35_PF_L")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let steps: Option<usize> = std::env::var("Q35_DEC_STEPS").ok().and_then(|s| s.parse().ok());

    let t0 = Instant::now();
    let mut m = CudaQwen35::load(Path::new(&dir)).expect("load");
    eprintln!("[gpu] loaded in {:.2?}", t0.elapsed());

    // incremental self-consistency: prefill(l-1)+decode(last) vs prefill(l)
    let mut inc: Option<Vec<f32>> = None;
    if std::env::var_os("Q35_INCCHECK").is_some() && l >= 2 {
        let sub = gen_prompt(l - 1);
        let last = gen_prompt(l)[l - 1];
        let t = Instant::now();
        let _ = m.prefill_logits(&sub).expect("prefill l-1");
        let inc_logits = m.decode_logits(last).expect("decode last");
        eprintln!("[gpu] L={}-1 prefill + decode(last={last}) in {:.2?}", l, t.elapsed());
        let (tok, _) = top(&inc_logits);
        eprintln!("[gpu] incremental argmax={tok}");
        inc = Some(inc_logits);
    }

    let prompt = gen_prompt(l);
    let t1 = Instant::now();
    let logits = m.prefill_logits(&prompt).expect("prefill");
    eprintln!("[gpu] L={l} prefill in {:.2?}", t1.elapsed());
    dump(&format!("/tmp/gpu_logits_L{l}.bin"), &logits);
    let (tok, top5) = top(&logits);
    let maxv = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let minv = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let norm = (logits.iter().map(|v| v * v).sum::<f32>()).sqrt();
    eprintln!("[gpu] L={l} argmax={tok} top5={top5:?} max={maxv:.2} min={minv:.2} norm={norm:.2}");
    if let Some(b) = &inc {
        let mut diff = 0.0f32;
        for i in 0..logits.len() {
            diff = diff.max((logits[i] - b[i]).abs());
        }
        eprintln!("[gpu] INCCHECK L={l} cosine={:.6} maxdiff={:.4} argmax_match={}", cosine(&logits, b), diff, top(&b).0 == tok);
    }

    if let Some(n) = steps {
        let mut seq = prompt;
        let mut cur = tok as u32;
        let mut t_sum = 0.0f64;
        let mut bad = 0usize;
        for i in 0..n {
            let ts = Instant::now();
            let lg = m.decode_logits(cur).expect("decode");
            let dt = ts.elapsed().as_secs_f64();
            t_sum += dt;
            if lg.iter().any(|v| !v.is_finite()) {
                bad += 1;
            }
            let mv = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let nv = lg.iter().cloned().fold(f32::INFINITY, f32::min);
            let norm = (lg.iter().map(|v| v * v).sum::<f32>()).sqrt();
            let (nx, _) = top(&lg);
            seq.push(cur);
            cur = nx as u32;
            if (i + 1) % 100 == 0 || i + 1 == n {
                eprintln!(
                    "[decode] step={} tok={} logit max={:.2} min={:.2} norm={:.2} avg={:.3}ms",
                    i + 1,
                    nx,
                    mv,
                    nv,
                    norm,
                    dt * 1000.0
                );
            }
        }
        eprintln!("[decode] {} steps, avg {:.3}ms/token, non-finite logits at {} steps", n, t_sum / n as f64 * 1000.0, bad);
        let seq_s: Vec<String> = seq.iter().map(|t| t.to_string()).collect();
        let _ = std::fs::write("/tmp/gpu_gen_seq.txt", seq_s.join(","));
        eprintln!("[decode] final seq len={} next={} -> /tmp/gpu_gen_seq.txt", seq.len(), cur);
    }
}

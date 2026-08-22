//! Smoke test: load Qwen3.8-27B-AWQ-INT4 and run the CPU forward on an
//! 8-token prompt, printing top-5 logits and the greedy next token.

use std::path::Path;
use std::time::Instant;

use apxinf_model::qwen35::cpu::CpuQwen35;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: qwen35_smoke <model_dir>");
    let t0 = Instant::now();
    let mut model = CpuQwen35::load(Path::new(&dir)).expect("load model");
    eprintln!("loaded in {:.2?}", t0.elapsed());

    // First 8 tokens of the chat template for "<|im_start|>user\nHi..."
    let prompt: [u32; 8] = [248045, 8678, 198, 24342, 286, 4879, 369, 716];

    let t1 = Instant::now();
    let logits = model.forward_last_logits(&prompt).expect("forward");
    eprintln!("8-token prefill forward in {:.2?}", t1.elapsed());

    // Dump the full logits vector for cross-checking against the PyTorch
    // reference (little-endian f32).
    let mut buf: Vec<u8> = Vec::with_capacity(logits.len() * 4);
    for v in &logits {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("/tmp/rust_logits.bin", &buf).expect("write logits");
    eprintln!("dumped {} logits to /tmp/rust_logits.bin", logits.len());

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    println!("top-5 next-token candidates:");
    for i in 0..5 {
        println!("  #{}: token_id={} logit={:.4}", i + 1, order[i], logits[order[i]]);
    }

    let t2 = Instant::now();
    let next = model.generate_greedy(&prompt, 1, false).expect("greedy");
    println!("greedy next token (ignore_eos=false): {}  [{:.2?}]", next[0], t2.elapsed());
}

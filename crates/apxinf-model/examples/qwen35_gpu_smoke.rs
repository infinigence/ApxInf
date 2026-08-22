use std::path::Path;
use std::time::Instant;

use apxinf_model::qwen35::gpu::CudaQwen35;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: qwen35_gpu_smoke <model_dir>");
    eprintln!("[gpu] loading model from {dir}");
    let t0 = Instant::now();
    let m = CudaQwen35::load(Path::new(&dir)).expect("load");
    eprintln!("[gpu] loaded in {:.1?}", t0.elapsed());

    let ids = [248045u32, 8678, 198, 24342, 286, 4879, 369, 716];
    let t1 = Instant::now();
    let logits = m.forward_last_logits(&ids).expect("forward");
    let dt8 = t1.elapsed();
    eprintln!("[gpu] 8-token forward in {dt8:.2?}");

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    eprintln!("[gpu] top5: {:?}", order[..5].iter().map(|&i| (i, logits[i])).collect::<Vec<_>>());
    eprintln!("[gpu] argmax: {}", order[0]);

    let mut buf = Vec::with_capacity(logits.len() * 4);
    for v in &logits {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write("/tmp/gpu_logits.bin", &buf).expect("write logits");
    eprintln!("[gpu] dumped {} logits to /tmp/gpu_logits.bin", logits.len());

    for (env_name, want) in [("Q35_L128", 128usize), ("Q35_L1024", 1024usize)] {
        if std::env::var_os(env_name).is_some() {
            let mut long: Vec<u32> = Vec::with_capacity(want);
            for i in 0..want {
                long.push(ids[i % ids.len()]);
            }
            let t2 = Instant::now();
            let _ = m.forward_last_logits(&long).expect("forward long");
            eprintln!("[gpu] {want}-token forward in {:.2?}", t2.elapsed());
        }
    }
}

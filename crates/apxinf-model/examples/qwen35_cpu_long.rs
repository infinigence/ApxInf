//! CPU long-context reference dump for qwen35.
//! Q35_CPU_L=512,1024,...  (one or more; comma-separated)
//! For each L: forward_last_logits(gen_prompt(L)) -> /tmp/cpu_logits_L<L>.bin

use std::path::Path;
use std::time::Instant;

use apxinf_model::qwen35::cpu::CpuQwen35;

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

fn main() {
    let dir = std::env::args().nth(1).expect("usage: qwen35_cpu_long <model_dir>");
    let lens: Vec<usize> = std::env::var("Q35_CPU_L")
        .expect("set Q35_CPU_L=512,1024,...")
        .split(',')
        .map(|s| s.trim().parse().expect("len"))
        .collect();
    let t0 = Instant::now();
    let mut model = CpuQwen35::load(Path::new(&dir)).expect("load model");
    eprintln!("[cpu] loaded in {:.2?}", t0.elapsed());

    if let Some(path) = std::env::var_os("Q35_IDS_FILE") {
        let ids: Vec<u32> = std::fs::read_to_string(std::path::Path::new(&path))
            .expect("read ids")
            .split_whitespace()
            .map(|s| s.parse::<u32>().unwrap())
            .collect();
        let t = Instant::now();
        let logits = model.forward_last_logits(&ids).expect("forward");
        let dt = t.elapsed();
        let argmax = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        dump("/tmp/cpu_ids_logits.bin", &logits);
        eprintln!("[cpu] ids-file L={} forward in {:.2?} argmax={argmax} -> /tmp/cpu_ids_logits.bin", ids.len(), dt);
        return;
    }

    for l in lens {
        let p = gen_prompt(l);
        let t = Instant::now();
        let logits = model.forward_last_logits(&p).expect("forward");
        let dt = t.elapsed();
        let maxv = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let minv = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        dump(&format!("/tmp/cpu_logits_L{l}.bin"), &logits);
        eprintln!("[cpu] L={l} forward in {:.2?} argmax={argmax} max={maxv:.2} min={minv:.2} -> /tmp/cpu_logits_L{l}.bin", dt);
    }
}

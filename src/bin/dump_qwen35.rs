//! Debug tool: run one Qwen3.5 forward pass and dump per-layer hidden states
//! plus the final logits as raw little-endian f32 files for comparison against
//! a torch reference.
//!
//! Usage:
//!   dump_qwen35 --model <dir> --input <json array file> --out <dir> [--device cuda|cpu] [--start-pos N]
//!   dump_qwen35 --model <dir> --input <json array file> --out <dir> --gen-tokens K [--forced-tokens <json>]
//!
//! With `--gen-tokens K` the binary runs a greedy decode loop (prefill + K
//! steps) and dumps the logits of each step, the per-layer hidden state of
//! each step (`state_sNN_lMM.f32`), plus `gen.jsonl` with the chosen token and
//! its margin over the runner-up. With `--forced-tokens`, step s feeds the
//! forced token instead of the greedy choice so both sides can be compared
//! step-by-step on identical inputs.

use std::path::PathBuf;
use std::sync::Arc;

use apxinf_core::{Backend, Device};
use apxinf_model::llm_trait::LlmTrait;
use apxinf_model::qwen35::{GeneralQwen35, Qwen35Config};

fn main() {
    let mut model_dir = PathBuf::new();
    let mut input_file = PathBuf::new();
    let mut out_dir = PathBuf::new();
    let mut device = Device::Cuda(0);
    let mut start_pos: u32 = 0;
    let mut gen_tokens: usize = 0;
    let mut forced_tokens: Vec<u32> = Vec::new();
    let mut gpu_dump = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gpu-dump" => gpu_dump = true,
            "--model" => model_dir = PathBuf::from(args.next().expect("--model value")),
            "--input" => input_file = PathBuf::from(args.next().expect("--input value")),
            "--out" => out_dir = PathBuf::from(args.next().expect("--out value")),
            "--start-pos" => {
                start_pos = args.next().expect("--start-pos value").parse().expect("integer")
            }
            "--gen-tokens" => {
                gen_tokens = args.next().expect("--gen-tokens value").parse().expect("integer")
            }
            "--forced-tokens" => {
                let raw = std::fs::read_to_string(args.next().expect("--forced-tokens value"))
                    .expect("read forced tokens");
                forced_tokens =
                    serde_json::from_str(&raw).expect("forced tokens must be a JSON array");
            }
            "--device" => {
                device = match args.next().expect("--device value").as_str() {
                    "cpu" => Device::Cpu,
                    "cuda" | "gpu" => Device::Cuda(0),
                    other => panic!("unknown device {other}"),
                }
            }
            other => panic!("unknown arg {other}"),
        }
    }


    let raw = std::fs::read_to_string(&input_file).expect("read input file");
    let token_ids: Vec<u32> = serde_json::from_str(&raw).expect("input must be a JSON array of u32");

    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))
        .expect("parse config.json");
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(&model_dir)
        .expect("load safetensors");
    let backend: Arc<dyn Backend> = apxinf_model::accelerator::create_backend(device)
        .expect("create backend");
    let mut model =
        GeneralQwen35::from_weights_with_backend(config, tensors, backend)
            .expect("construct model");

    let n = model.config.text.n_layers;
    // GPU-path per-layer dump (state_lNN.f32): bisection aid.
    if gpu_dump {
        #[cfg(feature = "cuda")]
        {
            model
                .forward_dump_gpu(&token_ids, start_pos, &out_dir)
                .expect("gpu forward dump");
            eprintln!("dumped {n} GPU layer states to {}", out_dir.display());
            return;
        }
        #[cfg(not(feature = "cuda"))]
        panic!("--gpu-dump requires a cuda build");
    }
    if gen_tokens > 0 {
        run_generate_dump(&mut model, &token_ids, &out_dir, gen_tokens, n, &forced_tokens);
        return;
    }

    let logits = model
        .forward_with_hook(&token_ids, start_pos, |layer, x| {
            let data = x.to_f32_vec().expect("layer state as f32");
            let path = out_dir.join(format!("layer_{layer:02}.f32"));
            write_f32(&path, &data);
        })
        .expect("forward");

    let logits_data = logits.to_f32_vec().expect("logits as f32");
    write_f32(&out_dir.join("logits.f32"), &logits_data);
    std::fs::write(
        out_dir.join("shape.json"),
        serde_json::json!({
            "seq": token_ids.len(),
            "vocab": logits_data.len() / token_ids.len().max(1),
            "layers": n,
            "start_pos": start_pos,
        })
        .to_string(),
    )
    .expect("write shape.json");
    eprintln!("dumped {n} layer states + logits to {}", out_dir.display());
}

fn run_generate_dump(
    model: &mut GeneralQwen35,
    token_ids: &[u32],
    out_dir: &std::path::Path,
    gen_tokens: usize,
    n: usize,
    forced: &[u32],
) {
    let vocab = model.vocab_size();
    let mut greedy_prev: u32 = 0;
    let prompt_len = token_ids.len();
    for step in 0..=gen_tokens {
        let t0 = std::time::Instant::now();
        let logits = if step == 0 {
            model.forward(token_ids, 0)
        } else {
            // Feed the forced token when provided, otherwise the previous greedy pick.
            let fed = if step - 1 < forced.len() {
                forced[step - 1]
            } else {
                greedy_prev
            };
            model.forward(&[fed], (prompt_len + step - 1) as u32)
        }
        .expect("forward");
        eprintln!("[step {step}] fwd: {:.2} ms", (std::time::Instant::now() - t0).as_secs_f32() * 1000.0);
        let t0 = std::time::Instant::now();
        let data = {
            #[cfg(feature = "cuda")]
            {
                if logits.device().is_gpu() {
                    apxinf_cuda::transfers::to_cpu(&logits).expect("to_cpu")
                } else {
                    logits
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                logits
            }
        };
        eprintln!("[step {step}] fwd+d2h: {:.2} ms", (std::time::Instant::now() - t0).as_secs_f32() * 1000.0);
        let data = data.to_f32_vec().expect("logits as f32");
        let (tok, margin) = argmax_with_margin(&data, vocab);
        greedy_prev = tok;
        let fed = if step == 0 {
            0
        } else if step - 1 < forced.len() {
            forced[step - 1]
        } else {
            tok
        };
        let name = if step == 0 {
            "logits_prefill.f32".to_string()
        } else {
            format!("logits_step_{:02}.f32", step - 1)
        };
        write_f32(&out_dir.join(&name), &data);
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out_dir.join("gen.jsonl"))
            .expect("open gen.jsonl");
        writeln!(
            f,
            "{{\"step\":{step},\"token\":{tok},\"fed\":{fed},\"margin\":{margin}}}"
        )
        .expect("append");
    }
    std::fs::write(
        out_dir.join("shape.json"),
        serde_json::json!({
            "seq": token_ids.len(), "vocab": vocab,
            "layers": n, "start_pos": 0, "gen_tokens": gen_tokens,
        })
        .to_string(),
    )
    .expect("write shape.json");
    eprintln!("dumped prefill + {gen_tokens} decode logits to {}", out_dir.display());
}

fn argmax_with_margin(data: &[f32], vocab: usize) -> (u32, f32) {
    let row = &data[data.len() - vocab..];
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    let mut best_i = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            second = best;
            best = v;
            best_i = i;
        } else if v > second {
            second = v;
        }
    }
    (best_i as u32, best - second)
}

fn write_f32(path: &std::path::Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write f32 file");
}

//! Multimodal (text+image) verification tool.
//!
//! Loads Qwen3-VL, reads the reference pixel_values + grid_thw + prompt
//! tokens from the Phase 0 dump, runs the unified LLM/VLM generation
//! interface, and prints the first 10 greedy tokens.
//!
//! Usage:
//!   cargo run --example multimodal_check --features cuda -- \
//!       /hanjinchen/models/Qwen3-VL-2B-Instruct

use std::io::Read;
use std::path::PathBuf;

use apxinf_core::{Device, Tensor};
use apxinf_model::{GeneralQwen3VL, ImageInput, LlmInput, LlmTrait};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: multimodal_check <model_dir>");
        std::process::exit(1);
    }
    let model_dir = PathBuf::from(&args[1]);

    // Load reference data from the Phase 0 .npz dump.
    // We need: tokens, image_pixel_values, image_grid_thw, greedy_tokens (for comparison).
    let ref_npz = "tests/qwen3vl_reference/qwen3vl_image.npz";
    let (tokens, pv_data, pv_shape, grid, expected_greedy) = load_ref(ref_npz);
    eprintln!(
        "prompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(15)]
    );
    eprintln!("pixel_values shape: {:?}", pv_shape);
    eprintln!("grid_thw: {:?}", grid);
    eprintln!("expected greedy: {:?}", expected_greedy);

    // Load model.
    let mut model = GeneralQwen3VL::from_dir(&model_dir, Device::Cuda(0)).expect("load model");
    eprintln!("Model loaded.");

    // The unified prefill accepts processor output on CPU and uploads it once.
    let pv_cpu = Tensor::from_bf16(pv_shape.clone(), &pv_data).expect("build pv tensor");
    let grids = [grid];
    let (generated, _) = model
        .generate_streaming(
            LlmInput::with_image(&tokens, ImageInput::new(&pv_cpu, &grids)),
            10,
            |_| {},
            None,
        )
        .expect("multimodal generation");
    eprintln!("apxinf greedy (10): {:?}", generated);

    // Compare.
    if generated == expected_greedy {
        println!("PASS: first 10 greedy tokens match HF reference");
    } else {
        let matches: usize = generated
            .iter()
            .zip(expected_greedy.iter())
            .take_while(|(a, b)| a == b)
            .count();
        println!("MISMATCH: {matches}/10 tokens match");
        println!("  apxinf:  {:?}", generated);
        println!("  HF:     {:?}", expected_greedy);
    }
}

fn load_ref(npz_path: &str) -> (Vec<u32>, Vec<half::bf16>, Vec<usize>, [u32; 3], Vec<u32>) {
    // The .npz is a ZIP file. We need to extract the arrays. Use Python
    // via a subprocess to dump them as raw .npy, then read with our .npy
    // parser. Simpler: call a Python one-liner to save the arrays as
    // separate .npy files, then read them here.
    //
    // Actually — let's just read the .npz directly. It's a ZIP archive.
    // We'll use the `zip` crate... but we don't have it. Let me use
    // Python to extract.
    use std::process::Command;
    let tmp = "/tmp/multimodal_ref";
    std::fs::create_dir_all(tmp).ok();
    let script = format!(
        "import numpy as np; d=np.load('{npz_path}'); np.save('{tmp}/tokens.npy', d['tokens']); np.save('{tmp}/pv.npy', d['image_pixel_values'].astype(np.float32)); np.save('{tmp}/greedy.npy', d['greedy_tokens']); print(d['image_grid_thw'][0])");
    let out = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .output()
        .expect("python");
    let grid_str = String::from_utf8(out.stdout).unwrap().trim().to_string();
    let grid_vals: Vec<u32> = grid_str
        .replace('[', "")
        .replace(']', "")
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let grid = [grid_vals[0], grid_vals[1], grid_vals[2]];

    let tokens = read_npy_i64(&format!("{tmp}/tokens.npy"));
    let (pv_shape, pv_data) = read_npy_bf16(&format!("{tmp}/pv.npy"));
    let greedy = read_npy_i64(&format!("{tmp}/greedy.npy"));
    (tokens, pv_data, pv_shape, grid, greedy)
}

fn read_npy_i64(path: &str) -> Vec<u32> {
    let mut f = std::fs::File::open(path).expect("open");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("read");
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let data_start = 10 + header_len;
    let n = (buf.len() - data_start) / 8;
    (0..n)
        .map(|i| {
            let off = data_start + i * 8;
            u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()) as u32
        })
        .collect()
}

fn read_npy_bf16(path: &str) -> (Vec<usize>, Vec<half::bf16>) {
    let mut f = std::fs::File::open(path).expect("open");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("read");
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = std::str::from_utf8(&buf[10..10 + header_len]).unwrap();
    let shape = parse_shape(header);
    let data_start = 10 + header_len;
    let raw = &buf[data_start..];
    // f32 .npy — read as f32 then convert to bf16.
    let n = raw.len() / 4;
    let data: Vec<half::bf16> = (0..n)
        .map(|i| {
            let off = i * 4;
            let v = f32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
            half::bf16::from_f32(v)
        })
        .collect();
    (shape, data)
}

fn parse_shape(header: &str) -> Vec<usize> {
    let idx = header.find("shape").unwrap();
    let paren = header[idx..].find('(').unwrap();
    let close = header[idx + paren..].find(')').unwrap();
    let inner = &header[idx + paren + 1..idx + paren + close];
    if inner.trim().is_empty() {
        return vec![];
    }
    inner
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect()
}

//! Vision tower verification tool.
//!
//! Loads Qwen3-VL from a model directory, reads preprocessed pixel_values
//! (as a .npy file) + grid_thw, runs the vision tower, and writes the
//! primary + 3 deepstack embeddings as .npy files for external comparison.
//!
//! Usage:
//!   cargo run --example vision_check --features cuda -- \
//!       /hanjinchen/models/Qwen3-VL-2B-Instruct \
//!       tests/qwen3vl_reference/image_pixel_values.npy \
//!       "1 20 20" \
//!       /tmp/vision_out

use std::path::PathBuf;
use std::io::Read;

use apxinf_core::{Device, Tensor};
use apxinf_model::GeneralQwen3VL;
use apxinf_model::qwen3vl::vision;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: vision_check <model_dir> <pixel_values.npy> <grid_thw 'T H W'> <output_dir>");
        std::process::exit(1);
    }
    let model_dir = PathBuf::from(&args[1]);
    let pv_path = &args[2];
    let grid: Vec<u32> = args[3].split_whitespace().map(|s| s.parse().unwrap()).collect();
    let out_dir = PathBuf::from(&args[4]);

    // Read the .npy pixel_values file.
    let (pv_shape, pv_data) = read_npy_bf16(pv_path).expect("read pixel_values.npy");
    eprintln!("pixel_values shape: {:?}, {} bytes", pv_shape, pv_data.len());

    // Load model.
    let mut model = GeneralQwen3VL::from_dir(&model_dir, Device::Cuda(0)).expect("load model");
    eprintln!("Model loaded.");

    // Upload pixel_values to GPU.
    let pixel_values_cpu = Tensor::from_bf16(pv_shape.clone(), &pv_data).expect("build tensor");
    let pixel_values = model.backend().to_device(&pixel_values_cpu).expect("upload pixel_values");

    // Run vision tower with debug dumps.
    let grid_thw = vec![[grid[0], grid[1], grid[2]]];
    let vis = vision::forward_debug(
        &model.config_ref(), &model.vision_weights_ref(),
        model.backend(), &pixel_values, &grid_thw, "/tmp/apxinf",
    ).expect("vision forward");
    eprintln!("primary shape: {:?}", vis.primary.shape().dims());

    // Download outputs to CPU and write as .npy.
    std::fs::create_dir_all(&out_dir).ok();
    let be = model.backend();
    let primary_cpu = be.to_cpu(&vis.primary).expect("download primary");
    write_npy_bf16(&out_dir.join("vision_primary.npy"), &primary_cpu).unwrap();
    for (i, ds) in vis.deepstack.iter().enumerate() {
        let ds_cpu = be.to_cpu(ds).expect("download deepstack");
        write_npy_bf16(&out_dir.join(format!("vision_deepstack_{i}.npy")), &ds_cpu).unwrap();
    }
    eprintln!("Wrote outputs to {}", out_dir.display());
}

// ── Minimal .npy reader/writer for bf16 tensors ──────────────────────────

fn read_npy_bf16(path: &str) -> Result<(Vec<usize>, Vec<half::bf16>), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;

    // .npy magic: \x93NUMPY + version(2 bytes) + header_len + header
    if &buf[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }
    let major = buf[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    } else {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&buf[header_start..header_start + header_len])
        .map_err(|e| format!("header utf8: {e}"))?;
    let data_start = header_start + header_len;

    // Parse shape from header. Simple: find "shape': (..." and parse the tuple.
    let shape = parse_npy_shape(header)?;
    let dtype = if header.contains("'<f2'") || header.contains("'<bfloat16'") {
        "bf16"
    } else if header.contains("<f4") {
        "f32"
    } else {
        return Err(format!("unsupported dtype in {path}"));
    };

    let total: usize = shape.iter().product();
    let raw = &buf[data_start..];
    if dtype == "bf16" {
        let data: Vec<half::bf16> = raw.chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
            .collect();
        if data.len() != total {
            return Err(format!("expected {total} elements, got {}", data.len()));
        }
        Ok((shape, data))
    } else {
        // f32 → upcast to bf16
        let data: Vec<half::bf16> = raw.chunks_exact(4)
            .map(|c| half::bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        Ok((shape, data))
    }
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>, String> {
    let idx = header.find("shape").ok_or("no shape in header")?;
    let paren = header[idx..].find('(').ok_or("no ( after shape")?;
    let close = header[idx + paren..].find(')').ok_or("no ) after shape")?;
    let inner = &header[idx + paren + 1..idx + paren + close];
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }
    inner.split(',')
        .map(|s| s.trim().parse::<usize>()
            .map_err(|e| format!("parse shape dim '{s}': {e}")))
        .collect()
}

fn write_npy_bf16(path: &PathBuf, t: &Tensor) -> Result<(), String> {
    let f32_data = t.to_f32_vec().map_err(|e| format!("to_f32_vec: {e}"))?;
    let dims = t.shape().dims();
    let shape_str = dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}");
    // Pad header to multiple of 64 bytes (numpy convention).
    let pad = (64 - ((header.len() + 10) % 64)) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');

    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1); // version major
    out.push(0); // version minor
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    // Data as f32 LE.
    let f32_bytes: Vec<u8> = f32_data.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    out.extend_from_slice(&f32_bytes);

    std::fs::write(path, &out).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

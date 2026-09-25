//! Gated DeltaNet chunked prefill scan: equivalence to the single-token
//! recurrent path, and agreement with the official reference tensors.
//!
//! The whole point of the chunked scan is that a prompt run through it leaves
//! *exactly* the recurrent state the single-token `gdn_recurrent_step` would
//! have reached feeding those same tokens one at a time. If the two disagree,
//! generation after a prompt continues from a corrupted state. The first test
//! here proves that equivalence per layer-of-state; the second checks the
//! chunk scan's output and final state against `torch_chunk_gated_delta_rule`.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

fn upload_bf16(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::BF16).unwrap()
}

fn upload_f32(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::F32).unwrap()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| half::bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
        .collect()
}

/// Cosine similarity and relative L2 error between two vectors, in f64.
fn cosine_rel_l2(got: &[f32], want: &[f32]) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut diff = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let (g, w) = (*g as f64, *w as f64);
        dot += g * w;
        ng += g * g;
        nw += w * w;
        diff += (g - w) * (g - w);
    }
    let cos = if ng > 0.0 && nw > 0.0 {
        dot / (ng.sqrt() * nw.sqrt())
    } else {
        1.0
    };
    let rel = if nw > 0.0 { (diff / nw).sqrt() } else { diff.sqrt() };
    (cos, rel)
}

/// L2-normalize a single head-vector in fp32 with eps=1e-6, matching both the
/// chunk kernel's in-kernel norm and `l2norm` in the reference.
fn l2norm_vec(v: &[f32]) -> Vec<f32> {
    let ss: f32 = v.iter().map(|x| x * x).sum();
    let inv = 1.0f32 / (ss + 1e-6f32).sqrt();
    v.iter().map(|x| x * inv).collect()
}

// Head geometry of this checkpoint's GDN layers.
const V_HEADS: usize = 48;
const K_HEADS: usize = 16;
const DIM: usize = 128;
const CHUNK: usize = 64;

/// Deterministic pseudo-random activations for a token, distinct per token/head,
/// with decay values that are safely <= 0 (as the model guarantees for g).
#[allow(clippy::type_complexity)]
fn token_inputs(token: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let q: Vec<f32> = (0..K_HEADS * DIM)
        .map(|i| (((i + token * 13) % 17) as f32 - 8.0) / 11.0)
        .collect();
    let k: Vec<f32> = (0..K_HEADS * DIM)
        .map(|i| (((i * 3 + token * 7) % 19) as f32 - 9.0) / 13.0)
        .collect();
    let v: Vec<f32> = (0..V_HEADS * DIM)
        .map(|i| (((i * 5 + token * 11) % 23) as f32 - 11.0) / 17.0)
        .collect();
    // g (log decay) must be <= 0; spread a small negative range per head/token.
    let g: Vec<f32> = (0..V_HEADS)
        .map(|h| -0.05 - 0.01 * ((h + token) % 7) as f32)
        .collect();
    let beta: Vec<f32> = (0..V_HEADS)
        .map(|h| 0.3 + 0.05 * ((h + token) % 5) as f32)
        .collect();
    (q, k, v, g, beta)
}

/// Chunked prefill of N tokens must leave the same per-head recurrent state as
/// feeding those same N tokens one at a time through `gdn_recurrent_step`.
///
/// N is chosen to span more than one chunk (chunk_size = 64) so the sequential
/// cross-chunk state carry is exercised, and to leave a partial final chunk so
/// the padding path is exercised too. The single-token path defines the
/// contract; the chunk path must match it within cosine >= 0.9999, relL2 <= 1%.
#[test]
fn chunked_prefill_state_matches_single_token_decode() {
    let ctx = CudaContext::new(0).unwrap();
    let n_tokens = 80usize; // > 64: two chunks, second one padded.
    let pad = (CHUNK - n_tokens % CHUNK) % CHUNK;
    let seq_padded = n_tokens + pad;
    let num_chunks = seq_padded / CHUNK;

    // --- Single-token decode path: build the ground-truth state. -------------
    let state_len = ops::gdn_state_elements(V_HEADS, DIM, DIM);
    let dec_state = upload_f32(&ctx, &vec![0.0f32; state_len], vec![V_HEADS, DIM, DIM]);
    let dec_out = upload_bf16(&ctx, &vec![0.0f32; V_HEADS * DIM], vec![V_HEADS, DIM]);

    // Assemble the padded prefill activations while we walk the decode path, so
    // both paths consume byte-identical (BF16-rounded, L2-normed) q/k.
    let mut pf_q = vec![0.0f32; seq_padded * K_HEADS * DIM];
    let mut pf_k = vec![0.0f32; seq_padded * K_HEADS * DIM];
    let mut pf_v = vec![0.0f32; seq_padded * V_HEADS * DIM];
    let mut pf_g = vec![0.0f32; seq_padded * V_HEADS];
    let mut pf_b = vec![0.0f32; seq_padded * V_HEADS];

    for token in 0..n_tokens {
        let (q, k, v, g, beta) = token_inputs(token);

        // L2-normalize q/k per head in fp32 (what the chunk kernel does inside).
        // Feed the *normalized* q/k to both paths so the only thing under test
        // is the recurrence, not where the norm happens.
        let mut qn = vec![0.0f32; K_HEADS * DIM];
        let mut kn = vec![0.0f32; K_HEADS * DIM];
        for h in 0..K_HEADS {
            qn[h * DIM..(h + 1) * DIM].copy_from_slice(&l2norm_vec(&q[h * DIM..(h + 1) * DIM]));
            kn[h * DIM..(h + 1) * DIM].copy_from_slice(&l2norm_vec(&k[h * DIM..(h + 1) * DIM]));
        }

        // Decode step. gdn_recurrent_step takes already-normalized q/k and g as
        // the decay argument; it applies the 1/sqrt(k_dim) readout scaling.
        let dq = upload_bf16(&ctx, &qn, vec![K_HEADS, DIM]);
        let dk = upload_bf16(&ctx, &kn, vec![K_HEADS, DIM]);
        let dv = upload_bf16(&ctx, &v, vec![V_HEADS, DIM]);
        let dg = upload_f32(&ctx, &g, vec![V_HEADS]);
        let db = upload_f32(&ctx, &beta, vec![V_HEADS]);
        ops::gdn_recurrent_step(&ctx, &dec_state, &dq, &dk, &dv, &dg, &db, &dec_out, K_HEADS)
            .unwrap();

        // Stash the SAME normalized-then-BF16-rounded q/k for prefill. The
        // chunk kernel L2-normalizes again, but normalizing an already-unit
        // vector is idempotent up to eps, so the two paths stay aligned.
        for h in 0..K_HEADS {
            for d in 0..DIM {
                let src = h * DIM + d;
                let dst = (token * K_HEADS + h) * DIM + d;
                pf_q[dst] = half::bf16::from_f32(qn[src]).to_f32();
                pf_k[dst] = half::bf16::from_f32(kn[src]).to_f32();
            }
        }
        for i in 0..V_HEADS * DIM {
            pf_v[token * V_HEADS * DIM + i] = v[i];
        }
        for h in 0..V_HEADS {
            pf_g[token * V_HEADS + h] = g[h];
            pf_b[token * V_HEADS + h] = beta[h];
        }
    }
    ctx.synchronize().unwrap();
    let decode_state = read_f32(&dec_state);

    // --- Chunked prefill path over the padded sequence. ----------------------
    let pf_state = upload_f32(&ctx, &vec![0.0f32; state_len], vec![V_HEADS, DIM, DIM]);
    let pf_out = upload_bf16(
        &ctx,
        &vec![0.0f32; seq_padded * V_HEADS * DIM],
        vec![seq_padded, V_HEADS, DIM],
    );
    let dq = upload_bf16(&ctx, &pf_q, vec![seq_padded, K_HEADS, DIM]);
    let dk = upload_bf16(&ctx, &pf_k, vec![seq_padded, K_HEADS, DIM]);
    let dv = upload_bf16(&ctx, &pf_v, vec![seq_padded, V_HEADS, DIM]);
    let dg = upload_f32(&ctx, &pf_g, vec![seq_padded, V_HEADS]);
    let db = upload_f32(&ctx, &pf_b, vec![seq_padded, V_HEADS]);
    ops::gdn_chunk_scan(
        &ctx, &dq, &dk, &dv, &dg, &db, &pf_out, &pf_state, V_HEADS, K_HEADS, CHUNK,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let prefill_state = read_f32(&pf_state);

    println!(
        "prefill/decode state equivalence: {n_tokens} tokens, {num_chunks} chunks (pad {pad})"
    );
    let head_stride = DIM * DIM;
    let mut worst_cos = 1.0f64;
    let mut worst_rel = 0.0f64;
    let mut worst_head = 0usize;
    for h in 0..V_HEADS {
        let a = &prefill_state[h * head_stride..(h + 1) * head_stride];
        let b = &decode_state[h * head_stride..(h + 1) * head_stride];
        let (cos, rel) = cosine_rel_l2(a, b);
        if cos < worst_cos {
            worst_cos = cos;
            worst_head = h;
        }
        worst_rel = worst_rel.max(rel);
        if h < 4 || cos < 0.9999 || rel > 0.01 {
            println!("  head {h:2}: cosine {cos:.6}  relL2 {rel:.5}");
        }
    }
    println!("  WORST: cosine {worst_cos:.6} (head {worst_head})  relL2 {worst_rel:.5}");

    assert!(
        worst_cos >= 0.9999,
        "prefill state diverged from decode: worst cosine {worst_cos} at head {worst_head}"
    );
    assert!(
        worst_rel <= 0.01,
        "prefill state diverged from decode: worst relL2 {worst_rel}"
    );
}

// -------------------------------------------------------------------------
// Reference-tensor validation against torch_chunk_gated_delta_rule (seq8).
// -------------------------------------------------------------------------

/// Minimal .npy reader for little-endian float32 C-order arrays.
fn load_npy_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "{path}: not a npy file");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(
        header.contains("'<f4'") || header.contains("\"<f4\""),
        "{path}: expected <f4, header={header}"
    );
    let shape_str = header
        .split("'shape':")
        .nth(1)
        .unwrap()
        .split('(')
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    let data_start = 10 + header_len;
    let data: Vec<f32> = bytes[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (shape, data)
}

fn ref_dir() -> Option<String> {
    for base in [
        "devlocal/qwen38-nvfp4/reference-tensors/seq8",
        "../../devlocal/qwen38-nvfp4/reference-tensors/seq8",
    ] {
        if std::path::Path::new(&format!("{base}/l0_16_core_attn_out.npy")).exists() {
            return Some(base.to_string());
        }
    }
    None
}

/// The chunk scan's output (core_attn_out) and final recurrent state must match
/// the official reference tensors captured from torch_chunk_gated_delta_rule on
/// layer 0 of the seq8 sequence.
#[test]
fn chunk_scan_matches_reference_tensors_layer0() {
    let dir = match ref_dir() {
        Some(d) => d,
        None => {
            // Silently returning here reports a pass having asserted nothing.
            // That is how this test spent a session validating kernels it
            // never compared: the tensors live under devlocal, which is
            // gitignored and so absent from every fresh worktree.
            // APXINF_QWEN38_ALLOW_NO_REFERENCE=1 opts out deliberately.
            if std::env::var("APXINF_QWEN38_ALLOW_NO_REFERENCE").is_ok() {
                eprintln!("reference tensors not found; skipping by request");
                return;
            }
            panic!(
                "reference tensors not found under devlocal/qwen38-nvfp4/\
reference-tensors/seq8. This test compares nothing without them. Link or copy \
devlocal into this checkout, or set APXINF_QWEN38_ALLOW_NO_REFERENCE=1 to skip \
on purpose."
            );
        }
    };
    let ctx = CudaContext::new(0).unwrap();

    // Inputs to the GDN scan, as the reference splits them just before it:
    //   q/k split: [1, seq, k_heads, dim]; v split: [1, seq, v_heads, dim]
    //   g_log_decay / beta: [1, seq, v_heads]
    let (q_shape, q_data) = load_npy_f32(&format!("{dir}/l0_08_q_split.npy"));
    let (_k_shape, k_data) = load_npy_f32(&format!("{dir}/l0_08_k_split.npy"));
    let (_v_shape, v_data) = load_npy_f32(&format!("{dir}/l0_08_v_split.npy"));
    let (_g_shape, g_data) = load_npy_f32(&format!("{dir}/l0_10_g_log_decay.npy"));
    let (_b_shape, b_data) = load_npy_f32(&format!("{dir}/l0_09_beta.npy"));

    let seq = q_shape[1];
    let k_heads = q_shape[2];
    let dim = q_shape[3];
    let v_heads = V_HEADS;
    assert_eq!(dim, DIM);
    assert_eq!(k_heads, K_HEADS);

    let pad = (CHUNK - seq % CHUNK) % CHUNK;
    let seq_padded = seq + pad;

    // Lay out padded prefill buffers. q/k are fed raw (the kernel L2-norms).
    let mut pf_q = vec![0.0f32; seq_padded * k_heads * dim];
    let mut pf_k = vec![0.0f32; seq_padded * k_heads * dim];
    let mut pf_v = vec![0.0f32; seq_padded * v_heads * dim];
    let mut pf_g = vec![0.0f32; seq_padded * v_heads]; // padded g stays 0 (exp(0)=1, harmless: padded tokens have beta 0)
    let mut pf_b = vec![0.0f32; seq_padded * v_heads]; // padded beta 0 -> no state contribution

    for t in 0..seq {
        for h in 0..k_heads {
            for d in 0..dim {
                let src = ((t * k_heads) + h) * dim + d;
                let dst = ((t * k_heads) + h) * dim + d;
                pf_q[dst] = q_data[src];
                pf_k[dst] = k_data[src];
            }
        }
        for h in 0..v_heads {
            for d in 0..dim {
                let src = ((t * v_heads) + h) * dim + d;
                pf_v[((t * v_heads) + h) * dim + d] = v_data[src];
            }
            pf_g[t * v_heads + h] = g_data[t * v_heads + h];
            pf_b[t * v_heads + h] = b_data[t * v_heads + h];
        }
    }

    let state_len = ops::gdn_state_elements(v_heads, dim, dim);
    let pf_state = upload_f32(&ctx, &vec![0.0f32; state_len], vec![v_heads, dim, dim]);
    let pf_out = upload_bf16(
        &ctx,
        &vec![0.0f32; seq_padded * v_heads * dim],
        vec![seq_padded, v_heads, dim],
    );
    let dq = upload_bf16(&ctx, &pf_q, vec![seq_padded, k_heads, dim]);
    let dk = upload_bf16(&ctx, &pf_k, vec![seq_padded, k_heads, dim]);
    let dv = upload_bf16(&ctx, &pf_v, vec![seq_padded, v_heads, dim]);
    let dg = upload_f32(&ctx, &pf_g, vec![seq_padded, v_heads]);
    let db = upload_f32(&ctx, &pf_b, vec![seq_padded, v_heads]);
    ops::gdn_chunk_scan(
        &ctx, &dq, &dk, &dv, &dg, &db, &pf_out, &pf_state, v_heads, k_heads, CHUNK,
    )
    .unwrap();
    ctx.synchronize().unwrap();

    // --- core_attn_out: kernel [seq_padded, v_heads, dim] vs ref [1, seq, v_heads, dim]
    let out = read_bf16(&pf_out);
    let (_o_shape, ref_out) = load_npy_f32(&format!("{dir}/l0_16_core_attn_out.npy"));
    let mut got_out = Vec::with_capacity(seq * v_heads * dim);
    for t in 0..seq {
        for h in 0..v_heads {
            for d in 0..dim {
                got_out.push(out[((t * v_heads) + h) * dim + d]);
            }
        }
    }
    let (out_cos, out_rel) = cosine_rel_l2(&got_out, &ref_out);
    println!("core_attn_out (l0_16): cosine {out_cos:.6}  relL2 {out_rel:.5}");

    // --- final state: kernel port [v_heads, v_dim, k_dim] = S[v][k]
    //     ref [1, v_heads, k_dim, v_dim] = S_ref[k][v]. Transpose to compare.
    let state = read_f32(&pf_state);
    let (_s_shape, ref_state) = load_npy_f32(&format!("{dir}/l0_15_recurrent_state_final.npy"));
    let mut got_state = vec![0.0f32; v_heads * dim * dim];
    for h in 0..v_heads {
        for kk in 0..dim {
            for vv in 0..dim {
                // ref index [h][k][v] -> our port [h][v][k]
                got_state[(h * dim + kk) * dim + vv] = state[(h * dim + vv) * dim + kk];
            }
        }
    }
    let (st_cos, st_rel) = cosine_rel_l2(&got_state, &ref_state);
    println!("recurrent_state_final (l0_15): cosine {st_cos:.6}  relL2 {st_rel:.5}");

    assert!(
        out_cos >= 0.9999 && out_rel <= 0.02,
        "core_attn_out off reference: cosine {out_cos} relL2 {out_rel}"
    );
    assert!(
        st_cos >= 0.9999 && st_rel <= 0.02,
        "final state off reference: cosine {st_cos} relL2 {st_rel}"
    );
}

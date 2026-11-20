//! Parity tests against HuggingFace reference fixtures.
//!
//! Everything is `#[ignore]`-gated and driven by env vars so the suite only
//! runs on the GPU host with the fixtures present:
//!
//! - `APXINF_QWEN3ASR_FIXTURES`: dir holding the npy reference fixtures.
//! - `APXINF_QWEN3ASR_CHECKPOINT`: converted checkpoint dir.
//!
//! There are no default paths: a run without both variables reports the
//! missing one instead of silently skipping.

use std::path::{Path, PathBuf};

use apxinf_core::{Error, Result, Shape, Tensor};
use apxinf_cuda_new::CudaBuffer;
use apxinf_cuda_new::CudaContext as Context;

use super::config::*;
use super::decoder::{self, Decoder, DecoderStages, KvCaches, PrefillWorkspace};
use super::encoder::{self, EncoderStages};
use super::frontend::Frontend;
use super::model::Qwen3AsrModel;
use super::weights::{DTYPE, Weights};
use crate::asr_trait::{AsrRequest, AsrTrait};

fn env_dir(variable: &str) -> PathBuf {
    std::env::var(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|_| panic!("{variable} must point at the reference data"))
}

fn fixtures_dir() -> PathBuf {
    env_dir("APXINF_QWEN3ASR_FIXTURES")
}

fn checkpoint_dir() -> PathBuf {
    env_dir("APXINF_QWEN3ASR_CHECKPOINT")
}

fn require(path: &Path) -> Result<()> {
    if path.exists() {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "fixture {} is missing; set APXINF_QWEN3ASR_FIXTURES",
            path.display()
        )))
    }
}

/// One npy file: shape, fortran-order flag, and f32-decoded data.
struct Npy {
    data: Vec<f32>,
    shape: Vec<usize>,
    fortran: bool,
}

impl Npy {
    fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        let magic: &[u8] = b"\x93NUMPY";
        if bytes.len() < 10 || &bytes[..6] != magic {
            return Err(Error::Other(format!("{}: not an npy file", path.display())));
        }
        let (header_len, data_start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            ),
            v => return Err(Error::Other(format!("{}: npy version {v}", path.display()))),
        };
        let header = std::str::from_utf8(&bytes[data_start..data_start + header_len])
            .map_err(|e| Error::Other(format!("{}: npy header utf8: {e}", path.display())))?;
        let descr = header
            .split("'descr':")
            .nth(1)
            .and_then(|rest| rest.split('\'').nth(1))
            .ok_or_else(|| Error::Other(format!("{}: descr missing", path.display())))?
            .to_owned();
        let fortran = header.contains("'fortran_order': True");
        let shape_text = header
            .split("'shape':")
            .nth(1)
            .and_then(|rest| rest.split(')').next())
            .unwrap_or("")
            .trim_start_matches([' ', '('])
            .to_owned();
        let shape: Vec<usize> = shape_text
            .split(',')
            .filter_map(|token| token.trim().parse::<usize>().ok())
            .collect();
        let body = &bytes[data_start + header_len..];
        let count: usize = shape.iter().product();
        let data: Vec<f32> = match descr.as_str() {
            "<f4" | "|f4" => body
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "<f2" | "|f2" => body
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            "<i8" | "|i8" => body
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect(),
            other => {
                return Err(Error::Other(format!(
                    "{}: unsupported descr {other}",
                    path.display()
                )))
            }
        };
        if data.len() != count {
            return Err(Error::Other(format!(
                "{}: {} elements, header says {count}",
                path.display(),
                data.len()
            )));
        }
        Ok(Self { data, shape, fortran })
    }

    /// Row-major element access with fortran-order decoding.
    fn at(&self, index: &[usize]) -> f32 {
        assert_eq!(index.len(), self.shape.len(), "npy index rank mismatch");
        let mut offset = 0usize;
        if self.fortran {
            // Column-major: axis strides are [1, d0, d0*d1, ...].
            let mut stride = 1usize;
            for (axis, &idx) in index.iter().enumerate() {
                offset += idx * stride;
                stride *= self.shape[axis];
            }
        } else {
            // Row-major: axis strides are [prod(d[1..]), ..., d[-1], 1].
            let mut stride = 1usize;
            for axis in (0..index.len()).rev() {
                offset += index[axis] * stride;
                stride *= self.shape[axis];
            }
        }
        self.data[offset]
    }
}

/// Raw f32 little-endian waveform file (480000 samples).
fn load_waveform(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_bf16(ctx: &Context, tensor: &Tensor) -> Result<Vec<f32>> {
    // A device-wide barrier, not just the context's own stream: the synchronous
    // cudaMemcpy runs on the legacy default stream, which has no implicit order
    // against this crate's non-blocking stream, so draining only that stream
    // lets the D2H read observe a buffer still being written (which showed up
    // as a few randomly all-zero rows). See the `transfers::synchronize_device`
    // doc comment.
    apxinf_cuda_new::transfers::synchronize_device()
        .map_err(|e| Error::Other(format!("device sync: {e}")))?;
    ctx.synchronize().map_err(|e| Error::Other(format!("sync: {e}")))?;
    let mut bytes = vec![0u8; tensor.numel() * 2];
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_to_host(&mut bytes)
        .map_err(|e| Error::Other(format!("readback: {e}")))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// Print diff stats and assert max abs error below `tol`.
fn assert_close(label: &str, ours: &[f32], reference: &[f32], tol: f32) {
    let (max_err, mean, max_at) = stats(ours, reference);
    assert_eq!(
        ours.len(),
        reference.len(),
        "{label}: element count {} vs {}",
        ours.len(),
        reference.len()
    );
    println!(
        "{label}: max |err| = {max_err:.4} at [{max_at}], mean |err| = {mean:.5} (tol {tol})"
    );
    assert!(
        max_err <= tol,
        "{label}: max |err| {max_err:.4} at [{max_at}] exceeds tol {tol}"
    );
}

/// (max abs err, mean abs err, argmax err index) without asserting.
fn stats(ours: &[f32], reference: &[f32]) -> (f32, f64, usize) {
    let mut max_err = 0.0f32;
    let mut sum_err = 0.0f64;
    let mut max_at = 0usize;
    for (i, (a, b)) in ours.iter().zip(reference).enumerate() {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
            max_at = i;
        }
        sum_err += err as f64;
    }
    (max_err, sum_err / ours.len().max(1) as f64, max_at)
}

/// 30 s speech fixture (`f32`, 16 kHz) inside the fixture directory.
fn waveform_path() -> PathBuf {
    let path = fixtures_dir().join("waveform_speech_30s.f32");
    require(&path).unwrap();
    path
}

struct Runtime {
    ctx: Context,
    weights: Weights,
    frontend: Frontend,
}

fn runtime() -> Runtime {
    let ctx = Context::new_with_autotune(0, false).map_err(Error::Cuda).unwrap();
    let mut weights = Weights::load(&checkpoint_dir()).unwrap();
    let frontend = Frontend::load(&mut weights, &ctx).unwrap();
    Runtime { ctx, weights, frontend }
}

#[test]
#[ignore]
fn qwen3_asr_mel_matches_hf() {
    let fixtures = fixtures_dir();
    let mel_npy = fixtures.join("mel_f16.npy");
    require(&mel_npy).unwrap();
    let reference = Npy::load(&mel_npy).unwrap();
    // HF layout [1, 128, 3000] (mels-major); ours is [3000, 128].
    assert_eq!(reference.shape, vec![1, MEL_BINS, FRAMES]);

    let rt = runtime();
    let waveform = load_waveform(&waveform_path()).unwrap();
    assert_eq!(waveform.len(), SAMPLES);
    let host = Tensor::from_f32(vec![SAMPLES], &waveform).unwrap();
    let wave_dev = apxinf_cuda_new::transfers::to_cuda(&host, rt.ctx.device_id()).unwrap();
    let valid = valid_frames(SAMPLES);
    let mel = rt.frontend.run(&rt.ctx, &wave_dev, valid).unwrap();

    let ours = read_bf16(&rt.ctx, &mel).unwrap();
    let mut reference_t = vec![0.0f32; FRAMES * MEL_BINS];
    for f in 0..FRAMES {
        for m in 0..MEL_BINS {
            reference_t[f * MEL_BINS + m] = reference.at(&[0, m, f]);
        }
    }
    assert_close("mel", &ours, &reference_t, 0.03);
}

#[test]
#[ignore]
fn qwen3_asr_pos_emb_readback() {
    // Isolated check: the device upload + readback of audio.pos_emb must
    // return the sinusoid table (row 0 = [0..0, 1..1]).
    let mut rt = runtime();
    let pos_emb = rt.weights.get(&rt.ctx, "audio.pos_emb").unwrap();
    assert_eq!(pos_emb.shape().dims(), vec![POST_CNN_TIME, ENC_DMODEL]);
    let pos = read_bf16(&rt.ctx, &pos_emb).unwrap();
    println!("pos row0 head: {:?}", &pos[..8]);
    println!("pos row0 tail: {:?}", &pos[ENC_DMODEL - 8..ENC_DMODEL]);
    for v in &pos[..ENC_DMODEL / 2] {
        assert!(*v == 0.0, "sin half of row 0 must be zero, got {v}");
    }
    for v in &pos[ENC_DMODEL / 2..ENC_DMODEL] {
        assert!(*v == 1.0, "cos half of row 0 must be one, got {v}");
    }
}

#[test]
#[ignore]
fn qwen3_asr_encoder_stages_match_hf() {
    let fixtures = fixtures_dir();
    for name in ["conv2d3_out.npy", "layer0_in.npy", "audio_embeds.npy", "projected.npy"] {
        require(&fixtures.join(name)).unwrap();
    }

    let mut rt = runtime();
    let waveform = load_waveform(&waveform_path()).unwrap();
    assert_eq!(waveform.len(), SAMPLES);
    let host = Tensor::from_f32(vec![SAMPLES], &waveform).unwrap();
    let wave_dev = apxinf_cuda_new::transfers::to_cuda(&host, rt.ctx.device_id()).unwrap();
    let valid = valid_frames(SAMPLES);
    let mel = rt.frontend.run(&rt.ctx, &wave_dev, valid).unwrap();

    let mut stages = EncoderStages {
        conv3_out: rt.ctx.allocate_output(Shape::new(vec![1]), super::weights::DTYPE).unwrap(),
        conv_out: rt.ctx.allocate_output(Shape::new(vec![1]), super::weights::DTYPE).unwrap(),
        packed: rt.ctx.allocate_output(Shape::new(vec![1]), super::weights::DTYPE).unwrap(),
        ln_post: rt.ctx.allocate_output(Shape::new(vec![1]), super::weights::DTYPE).unwrap(),
    };
    // encode now takes a persistent workspace (reused across calls when the
    // signature matches); the test passes None so it is built for
    // (valid, 390) on the spot.
    let mut ws = None;
    let encoded =
        encoder::encode(&rt.ctx, &mut rt.weights, &mut ws, &mel, valid, Some(&mut stages))
            .unwrap();
    assert_eq!(encoded.n, 390);

    // conv3: fixture [30, 13, 7680] uses HF columns d = c*16 + f; ours are
    // n = f*480 + c. Permute the fixture columns into our order.
    let conv_ref = Npy::load(&fixtures.join("conv2d3_out.npy")).unwrap();
    assert_eq!(
        conv_ref.shape,
        vec![N_CHUNKS, POST_CNN_TIME, CONV_CH * FREQ_BINS]
    );
    let ours_conv = read_bf16(&rt.ctx, &stages.conv3_out).unwrap();
    let mut conv_perm = vec![0.0f32; ours_conv.len()];
    for row in 0..N_CHUNKS * POST_CNN_TIME {
        for n in 0..CONV_CH * FREQ_BINS {
            let c = n % CONV_CH;
            let f = n / CONV_CH;
            let d = c * FREQ_BINS + f;
            conv_perm[row * CONV_CH * FREQ_BINS + n] =
                conv_ref.at(&[row / POST_CNN_TIME, row % POST_CNN_TIME, d]);
        }
    }
    assert_close("conv3", &ours_conv, &conv_perm, 0.25);

    // conv_out gemm: expected = layer0_in - sinusoids (t within chunk).
    let packed_ref = Npy::load(&fixtures.join("layer0_in.npy")).unwrap();
    assert_eq!(packed_ref.shape, vec![390, ENC_DMODEL]);
    let ours_conv_out = read_bf16(&rt.ctx, &stages.conv_out).unwrap();
    let mut conv_out_ref = vec![0.0f32; 390 * ENC_DMODEL];
    {
        let half = ENC_DMODEL / 2;
        let inc = (10000.0f64).ln() / (half - 1) as f64;
        let inv: Vec<f64> = (0..half).map(|i| (-inc * i as f64).exp()).collect();
        for row in 0..390 {
            let t = row % POST_CNN_TIME;
            for c in 0..ENC_DMODEL {
                let angle = t as f64 * inv[c % half];
                let pos = if c < half { angle.sin() } else { angle.cos() };
                conv_out_ref[row * ENC_DMODEL + c] =
                    packed_ref.data[row * ENC_DMODEL + c] - pos as f32;
            }
        }
    }
    assert_close("conv_out", &ours_conv_out, &conv_out_ref, 0.35);

    // packed input to layer 0.
    let ours_packed = read_bf16(&rt.ctx, &stages.packed).unwrap();
    {
        // Diagnostics: implied pos = ours_packed - ours_conv_out per row.
        let i = 142_687usize;
        let (row, col) = (i / ENC_DMODEL, i % ENC_DMODEL);
        let t = row % POST_CNN_TIME;
        let implied_pos: Vec<f32> = ours_packed
            .iter()
            .zip(ours_conv_out.iter())
            .map(|(p, c)| p - c)
            .collect();
        let half = ENC_DMODEL / 2;
        let inc = (10000.0f64).ln() / (half - 1) as f64;
        let angle = t as f64 * (-inc * (col % half) as f64).exp();
        let expected_pos =
            (if col < half { angle.sin() } else { angle.cos() }) as f32;
        println!(
            "diag[{i}] row={row} col={col} t={t}: packed={:.4} conv_out={:.4} ref={:.4} \
             implied_pos={:.4} expected_pos={:.4}",
            ours_packed[i], ours_conv_out[i], packed_ref.data[i], implied_pos[i], expected_pos
        );
        let big = implied_pos
            .iter()
            .zip(0..)
            .filter(|(v, _)| v.abs() > 1.5)
            .count();
        let pos_rows: Vec<f32> = implied_pos[..ENC_DMODEL].to_vec();
        println!(
            "implied_pos rows>1.5: {big}/390*1024; first row head: {:?}",
            &pos_rows[..8]
        );
        // implied pos of row r should equal pos[r % 13]: check per-row max dev.
        let mut worst = 0.0f32;
        let mut worst_row = 0usize;
        for r in 0..390 {
            let tr = r % POST_CNN_TIME;
            for c in 0..ENC_DMODEL {
                let a = tr as f64 * (-inc * (c % half) as f64).exp();
                let exp = (if c < half { a.sin() } else { a.cos() }) as f32;
                let dev = (implied_pos[r * ENC_DMODEL + c] - exp).abs();
                if dev > worst {
                    worst = dev;
                    worst_row = r;
                }
            }
        }
        println!("implied_pos worst dev {worst} at row {worst_row}");
    }
    assert_close("packed", &ours_packed, &packed_ref.data, 0.3);

    // ln_post output (== HF `audio_embeds` before the projector).
    let ln_ref = Npy::load(&fixtures.join("audio_embeds.npy")).unwrap();
    assert_eq!(ln_ref.shape, vec![390, ENC_DMODEL]);
    let ours_ln = read_bf16(&rt.ctx, &stages.ln_post).unwrap();
    assert_close("ln_post", &ours_ln, &ln_ref.data, 0.6);

    // projector output.
    let proj_ref = Npy::load(&fixtures.join("projected.npy")).unwrap();
    assert_eq!(proj_ref.shape, vec![390, OUTPUT_DIM]);
    let ours_proj = read_bf16(&rt.ctx, &encoded.projected).unwrap();
    assert_close("projected", &ours_proj, &proj_ref.data, 0.6);
}

#[test]
#[ignore]
fn qwen3_asr_prompt_matches_hf() {
    let fixtures = fixtures_dir();
    let prompt_ref = Npy::load(&fixtures.join("input_ids.npy")).unwrap();
    assert_eq!(prompt_ref.shape, vec![408]);
    let reference: Vec<u32> = prompt_ref.data.iter().map(|&v| v as u32).collect();

    let mut ours = prompt_prefix();
    ours.extend((0..390).map(|_| AUDIO_TOKEN));
    ours.extend(prompt_suffix("Chinese").unwrap());
    assert_eq!(ours, reference, "hard-coded prompt template diverges from HF");
}

#[test]
#[ignore]
fn qwen3_asr_decoder_prefill_match_hf() {
    let fixtures = fixtures_dir();
    for name in [
        "input_ids.npy",
        "projected.npy",
        "dec_in.npy",
        "dec_layer0.npy",
        "dec_layer27.npy",
        "dec_final.npy",
    ] {
        require(&fixtures.join(name)).unwrap();
    }
    let ids_npy = Npy::load(&fixtures.join("input_ids.npy")).unwrap();
    assert_eq!(ids_npy.shape, vec![408]);
    let ids: Vec<i32> = ids_npy.data.iter().map(|&v| v as i32).collect();
    let proj_npy = Npy::load(&fixtures.join("projected.npy")).unwrap();
    assert_eq!(proj_npy.shape, vec![390, OUTPUT_DIM]);

    let mut rt = runtime();
    let mut caches = KvCaches::alloc(&rt.ctx).unwrap();
    let decoder = Decoder::new();

    // Embed + splice the projected audio, then compare with HF `dec_in`.
    // prefill intermediates go into the persistent PrefillWorkspace, built
    // for T=408 on the spot.
    let mut ws = PrefillWorkspace::alloc(&rt.ctx, ids.len()).unwrap();
    decoder.embed_into(&rt.ctx, &mut rt.weights, &ids, &mut ws).unwrap();
    {
        let data: Vec<half::bf16> = proj_npy
            .data
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        let host = Tensor::from_bf16(vec![390, OUTPUT_DIM], &data).unwrap();
        let proj = apxinf_cuda_new::transfers::to_cuda_on(&rt.ctx, &host).unwrap();
        let dest = decoder::view_rows(&ws.hidden[0], 9, 390, OUTPUT_DIM).unwrap();
        apxinf_cuda_new::transfers::copy_cuda_to_cuda(&rt.ctx, &proj, &dest).unwrap();
    }
    let dec_in_ref = Npy::load(&fixtures.join("dec_in.npy")).unwrap();
    assert_eq!(dec_in_ref.shape, vec![408, TXT_HIDDEN]);
    let ours_in = read_bf16(&rt.ctx, &ws.hidden[0]).unwrap();
    assert_close("dec_in", &ours_in, &dec_in_ref.data, 0.05);

    let dummy = |rt: &Runtime| rt.ctx.allocate_output(Shape::new(vec![1]), DTYPE).unwrap();
    let mut stages = DecoderStages {
        layer0: dummy(&rt),
        layer27: dummy(&rt),
        final_norm: dummy(&rt),
        normed0: dummy(&rt),
        q0: dummy(&rt),
        k0: dummy(&rt),
        v0: dummy(&rt),
        qrot0: dummy(&rt),
        krot0: dummy(&rt),
        attn0: dummy(&rt),
        o_proj0: dummy(&rt),
        normed2_0: dummy(&rt),
        gate_up0: dummy(&rt),
        inner0: dummy(&rt),
    };
    // positions (0..408) are written into ws.positions by the layers_staged
    // entry point.
    decoder
        .layers_staged(
            &rt.ctx,
            &mut rt.weights,
            &mut caches,
            &mut ws,
            0,
            true,
            Some(&mut stages),
        )
        .unwrap();

    // Layer-0 internals: report everything, then assert, so one run shows
    // which sub-op diverges.
    let load = |name: &str| Npy::load(&fixtures.join(name)).unwrap();
    let read = |t: &Tensor| read_bf16(&rt.ctx, t).unwrap();
    let mut checks: Vec<(&str, Vec<f32>, Vec<f32>, f32)> = Vec::new();
    checks.push(("l0_normed", read(&stages.normed0), load("l0_normed.npy").data, 0.1));
    checks.push(("l0_q", read(&stages.q0), load("l0_q.npy").data, 0.1));
    checks.push(("l0_k", read(&stages.k0), load("l0_k.npy").data, 0.1));
    checks.push(("l0_v", read(&stages.v0), load("l0_v.npy").data, 0.1));
    // bf16 rounding on large-magnitude outliers lands 1-2 ulp apart; mean err
    // is ~5e-3, so 0.5 only fails on genuine divergence.
    checks.push(("l0_qrot", read(&stages.qrot0), load("l0_qrot.npy").data, 0.5));
    checks.push(("l0_krot", read(&stages.krot0), load("l0_krot.npy").data, 2.5));
    // o_proj output == HF self_attn output (the module returns post-o values).
    checks.push(("l0_o_proj", read(&stages.o_proj0), load("l0_attn_post_o.npy").data, 0.3));
    checks.push(("l0_normed2", read(&stages.normed2_0), load("l0_normed2.npy").data, 0.3));
    {
        // fused gate_up: first 6144 cols = gate, last 6144 = up.
        let ours = read(&stages.gate_up0);
        let gate_ref = load("l0_gate.npy").data;
        let up_ref = load("l0_up.npy").data;
        assert_eq!(ours.len(), 408 * TXT_INTER * 2);
        let mut gate = Vec::with_capacity(408 * TXT_INTER);
        let mut up = Vec::with_capacity(408 * TXT_INTER);
        for row in 0..408 {
            gate.extend_from_slice(&ours[row * 2 * TXT_INTER..row * 2 * TXT_INTER + TXT_INTER]);
            up.extend_from_slice(&ours[row * 2 * TXT_INTER + TXT_INTER..(row + 1) * 2 * TXT_INTER]);
        }
        checks.push(("l0_gate", gate, gate_ref, 0.3));
        checks.push(("l0_up", up, up_ref, 0.3));
    }
    checks.push(("l0_inner", read(&stages.inner0), {
        // reference inner = silu(gate) * up
        let gate = load("l0_gate.npy").data;
        let up = load("l0_up.npy").data;
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| g * (1.0 + (-g).exp()).recip() * u)
            .collect()
    }, 0.3));
    checks.push(("dec_layer0", read(&stages.layer0), load("l0_out.npy").data, 0.5));
    // Layers 1..27 accumulate bf16 chaos (mean err grows), so these are
    // observation-only loose tolerances; correctness is judged by argmax.
    checks.push(("dec_layer27", read(&stages.layer27), load("dec_layer27.npy").data, 1200.0));
    checks.push(("dec_final", read(&stages.final_norm), load("dec_final.npy").data, 400.0));

    for (label, ours, reference, tol) in &checks {
        let (max_err, mean, max_at) = stats(ours, reference);
        println!("{label}: max |err| = {max_err:.4} at [{max_at}], mean = {mean:.5} (tol {tol})");
    }
    {
        // Drill the qrot/krot outliers: where and how many.
        for (label, tensor, ref_name) in [
            ("qrot", &stages.qrot0, "l0_qrot.npy"),
            ("krot", &stages.krot0, "l0_krot.npy"),
        ] {
            let ours = read_bf16(&rt.ctx, tensor).unwrap();
            let reference = Npy::load(&fixtures.join(ref_name)).unwrap();
            let big: Vec<usize> = ours
                .iter()
                .zip(reference.data.iter())
                .enumerate()
                .filter(|(_, (a, b))| (**a - **b).abs() > 0.5)
                .map(|(i, _)| i)
                .collect();
            println!(
                "{label}: {} elems > 0.5; rows: {:?}",
                big.len(),
                big.iter().take(12).map(|&i| i / reference.shape[1]).collect::<Vec<_>>()
            );
            if let Some(&i) = big.first() {
                let cols = reference.shape[1];
                let (row, col) = (i / cols, i % cols);
                println!(
                    "{label} worst [{i}] row={row} col={col}: ours={} ref={}",
                    ours[i], reference.data[i]
                );
                let lo = col.saturating_sub(3);
                println!(
                    "{label} ours row{row} [{lo}..{}]: {:?}",
                    col + 4,
                    &ours[row * cols + lo..row * cols + col + 4]
                );
                println!(
                    "{label} ref  row{row} [{lo}..{}]: {:?}",
                    col + 4,
                    &reference.data[row * cols + lo..row * cols + col + 4]
                );
            }
        }
    }
    for (label, ours, reference, tol) in checks {
        assert_close(&label, &ours, &reference, tol);
    }

    // The last row's logits and argmax also go to workspace buffers
    // (logits_argmax_into).
    let next = decoder
        .logits_argmax_into(
            &rt.ctx,
            &mut rt.weights,
            &stages.final_norm,
            407,
            &mut ws.logits,
            &ws.index,
        )
        .unwrap();
    println!("prefill argmax: {next} (hf first token 40820)");
    assert_eq!(next, 40820, "prefill greedy token diverges from HF");
}

#[test]
#[ignore]
fn qwen3_asr_e2e_tokens_match_hf() {
    let fixtures = fixtures_dir();
    // HF `generate()` output: the 48 generated ids, prompt excluded.
    let ids_ref = Npy::load(&fixtures.join("generated_ids.npy")).unwrap();
    assert_eq!(ids_ref.shape, vec![48]);
    let hf_tail: Vec<u32> = ids_ref.data.iter().map(|&v| v as u32).collect();

    let waveform = load_waveform(&waveform_path()).unwrap();
    assert_eq!(waveform.len(), SAMPLES);

    let model = Qwen3AsrModel::load(&checkpoint_dir(), 0, false).unwrap();
    let request = AsrRequest {
        waveform: &waveform,
        language: Some("Chinese"),
        task: None,
        prompt_token_ids: &[],
        max_new_tokens: MAX_NEW_TOKENS,
    };
    let result = model.transcribe(&request).unwrap();

    println!(
        "ours ({}): {:?}",
        result.token_ids.len(),
        &result.token_ids[..result.token_ids.len().min(60)]
    );
    println!(
        "hf   ({}): {:?}",
        hf_tail.len(),
        &hf_tail[..hf_tail.len().min(60)]
    );
    let matches = result
        .token_ids
        .iter()
        .zip(hf_tail.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "token prefix agreement: {}/{}",
        matches,
        hf_tail.len().min(result.token_ids.len())
    );
    assert_eq!(
        result.token_ids, hf_tail,
        "generated ids diverge from the HF reference after {matches} tokens"
    );
}

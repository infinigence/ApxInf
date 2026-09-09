//! Device-resident Qwen3-MoE AutoAWQ weights and the checkpoint packer.
//!
//! Weights keep the on-disk AutoAWQ `gemm` packing (`qweight` `[K, N/8]` i32,
//! `qzeros` `[K/G, N/8]` i32, `scales` `[K/G, N]` f16); the kernels in
//! `apxinf-cuda` understand the nibble interleave, so no repacking pass runs
//! on the GPU. Two host-side concatenations are made while loading:
//!
//! * `q_proj | k_proj | v_proj` along the output dimension (one GEMM/GEMV per
//!   layer instead of three);
//! * per expert `gate_proj | up_proj` along the output dimension, so a single
//!   projection feeds the fused SiLU·mul.
//!
//! All 128 experts of a layer live in one allocation per tensor kind so that
//! routing can address an expert by index on the device.
//!
//! FP16 auxiliary tensors (norms, router, embedding, `lm_head`) are converted
//! to BF16 on the host because every activation kernel in this runtime is
//! BF16.

use std::path::Path;

use apxinf_core::{Error, Result};
use apxinf_loader::{SafetensorsArchive, TensorBytes};
use half::{bf16, f16};

use super::config::Qwen3MoeConfig;
use crate::accelerator::cuda::DeviceBuffer;

/// One AutoAWQ linear (or a stack of `experts` linears) on the device.
pub struct AwqLinear {
    pub qweight: DeviceBuffer,
    pub qzeros: DeviceBuffer,
    pub scales: DeviceBuffer,
    pub in_dim: usize,
    pub out_dim: usize,
    pub experts: usize,
}

impl AwqLinear {
    pub fn packed_cols(&self) -> usize {
        self.out_dim / 8
    }

    /// `i32` words per expert in `qweight`.
    pub fn stride_q(&self) -> usize {
        self.in_dim * self.packed_cols()
    }

    /// `i32` words per expert in `qzeros`.
    pub fn stride_z(&self, group_size: usize) -> usize {
        self.in_dim / group_size * self.packed_cols()
    }

    /// `f16` elements per expert in `scales`.
    pub fn stride_s(&self, group_size: usize) -> usize {
        self.in_dim / group_size * self.out_dim
    }
}

pub struct Qwen3MoeLayerWeights {
    pub attn_norm: DeviceBuffer, // [hidden], dtype selected by rms_weights_f16
    pub ffn_norm: DeviceBuffer,  // [hidden], dtype selected by rms_weights_f16
    pub q_norm: DeviceBuffer,    // bf16 [head_dim]
    pub k_norm: DeviceBuffer,    // bf16 [head_dim]
    /// `[hidden] -> [q | k | v]`
    pub qkv: AwqLinear,
    /// `[n_heads * head_dim] -> [hidden]`
    pub o: AwqLinear,
    /// Router weight transposed to bf16 `[hidden, experts]` (GEMM `B` operand).
    pub router: DeviceBuffer,
    /// Per expert `[hidden] -> [gate | up]` (`experts` stacked).
    pub gate_up: AwqLinear,
    /// Per expert `[inter] -> [hidden]` (`experts` stacked).
    pub down: AwqLinear,
}

pub struct Qwen3MoeWeights {
    /// bf16 `[vocab, hidden]`
    pub embed_tokens: DeviceBuffer,
    /// `[hidden]`, dtype selected by `rms_weights_f16`.
    pub final_norm: DeviceBuffer,
    /// Preserve checkpoint FP16 RMS scale values; activations remain BF16.
    pub rms_weights_f16: bool,
    /// bf16 `[vocab, hidden]` (row-major, consumed with a transposed GEMM).
    pub lm_head: DeviceBuffer,
    pub layers: Vec<Qwen3MoeLayerWeights>,
    pub device_bytes: usize,
}

fn cuda(error: String) -> Error {
    Error::Cuda(error)
}

fn other(message: String) -> Error {
    Error::Other(message)
}

/// Where a tensor is allocated.
///
/// Thor's `cudaMalloc` pool is a GPU carveout much smaller than system RAM
/// (14.4 GiB of 57.7 GiB on Thor-U), and this checkpoint needs ~16.7 GiB of
/// weights, so some of it has to live in pinned host memory mapped into the
/// device address space. Both are the same physical LPDDR5X; the mapped path
/// measured 230 GB/s against 259 GB/s for the carveout, so the policy spends
/// the carveout on what decode reads every single token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// `cudaMalloc` — the fast, scarce pool.
    Carveout,
    /// Pinned host memory mapped for the device — plentiful, ~11% slower.
    Mapped,
}

/// Hands out [`Placement`] decisions against a byte budget.
///
/// Tensors are requested in priority order: whatever is asked for first gets
/// the carveout until the budget runs out, and everything after it spills.
struct Arena {
    device: usize,
    carveout_left: usize,
    carveout_used: usize,
    mapped_used: usize,
}

impl Arena {
    /// `reserve` is memory the runtime still needs after loading (KV cache and
    /// workspaces), held back from the weight budget.
    fn new(device: usize, reserve: usize) -> Result<Self> {
        let (free, total) = DeviceBuffer::pool_memory(device).map_err(cuda)?;
        // Leave a margin for allocator fragmentation and the CUDA context's
        // own bookkeeping; running the carveout to zero fails in cuBLAS later
        // rather than here, where the error would be legible.
        let margin = 512 << 20;
        let mut budget = free.saturating_sub(reserve + margin);
        // Diagnostic knob: capping the carveout pushes more weights into
        // mapped host memory, which is how you measure what that placement
        // actually costs. Decode reads roughly 2 GiB of weights per token, so
        // if mapped pages were as fast as the carveout for this access
        // pattern, sweeping this would leave throughput flat.
        if let Ok(cap) = std::env::var("APXINF_QWEN3MOE_CARVEOUT_GIB") {
            match cap.trim().parse::<f64>() {
                Ok(gib) if gib >= 0.0 => {
                    let capped = (gib * (1u64 << 30) as f64) as usize;
                    eprintln!(
                        "[apxinf] qwen3moe: APXINF_QWEN3MOE_CARVEOUT_GIB caps the weight \
                         budget at {gib:.2} GiB (was {:.2} GiB)",
                        self::gib(budget)
                    );
                    budget = budget.min(capped);
                }
                _ => eprintln!(
                    "[apxinf] qwen3moe: ignoring APXINF_QWEN3MOE_CARVEOUT_GIB=`{cap}`, \
                     expected a number of GiB"
                ),
            }
        }
        eprintln!(
            "[apxinf] qwen3moe: CUDA pool {:.2} GiB free of {:.2} GiB; \
             reserving {:.2} GiB for runtime, {:.2} GiB budget for weights",
            gib(free),
            gib(total),
            gib(reserve),
            gib(budget)
        );
        Ok(Self {
            device,
            carveout_left: budget,
            carveout_used: 0,
            mapped_used: 0,
        })
    }

    fn place(&mut self, bytes: usize) -> Placement {
        if bytes <= self.carveout_left {
            Placement::Carveout
        } else {
            Placement::Mapped
        }
    }

    fn upload(&mut self, bytes: &[u8]) -> Result<DeviceBuffer> {
        let len = bytes.len().max(1);
        let placement = self.place(len);
        let buffer = match placement {
            Placement::Carveout => {
                self.carveout_left -= len;
                self.carveout_used += len;
                DeviceBuffer::alloc(len, self.device)
            }
            Placement::Mapped => {
                self.mapped_used += len;
                DeviceBuffer::alloc_mapped(len, self.device)
            }
        }
        .map_err(|error| {
            cuda(format!(
                "{error} (allocating {:.1} MiB as {placement:?}; {})",
                len as f64 / (1u64 << 20) as f64,
                host_memory_note()
            ))
        })?;
        buffer.copy_from_host(bytes).map_err(cuda)?;
        Ok(buffer)
    }
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Unified memory means a failed `cudaMalloc` is usually a host-memory story;
/// quote the kernel's own numbers so the error explains itself.
fn host_memory_note() -> String {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| -> Option<u64> {
        meminfo
            .lines()
            .find(|line| line.starts_with(name))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()
    };
    match (field("MemTotal:"), field("MemAvailable:")) {
        (Some(total), Some(available)) => format!(
            "host memory {:.1} GiB available of {:.1} GiB",
            available as f64 / (1u64 << 20) as f64,
            total as f64 / (1u64 << 20) as f64
        ),
        _ => "host memory unknown".to_string(),
    }
}

fn f16_bytes_to_bf16(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for chunk in bytes.chunks_exact(2) {
        let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        out.extend_from_slice(&bf16::from_f32(value).to_le_bytes());
    }
    out
}

/// Upload an F16 (or BF16) checkpoint tensor as BF16 after validating its shape.
fn upload_bf16(
    archive: &SafetensorsArchive,
    arena: &mut Arena,
    name: &str,
    shape: &[usize],
) -> Result<DeviceBuffer> {
    let view = archive.get(name).map_err(other)?;
    if view.shape != shape {
        return Err(other(format!(
            "qwen3moe: `{name}` has shape {:?}, expected {shape:?}",
            view.shape
        )));
    }
    match view.dtype {
        "F16" => arena.upload(&f16_bytes_to_bf16(view.bytes)),
        "BF16" => arena.upload(view.bytes),
        dtype => Err(other(format!(
            "qwen3moe: `{name}` has dtype {dtype}, expected F16 or BF16"
        ))),
    }
}

/// Preserve the quantized checkpoint's calibrated FP16 RMS scale values.
fn upload_rms_weight(
    archive: &SafetensorsArchive,
    arena: &mut Arena,
    name: &str,
    shape: &[usize],
    preserve_f16: bool,
) -> Result<DeviceBuffer> {
    if preserve_f16 {
        let value = archive.get_checked(name, "F16", shape).map_err(other)?;
        arena.upload(value.bytes)
    } else {
        upload_bf16(archive, arena, name, shape)
    }
}

/// Convert a row-major F16 `[rows, cols]` tensor to bf16 `[cols, rows]`.
fn transpose_f16_to_bf16(view: TensorBytes<'_>, rows: usize, cols: usize) -> Result<Vec<u8>> {
    if view.dtype != "F16" || view.shape != [rows, cols] {
        return Err(other(format!(
            "qwen3moe: expected F16 [{rows}, {cols}], got {} {:?}",
            view.dtype, view.shape
        )));
    }
    let src: Vec<f16> = view
        .bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut out = vec![0u8; rows * cols * 2];
    for r in 0..rows {
        for c in 0..cols {
            let value = bf16::from_f32(src[r * cols + c].to_f32()).to_le_bytes();
            let at = (c * rows + r) * 2;
            out[at] = value[0];
            out[at + 1] = value[1];
        }
    }
    Ok(out)
}

/// The three AutoAWQ tensors of one checkpoint linear, borrowed from the archive.
struct AwqSource<'a> {
    qweight: TensorBytes<'a>,
    qzeros: TensorBytes<'a>,
    scales: TensorBytes<'a>,
    in_dim: usize,
    out_dim: usize,
}

fn awq_source<'a>(
    archive: &'a SafetensorsArchive,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
) -> Result<AwqSource<'a>> {
    let groups = in_dim / group_size;
    let qweight = archive
        .get_checked(&format!("{prefix}.qweight"), "I32", &[in_dim, out_dim / 8])
        .map_err(other)?;
    let qzeros = archive
        .get_checked(&format!("{prefix}.qzeros"), "I32", &[groups, out_dim / 8])
        .map_err(other)?;
    let scales = archive
        .get_checked(&format!("{prefix}.scales"), "F16", &[groups, out_dim])
        .map_err(other)?;
    Ok(AwqSource {
        qweight,
        qzeros,
        scales,
        in_dim,
        out_dim,
    })
}

/// Host staging area for one (possibly stacked) AWQ linear built by
/// concatenating several checkpoint linears along the output dimension.
struct AwqStaging {
    qweight: Vec<u8>,
    qzeros: Vec<u8>,
    scales: Vec<u8>,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    experts: usize,
}

impl AwqStaging {
    fn new(in_dim: usize, out_dim: usize, group_size: usize, experts: usize) -> Self {
        let groups = in_dim / group_size;
        Self {
            qweight: Vec::with_capacity(experts * in_dim * out_dim / 8 * 4),
            qzeros: Vec::with_capacity(experts * groups * out_dim / 8 * 4),
            scales: Vec::with_capacity(experts * groups * out_dim * 2),
            in_dim,
            out_dim,
            group_size,
            experts,
        }
    }

    /// Append one expert made of `parts` concatenated along the output dim.
    fn push_expert(&mut self, parts: &[AwqSource<'_>]) -> Result<()> {
        let total_out: usize = parts.iter().map(|p| p.out_dim).sum();
        if total_out != self.out_dim || parts.iter().any(|p| p.in_dim != self.in_dim) {
            return Err(other(format!(
                "qwen3moe: AWQ concat mismatch, parts sum to {total_out} of {} outputs",
                self.out_dim
            )));
        }
        let groups = self.in_dim / self.group_size;
        // qweight rows: [K][sum of N_i/8 words]
        for k in 0..self.in_dim {
            for part in parts {
                let words = part.out_dim / 8;
                let start = k * words * 4;
                self.qweight
                    .extend_from_slice(&part.qweight.bytes[start..start + words * 4]);
            }
        }
        for g in 0..groups {
            for part in parts {
                let words = part.out_dim / 8;
                let start = g * words * 4;
                self.qzeros
                    .extend_from_slice(&part.qzeros.bytes[start..start + words * 4]);
            }
        }
        for g in 0..groups {
            for part in parts {
                let start = g * part.out_dim * 2;
                self.scales
                    .extend_from_slice(&part.scales.bytes[start..start + part.out_dim * 2]);
            }
        }
        Ok(())
    }

    fn upload(self, arena: &mut Arena) -> Result<AwqLinear> {
        let groups = self.in_dim / self.group_size;
        let expect_q = self.experts * self.in_dim * self.out_dim / 8 * 4;
        let expect_z = self.experts * groups * self.out_dim / 8 * 4;
        let expect_s = self.experts * groups * self.out_dim * 2;
        if self.qweight.len() != expect_q
            || self.qzeros.len() != expect_z
            || self.scales.len() != expect_s
        {
            return Err(other(format!(
                "qwen3moe: staged AWQ sizes {}/{}/{} != expected {expect_q}/{expect_z}/{expect_s}",
                self.qweight.len(),
                self.qzeros.len(),
                self.scales.len()
            )));
        }
        Ok(AwqLinear {
            qweight: arena.upload(&self.qweight)?,
            qzeros: arena.upload(&self.qzeros)?,
            scales: arena.upload(&self.scales)?,
            in_dim: self.in_dim,
            out_dim: self.out_dim,
            experts: self.experts,
        })
    }
}

/// Attention-side weights of one layer; loaded before the experts because
/// decode reads every byte of them on every token.
struct DenseLayer {
    attn_norm: DeviceBuffer,
    ffn_norm: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    qkv: AwqLinear,
    o: AwqLinear,
    router: DeviceBuffer,
}

impl Qwen3MoeWeights {
    /// Load and pack an AutoAWQ checkpoint directory onto CUDA device `device`.
    ///
    /// `runtime_reserve` is how many bytes of the `cudaMalloc` pool the caller
    /// still needs after loading (KV cache and workspaces); it is held back
    /// from the weight budget so the spill lands on weights rather than on the
    /// allocations the runtime cannot place anywhere else.
    pub fn load(
        config: &Qwen3MoeConfig,
        model_dir: &Path,
        device: usize,
        runtime_reserve: usize,
    ) -> Result<Self> {
        let archive = SafetensorsArchive::open(model_dir).map_err(other)?;
        Self::from_archive(config, &archive, device, runtime_reserve)
    }

    /// Weights are uploaded in decreasing order of bytes-read-per-token, so
    /// that when the carveout runs out the spill hits the coldest tensors:
    ///
    /// 1. `lm_head` — the whole matrix is read for every token.
    /// 2. per-layer norms, `qkv`, `o`, `router` — likewise, all dense.
    /// 3. the experts — only 8 of 128 are touched per token, so a spilled
    ///    expert costs 1/16th of what a spilled dense projection would.
    /// 4. `embed_tokens` — one row per token; the coldest tensor by far.
    pub fn from_archive(
        config: &Qwen3MoeConfig,
        archive: &SafetensorsArchive,
        device: usize,
        runtime_reserve: usize,
    ) -> Result<Self> {
        let rms_weights_f16 =
            std::env::var("APXINF_QWEN3MOE_F16_RMS_WEIGHTS").as_deref() == Ok("1");
        let hidden = config.hidden_size;
        let group = config.group_size();
        let inter = config.moe_intermediate_size;
        let experts = config.num_experts;
        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.kv_dim();
        let arena = &mut Arena::new(device, runtime_reserve)?;
        let tied = config.tie_word_embeddings || !archive.contains("lm_head.weight");

        // Whichever tensor produces the logits is read in full on every token,
        // so it goes first. With tied weights that is the embedding table.
        let hot_head = upload_bf16(
            archive,
            arena,
            if tied {
                "model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            },
            &[config.vocab_size, hidden],
        )?;
        let final_norm = upload_rms_weight(
            archive,
            arena,
            "model.norm.weight",
            &[hidden],
            rms_weights_f16,
        )?;

        let mut dense = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_norm = upload_rms_weight(
                archive,
                arena,
                &format!("{p}.input_layernorm.weight"),
                &[hidden],
                rms_weights_f16,
            )?;
            let ffn_norm = upload_rms_weight(
                archive,
                arena,
                &format!("{p}.post_attention_layernorm.weight"),
                &[hidden],
                rms_weights_f16,
            )?;
            let q_norm = upload_bf16(
                archive,
                arena,
                &format!("{p}.self_attn.q_norm.weight"),
                &[config.head_dim],
            )?;
            let k_norm = upload_bf16(
                archive,
                arena,
                &format!("{p}.self_attn.k_norm.weight"),
                &[config.head_dim],
            )?;

            let mut qkv = AwqStaging::new(hidden, q_dim + 2 * kv_dim, group, 1);
            qkv.push_expert(&[
                awq_source(
                    archive,
                    &format!("{p}.self_attn.q_proj"),
                    hidden,
                    q_dim,
                    group,
                )?,
                awq_source(
                    archive,
                    &format!("{p}.self_attn.k_proj"),
                    hidden,
                    kv_dim,
                    group,
                )?,
                awq_source(
                    archive,
                    &format!("{p}.self_attn.v_proj"),
                    hidden,
                    kv_dim,
                    group,
                )?,
            ])?;
            let qkv = qkv.upload(arena)?;

            let mut o = AwqStaging::new(q_dim, hidden, group, 1);
            o.push_expert(&[awq_source(
                archive,
                &format!("{p}.self_attn.o_proj"),
                q_dim,
                hidden,
                group,
            )?])?;
            let o = o.upload(arena)?;

            let router_view = archive
                .get(&format!("{p}.mlp.gate.weight"))
                .map_err(other)?;
            let router = arena.upload(&transpose_f16_to_bf16(router_view, experts, hidden)?)?;

            dense.push(DenseLayer {
                attn_norm,
                ffn_norm,
                q_norm,
                k_norm,
                qkv,
                o,
                router,
            });
        }
        eprintln!(
            "[apxinf] qwen3moe: dense weights packed ({:.2} GiB carveout used)",
            gib(arena.carveout_used)
        );

        let mut layers = Vec::with_capacity(config.n_layers);
        for (layer, dense) in dense.into_iter().enumerate() {
            let p = format!("model.layers.{layer}");
            let mut gate_up = AwqStaging::new(hidden, 2 * inter, group, experts);
            let mut down = AwqStaging::new(inter, hidden, group, experts);
            for e in 0..experts {
                let ep = format!("{p}.mlp.experts.{e}");
                gate_up.push_expert(&[
                    awq_source(archive, &format!("{ep}.gate_proj"), hidden, inter, group)?,
                    awq_source(archive, &format!("{ep}.up_proj"), hidden, inter, group)?,
                ])?;
                down.push_expert(&[awq_source(
                    archive,
                    &format!("{ep}.down_proj"),
                    inter,
                    hidden,
                    group,
                )?])?;
            }
            let gate_up = gate_up.upload(arena)?;
            let down = down.upload(arena)?;

            if layer % 8 == 0 || layer + 1 == config.n_layers {
                eprintln!(
                    "[apxinf] qwen3moe: experts {}/{} packed ({:.2} GiB carveout, \
                     {:.2} GiB mapped; {})",
                    layer + 1,
                    config.n_layers,
                    gib(arena.carveout_used),
                    gib(arena.mapped_used),
                    host_memory_note()
                );
            }

            layers.push(Qwen3MoeLayerWeights {
                attn_norm: dense.attn_norm,
                ffn_norm: dense.ffn_norm,
                q_norm: dense.q_norm,
                k_norm: dense.k_norm,
                qkv: dense.qkv,
                o: dense.o,
                router: dense.router,
                gate_up,
                down,
            });
        }

        let (embed_tokens, lm_head) = if tied {
            (hot_head.clone(), hot_head)
        } else {
            let embed = upload_bf16(
                archive,
                arena,
                "model.embed_tokens.weight",
                &[config.vocab_size, hidden],
            )?;
            (embed, hot_head)
        };

        let device_bytes = arena.carveout_used + arena.mapped_used;
        eprintln!(
            "[apxinf] qwen3moe: {:.2} GiB of weights ({:.2} GiB in the CUDA pool, \
             {:.2} GiB in mapped host memory)",
            gib(device_bytes),
            gib(arena.carveout_used),
            gib(arena.mapped_used)
        );

        Ok(Self {
            embed_tokens,
            final_norm,
            rms_weights_f16,
            lm_head,
            layers,
            device_bytes,
        })
    }
}

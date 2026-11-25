//! The Qwen3-VL 32B language stack of the text encoder (layers 0-49, INT8 ConvRot).
//!
//! The 22.7 GiB of layer weights do not fit next to the DiT on a 24 GiB card, so they stay in the checkpoint file and every cold
//! prompt streams them through the device. The file is memory-mapped once at load and, when the host can afford it, page-locked
//! (`cudaHostRegister`), so a layer reaches the GPU by DMA straight from the page cache: no read into the heap, no staging copy and
//! no repack, because the checkpoint's `[N, K]` bytes are the layout the W8A8 kernels execute. Two device-resident sets of
//! fixed-address slots form a ring: a dedicated copy stream fills one set with layer `i + 1` while the compute stream runs layer
//! `i`, ordered by events, so the prompt costs about one pass over the PCIe link.
//!
//! The embedding table never moves: only the rows of the prompt are read from the map.
use super::bf16::{check_convrot_meta, Int8Linear};
use super::host::{available_host_memory, Checkpoint};
use crate::cuda_minimax_h3::backend::ops::OutputMajorSlot;
use crate::cuda_minimax_h3::backend::{
    CudaBuffer, CudaContext, CudaEvent, CudaStream, PinnedRegion,
};
use crate::cuda_minimax_h3::config::*;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_loader::MappedFile;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

pub(in crate::cuda_minimax_h3) struct TextLayer {
    pub input_norm: Tensor,
    pub post_norm: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub q: Int8Linear,
    pub k: Int8Linear,
    pub v: Int8Linear,
    pub o: Int8Linear,
    pub gate: Int8Linear,
    pub up: Int8Linear,
    pub down: Int8Linear,
}

/// `(name, n, k)` of the seven INT8 linears of a layer, in file order.
const LINEARS: [(&str, usize, usize); 7] = [
    ("self_attn.q_proj", TE_Q, TE_DIM),
    ("self_attn.k_proj", TE_KV, TE_DIM),
    ("self_attn.v_proj", TE_KV, TE_DIM),
    ("self_attn.o_proj", TE_DIM, TE_Q),
    ("mlp.gate_proj", TE_FFN, TE_DIM),
    ("mlp.up_proj", TE_FFN, TE_DIM),
    ("mlp.down_proj", TE_DIM, TE_FFN),
];

/// `(name, len)` of the four BF16 norm vectors of a layer.
const NORMS: [(&str, usize); 4] = [
    ("input_layernorm", TE_DIM),
    ("post_attention_layernorm", TE_DIM),
    ("self_attn.q_norm", TE_HEAD_DIM),
    ("self_attn.k_norm", TE_HEAD_DIM),
];

/// Layers of device storage. Two let the copy of the next layer run while the current one computes.
const RING: usize = 2;

/// Host memory the system must keep available beyond the file itself before the file is page-locked.
const PIN_HEADROOM: u64 = 8 << 30;

/// Byte range of one tensor inside the mapped file.
#[derive(Clone, Copy, Debug)]
struct At {
    start: usize,
    len: usize,
}

#[derive(Clone, Copy, Debug)]
struct LinearAt {
    weight: At,
    scales: At,
}

#[derive(Clone, Copy, Debug)]
struct LayerAt {
    norms: [At; 4],
    linears: [LinearAt; 7],
}

/// How the file is exposed to the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cuda_minimax_h3) enum Pinning {
    /// Page-lock when the host has memory to spare, otherwise stream through the pageable path.
    Auto,
    /// Never page-lock.
    Off,
    /// Page-lock or fail the load.
    Require,
}

impl Pinning {
    /// `APXINF_H3_TE_PIN`: `0`/`off`, `1`/`on` (required), anything else or unset: automatic.
    pub fn from_env() -> Self {
        match std::env::var("APXINF_H3_TE_PIN").as_deref() {
            Ok("0" | "off") => Self::Off,
            Ok("1" | "on") => Self::Require,
            _ => Self::Auto,
        }
    }
}

/// One device-resident set of layer slots (fixed addresses, refilled for every layer that maps to it).
struct LayerSet {
    layer: TextLayer,
    slots: Vec<OutputMajorSlot>,
}

fn bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn device_tensor(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

impl LayerSet {
    fn new(ctx: &CudaContext) -> Result<Self> {
        let mut slots = Vec::with_capacity(LINEARS.len());
        let mut linears = Vec::with_capacity(LINEARS.len());
        for (_, n, k) in LINEARS {
            let slot = OutputMajorSlot::new(ctx, n, k)?;
            linears.push(Int8Linear {
                weight: slot.weight.clone(),
                layout: slot.layout.clone(),
                channel_scales: device_tensor(ctx, vec![n], DType::F32)?,
                k,
                n,
            });
            slots.push(slot);
        }
        let mut linears = linears.into_iter();
        let mut next = || linears.next().expect("seven linears");
        let layer = TextLayer {
            input_norm: device_tensor(ctx, vec![TE_DIM], DType::BF16)?,
            post_norm: device_tensor(ctx, vec![TE_DIM], DType::BF16)?,
            q_norm: device_tensor(ctx, vec![TE_HEAD_DIM], DType::BF16)?,
            k_norm: device_tensor(ctx, vec![TE_HEAD_DIM], DType::BF16)?,
            q: next(),
            k: next(),
            v: next(),
            o: next(),
            gate: next(),
            up: next(),
            down: next(),
        };
        Ok(Self { layer, slots })
    }

    fn norms(&self) -> [&Tensor; 4] {
        let l = &self.layer;
        [&l.input_norm, &l.post_norm, &l.q_norm, &l.k_norm]
    }

    fn linears(&self) -> [&Int8Linear; 7] {
        let l = &self.layer;
        [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down]
    }
}

/// The text encoder's weights: the mapped checkpoint and the device ring they stream through.
pub(in crate::cuda_minimax_h3) struct TextWeights {
    ctx: Rc<CudaContext>,
    layers: Vec<LayerAt>,
    embed: At,
    sets: Vec<LayerSet>,
    copy: CudaStream,
    /// Recorded on the copy stream when a set is full, and on the compute stream when a set may be refilled.
    copied: Vec<CudaEvent>,
    consumed: Vec<CudaEvent>,
    // Dropped last: waits for the device, then unregisters.
    region: PinnedRegion,
}

impl TextWeights {
    /// Verify the file's whole layout (metadata, the embedding table, exactly the tensors of layers 0-49 and the vision tower,
    /// which is ignored because edit mode is not supported), map it, and allocate the device ring.
    pub fn open(path: &Path, ctx: Rc<CudaContext>, pinning: Pinning) -> Result<Self> {
        let mut file = Checkpoint::open(path)?;
        check_layout(&file, path)?;
        verify_convrot(&mut file)?;
        let (layers, embed) = locate(&file)?;
        drop(file);

        let mapped = MappedFile::open(path).map_err(Error::Other)?;
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(mapped);
        let region = pin(owner, ctx.device_id(), pinning, path)?;

        let sets = (0..RING)
            .map(|_| LayerSet::new(&ctx))
            .collect::<Result<Vec<_>>>()?;
        let copy = CudaStream::new_on(ctx.device_id()).map_err(Error::Cuda)?;
        let events = || {
            (0..RING)
                .map(|_| CudaEvent::new(ctx.device_id()).map_err(Error::Cuda))
                .collect::<Result<Vec<_>>>()
        };
        let (copied, consumed) = (events()?, events()?);
        Ok(Self {
            ctx,
            layers,
            embed,
            sets,
            copy,
            copied,
            consumed,
            region,
        })
    }

    /// Whether layer copies are DMA transfers.
    pub fn is_pinned(&self) -> bool {
        self.region.is_pinned()
    }

    /// BF16 embedding rows of `ids`, row-major `[ids.len(), TE_DIM]`.
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<half::bf16>> {
        let bytes = self.region.bytes();
        let mut out = Vec::with_capacity(ids.len() * TE_DIM);
        for &id in ids {
            if id as usize >= TE_VOCAB {
                return Err(Error::Other(format!(
                    "token id {id} is outside the {TE_VOCAB}-entry vocabulary"
                )));
            }
            let start = self.embed.start + id as usize * TE_DIM * 2;
            out.extend(bf16_vec(&bytes[start..start + TE_DIM * 2]));
        }
        Ok(out)
    }

    /// Queue everything layer `layer` needs on the copy stream, after the compute that last read its set.
    fn enqueue(&self, layer: usize) -> Result<()> {
        let set = layer % RING;
        let (at, target) = (&self.layers[layer], &self.sets[set]);
        self.copy
            .wait_for(&self.consumed[set])
            .map_err(Error::Cuda)?;
        for ((slot, linear), at) in target.slots.iter().zip(target.linears()).zip(&at.linears) {
            slot.fill(&self.region, at.weight.start, &self.copy)?;
            self.copy_into(&linear.channel_scales, at.scales)?;
        }
        for (tensor, at) in target.norms().into_iter().zip(&at.norms) {
            self.copy_into(tensor, *at)?;
        }
        self.copy.record(&self.copied[set]).map_err(Error::Cuda)
    }

    fn copy_into(&self, tensor: &Tensor, at: At) -> Result<()> {
        CudaBuffer::from_tensor(tensor)
            .map_err(Error::Cuda)?
            .copy_from_region_async(0, &self.region, at.start, at.len, &self.copy)
            .map_err(Error::Cuda)
    }

    /// Run `layer_fn(index, layer)` for layers 0-49 in order. Each call must enqueue all of the layer's work on the context
    /// stream before returning (the slots are refilled as soon as the compute stream passes that point), and may not keep
    /// the layer past the call.
    pub fn run_layers(
        &self,
        mut layer_fn: impl FnMut(usize, &TextLayer) -> Result<()>,
    ) -> Result<()> {
        let result = self.pass(&mut layer_fn);
        // an abandoned pass must not leave copies in flight into slots the next request refills
        let drained = self.copy.synchronize().map_err(Error::Cuda);
        result.and(drained)
    }

    fn pass(&self, layer_fn: &mut impl FnMut(usize, &TextLayer) -> Result<()>) -> Result<()> {
        self.enqueue(0)?;
        for index in 0..TE_LAYERS {
            if index + 1 < TE_LAYERS {
                // queued before this layer's compute so the DMA of the next layer overlaps it
                self.enqueue(index + 1)?;
            }
            let set = index % RING;
            self.ctx
                .stream()
                .wait_for(&self.copied[set])
                .map_err(Error::Cuda)?;
            layer_fn(index, &self.sets[set].layer)?;
            self.ctx
                .stream()
                .record(&self.consumed[set])
                .map_err(Error::Cuda)?;
        }
        Ok(())
    }
}

/// Page-lock the mapping according to `pinning`; a refusal in automatic mode falls back to the pageable path, loudly.
fn pin(
    owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
    device: usize,
    pinning: Pinning,
    path: &Path,
) -> Result<PinnedRegion> {
    let len = (*owner).as_ref().len() as u64;
    let attempt = match pinning {
        Pinning::Off => return Ok(PinnedRegion::unregistered(owner, device)),
        Pinning::Require => true,
        Pinning::Auto => available_host_memory().is_none_or(|free| free >= len + PIN_HEADROOM),
    };
    if !attempt {
        eprintln!(
            "MiniMax-H3: not page-locking {} ({:.1} GiB): less than {:.0} GiB of host memory would remain; \
             text layers stream through the pageable path, which is about 3x slower per cold prompt",
            path.display(),
            len as f64 / 1073741824.0,
            PIN_HEADROOM as f64 / 1073741824.0
        );
        return Ok(PinnedRegion::unregistered(owner, device));
    }
    match PinnedRegion::register(Arc::clone(&owner), device) {
        Ok(region) => Ok(region),
        Err(e) if pinning == Pinning::Require => Err(Error::Other(format!(
            "{}: page-locking the text encoder file failed: {e}",
            path.display()
        ))),
        Err(e) => {
            eprintln!(
                "MiniMax-H3: page-locking {} failed ({e}); text layers stream through the pageable path, which is about 3x slower per cold prompt",
                path.display()
            );
            Ok(PinnedRegion::unregistered(owner, device))
        }
    }
}

fn check_layout(file: &Checkpoint, path: &Path) -> Result<()> {
    let meta = file.metadata("minimax_h3_te").ok_or_else(|| {
        Error::Other(format!(
            "{}: missing minimax_h3_te metadata",
            path.display()
        ))
    })?;
    let parsed: serde_json::Value = serde_json::from_str(meta)
        .map_err(|e| Error::Other(format!("{}: minimax_h3_te metadata: {e}", path.display())))?;
    if parsed["num_hidden_layers"] != TE_LAYERS
        || parsed["output"] != "unnormalized_hidden_after_layer_50"
    {
        return Err(Error::Other(format!(
            "{}: expected the 50-layer unnormalized MiniMax-H3 text encoder, got {meta}",
            path.display()
        )));
    }
    let mut expected: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    expected.insert("model.embed_tokens.weight".into());
    for layer in 0..TE_LAYERS {
        let p = format!("model.layers.{layer}");
        for (norm, _) in NORMS {
            expected.insert(format!("{p}.{norm}.weight"));
        }
        for (name, _, _) in LINEARS {
            for part in ["weight", "weight_scale", "comfy_quant"] {
                expected.insert(format!("{p}.{name}.{part}"));
            }
        }
    }
    let present: std::collections::BTreeSet<String> = file
        .names()
        .into_iter()
        .filter(|n| !n.starts_with("visual."))
        .collect();
    if present != expected {
        let missing: Vec<_> = expected.difference(&present).take(4).collect();
        let extra: Vec<_> = present.difference(&expected).take(4).collect();
        return Err(Error::Other(format!(
            "{}: text encoder layout differs from the supported 50-layer file (missing {missing:?}, unexpected {extra:?})",
            path.display()
        )));
    }
    Ok(())
}

/// Byte ranges of every tensor the stream moves, with dtype and shape checked.
fn locate(file: &Checkpoint) -> Result<(Vec<LayerAt>, At)> {
    let at = |(start, len): (usize, usize)| At { start, len };
    let embed = at(file.locate(
        "model.embed_tokens.weight",
        Some(DType::BF16),
        &[TE_VOCAB, TE_DIM],
    )?);
    let mut layers = Vec::with_capacity(TE_LAYERS);
    for layer in 0..TE_LAYERS {
        let p = format!("model.layers.{layer}");
        let mut norms = [At { start: 0, len: 0 }; 4];
        for (slot, (name, len)) in norms.iter_mut().zip(NORMS) {
            *slot = at(file.locate(&format!("{p}.{name}.weight"), Some(DType::BF16), &[len])?);
        }
        let mut linears = [LinearAt {
            weight: norms[0],
            scales: norms[0],
        }; 7];
        for (slot, (name, n, k)) in linears.iter_mut().zip(LINEARS) {
            *slot = LinearAt {
                weight: at(file.locate(&format!("{p}.{name}.weight"), Some(DType::I8), &[n, k])?),
                scales: at(file.locate(
                    &format!("{p}.{name}.weight_scale"),
                    Some(DType::F32),
                    &[n, 1],
                )?),
            };
        }
        layers.push(LayerAt { norms, linears });
    }
    Ok((layers, embed))
}

/// Every linear must be the supported ConvRot INT8 format; checked once at load, not per request.
fn verify_convrot(file: &mut Checkpoint) -> Result<()> {
    for layer in 0..TE_LAYERS {
        for (name, _, _) in LINEARS {
            let prefix = format!("model.layers.{layer}.{name}");
            let meta = file.read_u8(&format!("{prefix}.comfy_quant"))?;
            check_convrot_meta(&prefix, &meta)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use std::path::PathBuf;

    /// `APXINF_H3_TE=/path/qwen3vl_32b_minimax_h3_int8_convrot.safetensors cargo test --release -p apxinf-model \
    ///  --features cuda --lib -- --ignored cuda_minimax_h3::weights::text`
    #[test]
    #[ignore = "requires the Comfy-Org text-encoder checkpoint (27 GB) and a CUDA GPU"]
    fn streamed_layers_equal_the_file_bytes_and_ids_are_checked() {
        let path = PathBuf::from(std::env::var("APXINF_H3_TE").expect("APXINF_H3_TE"));
        let ctx = Rc::new(CudaContext::new(0).unwrap());
        let weights = TextWeights::open(&path, ctx.clone(), Pinning::from_env()).unwrap();
        let rows = weights.embed(&[32, 2518, 13]).unwrap();
        assert_eq!(rows.len(), 3 * TE_DIM);
        assert!(rows.iter().all(|v| v.to_f32().is_finite()));
        assert!(
            weights.embed(&[TE_VOCAB as u32]).is_err(),
            "ids beyond the vocabulary are rejected"
        );

        let mut seen = 0;
        let started = std::time::Instant::now();
        weights
            .run_layers(|index, layer| {
                assert_eq!(index, seen, "layers arrive in order");
                seen += 1;
                ctx.synchronize().map_err(Error::Cuda)?;
                let at = &weights.layers[index];
                let bytes = weights.region.bytes();
                let direct = weights.sets[index % RING].slots[0].is_direct();
                for (linear, at) in [
                    &layer.q,
                    &layer.k,
                    &layer.v,
                    &layer.o,
                    &layer.gate,
                    &layer.up,
                    &layer.down,
                ]
                .into_iter()
                .zip(&at.linears)
                {
                    let mut scales = vec![0u8; at.scales.len];
                    CudaBuffer::from_tensor(&linear.channel_scales)
                        .unwrap()
                        .copy_to_host(&mut scales)
                        .unwrap();
                    assert_eq!(
                        scales,
                        &bytes[at.scales.start..at.scales.start + at.scales.len],
                        "scales of layer {index}"
                    );
                    // the large weights are compared in full for the small projections and on every layer's q and down
                    if direct && matches!(linear.n, TE_Q | TE_KV | TE_DIM) {
                        let mut got = vec![0u8; at.weight.len];
                        CudaBuffer::from_tensor(&linear.weight)
                            .unwrap()
                            .copy_to_host(&mut got)
                            .unwrap();
                        assert_eq!(
                            got,
                            &bytes[at.weight.start..at.weight.start + at.weight.len],
                            "weight of layer {index}"
                        );
                    }
                }
                for (tensor, at) in [
                    &layer.input_norm,
                    &layer.post_norm,
                    &layer.q_norm,
                    &layer.k_norm,
                ]
                .into_iter()
                .zip(&at.norms)
                {
                    let mut got = vec![0u8; at.len];
                    CudaBuffer::from_tensor(tensor)
                        .unwrap()
                        .copy_to_host(&mut got)
                        .unwrap();
                    assert_eq!(
                        got,
                        &bytes[at.start..at.start + at.len],
                        "norm of layer {index}"
                    );
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, TE_LAYERS);
        eprintln!(
            "{TE_LAYERS} layers streamed and verified in {:.2} s (pinned: {})",
            started.elapsed().as_secs_f64(),
            weights.is_pinned()
        );
    }
}

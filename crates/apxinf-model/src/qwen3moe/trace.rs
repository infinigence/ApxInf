//! Opt-in layer diagnostics for comparisons with the independent FP32 reference.
//! Synchronous host reads require eager execution; never use this for timing.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::Mutex;

use crate::accelerator::cuda::{Context, DeviceBuffer};
use apxinf_core::{Error, Result};

pub(super) struct LayerTrace(Mutex<BufWriter<File>>);

pub(super) struct LayerBuffers<'a> {
    pub residual: &'a DeviceBuffer,
    pub ffn_normed: &'a DeviceBuffer,
    pub router_logits: &'a DeviceBuffer,
    pub topk_idx: &'a DeviceBuffer,
    pub topk_weight: &'a DeviceBuffer,
}

impl LayerTrace {
    pub fn from_env() -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("APXINF_QWEN3MOE_LAYER_TRACE") else {
            return Ok(None);
        };
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| Error::Other(format!("create layer trace {path:?}: {e}")))?;
        eprintln!("[apxinf] qwen3moe: layer trace enables synchronous eager diagnostics; timings are invalid");
        Ok(Some(Self(Mutex::new(BufWriter::new(file)))))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        ctx: &Context,
        phase: &str,
        position: usize,
        layer: usize,
        row: usize,
        hidden: usize,
        experts: usize,
        topk: usize,
        buffers: LayerBuffers<'_>,
    ) -> Result<()> {
        ctx.synchronize().map_err(Error::Cuda)?;
        let read = |buffer: &DeviceBuffer, width: usize, item: usize| -> Result<Vec<u8>> {
            let mut bytes = vec![0; width * item];
            buffer
                .view(row * width * item, bytes.len())
                .map_err(Error::Cuda)?
                .copy_to_host(&mut bytes)
                .map_err(Error::Cuda)?;
            Ok(bytes)
        };
        let bf16 = |buffer: &DeviceBuffer, width: usize| -> Result<Vec<f32>> {
            Ok(read(buffer, width, 2)?
                .chunks_exact(2)
                .map(|v| f32::from_bits((u16::from_ne_bytes([v[0], v[1]]) as u32) << 16))
                .collect())
        };
        let ids: Vec<i32> = read(buffers.topk_idx, topk, 4)?
            .chunks_exact(4)
            .map(|v| i32::from_ne_bytes(v.try_into().unwrap()))
            .collect();
        let weights: Vec<f32> = read(buffers.topk_weight, topk, 4)?
            .chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
            .collect();
        let record = serde_json::json!({
            "phase": phase, "position": position, "layer": layer,
            "residual": bf16(buffers.residual, hidden)?,
            "ffn_normed": bf16(buffers.ffn_normed, hidden)?,
            "router_logits": bf16(buffers.router_logits, experts)?,
            "router_topk": ids, "router_weights": weights,
        });
        let mut file = self
            .0
            .lock()
            .map_err(|_| Error::Other("layer trace lock poisoned".into()))?;
        serde_json::to_writer(&mut *file, &record)
            .map_err(|e| Error::Other(format!("write layer trace: {e}")))?;
        file.write_all(b"\n")
            .and_then(|_| file.flush())
            .map_err(|e| Error::Other(format!("flush layer trace: {e}")))
    }
}

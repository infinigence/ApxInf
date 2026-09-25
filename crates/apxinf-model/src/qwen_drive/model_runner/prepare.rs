//! Existing opt-in per-GDN-layer graphs. This is not full-model preparation.
use crate::qwen_drive::backend::{kernels, Context, DeviceBuffer};
use crate::qwen_drive::model::{GdnExecution, GdnRequest};
use apxinf_core::{Backend, DType, Error, Result, Shape, Tensor};
use std::sync::Arc;
const DECODE_GRAPH_ARENA_BYTES: usize = 64 * 1024 * 1024;
const DECODE_GRAPH_WARMUP_STEPS: usize = 2;
fn persistent_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let elements: usize = shape.iter().product();
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen_drive: persistent tensor size overflow".into()))?;
    DeviceBuffer::alloc(bytes.max(1), ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}
fn device_copy(ctx: &Context, destination: &Tensor, source: &Tensor, bytes: usize) -> Result<()> {
    let dst = DeviceBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    let src = DeviceBuffer::from_tensor(source).map_err(Error::Cuda)?;
    dst.copy_from_device_async(&src, bytes, ctx.stream())
        .map_err(Error::Cuda)
}
fn decode_graph_layers() -> Option<Option<usize>> {
    static SETTING: std::sync::OnceLock<Option<Option<usize>>> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| match std::env::var("APXINF_QWEN_DECODE_GRAPH") {
        Err(_) => None,
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(layer) => Some(Some(layer)),
            Err(_) => Some(None),
        },
    })
}
fn decode_graph_enabled_for(layer_idx: usize) -> bool {
    match decode_graph_layers() {
        None => false,
        Some(None) => true,
        Some(Some(only)) => only == layer_idx,
    }
}
#[derive(Default)]
pub(super) struct GdnGraphs {
    /// Captured GDN decode graphs, by layer and by the conv double-buffer
    /// parity the capture baked in.
    gdn_graphs: Vec<[Option<Box<dyn apxinf_core::Graph>>; 2]>,
    /// Whether one eager pass has already run under the arena for this layer
    /// and parity. cuBLASLt plans bind buffer addresses when they are prepared,
    /// and `may_prepare_native_resources` reports false once a workspace is
    /// bound unless the pass is a declared preflight, so a capture taken before
    /// that preflight records kernels set up against the driver-allocated
    /// buffers the eager steps used.
    gdn_graph_prepared: Vec<[bool; 2]>,
    /// Where a captured layer leaves its result: a workspace view, valid only
    /// until the arena is reused, so it is copied out after every replay.
    gdn_graph_out: Vec<[Option<Tensor>; 2]>,
    /// Fixed-address input a captured layer reads, and the per-layer landing
    /// buffer a replayed result is copied into.
    gdn_graph_in: Option<Tensor>,
    gdn_graph_result: Vec<Option<Tensor>>,
    /// Stable-address arena the captured bodies allocate from.
    gdn_graph_workspace: Option<kernels::GraphWorkspace>,
    #[cfg(test)]
    force_decode_graph: bool,
}
impl GdnGraphs {
    fn enabled_for(&self, layer: usize) -> bool {
        #[cfg(test)]
        if self.force_decode_graph {
            return true;
        }
        decode_graph_enabled_for(layer)
    }

    fn forward_gdn_captured(
        &mut self,
        request: &GdnRequest<'_>,
        eager: &mut dyn FnMut(Tensor, Option<usize>) -> Result<Tensor>,
        x: Tensor,
        layer_idx: usize,
        parity: usize,
    ) -> Result<(Tensor, bool)> {
        let cuda = Arc::clone(request.backend);
        let ctx = cuda.context();
        let dims = x.shape().dims().to_vec();
        let bytes = x.numel() * DType::BF16.size_in_bytes();

        if self.gdn_graphs.len() != request.layer_count {
            self.gdn_graphs = (0..request.layer_count).map(|_| [None, None]).collect();
            self.gdn_graph_out = (0..request.layer_count).map(|_| [None, None]).collect();
            self.gdn_graph_result = vec![None; request.layer_count];
            self.gdn_graph_prepared = vec![[false, false]; request.layer_count];
        }
        if self.gdn_graph_workspace.is_none() {
            self.gdn_graph_workspace = Some(kernels::GraphWorkspace::new(
                DECODE_GRAPH_ARENA_BYTES,
                ctx.device_id(),
            )?);
        }
        // Diagnostic arm: take the arena and then run the body exactly as the
        // eager path does, with no workspace bound, no staging, no capture. The
        // only difference from a clean run is that a 64MB block now exists.
        // Every hypothesis that blames what the arena path *does* predicts this
        // is correct; a layout hypothesis -- some kernel writing past the end of
        // a buffer whose neighbour this allocation changed -- predicts it is
        // wrong, and that would explain a full-attention layer going bad while
        // the GDN layer that uses the arena stays exact.
        if std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
            .map(|value| value.contains("alloc"))
            .unwrap_or(false)
        {
            return eager(x, None).map(|x| (x, false));
        }

        if self.gdn_graph_in.is_none() {
            self.gdn_graph_in = Some(persistent_tensor(ctx, &dims, DType::BF16)?);
        }
        if self.gdn_graph_result[layer_idx].is_none() {
            self.gdn_graph_result[layer_idx] = Some(persistent_tensor(ctx, &dims, DType::BF16)?);
        }

        // The capture reads this address, so every step stages its input here.
        let staged = self.gdn_graph_in.clone().expect("staging input");
        device_copy(ctx, &staged, &x, bytes)?;

        // Diagnostic arm: run the body inside the arena with no capture and no
        // replay. If this is already wrong, the fault is in the workspace
        // allocation path rather than in anything CUDA graphs do.
        if std::env::var_os("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY").is_some() {
            // `raw` drops the staging copies as well, which separates the arena
            // itself from the copies in and out of it.
            let raw = std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
                .map(|value| value.contains("raw"))
                .unwrap_or(false);
            let input = if raw { x.clone() } else { staged.clone() };
            let prepare = std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
                .map(|value| value.contains("prepare"))
                .unwrap_or(false);
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let produced = if prepare {
                kernels::prepare_with_workspace(&workspace, || eager(input, None))
            } else {
                kernels::with_workspace(&workspace, || eager(input, None))
            };
            self.gdn_graph_workspace = Some(workspace);
            let output = produced?;
            if raw {
                return Ok((output, false));
            }
            let landing = self.gdn_graph_result[layer_idx]
                .clone()
                .expect("landing buffer");
            device_copy(ctx, &landing, &output, bytes)?;
            return Ok((landing, false));
        }

        if !self.gdn_graph_prepared[layer_idx][parity] {
            // Declared preflight: executes, so its output is this step's real
            // result and the recurrent and conv state advance exactly once.
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let prepared =
                kernels::prepare_with_workspace(&workspace, || eager(staged.clone(), None));
            self.gdn_graph_workspace = Some(workspace);
            let output = prepared?;
            cuda.synchronize()?;
            self.gdn_graph_prepared[layer_idx][parity] = true;
            let landing = self.gdn_graph_result[layer_idx]
                .clone()
                .expect("landing buffer");
            device_copy(ctx, &landing, &output, bytes)?;
            return Ok((landing, false));
        }

        if self.gdn_graphs[layer_idx][parity].is_none() {
            cuda.synchronize()?;
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let captured = cuda.capture_graph(|| {
                kernels::with_workspace(&workspace, || eager(staged.clone(), Some(parity)))
            });
            self.gdn_graph_workspace = Some(workspace);
            let (graph, output) = captured?;
            self.gdn_graph_out[layer_idx][parity] = Some(output);
            self.gdn_graphs[layer_idx][parity] = Some(graph);
        }

        self.gdn_graphs[layer_idx][parity]
            .as_ref()
            .expect("captured graph")
            .replay()?;

        // The captured output lives in the arena, which the next captured layer
        // reuses, so it is copied into this layer's own buffer before returning.
        let produced = self.gdn_graph_out[layer_idx][parity]
            .clone()
            .expect("captured output");
        let landing = self.gdn_graph_result[layer_idx]
            .clone()
            .expect("landing buffer");
        device_copy(ctx, &landing, &produced, bytes)?;
        Ok((landing, true))
    }
}
impl GdnExecution for GdnGraphs {
    fn run(
        &mut self,
        request: &GdnRequest<'_>,
        input: Tensor,
        eager: &mut dyn FnMut(Tensor, Option<usize>) -> Result<Tensor>,
    ) -> Result<(Tensor, bool)> {
        if !request.decode
            || request.decode_step < DECODE_GRAPH_WARMUP_STEPS
            || !self.enabled_for(request.layer)
        {
            return eager(input, None).map(|output| (output, false));
        }
        self.forward_gdn_captured(request, eager, input, request.layer, request.parity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen_drive::backend::{transfers, RuntimeBackend};
    use half::bf16;

    #[test]
    fn gdn_graphs_run_binds_parity_and_advances_state_once() -> Result<()> {
        let backend = Arc::new(RuntimeBackend::new(0)?);
        let input = Tensor::from_bf16(vec![1, 4], &[bf16::from_f32(1.0); 4])?;
        let input = transfers::to_cuda(&input, backend.device_id())?;
        let sources = [2.0f32, 3.0]
            .map(|value| {
                let source = Tensor::from_bf16(vec![1, 4], &[bf16::from_f32(value); 4])?;
                transfers::to_cuda(&source, backend.device_id())
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let outputs = [
            persistent_tensor(backend.context(), &[1, 4], DType::BF16)?,
            persistent_tensor(backend.context(), &[1, 4], DType::BF16)?,
        ];
        let mut graphs = GdnGraphs {
            force_decode_graph: true,
            ..GdnGraphs::default()
        };
        let mut flip = false;
        let mut body_calls = Vec::new();

        // parity-0 preflight, parity-1 preflight, capture+replay of both
        // parities, then replay both already-captured graphs.
        for (step, expected) in [2.0f32, 3.0, 2.0, 3.0, 2.0, 3.0]
            .into_iter()
            .enumerate()
        {
            let request = GdnRequest {
                backend: &backend,
                layer: 0,
                layer_count: 1,
                parity: usize::from(flip),
                decode: true,
                decode_step: DECODE_GRAPH_WARMUP_STEPS + step,
            };
            let (output, replayed) = graphs.run(&request, input.clone(), &mut |x, explicit| {
                body_calls.push(explicit);
                let parity = explicit.unwrap_or(usize::from(flip));
                if explicit.is_none() {
                    flip = !flip;
                }
                device_copy(
                    backend.context(),
                    &outputs[parity],
                    &sources[parity],
                    x.size_in_bytes(),
                )?;
                Ok(outputs[parity].clone())
            })?;
            if replayed {
                flip = !flip;
            }

            backend.synchronize()?;
            let output = transfers::to_cpu(&output)?.to_f32_vec()?;
            assert_eq!(output, vec![expected; 4]);
            assert_eq!(flip, step % 2 == 0);
        }

        assert_eq!(body_calls, vec![None, None, Some(0), Some(1)]);
        assert!(graphs.gdn_graphs[0][0].is_some());
        assert!(graphs.gdn_graphs[0][1].is_some());
        Ok(())
    }
}

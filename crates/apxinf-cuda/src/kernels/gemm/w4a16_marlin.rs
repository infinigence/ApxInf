//! Narrow grouped AutoAWQ/BF16 Marlin contract. Repacking is load-time only.
use super::super::contracts::{check_cuda, require_buffers};
use super::w4a16::AwqWeightView;
use crate::{buffer::CudaBuffer, context::CudaContext, ffi};
use apxinf_core::{Error, Result};

fn error(message: &str) -> Error {
    Error::Other(format!("Marlin: {message}"))
}
fn alloc(device: usize, bytes: usize) -> Result<CudaBuffer> {
    CudaBuffer::alloc_zeros(bytes.max(16), device).map_err(Error::Cuda)
}
fn bytes(rows: usize, cols: usize, width: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| error("size overflow"))
}

/// Owns a repacked copy; original AWQ storage remains available to decode.
/// FP16 scales are retained; GEMM reduction order still requires numerical acceptance.
pub struct Weights {
    q: CudaBuffer,
    z: CudaBuffer,
    s: CudaBuffer,
    k: usize,
    n: usize,
    experts: usize,
    fused_silu: bool,
}
impl Weights {
    pub fn repack(ctx: &CudaContext, source: AwqWeightView<'_>, mapped: bool) -> Result<Self> {
        Self::repack_with_silu(ctx, source, mapped, false)
    }
    /// Pair gate/up columns and emit rounded SwiGLU directly from FP16 GEMM.
    pub fn repack_with_silu(
        ctx: &CudaContext,
        source: AwqWeightView<'_>,
        mapped: bool,
        fused_silu: bool,
    ) -> Result<Self> {
        source.validate(ctx, "Marlin repack")?;
        if ctx.caps().compute_major < 8
            || source.group_size != 128
            || source.in_dim <= 128
            || source.out_dim % 256 != 0
            || source.in_dim > i32::MAX as usize
            || source.out_dim > i32::MAX as usize
            || source.experts > i32::MAX as usize
            || (source.experts > 1
                && (source.stride_q != source.qweight_words()
                    || source.stride_z != source.qzeros_words()
                    || source.stride_s != source.scales_elements()))
        {
            return Err(error(
                "requires SM80+, contiguous experts, group 128, K>128 and N divisible by 256",
            ));
        }
        check_cuda(unsafe { ffi::apxinf_marlin_prepare() })?;
        let allocate = |size| {
            if mapped {
                CudaBuffer::alloc_mapped(size, ctx.device_id())
            } else {
                CudaBuffer::alloc(size, ctx.device_id())
            }
            .map_err(Error::Cuda)
        };
        let result = Self {
            q: allocate(bytes(source.experts, source.qweight_words(), 4)?)?,
            z: allocate(bytes(source.experts, source.qzeros_words(), 4)?)?,
            s: allocate(bytes(source.experts, source.scales_elements(), 2)?)?,
            k: source.in_dim,
            n: source.out_dim,
            experts: source.experts,
            fused_silu,
        };
        // One expert's transient scratch is reused in stream order at load time.
        let scratch = if fused_silu {
            Some(alloc(
                ctx.device_id(),
                bytes(1, source.qweight_words(), 4)?,
            )?)
        } else {
            None
        };
        check_cuda(unsafe {
            ffi::apxinf_marlin_repack(
                source.qweight.ptr(),
                source.qzeros.ptr(),
                source.scales.ptr(),
                result.q.ptr(),
                result.z.ptr(),
                result.s.ptr(),
                result.k as i32,
                result.n as i32,
                result.experts as i32,
                i32::from(fused_silu),
                scratch
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |buffer| buffer.ptr()),
                ctx.stream().handle(),
            )
        })?;
        if scratch.is_some() {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        Ok(result)
    }
}

/// Fixed scratch and bounded routing metadata; no allocation in a GEMM launch.
pub struct Workspace {
    ids: CudaBuffer,
    experts: CudaBuffer,
    count: CudaBuffer,
    route_counts: CudaBuffer,
    route_offsets: CudaBuffer,
    temporary: CudaBuffer,
    locks: CudaBuffer,
    max_rows: usize,
    max_experts: usize,
    blocks: usize,
    tile_m: usize,
    use_f16: bool,
    variant: i32,
}
impl Workspace {
    /// Allocated routing, split-reduction, and lock storage (excluding weights).
    pub fn bytes(&self) -> usize {
        [
            &self.ids,
            &self.experts,
            &self.count,
            &self.route_counts,
            &self.route_offsets,
            &self.temporary,
            &self.locks,
        ]
        .into_iter()
        .map(CudaBuffer::len)
        .sum()
    }

    pub fn new(
        ctx: &CudaContext,
        max_rows: usize,
        max_experts: usize,
        tile_m: usize,
        use_f16: bool,
    ) -> Result<Self> {
        let variant = match std::env::var("APXINF_QWEN3MOE_MARLIN_TUNE").as_deref() {
            Ok("n128") => 1,
            Ok("n256") => 2,
            Ok("constants") => 3,
            _ => 0,
        };
        let tile_m = if matches!(variant, 1 | 2) { 32 } else { tile_m };
        if variant == 3 && tile_m != 64 {
            return Err(error("constant AWQ options require tile size 64"));
        }
        if variant > 0 && !use_f16 {
            return Err(error("tuned tiles require FP16 compute"));
        }
        if max_rows == 0 || max_experts == 0 || ![32, 64].contains(&tile_m) {
            return Err(error("invalid workspace capacity or tile size"));
        }
        let slots = max_rows
            .checked_add(
                max_experts
                    .checked_mul(tile_m)
                    .ok_or_else(|| error("size overflow"))?,
            )
            .ok_or_else(|| error("size overflow"))?;
        if slots > i32::MAX as usize {
            return Err(error("schedule too large"));
        }
        let blocks = ctx.caps().multiprocessor_count as usize
            * match variant {
                1 => 6,
                2 => 3,
                _ => 2,
            };
        Ok(Self {
            ids: alloc(ctx.device_id(), slots * 4)?,
            experts: alloc(ctx.device_id(), slots.div_ceil(tile_m) * 4)?,
            count: alloc(ctx.device_id(), 4)?,
            route_counts: alloc(ctx.device_id(), max_experts * 4)?,
            route_offsets: alloc(ctx.device_id(), (max_experts + 1) * 4)?,
            temporary: alloc(ctx.device_id(), blocks * 2 * 32 * 256 * 4)?,
            locks: alloc(ctx.device_id(), blocks * 2 * 4)?,
            max_rows,
            max_experts,
            blocks,
            tile_m,
            use_f16,
            variant,
        })
    }

    /// Produce stable expert routing directly on the device. IDs come from router_topk.
    /// Invalid expert IDs are omitted without writing beyond workspace bounds.
    pub fn permute(
        &self,
        ctx: &CudaContext,
        ids: &CudaBuffer,
        slots: usize,
        experts: usize,
    ) -> Result<Rows<'_>> {
        if slots == 0
            || slots > self.max_rows
            || experts == 0
            || experts > 128
            || experts > self.max_experts
        {
            return Err(error("invalid routing geometry"));
        }
        require_buffers(
            ctx,
            "Marlin routing",
            &[
                ("ids", ids, slots * 4),
                ("counts", &self.route_counts, experts * 4),
                ("offsets", &self.route_offsets, (experts + 1) * 4),
                ("sorted", &self.ids, (slots + experts * self.tile_m) * 4),
                (
                    "expert tiles",
                    &self.experts,
                    (slots + experts * self.tile_m).div_ceil(self.tile_m) * 4,
                ),
                ("padded count", &self.count, 4),
            ],
        )?;
        check_cuda(unsafe {
            ffi::apxinf_moe_permute_marlin(
                ids.ptr(),
                self.route_counts.ptr(),
                self.route_offsets.ptr(),
                self.ids.ptr(),
                self.experts.ptr(),
                self.count.ptr(),
                slots as i32,
                experts as i32,
                self.tile_m as i32,
                ctx.stream().handle(),
            )
        })?;
        Ok(Rows {
            workspace: self,
            rows: slots,
            experts,
        })
    }

    /// Host schedule for M1. A later device permutation can replace this producer.
    pub fn schedule(&self, offsets: &[usize]) -> Result<Rows<'_>> {
        if offsets.len() < 2
            || offsets.len() - 1 > self.max_experts
            || offsets[0] != 0
            || offsets.windows(2).any(|p| p[0] > p[1])
        {
            return Err(error("invalid expert offsets"));
        }
        let rows = *offsets.last().unwrap();
        if rows == 0 || rows > self.max_rows {
            return Err(error("row count exceeds workspace"));
        }
        let mut ids = Vec::<i32>::new();
        let mut experts = Vec::<i32>::new();
        for (expert, pair) in offsets.windows(2).enumerate() {
            for start in (pair[0]..pair[1]).step_by(self.tile_m) {
                experts.push(expert as i32);
                for row in start..start + self.tile_m {
                    ids.push(if row < pair[1] {
                        row as i32
                    } else {
                        rows as i32
                    });
                }
            }
        }
        let to_bytes = |values: &[i32]| {
            values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<u8>>()
        };
        self.ids
            .copy_from_host(&to_bytes(&ids))
            .map_err(Error::Cuda)?;
        self.experts
            .copy_from_host(&to_bytes(&experts))
            .map_err(Error::Cuda)?;
        self.count
            .copy_from_host(&(ids.len() as i32).to_ne_bytes())
            .map_err(Error::Cuda)?;
        Ok(Rows {
            workspace: self,
            rows,
            experts: offsets.len() - 1,
        })
    }
}

pub struct Rows<'a> {
    workspace: &'a Workspace,
    rows: usize,
    experts: usize,
}

pub fn grouped_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    weight: &Weights,
    rows: &Rows<'_>,
    output: &CudaBuffer,
) -> Result<()> {
    indexed_into(ctx, x, weight, rows, output, 1)
}

/// Read each token once logically, expanding its routed slots through indexed A loads.
pub fn indexed_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    weight: &Weights,
    rows: &Rows<'_>,
    output: &CudaBuffer,
    top_k: usize,
) -> Result<()> {
    if top_k == 0 || top_k > i32::MAX as usize || rows.rows % top_k != 0 {
        return Err(error("invalid top-k"));
    }
    if rows.experts != weight.experts {
        return Err(error("expert count mismatch"));
    }
    let ws = rows.workspace;
    if weight.fused_silu && (!ws.use_f16 || ws.variant != 0) {
        return Err(error("SwiGLU epilogue requires the original FP16 tiles"));
    }
    require_buffers(
        ctx,
        "Marlin grouped GEMM",
        &[
            ("input", x, bytes(rows.rows / top_k, weight.k, 2)?),
            (
                "output",
                output,
                bytes(
                    rows.rows,
                    if weight.fused_silu {
                        weight.n / 2
                    } else {
                        weight.n
                    },
                    2,
                )?,
            ),
            ("weight", &weight.q, weight.q.len()),
            ("zeros", &weight.z, weight.z.len()),
            ("scales", &weight.s, weight.s.len()),
            ("ids", &ws.ids, ws.ids.len()),
            ("experts", &ws.experts, ws.experts.len()),
            ("count", &ws.count, 4),
            ("temporary", &ws.temporary, ws.temporary.len()),
            ("locks", &ws.locks, ws.locks.len()),
        ],
    )?;
    check_cuda(unsafe {
        ffi::apxinf_marlin_grouped(
            x.ptr(),
            weight.q.ptr(),
            weight.z.ptr(),
            weight.s.ptr(),
            ws.ids.ptr(),
            ws.experts.ptr(),
            ws.count.ptr(),
            output.ptr(),
            ws.temporary.ptr(),
            ws.locks.ptr(),
            (rows.rows / top_k) as i32,
            weight.n as i32,
            weight.k as i32,
            ws.blocks as i32,
            top_k as i32,
            ws.tile_m as i32,
            i32::from(ws.use_f16),
            i32::from(weight.fused_silu),
            ws.variant,
            ctx.stream().handle(),
        )
    })
}

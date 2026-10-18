//! The GPU side of a layer split between the GPU and the Neural Engine
//! (EngineTailor M4 frontier m4a012, A14 boundary kernels): one kernel writes
//! both shares' inputs, one restores the Neural Engine share and adds it to
//! the GPU share. Adapted from EngineTailor (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

fn int(value: usize, message: &'static str) -> Result<TemplateArg> {
    i32::try_from(value)
        .map(TemplateArg::Int)
        .map_err(|_| Error::Contract(message))
}

/// Both boundary kernels move 32 x 32 blocks of tokens and channels.
fn check_geometry(tokens: usize, channels: usize, tile: usize) -> Result<usize> {
    if tokens == 0 || channels == 0 || channels % 32 != 0 || tile == 0 || tile % 64 != 0 {
        return Err(Error::Contract(
            "Neural Engine tiles need tokens, channels a multiple of 32 and a tile a multiple of 64",
        ));
    }
    let tiles = tokens.div_ceil(tile);
    tiles
        .checked_mul(tile)
        .and_then(|t| t.checked_mul(channels))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or(Error::Contract("Neural Engine tile index exceeds i32"))?;
    Ok(tiles)
}

/// BF16 `[1, T, C]` to (FP16 `[1, P, C]` rows padded to `P = 64 * ceil(T / 64)`
/// with zeros, FP16 `[ceil(T / TILE), C, TILE]` channel-first tiles zero past
/// `T`), in one pass.
pub struct AneTileHandoff {
    kernel: MetalKernel,
}

impl AneTileHandoff {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_ane_tile_handoff",
                &["h"],
                &["x", "packed"],
                include_str!("../native/ane_tile_handoff.metal"),
                "",
            )?,
        })
    }

    pub fn call(&self, h: &Array, tile: usize) -> Result<(Array, Array)> {
        let [1, tokens, channels] = *h.shape() else {
            return Err(Error::Contract(
                "Neural Engine handoff input must be [1, T, C]",
            ));
        };
        if h.dtype() != MlxDType::BF16 {
            return Err(Error::Contract("Neural Engine handoff input must be BF16"));
        }
        let tiles = check_geometry(tokens, channels, tile)?;
        let padded = tokens.div_ceil(64) * 64;
        // SAFETY: the grid covers channels `< C` and tokens `< tiles * TILE`
        // in 32 x 32 blocks (four rows of threads, eight tokens each). A
        // thread reads `h[token * C + channel]` only for `token < T`, writes
        // `x` only for `token < P <= tiles * TILE` (TILE a multiple of 64),
        // and writes each packed element of its block once; indices fit i32
        // (checked).
        let mut out = unsafe {
            self.kernel.call(
                &[h.clone()],
                &[
                    (&[1, padded, channels], MlxDType::F16),
                    (&[tiles, channels, tile], MlxDType::F16),
                ],
                [channels, tiles * tile / 32 * 4, 1],
                [32, 4, 1],
                &[
                    ("N", int(tokens, "handoff tokens exceed i32")?),
                    ("ROWS", int(padded, "handoff rows exceed i32")?),
                    ("C", int(channels, "handoff channels exceed i32")?),
                    ("TILE", int(tile, "handoff tile exceeds i32")?),
                ],
            )?
        }
        .into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }
}

/// `gpu_y + ane_y * gate_scale * up_scale` per token, in FP32, rounded once
/// to `out` (BF16 `[1, T, C]`). `ane_y` is the Neural Engine's FP16
/// channel-first `[ceil(T / TILE), C, TILE]` output and both scales are FP16
/// `[ceil(T / TILE) * TILE]`; `gpu_y` is FP16 `[1, T, C]`.
pub struct AneTileMerge {
    kernel: MetalKernel,
}

impl AneTileMerge {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_ane_tile_merge",
                &["gpu_y", "ane_y", "gate_scale", "up_scale"],
                &["out"],
                include_str!("../native/ane_tile_merge.metal"),
                "",
            )?,
        })
    }

    pub fn call(
        &self,
        gpu_y: &Array,
        ane_y: &Array,
        gate_scale: &Array,
        up_scale: &Array,
    ) -> Result<Array> {
        let [1, tokens, channels] = *gpu_y.shape() else {
            return Err(Error::Contract(
                "Neural Engine merge GPU share must be [1, T, C]",
            ));
        };
        let [tiles, ane_channels, tile] = *ane_y.shape() else {
            return Err(Error::Contract(
                "Neural Engine merge tiles must be [tiles, C, TILE]",
            ));
        };
        if check_geometry(tokens, channels, tile)? != tiles
            || ane_channels != channels
            || gate_scale.shape() != [tiles * tile]
            || up_scale.shape() != [tiles * tile]
            || [gpu_y, ane_y, gate_scale, up_scale]
                .iter()
                .any(|a| a.dtype() != MlxDType::F16)
        {
            return Err(Error::Contract(
                "Neural Engine merge needs FP16 shares, tiles covering T and one scale per tile token",
            ));
        }
        // SAFETY: the grid covers channels `< C` and tokens `< 32 *
        // ceil(T / 32)`; a thread reads `ane_y` within its token's tile for
        // `token < T` (TILE a multiple of 32, so a 32-token block stays in one
        // tile), reads `gpu_y`, both scales and writes `out` only for
        // `token < T`. Indices fit i32 (checked).
        let out = unsafe {
            self.kernel.call(
                &[
                    gpu_y.clone(),
                    ane_y.clone(),
                    gate_scale.clone(),
                    up_scale.clone(),
                ],
                &[(&[1, tokens, channels], MlxDType::BF16)],
                [channels, tokens.div_ceil(32) * 4, 1],
                [32, 4, 1],
                &[
                    ("C", int(channels, "merge channels exceed i32")?),
                    ("TILE", int(tile, "merge tile exceeds i32")?),
                    ("N", int(tokens, "merge tokens exceed i32")?),
                    ("OT", TemplateArg::DType(MlxDType::BF16)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

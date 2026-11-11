//! Deterministic tile choice for the MP22 INT8 (`s8mma`) GEMM assembly.
//!
//! `kernels/asm/qint8_*.asm` are six muDNN QY2 `s8mma` tiles, rendered from
//! `kernels/qint8_mma_tiles/templates/` with `io_dtype=int8` plus a local
//! template patch that narrows the epilogue to BF16 (the upstream
//! `output_dtype=half` configuration packed FP16), configured as a 1x1 GEMM,
//! then embedded by `build.rs`.
//! The templates are convolution kernels; the adapter drives their 67-argument
//! convolution ABI in the H=W=1 degenerate case, which is what makes them a plain
//! `[m,k] @ [n,k]^T -> [m,n]` with per-row and per-column dequantization scales.
//!
//! Unlike the BF16 tiles, these mask a partial `M` row but **not** a partial `N`
//! column or `K` row: a tile is only eligible when its `tile_n` divides `N` and
//! its `tile_k` divides `K`. That is why the tables here are narrower than
//! `asm_tiles`' -- several of the shapes the π0.5 L1 layer reaches (the ViT's
//! `n = 4304`, `k = 4304` and `k = 588`) have no eligible tile at all, and the
//! caller reports the gap rather than falling back to a different arithmetic.
//!
//! Choosing by measurement is what the probe did; the choice recorded here is a
//! table, not a run-time measurement, for the same reason as `asm_tiles`: the
//! default mode must not time anything.

/// One embedded tile: what the selector needs to know about it.
///
/// `tile_m` and `block_x` are the launch geometry the adapter needs, not
/// anything the *choice* between tiles turns on, so they live only in
/// `adapters/qint8_gemm_adapter.mu`'s `kTiles`. The shared contract with that
/// table and with `build.rs`'s `QINT8_ASM_VARIANTS` is the order, which is what
/// the tile index in the adapter ABI names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Qint8Tile {
    /// `tile_m x tile_n`, as the templates name them.
    pub(crate) name: &'static str,
    pub(crate) tile_n: u32,
    /// `tile_k`, the INT8 contraction depth one pass covers.
    pub(crate) tile_k: u32,
}

/// The embedded tiles, in the order `build.rs` lists them.
///
/// The index into this array is the index the adapter's `kTiles` must agree
/// with.
pub(crate) const QINT8_TILES: [Qint8Tile; 6] = [
    Qint8Tile { name: "256x128", tile_n: 128, tile_k: 64 },
    Qint8Tile { name: "256x256", tile_n: 256, tile_k: 64 },
    Qint8Tile { name: "128x128", tile_n: 128, tile_k: 64 },
    Qint8Tile { name: "256x64", tile_n: 64, tile_k: 64 },
    Qint8Tile { name: "32x128", tile_n: 128, tile_k: 64 },
    Qint8Tile { name: "32x64", tile_n: 64, tile_k: 64 },
];

/// The tile the probe timed fastest for each shape it was run on.
///
/// The key is `(m, n, k)` and the value is a [`QINT8_TILES`] name, measured on
/// M1000 (`mp_22`) with the reference launcher at `--warmup 5
/// --iterations 20`.
///
/// A five-tile sweep of the raw kernel over the four PI0.5 encode shapes:
/// `256x128` wins all four, while `128x128`/`256x256` lose 6%
/// (`522,2560,2048`), 6% (`522,2048,2048`) and 27% (`522,2048,16384`) of
/// end-to-end latency. `256x256` is a 512-thread tile with
/// one resident block per MP, which is what sinks it once `k` is deep.
///
/// The 522-row shapes are run as whole tiles plus a remainder (see the adapter's
/// row split), so the tile recorded here is the one that won for the *whole
/// block*; the adapter picks the remainder tile itself.
///
/// The `(10, 2560, 1024)` entry is the action expert's bare `m = 10` qkv, the
/// one decode shape where the short band's leading `32x64` loses: `32x128`
/// measured 28% faster (0.0460 vs 0.0596 ms, reference launcher at 50 warmup /
/// 200 launches) because 20 column blocks fit one wave where 40 need two. The
/// adapter's `kTailOverrides` only reaches a row split's remainder, so a bare
/// `m = 10` launch would otherwise take the band's pick; recording the shape
/// here fixes that without touching the ordering the other `m = 10` shapes
/// still win with.
const MEASURED: &[((u32, u32, u32), &str)] = &[
    ((512, 32768, 2048), "256x128"),
    ((522, 32768, 2048), "256x128"),
    ((522, 2560, 2048), "256x128"),
    ((522, 2048, 2048), "256x128"),
    ((522, 2048, 16384), "256x128"),
    ((512, 3456, 1152), "256x128"),
    ((512, 2048, 1152), "256x128"),
    ((512, 1152, 1152), "128x128"),
    ((10, 2560, 1024), "32x128"),
];

/// Tile preference for a short output (`m <= 32`), best first.
///
/// `32x64` leads because a short row block wastes most of a tall tile's height,
/// so the shortest-and-narrowest tile wins there -- the same ordering the BF16
/// set uses, and the tile the adapter's `select_tail_tile` picks for a row
/// split's ten-row remainder, which is the `m = 10` case this band mostly serves.
///
/// The ordering is measured for INT8 on the five π0.5 encode `(n, k)` pairs at
/// `m = 10`: `32x64` leads `32x128` by 5.1% on `(32768, 2048)`, 8.4% on
/// `(2048, 2048)` and 11.4% on `(2048, 16384)`, and trails by 26% on
/// `(2560, 2048)` and 28% on `(2560, 1024)` -- ahead on three of the five and on
/// their sum, which is why the adapter overrides those two pairs and not the
/// ordering. It is still not in [`MEASURED`], which keys whole row blocks: the
/// adapter picks the remainder tile itself, and a bare `m = 10` request is what
/// this band is for.
const PREFERRED_SHORT: &[u8] = &[5, 4, 3, 2, 0, 1];
/// Tile preference for a medium output (`32 < m <= 256`), best first.
const PREFERRED_MEDIUM: &[u8] = &[2, 0, 3, 1, 4, 5];
/// Tile preference for a tall output (`m > 256`), best first.
const PREFERRED_TALL: &[u8] = &[0, 1, 2, 3, 4, 5];

/// The index of the tile called `name` in [`QINT8_TILES`], or `None`.
fn index_of(name: &str) -> Option<u8> {
    QINT8_TILES
        .iter()
        .position(|tile| tile.name == name)
        .map(|index| index as u8)
}

/// Whether the tile at `index` can compute an `[m, k] @ [k, n]` product.
///
/// `m` is not an argument: the kernel masks a partial row tile. `n` and `k` must
/// each be a positive multiple of the tile's own extent, because these templates
/// mask neither.
pub(crate) fn eligible(index: u8, n: usize, k: usize) -> bool {
    let Some(tile) = QINT8_TILES.get(index as usize) else {
        return false;
    };
    n >= tile.tile_n as usize
        && n % tile.tile_n as usize == 0
        && k >= tile.tile_k as usize
        && k % tile.tile_k as usize == 0
}

/// Whether any embedded tile can compute an `[m, k] @ [k, n]` product.
///
/// `m` is not an argument, for the same reason [`eligible`] does not take it:
/// the kernel masks a partial row tile. The model layer asks this before it
/// commits a projection to INT8, because π0.5's ViT (`k = 588`) and action head
/// (`k = 32`, `n = 32`) reach shapes no tile can serve. The ViT MLP's
/// `k = n = 4304` misses here too, but `int8_dynamic` zero-pads it to 4352
/// before asking, so it does not end up on the BF16 route.
pub fn serves(n: usize, k: usize) -> bool {
    (0..QINT8_TILES.len() as u8).any(|index| eligible(index, n, k))
}

/// The tile for `[m, k] @ [k, n]`: the measured winner, else the band's first
/// eligible tile, else none.
fn pick(m: usize, n: usize, k: usize) -> Option<u8> {
    let shape = (
        u32::try_from(m).ok()?,
        u32::try_from(n).ok()?,
        u32::try_from(k).ok()?,
    );
    for (measured, name) in MEASURED {
        if *measured == shape {
            let index = index_of(name)?;
            if eligible(index, n, k) {
                return Some(index);
            }
        }
    }
    let preference = if m <= 32 {
        PREFERRED_SHORT
    } else if m <= 256 {
        PREFERRED_MEDIUM
    } else {
        PREFERRED_TALL
    };
    preference
        .iter()
        .copied()
        .find(|&index| eligible(index, n, k))
}

/// The tile to run on this build, or `None` when no tile can serve the shape.
///
/// The architecture gate mirrors `build.rs`: the six images are
/// [`crate::ASSEMBLY_ARCH`] assembly, so any other generation has nothing to
/// load.
pub(crate) fn select(m: usize, n: usize, k: usize) -> Option<u8> {
    if crate::MUSA_ARCH != crate::ASSEMBLY_ARCH {
        return None;
    }
    pick(m, n, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_are_the_six_the_build_embeds_in_order() {
        let names: Vec<&str> = QINT8_TILES.iter().map(|tile| tile.name).collect();
        assert_eq!(
            names,
            ["256x128", "256x256", "128x128", "256x64", "32x128", "32x64"]
        );
        assert!(QINT8_TILES.iter().all(|tile| tile.tile_k == 64));
    }

    #[test]
    fn every_measured_shape_resolves_to_an_eligible_tile() {
        for (shape, name) in MEASURED {
            let index = index_of(name).unwrap_or_else(|| panic!("{name} is not a tile"));
            assert!(
                eligible(index, shape.1 as usize, shape.2 as usize),
                "{name} cannot serve {shape:?}"
            );
            assert_eq!(
                pick(shape.0 as usize, shape.1 as usize, shape.2 as usize),
                Some(index),
                "{shape:?} does not select its measured winner {name}"
            );
        }
    }

    #[test]
    fn eligibility_needs_the_tile_to_divide() {
        // 2048 and 2560 are multiples of 128; 4304 is not.
        assert!(eligible(index_of("256x128").unwrap(), 2048, 2048));
        assert!(!eligible(index_of("256x128").unwrap(), 4304, 2048));
        // 588 and 4304 are not multiples of 64.
        assert!(!eligible(index_of("256x128").unwrap(), 1152, 588));
        // A width below every tile is eligible for none of them.
        assert_eq!(pick(10, 32, 1024), None);
    }

    #[test]
    fn serves_answers_for_the_shapes_the_model_layer_asks_about() {
        // The 18 encode layers' four projections, the vision qkv/output and the
        // action expert's four: all reachable, so `int8_dynamic` keeps them INT8.
        assert!(serves(2560, 2048));
        assert!(serves(2048, 16384));
        assert!(serves(1152, 1152));
        assert!(serves(4096, 1024));
        // The π0.5 misses: ViT `k = 588` patch embedding, its `n = 4304` FC1 and
        // `k = 4304` FC2, the action expert's `k = 32` in / `n = 32` out. The two
        // ViT MLP shapes are reachable once `int8_dynamic` pads them to 4352,
        // which is where the model layer asks about them instead of here.
        assert!(!serves(1152, 588));
        assert!(!serves(4304, 1152));
        assert!(!serves(1152, 4304));
        assert!(!serves(1024, 32));
        assert!(!serves(32, 1024));
        // The padded MLP the model layer actually asks about is served.
        assert!(serves(4352, 1152));
        assert!(serves(1152, 4352));
    }

    #[test]
    fn pick_falls_back_to_the_band() {
        assert_eq!(pick(10, 12800, 1024), index_of("32x64"));
        assert_eq!(pick(200, 2048, 2048), index_of("128x128"));
        assert_eq!(pick(1024, 2048, 2048), index_of("256x128"));
    }
}

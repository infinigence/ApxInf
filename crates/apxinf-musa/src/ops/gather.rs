//! Gather-style contracts: embedding lookup, position add, raw-image patches.

use apxinf_core::{Error, Result, Tensor};

use crate::kernels::embedding;
use crate::kernels::not_ported;
use crate::kernels::preprocess;
use crate::MusaBuffer;
use crate::MusaContext;
use crate::MusaDeviceAddress;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GatherSemantic {
    /// Token-embedding gather, scaled by `sqrt(cols)`.
    EmbeddingLookup,
    /// Projection + learned position embedding, plus an optional bias.
    BiasPosition,
    /// RGB u8 images to flattened patches in `[-1, 1]`.
    RgbToPatches,
    /// Replace selected matrix rows according to a u32 row map.
    ReplaceRows,
}

/// Geometry for [`GatherSemantic::RgbToPatches`].
#[derive(Clone, Copy, Debug)]
pub struct PatchGeometry {
    pub views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    /// True when the source images are NHWC rather than NCHW.
    pub nhwc: bool,
}

pub struct GatherArgs<'a> {
    pub semantic: GatherSemantic,
    /// Embedding table or activation, unused for raw-image preprocessing.
    pub input: Option<&'a Tensor>,
    /// Token ids for [`GatherSemantic::EmbeddingLookup`].
    pub ids: Option<&'a MusaBuffer>,
    /// Stable device address for graph-replayed token ids, mutually exclusive
    /// with [`Self::ids`]. Not served by this backend yet: the id buffer the
    /// model publishes is already a mapped allocation it reuses, and accepting
    /// a bare address would bypass the device check `ids` gets.
    pub ids_address: Option<MusaDeviceAddress>,
    /// Packed u8 images for [`GatherSemantic::RgbToPatches`].
    pub images: Option<&'a MusaBuffer>,
    /// Replacement matrix for [`GatherSemantic::ReplaceRows`].
    pub replacement: Option<&'a Tensor>,
    /// Per-output-row replacement index; `u32::MAX` keeps the input row.
    pub row_map: Option<&'a MusaBuffer>,
    pub bias: Option<&'a Tensor>,
    /// Learned position embedding for [`GatherSemantic::BiasPosition`].
    pub position: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub tokens_per_view: usize,
    pub patches: Option<PatchGeometry>,
}

impl<'a> GatherArgs<'a> {
    pub fn new(semantic: GatherSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input: Some(input),
            ids: None,
            ids_address: None,
            images: None,
            replacement: None,
            row_map: None,
            bias: None,
            position: None,
            out,
            tokens_per_view: 0,
            patches: None,
        }
    }

    /// Bind raw `uint8` images and the geometry that turns them into patches.
    pub fn rgb_to_patches(
        images: &'a MusaBuffer,
        out: &'a mut Tensor,
        geometry: PatchGeometry,
    ) -> Self {
        Self {
            semantic: GatherSemantic::RgbToPatches,
            input: None,
            ids: None,
            ids_address: None,
            images: Some(images),
            replacement: None,
            row_map: None,
            bias: None,
            position: None,
            out,
            tokens_per_view: 0,
            patches: Some(geometry),
        }
    }
}

pub fn gather(ctx: &MusaContext, args: GatherArgs<'_>) -> Result<()> {
    match args.semantic {
        GatherSemantic::EmbeddingLookup => {
            let table = args
                .input
                .ok_or_else(|| Error::Other("MUSA gather lookup requires a table".into()))?;
            if args.ids_address.is_some() {
                return Err(not_ported("gather with a stable ids address"));
            }
            let ids = args
                .ids
                .ok_or_else(|| Error::Other("MUSA gather lookup requires an id buffer".into()))?;
            let tokens = args.out.shape().dims().first().copied().unwrap_or(0);
            embedding::lookup_bf16_into(ctx, table, ids, args.out, tokens)
        }
        GatherSemantic::BiasPosition => {
            let projection = args
                .input
                .ok_or_else(|| Error::Other("MUSA gather position requires a projection".into()))?;
            let position = args.position.ok_or_else(|| {
                Error::Other("MUSA gather position requires a position table".into())
            })?;
            embedding::add_position_bf16_into(
                ctx,
                projection,
                args.bias,
                position,
                args.out,
                args.tokens_per_view,
            )
        }
        GatherSemantic::RgbToPatches => {
            let images = args
                .images
                .ok_or_else(|| Error::Other("MUSA gather patches requires image bytes".into()))?;
            let patches = args.patches.ok_or_else(|| {
                Error::Other("MUSA gather patches requires patch geometry".into())
            })?;
            let layout = if patches.nhwc {
                preprocess::ImageLayout::Nhwc
            } else {
                preprocess::ImageLayout::Nchw
            };
            preprocess::rgb_u8_to_patches_bf16(
                ctx,
                images,
                args.out,
                patches.views,
                patches.image_size,
                patches.patch_size,
                layout,
            )
        }
        GatherSemantic::ReplaceRows => Err(not_ported("gather replace_rows")),
    }
}

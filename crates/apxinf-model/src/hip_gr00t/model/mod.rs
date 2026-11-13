//! The loaded GR00T N1.7 model on a HIP device.
//!
//! A thin shell over the [`Blocks`] seam: the model owns the blocks, hands
//! out per-shape preparation, and runs the forward. Everything that outlives
//! a request lives in the blocks -- device weights, the static timestep
//! conditioning, the host-side position table -- while request-shaped state
//! (arenas, input buffers, prepared shapes, the embodiment cache) belongs to
//! [`crate::hip_gr00t::model_runner`].

pub mod blocks;
mod model;
pub mod token_groups;


use apxinf_core::Result;

pub use model::ModelVariant;
pub use model::validate_servable;

/// Wrap the BF16 blocks into the precision-dispatched variant the runner
/// holds. This is the only construction path today: gfx1151 has no FP8 tensor
/// cores, so BF16 is the only precision GR00T ships on HIP.
pub(in crate::hip_gr00t) fn build_bf16_model(
    blocks: blocks::Bf16Blocks,
    host: &mut crate::hip_gr00t::weights::host::Gr00tWeights,
) -> Result<ModelVariant> {
    Ok(ModelVariant::Bf16 {
        model: HipGr00tModel::new(blocks, host)?,
    })
}

use blocks::Blocks;
use crate::hip_gr00t::weights::Embodiment;

/// The loaded model over a block implementation `B`.
///
/// Generic in the blocks so precision variants construct and dispatch inside
/// [`model::ModelVariant`]; the runner holds the variant, never a concrete
/// blocks type.
pub struct HipGr00tModel<B: Blocks> {
    pub(super) blocks: B,
    /// The seven category tensors embodiment selection needs. The only host
    /// weight data that outlives upload; see [`EmbodimentSource`].
    embodiment_source: crate::hip_gr00t::weights::EmbodimentSource,
}

impl<B: Blocks> HipGr00tModel<B> {
    pub(in crate::hip_gr00t) fn new(blocks: B, host: &mut crate::hip_gr00t::weights::host::Gr00tWeights) -> Result<Self> {
        let embodiment_source = crate::hip_gr00t::weights::EmbodimentSource::take(host);
        // `host` is dropped by the caller after this; everything still in it
        // is already on the device.
        Ok(Self {
            blocks,
            embodiment_source,
        })
    }

    /// The requested embodiment's adapters, uploading them on demand.
    ///
    /// One deployment uses one embodiment, so this is normally a single
    /// upload. A caller that switches embodiment uploads the new set and
    /// drops the old one's allocations, so alternating costs bandwidth but
    /// does not grow memory. Caching the selection is runner policy, not
    /// model state, which is why this takes no cache.
    pub(in crate::hip_gr00t) fn embodiment(&self, id: usize) -> Result<Embodiment> {
        self.embodiment_source
            .select(id, self.blocks.config(), self.blocks.context())
    }
}

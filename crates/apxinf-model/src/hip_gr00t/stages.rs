//! Stage observation for accuracy work.
//!
//! The NVIDIA reference dumps every backbone layer and every DiT block of
//! every flow step, because that is what makes a 24-block tower and a 32-block
//! DiT bisectable when the final action disagrees. This is the seam the HIP
//! port emits the same stages through.
//!
//! Production inference passes [`NoStages`], a zero-sized no-op, so the
//! forward path pays nothing for it.

use apxinf_core::{Result, Tensor};

/// Receives each named stage as the forward pass produces it.
///
/// Implementations typically copy to the host, which synchronizes; nothing in
/// the port calls a sink on a path that matters for latency.
pub trait StageSink {
    fn stage(&mut self, name: &str, tensor: &Tensor) -> Result<()>;
}

/// Observes nothing.
pub struct NoStages;

impl StageSink for NoStages {
    fn stage(&mut self, _name: &str, _tensor: &Tensor) -> Result<()> {
        Ok(())
    }
}

impl<T: StageSink + ?Sized> StageSink for &mut T {
    fn stage(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        (**self).stage(name, tensor)
    }
}

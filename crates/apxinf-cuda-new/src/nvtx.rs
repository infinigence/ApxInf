//! NVTX range markers, no-op form.
//!
//! The legacy crate wires these to Nsight Systems when its `nvtx` feature and
//! toolkit are present. cuda-new's direct-launch shim keeps the call sites
//! compiling with free stubs; profiling runs use the legacy build.

/// RAII guard that would pop an NVTX range on drop.
pub struct Range;

impl Range {
    pub fn new(_name: &str) -> Self {
        Self
    }
}

/// `let _g = nvtx::range("name");` — a free no-op guard.
pub fn range(name: &str) -> Range {
    Range::new(name)
}

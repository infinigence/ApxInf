//! Registry of model loaders selected by model name and device suffix.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use apxinf_core::{Backend, Device, Result};
use once_cell::sync::Lazy;

use crate::auto::{LoadOptions, LoadedModel};

pub type ModelFactory = fn(&Path, Device, Arc<dyn Backend>, &LoadOptions) -> Result<LoadedModel>;

/// Device-independent option admission. A family must reject every
/// `model_variant` it does not implement; [`crate::AutoModel::load_model`]
/// calls it before constructing a backend or opening the checkpoint.
pub type Admission = fn(&LoadOptions) -> Result<()>;

/// A Metal family's admission and loader, registered as one entry so a
/// `metal_` name cannot exist without an admission function.
#[derive(Clone, Copy)]
pub struct MetalFamily {
    pub admit: Admission,
    pub load: ModelFactory,
}

static REGISTRY: Lazy<Mutex<HashMap<&'static str, ModelFactory>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static METAL_REGISTRY: Lazy<Mutex<HashMap<&'static str, MetalFamily>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Register a non-Metal model implementation under a stable name.
///
/// A Metal request never falls back here, so a `metal_` name registered with
/// this function is unreachable; Metal families register with
/// [`register_metal`] and admit their options first.
pub fn register(name: &'static str, factory: ModelFactory) {
    REGISTRY.lock().unwrap().insert(name, factory);
}
pub fn get(name: &str) -> Option<ModelFactory> {
    REGISTRY.lock().unwrap().get(name).copied()
}

pub fn list() -> Vec<&'static str> {
    let mut names = REGISTRY.lock().unwrap().keys().copied().collect::<Vec<_>>();
    names.sort_unstable();
    names
}

/// Register a Metal family with the admission that runs before device
/// construction.
pub fn register_metal(name: &'static str, admit: Admission, load: ModelFactory) {
    METAL_REGISTRY
        .lock()
        .unwrap()
        .insert(name, MetalFamily { admit, load });
}
pub fn get_metal(name: &str) -> Option<MetalFamily> {
    METAL_REGISTRY.lock().unwrap().get(name).copied()
}

pub fn list_metal() -> Vec<&'static str> {
    let mut names = METAL_REGISTRY
        .lock()
        .unwrap()
        .keys()
        .copied()
        .collect::<Vec<_>>();
    names.sort_unstable();
    names
}

//! Registry of model loaders selected by model name and device suffix.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use apxinf_core::{Backend, Device, Result};
use once_cell::sync::Lazy;

use crate::auto::{LoadOptions, LoadedModel};

pub type ModelFactory = fn(&Path, Device, Arc<dyn Backend>, &LoadOptions) -> Result<LoadedModel>;

static REGISTRY: Lazy<Mutex<HashMap<&'static str, ModelFactory>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Register a model implementation under a stable name.
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

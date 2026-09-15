//! GR00T's model-local bridge to the shared FP8 calibration workflow.
//!
//! The public runner lives in Python and owns dataset iteration, aggregation,
//! coverage checks, and manifest generation. This module only identifies the
//! real static-FP8 consumers and records BF16 inputs while a calibration call
//! is active.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Backend, Error, Result, Tensor};

use super::backbone::Qwen3VLConfig;
use super::backend::RuntimeBackend;
use super::Gr00tConfig;

thread_local! {
    static ACTIVE: RefCell<Option<Rc<Gr00tCalibrationCollector>>> = const { RefCell::new(None) };
}

pub(super) struct Gr00tCalibrationCollector {
    backend: Arc<RuntimeBackend>,
    records: RefCell<BTreeMap<String, f32>>,
}

pub(super) struct Gr00tCalibrationGuard;

impl Gr00tCalibrationCollector {
    pub(super) fn new(backend: Arc<RuntimeBackend>) -> Self {
        Self {
            backend,
            records: RefCell::new(BTreeMap::new()),
        }
    }

    pub(super) fn records(&self, expected: &[String]) -> Result<BTreeMap<String, f32>> {
        let records = self.records.borrow().clone();
        let expected = expected.iter().cloned().collect::<BTreeSet<_>>();
        let observed = records.keys().cloned().collect::<BTreeSet<_>>();
        if observed != expected {
            let missing = expected.difference(&observed).take(12).collect::<Vec<_>>();
            let unknown = observed.difference(&expected).take(12).collect::<Vec<_>>();
            return Err(Error::Other(format!(
                "GR00T calibration site coverage mismatch: missing={missing:?}, unknown={unknown:?}"
            )));
        }
        Ok(records)
    }

    fn observe(&self, name: &str, input: &Tensor) -> Result<()> {
        self.backend.synchronize()?;
        let input = if input.device() == self.backend.device() {
            self.backend.to_cpu(input)?
        } else {
            input.clone()
        };
        let mut maximum = 0.0f32;
        for value in input.to_f32_vec()? {
            if !value.is_finite() {
                return Err(Error::Other(format!(
                    "GR00T calibration site {name} produced non-finite activation {value}"
                )));
            }
            maximum = maximum.max(value.abs());
        }
        let mut records = self.records.borrow_mut();
        records
            .entry(name.to_owned())
            .and_modify(|current| *current = current.max(maximum))
            .or_insert(maximum);
        Ok(())
    }
}

impl Drop for Gr00tCalibrationGuard {
    fn drop(&mut self) {
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
    }
}

pub(super) fn install(collector: Rc<Gr00tCalibrationCollector>) -> Result<Gr00tCalibrationGuard> {
    ACTIVE.with(|slot| {
        let mut active = slot.borrow_mut();
        if active.is_some() {
            return Err(Error::Other(
                "nested GR00T calibration collection is not supported".into(),
            ));
        }
        *active = Some(collector);
        Ok(Gr00tCalibrationGuard)
    })
}

pub(super) fn observe(consumer: &str, input: &Tensor) -> Result<()> {
    let collector = ACTIVE.with(|slot| slot.borrow().clone());
    match collector {
        Some(collector) => collector.observe(&format!("{consumer}.input"), input),
        None => Ok(()),
    }
}

pub(super) fn is_active() -> bool {
    ACTIVE.with(|slot| slot.borrow().is_some())
}

/// Stable consumer identifiers for every linear executed by the static-FP8
/// path. The shared Python `QuantizationSpec` turns each consumer into its
/// conventional `<consumer>.input` capture site.
pub(super) fn fp8_consumers(config: &Gr00tConfig, backbone: &Qwen3VLConfig) -> Vec<String> {
    let mut consumers = BTreeSet::new();
    let mut add = |name: String| {
        consumers.insert(name);
    };

    for index in 0..backbone.text.n_layers {
        let prefix = format!("backbone.text.layers.{index}");
        for projection in ["query", "key", "value", "output", "gate", "up", "down"] {
            add(format!("{prefix}.{projection}"));
        }
    }

    add("backbone.vision.patch_embed".into());
    for index in 0..backbone.vision.depth {
        let prefix = format!("backbone.vision.blocks.{index}");
        for projection in ["qkv", "output", "fc1", "fc2"] {
            add(format!("{prefix}.{projection}"));
        }
    }
    for projection in ["fc1", "fc2"] {
        add(format!("backbone.vision.merger.{projection}"));
    }
    for index in 0..backbone.vision.deepstack_visual_indexes.len() {
        for projection in ["fc1", "fc2"] {
            add(format!("backbone.vision.deepstack.{index}.{projection}"));
        }
    }

    for name in [
        "action_head.state_encoder.input",
        "action_head.state_encoder.output",
        "action_head.action_encoder.input",
        "action_head.action_encoder.time",
        "action_head.action_encoder.output",
        "action_head.action_decoder.input",
        "action_head.action_decoder.output",
        "action_head.timestep_input",
        "action_head.timestep_output",
        "action_head.output_modulation",
        "action_head.output_projection",
    ] {
        add(name.into());
    }
    if let Some(vl) = &config.vl_self_attention {
        for index in 0..vl.num_layers {
            let prefix = format!("action_head.vl_self_attention.{index}");
            for projection in [
                "attention.query",
                "attention.key",
                "attention.value",
                "attention.output",
                "feed_forward.input",
                "feed_forward.output",
            ] {
                add(format!("{prefix}.{projection}"));
            }
        }
    }
    for index in 0..config.diffusion.num_layers {
        let prefix = format!("action_head.dit_blocks.{index}");
        for projection in [
            "adaptive_norm",
            "attention.query",
            "attention.key",
            "attention.value",
            "attention.output",
            "feed_forward.input",
            "feed_forward.output",
        ] {
            add(format!("{prefix}.{projection}"));
        }
    }
    consumers.into_iter().collect()
}

//! π0-FAST's model-local bridge to the shared FP8 calibration workflow.
//!
//! The public runner lives in Python and owns dataset iteration, aggregation,
//! coverage checks, and manifest generation. This module only names the real
//! static-FP8 consumers of the checkpoint and records their BF16 inputs while a
//! calibration call is active.
//!
//! Every π0-FAST projection except the FP32 SigLIP patch embedding leaves this
//! crate through `gemm::bf16` or `gemm::bf16_geglu_fused`, both of which offer
//! their activation and weight to the installed observer. The observer keys on
//! the *weight* tensor's device pointer, so it records exactly the operands the
//! FP8 runtime will later quantize, and nothing has to be threaded through the
//! layer signatures.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use apxinf_cuda::kernels::gemm::Bf16ActivationObserver;
use apxinf_core::{Backend, Error, Result, Tensor};

use super::backend::RuntimeBackend;
use super::fp8_calibration::Pi0FastCalibrationPlan;
use super::{Pi0FastConfig, StaticBf16Pi0FastWeights};

pub struct Pi0FastCalibrationObserver {
    backend: Arc<RuntimeBackend>,
    /// Device pointer of a BF16 projection weight -> stable logical site name.
    sites: HashMap<usize, String>,
    plan: Pi0FastCalibrationPlan,
    records: RefCell<BTreeMap<String, f32>>,
}

impl Pi0FastCalibrationObserver {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: &Pi0FastConfig,
        weights: &StaticBf16Pi0FastWeights,
    ) -> Result<Self> {
        let plan = Pi0FastCalibrationPlan::for_config(config);
        if weights.vision_layers.len() != plan.vision_layers().len()
            || weights.language_layers.len() != plan.language_layers().len()
        {
            return Err(Error::Other(
                "π0-FAST calibration plan does not match the loaded weight depth".into(),
            ));
        }
        let mut sites = HashMap::new();
        let mut insert = |tensor: &Tensor, name: &str| -> Result<()> {
            let handle = tensor.storage().as_gpu().ok_or_else(|| {
                Error::Other(format!("calibration weight for {name} is not on CUDA"))
            })?;
            sites.insert(handle.ptr, name.to_owned());
            Ok(())
        };
        for (layer, names) in weights.vision_layers.iter().zip(plan.vision_layers()) {
            insert(&layer.qkv.weight, &names.qkv_input)?;
            insert(&layer.output.weight, &names.attention_output)?;
            insert(&layer.fc1.weight, &names.fc1_input)?;
            insert(&layer.fc2.weight, &names.fc2_input)?;
        }
        insert(
            &weights.multimodal_projector.weight,
            super::fp8_calibration::MULTIMODAL_PROJECTOR_SITE,
        )?;
        for (layer, names) in weights.language_layers.iter().zip(plan.language_layers()) {
            insert(&layer.qkv.weight, &names.qkv_input)?;
            insert(&layer.output.weight, &names.attention_output)?;
            insert(&layer.gate_up.weight, &names.gate_up_input)?;
            insert(&layer.down.weight, &names.down_input)?;
        }
        insert(
            &weights.lm_head.weight,
            super::fp8_calibration::LM_HEAD_SITE,
        )?;

        Ok(Self {
            backend,
            sites,
            plan,
            records: RefCell::new(BTreeMap::new()),
        })
    }

    /// The aggregated per-site maxima, validated against the plan.
    pub fn records(&self) -> Result<BTreeMap<String, f32>> {
        let records = self.records.borrow().clone();
        let expected = self.plan.sites().iter().cloned().collect::<BTreeSet<_>>();
        let observed = records.keys().cloned().collect::<BTreeSet<_>>();
        if observed != expected {
            let missing = expected.difference(&observed).take(8).collect::<Vec<_>>();
            let unknown = observed.difference(&expected).take(8).collect::<Vec<_>>();
            return Err(Error::Other(format!(
                "π0-FAST calibration site coverage mismatch: missing={missing:?}, \
                 unknown={unknown:?}"
            )));
        }
        Ok(records)
    }
}

impl Bf16ActivationObserver for Pi0FastCalibrationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()> {
        let pointer = weight
            .storage()
            .as_gpu()
            .ok_or_else(|| Error::Other("observed BF16 weight is not on CUDA".into()))?
            .ptr;
        let Some(name) = self.sites.get(&pointer) else {
            return Ok(());
        };
        // The host vector is scoped to this reduction and dropped immediately;
        // the collector retains one scalar per logical site.
        let values = self.backend.to_cpu(activation)?.to_f32_vec()?;
        let amax = finite_amax(values, name)?;
        let mut records = self.records.borrow_mut();
        records
            .entry(name.clone())
            .and_modify(|current| *current = current.max(amax))
            .or_insert(amax);
        Ok(())
    }
}

fn finite_amax(values: impl IntoIterator<Item = f32>, name: &str) -> Result<f32> {
    let mut amax = 0.0f32;
    for value in values {
        if !value.is_finite() {
            return Err(Error::Other(format!(
                "calibration site {name} produced non-finite activation {value}"
            )));
        }
        amax = amax.max(value.abs());
    }
    Ok(amax)
}

#[cfg(test)]
mod tests {
    use super::finite_amax;

    #[test]
    fn activation_amax_rejects_non_finite_values() {
        assert!(finite_amax([1.0, f32::NAN], "test.site").is_err());
        assert!(finite_amax([f32::INFINITY], "test.site").is_err());
        assert_eq!(finite_amax([-2.0, 1.0], "test.site").unwrap(), 2.0);
    }
}

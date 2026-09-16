//! Native BF16 activation collection for PI0.5 FP8 calibration.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use apxinf_core::{Backend, Error, Result, Tensor};
use apxinf_cuda::kernels::gemm::Bf16ActivationObserver;

use super::{backend::RuntimeBackend, Bf16Weights, Pi05CalibrationPlan, Pi05Config};

pub struct Pi05CalibrationObserver {
    backend: Arc<RuntimeBackend>,
    sites: HashMap<usize, String>,
    plan: Pi05CalibrationPlan,
    records: RefCell<BTreeMap<String, f32>>,
}

impl Pi05CalibrationObserver {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: &Pi05Config,
        weights: &Bf16Weights,
    ) -> Result<Self> {
        let mut sites = HashMap::new();
        let plan = Pi05CalibrationPlan::for_config(config);
        let mut insert = |tensor: &Tensor, name: String| -> Result<()> {
            let handle = tensor
                .storage()
                .as_gpu()
                .ok_or_else(|| Error::Other(format!("calibration weight {name} is not on CUDA")))?;
            sites.insert(handle.ptr, name);
            Ok(())
        };

        insert(&weights.patch_embedding.weight, "vision.patch_input".into())?;
        for (layer, names) in weights.vision_layers.iter().zip(plan.vision_layers()) {
            insert(&layer.qkv.weight, names.attention_norm.clone())?;
            insert(
                &layer.output.weight,
                names.attention_output.clone().expect("vision tail site"),
            )?;
            insert(
                &layer.fc1.weight,
                names.mlp_norm.clone().expect("vision tail site"),
            )?;
            insert(
                &layer.fc2.weight,
                names.mlp_activation.clone().expect("vision tail site"),
            )?;
        }
        insert(
            &weights.multimodal_projector.weight,
            "vision.post_norm".into(),
        )?;
        for (layer, names) in weights.language_layers.iter().zip(plan.language_layers()) {
            insert(&layer.qkv.weight, names.attention_norm.clone())?;
            if let Some(attention_output) = &names.attention_output {
                insert(&layer.output.weight, attention_output.clone())?;
                insert(
                    &layer.gate_up.weight,
                    names.mlp_norm.clone().expect("language tail site"),
                )?;
                insert(
                    &layer.down.weight,
                    names.mlp_activation.clone().expect("language tail site"),
                )?;
            }
        }
        insert(&weights.action_in.weight, "action.input".into())?;
        insert(&weights.time_mlp_in.weight, "time.input".into())?;
        insert(&weights.time_mlp_out.weight, "time.hidden".into())?;
        for layer in &weights.action_layers {
            insert(&layer.input_style.weight, "action.conditioning".into())?;
            insert(
                &layer.post_attention_style.weight,
                "action.conditioning".into(),
            )?;
        }
        insert(
            &weights.action_final_style.weight,
            "action.conditioning".into(),
        )?;
        for (layer, names) in weights.action_layers.iter().zip(plan.action_layers()) {
            insert(&layer.qkv.weight, names.attention_norm.clone())?;
            insert(
                &layer.output.weight,
                names.attention_output.clone().expect("action tail site"),
            )?;
            insert(
                &layer.gate_up.weight,
                names.mlp_norm.clone().expect("action tail site"),
            )?;
            insert(
                &layer.down.weight,
                names.mlp_activation.clone().expect("action tail site"),
            )?;
        }
        insert(&weights.action_out.weight, "action.final_norm".into())?;

        Ok(Self {
            backend,
            sites,
            plan,
            records: RefCell::new(BTreeMap::new()),
        })
    }

    pub fn records(&self) -> Result<BTreeMap<String, f32>> {
        let records = self.records.borrow().clone();
        let expected = self.plan.sites().iter().cloned().collect::<BTreeSet<_>>();
        let observed = records.keys().cloned().collect::<BTreeSet<_>>();
        if observed != expected {
            let missing = expected.difference(&observed).take(8).collect::<Vec<_>>();
            let unknown = observed.difference(&expected).take(8).collect::<Vec<_>>();
            return Err(Error::Other(format!(
                "calibration site coverage mismatch: missing={missing:?}, unknown={unknown:?}"
            )));
        }
        Ok(records)
    }
}

impl Bf16ActivationObserver for Pi05CalibrationObserver {
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
        // the collector retains only one scalar per logical site.
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

use super::backend::DeviceBuffer as CudaBuffer;
use super::blocks::Bf16Blocks;
use super::network::Pi05Network;
impl Pi05Network<Bf16Blocks> {
    /// Explicit diagnostic traversal of the same Network; no second model body.
    pub fn calibrate(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
    ) -> Result<std::collections::BTreeMap<String, f32>> {
        use apxinf_core::Backend;
        let observer = std::rc::Rc::new(super::Pi05CalibrationObserver::new(
            self.blocks.backend.clone(),
            &self.blocks.config,
            &self.blocks.weights,
        )?);
        let _guard = super::backend::kernels::gemm::install_bf16_observer(observer.clone())?;
        self.infer(patches, ids, count, noise, embeddings)?;
        self.blocks.backend.synchronize()?;
        observer.records()
    }
}

//! Native BF16 activation collection for PI0.5 FP8 calibration.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{DType, Error, Result, Tensor};

use crate::pi05::{
    backend::{self, ops, Context, DeviceBuffer},
    Bf16Weights, Pi05CalibrationPlan, Pi05Config,
};

pub struct Pi05CalibrationObserver {
    backend: Arc<Context>,
    sites: HashMap<usize, String>,
    folded_rms_sites: BTreeSet<usize>,
    rms_identity: Tensor,
    rms_scratch: DeviceBuffer,
    plan: Pi05CalibrationPlan,
    records: RefCell<BTreeMap<String, f32>>,
}

thread_local! {
    static BF16_OBSERVER: RefCell<Option<Rc<Pi05CalibrationObserver>>> =
        const { RefCell::new(None) };
}

struct Bf16ObserverGuard;

impl Drop for Bf16ObserverGuard {
    fn drop(&mut self) {
        BF16_OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

fn install_bf16_observer(observer: Rc<Pi05CalibrationObserver>) -> Result<Bf16ObserverGuard> {
    BF16_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(Error::Other(
                "a PI0.5 BF16 activation observer is already installed".into(),
            ));
        }
        *slot = Some(observer);
        Ok(Bf16ObserverGuard)
    })
}

pub(in crate::pi05::model) fn observe_bf16_activation(
    activation: &Tensor,
    weight: &Tensor,
) -> Result<()> {
    BF16_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().as_ref() {
            observer.observe(activation, weight)?;
        }
        Ok(())
    })
}

/// FP8 folds language RMS scales into projection weights. Collect its input
/// before that scale without changing the BF16 reference traversal.
pub(in crate::pi05::model) fn observe_bf16_folded_rms(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<()> {
    BF16_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().as_ref() {
            observer.observe_folded_rms(input, weight, eps)?;
        }
        Ok(())
    })
}

impl Pi05CalibrationObserver {
    pub fn new(backend: Arc<Context>, config: &Pi05Config, weights: &Bf16Weights) -> Result<Self> {
        let mut sites = HashMap::new();
        let mut folded_rms_sites = BTreeSet::new();
        let plan = Pi05CalibrationPlan::for_config(config);
        let mut insert = |tensor: &Tensor, name: String| -> Result<usize> {
            let handle = tensor
                .storage()
                .as_gpu()
                .ok_or_else(|| Error::Other(format!("calibration weight {name} is not on CUDA")))?;
            sites.insert(handle.ptr, name);
            Ok(handle.ptr)
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
            folded_rms_sites.insert(insert(&layer.qkv.weight, names.attention_norm.clone())?);
            if let Some(attention_output) = &names.attention_output {
                insert(&layer.output.weight, attention_output.clone())?;
                folded_rms_sites.insert(insert(
                    &layer.gate_up.weight,
                    names.mlp_norm.clone().expect("language tail site"),
                )?);
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
            insert(&layer.input_modulation.weight, "action.conditioning".into())?;
            insert(
                &layer.post_attention_modulation.weight,
                "action.conditioning".into(),
            )?;
        }
        insert(
            &weights.action_final_modulation.weight,
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

        let tokens = config
            .num_views
            .checked_mul(config.patches_per_view())
            .and_then(|patches| patches.checked_add(config.max_token_len));
        let bytes = tokens
            .and_then(|tokens| tokens.checked_mul(config.language.width))
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("calibration RMS scratch size overflow".into()))?;
        let rms_identity = backend::to_device(
            &backend,
            &Tensor::from_f32(
                vec![config.language.width],
                &vec![1.0; config.language.width],
            )?,
        )?;
        // Keep diagnostic output outside the inference arena: additional arena
        // allocations would invalidate the prepared GEMM binding addresses.
        let rms_scratch = DeviceBuffer::alloc(bytes, backend.device_id()).map_err(Error::Cuda)?;
        Ok(Self {
            backend,
            sites,
            folded_rms_sites,
            rms_identity,
            rms_scratch,
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

impl Pi05CalibrationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()> {
        let pointer = weight
            .storage()
            .as_gpu()
            .ok_or_else(|| Error::Other("observed BF16 weight is not on CUDA".into()))?
            .ptr;
        let Some(name) = self.sites.get(&pointer) else {
            return Ok(());
        };
        if self.folded_rms_sites.contains(&pointer) {
            return Ok(());
        }
        self.record(activation, name)
    }

    fn observe_folded_rms(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<()> {
        let pointer = weight
            .storage()
            .as_gpu()
            .ok_or_else(|| Error::Other("observed BF16 weight is not on CUDA".into()))?
            .ptr;
        if !self.folded_rms_sites.contains(&pointer) {
            return Err(Error::Other(
                "unregistered folded RMS calibration site".into(),
            ));
        }
        let bytes = input
            .numel()
            .checked_mul(input.dtype().size_in_bytes())
            .ok_or_else(|| Error::Other("calibration RMS input size overflow".into()))?;
        let mut output = self
            .rms_scratch
            .view(0, bytes)
            .and_then(|buffer| buffer.as_tensor(input.shape().clone(), input.dtype()))
            .map_err(Error::Cuda)?;
        ops::rms_norm(
            &self.backend,
            ops::RmsNormArgs::new(input, &self.rms_identity, &mut output, eps),
        )?;
        self.record(&output, &self.sites[&pointer])
    }

    fn record(&self, activation: &Tensor, name: &str) -> Result<()> {
        // The host vector is scoped to this reduction and dropped immediately;
        // the collector retains only one scalar per logical site.
        let values = backend::to_cpu(activation)?.to_f32_vec()?;
        let amax = finite_amax(values, name)?;
        let mut records = self.records.borrow_mut();
        records
            .entry(name.to_owned())
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
    #[ignore = "requires CUDA"]
    fn calibration_observes_language_norm_before_folded_scale() {
        use super::*;
        use crate::pi05::model::blocks::bf16::language_layer_bf16;
        use crate::pi05::{GemmaVariantConfig, Pi05Weights};
        use apxinf_core::DType;

        let language = GemmaVariantConfig {
            width: 256,
            depth: 2,
            mlp_dim: 256,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 256,
        };
        let config = Pi05Config {
            language,
            action_expert: language,
            vocab_size: 16,
            num_views: 1,
            vision_depth: 1,
            vision_width: 64,
            vision_heads: 1,
            vision_head_dim: 64,
            vision_mlp_dim: 64,
            action_horizon: 2,
            num_flow_steps: 1,
            ..Pi05Config::default()
        };
        let backend = Arc::new(Context::new_with_autotune(0, false).unwrap());
        let mut host = Pi05Weights::synthetic(&config, 7).unwrap();
        let layer = &mut host.language_layers[0];
        layer.input_norm.offset = Tensor::from_f32(vec![256], &vec![-0.875; 256]).unwrap();
        layer.post_attention_norm.offset = Tensor::from_f32(vec![256], &vec![1.0; 256]).unwrap();
        // Zero attention projection keeps the second norm's input identical.
        layer.attention.output.weight = Tensor::zeros(vec![256, 256], DType::F32);
        let weights = Bf16Weights::from_host(&host, &backend).unwrap();
        let mut values = vec![half::bf16::ZERO; 32 * 256];
        for row in values.chunks_exact_mut(256) {
            row[0] = half::bf16::from_f32(16.0);
        }
        let input = backend::to_device(
            &backend,
            &Tensor::from_bf16(vec![32, 256], &values).unwrap(),
        )
        .unwrap();
        let run = || {
            language_layer_bf16(
                &backend,
                language,
                &weights.language_layers[0],
                &input,
                true,
                0,
                config.rms_norm_eps,
                config.rope_theta,
            )
            .unwrap()
        };
        let expected_output = backend::to_cpu(&run().hidden)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let observer =
            Rc::new(Pi05CalibrationObserver::new(backend.clone(), &config, &weights).unwrap());
        let guard = install_bf16_observer(observer.clone()).unwrap();
        let observed_output = backend::to_cpu(&run().hidden)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        drop(guard);
        assert_eq!(
            observed_output, expected_output,
            "calibration changed model output"
        );
        let records = observer.records.borrow();
        let expected = (16.0 / (1.0 + f64::from(config.rms_norm_eps)).sqrt()) as f32;
        for site in [
            "language.layers.0.attention_norm",
            "language.layers.0.mlp_norm",
        ] {
            assert!(
                (records[site] - expected).abs() < 1.0e-5,
                "FP8 folded norm range at {site}: {} != {expected}",
                records[site],
            );
        }
    }

    #[test]
    fn activation_amax_rejects_non_finite_values() {
        assert!(finite_amax([1.0, f32::NAN], "test.site").is_err());
        assert!(finite_amax([f32::INFINITY], "test.site").is_err());
        assert_eq!(finite_amax([-2.0, 1.0], "test.site").unwrap(), 2.0);
    }
}

use super::blocks::Bf16Blocks;
use super::Pi05Model;
use crate::pi05::backend::DeviceBuffer as CudaBuffer;
impl Pi05Model<Bf16Blocks> {
    /// Explicit diagnostic traversal of the same Model; no second model body.
    pub fn calibrate(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
    ) -> Result<std::collections::BTreeMap<String, f32>> {
        let observer = Rc::new(Pi05CalibrationObserver::new(
            self.blocks.backend.clone(),
            &self.blocks.config,
            &self.blocks.weights,
        )?);
        let _guard = install_bf16_observer(observer.clone())?;
        self.infer(patches, ids, count, noise, embeddings)?;
        backend::synchronize(&self.blocks.backend)?;
        observer.records()
    }
}

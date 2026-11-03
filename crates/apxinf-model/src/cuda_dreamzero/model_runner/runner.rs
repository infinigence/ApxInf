use super::prepare::Plan;
// Request ownership, transactional observation history, and three captured stages.
use crate::cuda_dreamzero::{
    backend::transfers,
    model::{DreamZeroModel, History, Profile, Workspace},
    DreamZeroConfig,
};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus,
    PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};
use apxinf_core::{DType, Device, Error, NormalGenerator, Result, Tensor};
use std::{cell::RefCell, rc::Rc};

const NOISE: usize = 16 * 2 * 44 * 80 + 24 * 32;
pub(super) type Session = Rc<RefCell<Option<Episode>>>;
#[derive(Clone)]
pub(super) struct HistoryIdentity {
    pub(super) identity: (String, String),
    pub(super) tokens: Vec<u32>,
    pub(super) mask: Vec<u8>,
    pub(super) next_frame: usize,
}
#[derive(Clone)]
pub(super) struct Episode {
    pub(super) key: HistoryIdentity,
    pub(super) history: History,
}
pub(super) struct HostInputs {
    pub(super) images: Tensor,
    pub(super) state: Tensor,
    pub(super) noise: Option<Tensor>,
    pub(super) ids: [Tensor; 2],
    pub(super) masks: [Tensor; 2],
    pub(super) valid: [Tensor; 2],
    pub(super) frames: usize,
    pub(super) identity: Option<(String, String)>,
    pub(super) mask: Vec<u8>,
}
fn bf16_input(t: &Tensor, shape: &[usize], name: &str) -> Result<Tensor> {
    if t.device() != Device::Cpu
        || !matches!(t.dtype(), DType::F32 | DType::BF16)
        || t.shape().dims() != shape
    {
        return Err(Error::Other(format!(
            "DreamZero {name} must be contiguous CPU F32/BF16 {shape:?}"
        )));
    }
    let values = t.to_f32_vec()?;
    let values: Vec<_> = values.into_iter().map(half::bf16::from_f32).collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Other(format!(
            "DreamZero {name} contains nonfinite BF16 values"
        )));
    }
    Tensor::from_bf16(shape.to_vec(), &values)
}
fn validate(request: &VlaRequest<'_>) -> Result<HostInputs> {
    let o = request.observation;
    let metadata = request.metadata;
    if o.action_mask.is_some()
        || metadata.planning.is_some()
        || metadata.embodiment_id.is_some_and(|id| id != 17)
    {
        return Err(Error::Other("DreamZero DROID supports embodiment 17, the fixed standard schedule, and no action mask".into()));
    }
    let grids = metadata
        .image_grid_thw
        .ok_or_else(|| Error::Other("DreamZero requires one montage grid [T,352,640]".into()))?;
    if grids.len() != 1 || grids[0][1..] != [352, 640] || ![1, 4, 9].contains(&grids[0][0]) {
        return Err(Error::Other(
            "DreamZero montage grid must be [1|4|9,352,640]".into(),
        ));
    }
    let frames = grids[0][0] as usize;
    let VisionObservation::Patches(pixels) = &o.vision else {
        return Err(Error::Other(
            "DreamZero requires policy-normalized montage rows".into(),
        ));
    };
    let images = bf16_input(pixels, &[frames * 352 * 640, 3], "montage")?;
    let state = bf16_input(
        o.state
            .as_ref()
            .ok_or_else(|| Error::Other("DreamZero requires normalized state".into()))?,
        &[1, 64],
        "state",
    )?;
    let noise = match request.initial_latent {
        InitialLatent::Provided(t) => Some(bf16_input(t, &[NOISE], "joint video/action noise")?),
        InitialLatent::Generate { .. } => None,
    };
    if o.token_ids.len() != 1024
        || o.token_ids
            .iter()
            .any(|&x| x as usize >= DreamZeroConfig::TEXT_VOCAB)
    {
        return Err(Error::Other(
            "DreamZero requires 512 positive + 512 negative in-vocabulary tokens".into(),
        ));
    }
    let mask = metadata
        .attention_mask
        .ok_or_else(|| Error::Other("DreamZero requires both prompt attention masks".into()))?;
    if mask.len() != 1024
        || mask.iter().any(|&x| x > 1)
        || mask
            .chunks(512)
            .any(|m| m[0] != 1 || m.windows(2).any(|w| w == [0, 1]))
    {
        return Err(Error::Other(
            "DreamZero masks must be two nonempty right-padded binary rows".into(),
        ));
    }
    let identity = match metadata.history {
        None => None,
        Some(h) if !h.session.is_empty() && !h.episode.is_empty() => {
            Some((h.session.into(), h.episode.into()))
        }
        Some(_) => {
            return Err(Error::Other(
                "DreamZero history identity must be nonempty".into(),
            ))
        }
    };
    let ids = |i: usize| {
        Tensor::from_i32(
            vec![512],
            &o.token_ids[i * 512..(i + 1) * 512]
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        )
    };
    let attention = |i: usize| {
        Tensor::from_bf16(
            vec![1, 512],
            &mask[i * 512..(i + 1) * 512]
                .iter()
                .map(|&x| {
                    if x == 1 {
                        half::bf16::ZERO
                    } else {
                        half::bf16::MIN
                    }
                })
                .collect::<Vec<_>>(),
        )
    };
    let valid = |i: usize| {
        Tensor::from_bf16(
            vec![512, 1],
            &mask[i * 512..(i + 1) * 512]
                .iter()
                .map(|&x| half::bf16::from_f32(x as f32))
                .collect::<Vec<_>>(),
        )
    };
    Ok(HostInputs {
        images,
        state,
        noise,
        ids: [ids(0)?, ids(1)?],
        masks: [attention(0)?, attention(1)?],
        valid: [valid(0)?, valid(1)?],
        frames,
        identity,
        mask: mask.to_vec(),
    })
}
pub(super) fn profile(
    host: &HostInputs,
    tokens: &[u32],
    episode: Option<&HistoryIdentity>,
) -> Profile {
    let next = episode
        .filter(|e| {
            host.identity.as_ref() == Some(&e.identity)
                && tokens == e.tokens
                && host.mask == e.mask
                && host.frames != 1
                && e.next_frame < 9
        })
        .map_or(1, |e| e.next_frame);
    Profile {
        input_frames: host.frames,
        start_frame: next,
    }
}

pub struct DreamZeroModelRunner {
    pub(super) model: Rc<DreamZeroModel>,
    pub(super) session: Session,
    implicit: RefCell<Option<Rc<Plan>>>,
}
impl DreamZeroModelRunner {
    pub(in crate::cuda_dreamzero) fn new(model: DreamZeroModel) -> Self {
        Self {
            model: Rc::new(model),
            session: Rc::new(RefCell::new(None)),
            implicit: RefCell::new(None),
        }
    }
    fn infer_host_with_download(
        &self,
        request: &VlaRequest<'_>,
        download: impl FnOnce(&Tensor) -> Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        // Public host inference is transactional through its output transfer
        // and validation, not merely through the last device launch. History
        // snapshots own immutable storage, so cloning retains the old episode
        // without copying or downloading its K/V tensors.
        let previous = self.session.borrow().clone();
        let result = (|| {
            let action = self.infer(request)?;
            self.model.ctx.synchronize().map_err(Error::Cuda)?;
            let values = download(action.tensor())?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::Other("DreamZero produced nonfinite actions".into()));
            }
            Ok(values)
        })();
        if result.is_err() {
            *self.session.borrow_mut() = previous;
        }
        result
    }
}

pub(super) fn bind(
    model: &DreamZeroModel,
    work: &Workspace,
    host: &HostInputs,
    request: &VlaRequest<'_>,
    normal: &mut dyn NormalGenerator,
) -> Result<()> {
    // Previous execution is complete before stable public-input storage is changed.
    model.ctx.synchronize().map_err(Error::Cuda)?;
    transfers::copy_cpu_to_cuda(&host.images, &work.images())?;
    transfers::copy_cpu_to_cuda(&host.state, &work.state())?;
    for i in 0..2 {
        transfers::copy_cpu_to_cuda(&host.ids[i], &work.ids(i))?;
        transfers::copy_cpu_to_cuda(&host.masks[i], &work.mask(i))?;
        transfers::copy_cpu_to_cuda(&host.valid[i], &work.valid_rows(i))?;
    }
    match (&host.noise, request.initial_latent) {
        (Some(t), InitialLatent::Provided(_)) => transfers::copy_cpu_to_cuda(t, &work.noise())?,
        (None, InitialLatent::Generate { rng }) => {
            normal.generate(rng)?;
        }
        _ => return Err(Error::Other("DreamZero latent binding changed".into())),
    }
    Ok(())
}
impl Plan {
    fn execute(&self, request: &VlaRequest<'_>, host: HostInputs) -> Result<Action> {
        let mut episode = self
            .session
            .try_borrow_mut()
            .map_err(|_| Error::Other("DreamZero session is already executing".into()))?;
        let actual = profile(
            &host,
            &request.observation.token_ids,
            episode.as_ref().map(|e| &e.key),
        );
        if !self.spec.matches(request.observation) || actual != self.profile {
            return Err(Error::Other("DreamZero prepared profile is stale for this observation/history; call prepare_for again".into()));
        }
        let mut work = self.work.borrow_mut();
        bind(
            &self.model,
            &work,
            &host,
            request,
            &mut **self.normal.borrow_mut(),
        )?;
        self.model.bind_history(
            &work,
            if actual.initial() {
                None
            } else {
                episode.as_ref().map(|e| &e.history)
            },
        )?;
        if let Some(graphs) = &self.graphs {
            for graph in graphs {
                graph.replay()?;
            }
        } else {
            self.sessions[0].run(|| self.model.vision(&mut work, false))?;
            self.sessions[1].run(|| self.model.language(&mut work, false))?;
            self.sessions[2].run(|| self.model.action(&mut work, false))?;
        }
        self.publish(request, host, actual, &work, &mut episode)
    }

    // The caller owns the session borrow until the synchronized result and its
    // immutable history snapshot are ready. Both a replay and a just-completed
    // real preparation use this single publication point.
    fn publish(
        &self,
        request: &VlaRequest<'_>,
        host: HostInputs,
        actual: Profile,
        work: &Workspace,
        episode: &mut Option<Episode>,
    ) -> Result<Action> {
        self.model.ctx.synchronize().map_err(Error::Cuda)?;
        let next = if let Some(identity) = host.identity {
            let history = self.model.snapshot(work)?;
            self.model.ctx.synchronize().map_err(Error::Cuda)?;
            Some(Episode {
                key: HistoryIdentity {
                    identity,
                    tokens: request.observation.token_ids.clone(),
                    mask: host.mask,
                    next_frame: actual.start_frame + 2,
                },
                history,
            })
        } else {
            None
        };
        // Sole publication point: invalid inputs, warmup, capture, or failed work
        // leave the previously committed episode intact.
        *episode = next;
        Ok(Action::new(work.actions().clone()))
    }
}
impl PreparedInference for Plan {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: if self.graphs.is_some() {
                ExecutionMode::Graph
            } else {
                ExecutionMode::Eager
            },
            fallback_reason: self.fallback_reason.clone(),
        }
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.execute(request, validate(request)?)
    }
}
impl VlaRuntime for DreamZeroModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [24, 32],
            patch_shape: [0, 3],
            max_token_len: 1024,
            num_views: 3,
            image_size: 0,
            patch_size: 1,
            accepts_rgb_u8: false,
        }
    }
    fn initial_latent_shape(&self) -> Vec<usize> {
        vec![NOISE]
    }
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let host = validate(request)?;
        let expected = profile(
            &host,
            &request.observation.token_ids,
            self.session.borrow().as_ref().map(|e| &e.key),
        );
        let cached = self
            .implicit
            .borrow()
            .as_ref()
            .filter(|p| p.profile == expected)
            .cloned();
        if let Some(plan) = cached {
            return plan.execute(request, host);
        }
        // Bound the implicit plan cache before allocating the next large profile.
        self.implicit.borrow_mut().take();
        let plan = Rc::new(self.make_plan(request, &host, ExecutionPolicy::RequireGraph)?);
        // make_plan has already evaluated this exact request on real data, then
        // successfully captured all three graphs without executing them. No
        // request can intervene here. Publish that result once instead of
        // rebinding and evaluating the same complete schedule a second time.
        // Explicit prepared handles and all later compatible calls still run
        // through execute, which rebinds every request value before replay.
        let action = {
            let mut episode = self
                .session
                .try_borrow_mut()
                .map_err(|_| Error::Other("DreamZero session is already executing".into()))?;
            let work = plan.work.borrow();
            plan.publish(request, host, expected, &work, &mut episode)?
        };
        *self.implicit.borrow_mut() = Some(plan);
        Ok(action)
    }
    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "DreamZero needs prepare_for with real montage, prompt, and history metadata".into(),
        ))
    }
    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        let host = validate(sample)?;
        Ok(Box::new(self.make_plan(sample, &host, policy)?))
    }
    fn clear_prepared(&self) -> Result<()> {
        self.implicit.borrow_mut().take();
        Ok(())
    }
    fn reset_history(&self) -> Result<()> {
        self.session.borrow_mut().take();
        Ok(())
    }
    fn execution_mode(&self) -> &'static str {
        if self.implicit.borrow().is_some() {
            "graph"
        } else {
            "unprepared"
        }
    }
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.infer_host_with_download(request, |tensor| transfers::to_cpu(tensor)?.to_f32_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vla::{HistoryContext, Observation, VlaMetadata};
    use apxinf_core::RngKey;

    fn observation() -> Observation {
        Observation {
            vision: VisionObservation::Patches(Tensor::zeros((4 * 352 * 640, 3), DType::F32)),
            state: Some(Tensor::zeros((1, 64), DType::F32)),
            token_ids: vec![1; 1024],
            action_mask: None,
        }
    }
    fn metadata<'a>(mask: &'a [u8], grid: &'a [[u32; 3]]) -> VlaMetadata<'a> {
        VlaMetadata {
            attention_mask: Some(mask),
            image_grid_thw: Some(grid),
            embodiment_id: Some(17),
            history: Some(HistoryContext {
                session: "caller-a",
                episode: "episode-a",
            }),
            ..Default::default()
        }
    }
    #[test]
    fn history_resets_on_identity_task_still_and_window_boundaries() {
        let observation = observation();
        let mask = vec![1; 1024];
        let grid = [[4, 352, 640]];
        let request = VlaRequest::generated_with_metadata(
            &observation,
            RngKey::new(1, 2, 3),
            metadata(&mask, &grid),
        );
        let mut host = validate(&request).unwrap();
        let mut key = HistoryIdentity {
            identity: host.identity.clone().unwrap(),
            tokens: observation.token_ids.clone(),
            mask,
            next_frame: 3,
        };
        let position = |h: &HostInputs, k: &HistoryIdentity| {
            profile(h, &observation.token_ids, Some(k)).start_frame
        };
        assert_eq!(position(&host, &key), 3);
        key.next_frame = 7;
        assert_eq!(position(&host, &key), 7);
        key.next_frame = 9;
        assert_eq!(position(&host, &key), 1);
        key.next_frame = 3;
        host.identity.as_mut().unwrap().0 = "caller-b".into();
        assert_eq!(position(&host, &key), 1);
        host.identity = Some(("caller-a".into(), "episode-b".into()));
        assert_eq!(position(&host, &key), 1);
        host.identity = Some(key.identity.clone());
        host.frames = 1;
        assert_eq!(position(&host, &key), 1);
        host.frames = 4;
        key.tokens[0] = 2;
        assert_eq!(position(&host, &key), 1);
        key.tokens[0] = 1;
        key.mask[511] = 0;
        assert_eq!(position(&host, &key), 1);
        key.mask[511] = 1;
        host.identity = None;
        assert_eq!(position(&host, &key), 1);
    }
    #[test]
    fn joint_input_validation_rejects_action_only_nonfinite_and_bad_masks() {
        let observation = observation();
        let mask = vec![1; 1024];
        let grid = [[4, 352, 640]];
        let joint = Tensor::zeros(vec![NOISE], DType::F32);
        let request =
            VlaRequest::provided_with_metadata(&observation, &joint, metadata(&mask, &grid));
        assert!(validate(&request).unwrap().noise.is_some());
        let action = Tensor::zeros((24, 32), DType::F32);
        assert!(validate(&VlaRequest {
            initial_latent: InitialLatent::Provided(&action),
            ..request
        })
        .is_err());
        let mut values = vec![0.; NOISE];
        values[0] = f32::NAN;
        let invalid = Tensor::from_f32(vec![NOISE], &values).unwrap();
        assert!(validate(&VlaRequest {
            initial_latent: InitialLatent::Provided(&invalid),
            ..request
        })
        .is_err());
        let mut invalid_mask = mask.clone();
        invalid_mask[7] = 0;
        assert!(validate(&VlaRequest {
            metadata: metadata(&invalid_mask, &grid),
            ..request
        })
        .is_err());
        let invalid_grid = [[2, 352, 640]];
        assert!(validate(&VlaRequest {
            metadata: metadata(&mask, &invalid_grid),
            ..request
        })
        .is_err());
    }
}

#[cfg(test)]
#[path = "acceptance.rs"]
mod acceptance;

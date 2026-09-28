//! Real-checkpoint graph contract. Fixtures are private canonical policy inputs.
use super::*;
use crate::{
    qwen_drive::backend::RuntimeBackend,
    vla::{Observation, PlanningOptions, VisionObservation, VlaMetadata},
    LoadOptions, LoadedModel,
};
use apxinf_core::RngKey;
use std::path::Path;

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn host(action: Action) -> Vec<f32> {
    transfers::to_cpu(action.tensor())
        .unwrap()
        .to_f32_vec()
        .unwrap()
}
#[test]
#[ignore = "requires CUDA, APXINF_QWEN_DRIVE_TEST_MODEL and APXINF_QWEN_DRIVE_TEST_INPUTS"]
fn whole_direct_graph_rebinds_inputs_and_owns_its_lifetime() {
    let checkpoint = std::env::var("APXINF_QWEN_DRIVE_TEST_MODEL").unwrap();
    let fixture = std::env::var("APXINF_QWEN_DRIVE_TEST_INPUTS").unwrap();
    let fixture = Path::new(&fixture);
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
    let grids: Vec<[u32; 3]> = serde_json::from_value(meta["grids"].clone()).unwrap();
    let shape: Vec<usize> = serde_json::from_value(meta["pixels_shape"].clone()).unwrap();
    let tokens: Vec<u32> = std::fs::read(fixture.join("tokens.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let conditioning = floats(&fixture.join("conditioning.bin"));
    let pixels = floats(&fixture.join("pixels.bin"));
    let noise_values = floats(&fixture.join("noise.bin"));
    let noise = Tensor::from_f32(vec![50, 3], &noise_values).unwrap();
    let changed_noise = Tensor::from_f32(
        vec![50, 3],
        &noise_values.iter().map(|x| -x).collect::<Vec<_>>(),
    )
    .unwrap();
    let original = Observation {
        vision: VisionObservation::Patches(Tensor::from_f32(shape.clone(), &pixels).unwrap()),
        token_ids: tokens,
        state: Some(Tensor::from_f32(vec![conditioning.len()], &conditioning).unwrap()),
        action_mask: None,
    };
    let mut images = original.clone();
    images.vision = VisionObservation::Patches(
        Tensor::from_f32(shape, &pixels.iter().map(|x| x * 0.5).collect::<Vec<_>>()).unwrap(),
    );
    let mut text = original.clone();
    // Change a plain-text token while retaining all image positions and length.
    let slot = text.token_ids.iter_mut().find(|id| **id < 100000).unwrap();
    *slot = if *slot == 42 { 43 } else { 42 };
    let mut state = original.clone();
    let mut changed_conditioning = conditioning;
    changed_conditioning[0] += 0.25;
    state.state =
        Some(Tensor::from_f32(vec![changed_conditioning.len()], &changed_conditioning).unwrap());
    let options = PlanningOptions {
        num_steps: Some(10),
        reasoning: None,
        raw_rgb_resize: None,
    };
    let metadata = VlaMetadata {
        image_grid_thw: Some(&grids),
        planning: Some(&options),
        ..Default::default()
    };
    let request = VlaRequest::provided_with_metadata(&original, &noise, metadata);
    let backend = Arc::new(RuntimeBackend::new(0).unwrap());
    let loaded = crate::qwen_drive::load::load_registered(
        Path::new(&checkpoint),
        Device::Cuda(0),
        backend.clone(),
        &LoadOptions::default(),
    )
    .unwrap();
    let LoadedModel::Vla(runner) = loaded else {
        panic!("VLA required")
    };
    let observations = [&original, &images, &text, &state, &original, &original];
    let noises = [&noise, &noise, &noise, &noise, &changed_noise, &noise];
    let eager = runner
        .prepare_for(&request, ExecutionPolicy::Eager)
        .unwrap();
    let reference: Vec<_> = observations
        .iter()
        .zip(noises)
        .map(|(obs, latent)| {
            host(
                eager
                    .run(&VlaRequest::provided_with_metadata(obs, latent, metadata))
                    .unwrap(),
            )
        })
        .collect();
    for (index, result) in reference[1..5].iter().enumerate() {
        assert_ne!(
            result,
            &reference[0],
            "fixture change {} must affect the output",
            index + 1
        );
    }
    assert_eq!(reference[0], reference[5]);
    let generated_reference = host(
        eager
            .run(&VlaRequest::generated_with_metadata(
                &original,
                RngKey::default(),
                metadata,
            ))
            .unwrap(),
    );
    drop(eager);
    let graph = runner
        .prepare_for(&request, ExecutionPolicy::RequireGraph)
        .unwrap();
    assert_eq!(
        graph.status(),
        PreparationStatus::Ready {
            mode: ExecutionMode::Graph,
            fallback_reason: None
        }
    );
    for (index, ((obs, latent), expected)) in
        observations.iter().zip(noises).zip(&reference).enumerate()
    {
        let actual = host(
            graph
                .run(&VlaRequest::provided_with_metadata(obs, latent, metadata))
                .unwrap(),
        );
        assert_eq!(&actual, expected, "changed-input case {index}");
    }
    // Same length is insufficient: changed image placement/geometry or steps
    // must reject without overwriting or evicting a healthy explicit plan.
    let mut invalid = original.clone();
    invalid.token_ids.pop();
    assert!(graph
        .run(&VlaRequest::provided_with_metadata(
            &invalid, &noise, metadata
        ))
        .is_err());
    let different_steps = PlanningOptions {
        num_steps: Some(4),
        reasoning: None,
        raw_rgb_resize: None,
    };
    let different_meta = VlaMetadata {
        planning: Some(&different_steps),
        ..metadata
    };
    assert!(graph
        .run(&VlaRequest::provided_with_metadata(
            &original,
            &noise,
            different_meta
        ))
        .is_err());
    assert_eq!(host(graph.run(&request).unwrap()), reference[0]);
    let rng = RngKey::default();
    let generated = VlaRequest::generated_with_metadata(&original, rng, metadata);
    let first = host(graph.run(&generated).unwrap());
    assert_eq!(first, generated_reference);
    assert_eq!(host(graph.run(&generated).unwrap()), first);
    for changed_key in [
        RngKey::new(1, 0, 0),
        RngKey::new(0, 1, 0),
        RngKey::new(0, 0, 1),
    ] {
        let changed = VlaRequest::generated_with_metadata(&original, changed_key, metadata);
        assert_ne!(host(graph.run(&changed).unwrap()), first);
    }
    assert_eq!(host(graph.run(&generated).unwrap()), first);
    assert_eq!(host(graph.run(&request).unwrap()), reference[0]);
    let generation = backend.context().tuning().generation();
    runner.clear_prepared().unwrap();
    drop(runner);
    // Explicit plan owns weights/state even after the originating runner drops.
    assert_eq!(host(graph.run(&request).unwrap()), reference[0]);
    assert_eq!(backend.context().tuning().generation(), generation);
    backend
        .context()
        .install_tuning(tuning::TuningSession::inference(
            tuning::TacticStore::default(),
        ))
        .unwrap();
    assert_eq!(graph.status(), PreparationStatus::Invalidated);
    assert!(graph.run(&request).is_err());
}

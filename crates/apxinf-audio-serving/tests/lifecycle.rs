use apxinf_audio_serving::{
    config::{capabilities, sha256, Config, Manifest},
    contracts::{AttemptKey, AudioOperation, MediaGrant},
    media::MediaRoot,
    service::AudioService,
};
use std::{
    collections::BTreeMap,
    fs,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[tokio::test]
async fn original_deadline_expires_without_public_success_or_replay() {
    let temporary = tempfile::tempdir().unwrap();
    let manifest: Manifest = serde_json::from_value(serde_json::json!({
        "schema":"apxinf-audio-manifest/1","task":"asr","model":"fixture-asr","provider":"cpu_fixture",
        "family":"fixture","precision":"fixture_pcm16","model_dir":temporary.path(),
        "runtime":{"python":"fixture","mlx":"none","mlx_audio":"none","mlx_audio_commit":"none","packages":BTreeMap::<String,String>::new()},
        "artifacts":[],"languages":["en"],"voices":BTreeMap::<String,String>::new()
    })).unwrap();
    let raw = serde_json::to_vec(&manifest).unwrap();
    let path = temporary.path().join("manifest.json");
    fs::write(&path, &raw).unwrap();
    let config = Config::load(
        &path,
        "python3".into(),
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/apxinf/apxinf/serving/audio_worker.py"),
        true,
        false,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert_eq!(config.revision, sha256(&raw));
    assert_eq!(
        config.capability_revision,
        sha256(&serde_json::to_vec(&capabilities(&manifest)).unwrap())
    );
    let media = MediaRoot::create(temporary.path().into()).unwrap();
    let (service, actor) = AudioService::start(config, media.clone(), 10000)
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !service.observations.ready.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let lease = media.reserve(4096).unwrap();
    let raw = b"RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00";
    let handle = uuid::Uuid::new_v4().to_string();
    lease.publish(&handle, raw).unwrap();
    let key = AttemptKey {
        request_id: uuid::Uuid::new_v4().to_string(),
        attempt: 1,
    };
    let operation = AudioOperation::Asr {
        input: MediaGrant {
            handle,
            worker_epoch: service.epoch.clone(),
            key: key.clone(),
            bytes: raw.len() as u64,
            sha256: sha256(raw),
            frames: 1,
            sample_rate: 16000,
        },
        language: "en".into(),
    };
    let (result, _cancellation) = service
        .submit(
            operation,
            lease,
            key,
            Instant::now() + Duration::from_millis(100),
        )
        .unwrap();
    let error = tokio::time::timeout(Duration::from_secs(2), result)
        .await
        .unwrap()
        .unwrap()
        .err()
        .unwrap();
    assert_eq!(error.code, "deadline_exceeded");
    tokio::time::timeout(Duration::from_secs(3), async {
        while media.reserved.load(Ordering::SeqCst) != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(service.observations.completed.load(Ordering::SeqCst), 0);
    assert!(service.observations.ready.load(Ordering::SeqCst));
    service.shutdown().await;
    actor.await.unwrap();
    assert!(service.observations.reaped.load(Ordering::SeqCst));
}

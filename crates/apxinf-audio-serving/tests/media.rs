use apxinf_audio_serving::{
    config::sha256,
    contracts::{AttemptKey, MediaGrant, MEDIA_QUOTA},
    media::{MediaRoot, ResponseMedia},
};
use std::{fs, os::unix::fs::symlink, sync::atomic::Ordering};

#[test]
fn response_buffer_retains_quota_through_last_clone() {
    let parent = tempfile::tempdir().unwrap();
    let root = MediaRoot::create(parent.path().into()).unwrap();
    let lease = root.reserve(8192).unwrap();
    let bytes = bytes::Bytes::from_owner(ResponseMedia {
        bytes: vec![1, 2, 3],
        lease,
    });
    let retained = bytes.clone();
    drop(bytes);
    assert_eq!(root.reserved.load(Ordering::SeqCst), 8192);
    assert_eq!(retained.as_ref(), [1, 2, 3]);
    drop(retained);
    assert_eq!(root.reserved.load(Ordering::SeqCst), 0);
}

#[test]
fn quota_and_media_identity_fail_closed() {
    let parent = tempfile::tempdir().unwrap();
    let root = MediaRoot::create(parent.path().into()).unwrap();
    let lease = root.reserve(MEDIA_QUOTA).unwrap();
    assert!(root.reserve(1).is_err());
    let raw = b"RIFF\x26\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\xc0\x5d\x00\x00\x80\xbb\x00\x00\x02\x00\x10\x00data\x02\x00\x00\x00\x00\x00";
    let handle = "55555555-5555-4555-8555-555555555555";
    lease.publish(handle, raw).unwrap();
    let grant = MediaGrant {
        handle: handle.into(),
        worker_epoch: "11111111-1111-4111-8111-111111111111".into(),
        key: AttemptKey {
            request_id: "22222222-2222-4222-8222-222222222222".into(),
            attempt: 1,
        },
        bytes: raw.len() as u64,
        sha256: sha256(raw),
        frames: 1,
        sample_rate: 24000,
    };
    assert_eq!(lease.read_output(&grant, handle).unwrap(), raw);
    let mut wrong = grant.clone();
    wrong.sha256 = "a".repeat(64);
    assert!(lease.read_output(&wrong, handle).is_err());
    assert!(lease.read_output(&grant, "another-handle").is_err());
    fs::remove_file(root.path.join(handle)).unwrap();
    let external = parent.path().join("external");
    fs::write(&external, raw).unwrap();
    symlink(&external, root.path.join(handle)).unwrap();
    assert!(lease.read_output(&grant, handle).is_err());
    drop(lease);
    assert_eq!(root.reserved.load(Ordering::SeqCst), 0);
    assert!(!root.path.join(handle).exists());
}

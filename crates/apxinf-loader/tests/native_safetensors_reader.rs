use apxinf_core::{DType, Device};
use apxinf_loader::safetensors::{load_native_path, safetensors_files, NativeSafetensorsReader};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

fn bytes(header: Value, payload: &[u8]) -> Vec<u8> {
    let mut header = serde_json::to_vec(&header).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend(header);
    out.extend(payload);
    out
}
fn checkpoint(path: &Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        json!({"format":"pt","config":"fixture"}),
    );
    let mut payload = Vec::new();
    for (name, dtype, shape, data) in tensors {
        let start = payload.len();
        payload.extend(*data);
        header.insert(
            (*name).into(),
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    fs::write(path, bytes(Value::Object(header), &payload)).unwrap();
}
fn index(path: &Path, mapping: Value) {
    fs::write(
        path.join("model.safetensors.index.json"),
        json!({"weight_map":mapping}).to_string(),
    )
    .unwrap();
}
fn error(path: &Path) -> String {
    NativeSafetensorsReader::open_path(path)
        .err()
        .expect("invalid fixture accepted")
}

#[test]
fn native_dtypes_and_payload_bits_match_eager_loading() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let f32_bits = [0x80000000u32, 0x7fc01234, 0x3f800000]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    checkpoint(
        &path,
        &[
            ("f32", "F32", &[3], &f32_bits),
            ("f16", "F16", &[2], &[1, 0x7e, 0, 0x80]),
            ("bf16", "BF16", &[2], &[0xc1, 0x7f, 0, 0x80]),
            ("fp8", "F8_E4M3", &[3], &[0, 0x80, 0x7f]),
            ("packed", "U8", &[3], &[0x21, 0xab, 0xff]),
        ],
    );
    let (eager, metadata) = load_native_path(&path).unwrap();
    let mut reader = NativeSafetensorsReader::open_path(dir.path()).unwrap();
    assert_eq!(reader.metadata(), &metadata);
    assert_eq!(reader.payload_bytes_copied(), 0);
    assert_eq!(
        reader.tensor_names().collect::<BTreeSet<_>>(),
        eager.keys().map(String::as_str).collect()
    );
    for (name, reference) in &eager {
        let info = reader.tensor_metadata(name).unwrap();
        assert_eq!(info.dtype, reference.dtype());
        assert_eq!(info.shape, reference.shape().dims());
        assert_eq!(info.byte_len, reference.storage().as_cpu().unwrap().len());
    }
    assert_eq!(reader.payload_bytes_copied(), 0);
    let mut total = 0;
    for (name, reference) in &eager {
        let loaded = reader.load_tensor(name).unwrap();
        assert_eq!(loaded.device(), Device::Cpu);
        assert_eq!(loaded.dtype(), reference.dtype());
        assert_eq!(loaded.shape(), reference.shape());
        assert_eq!(
            loaded.storage().as_cpu().unwrap(),
            reference.storage().as_cpu().unwrap()
        );
        total += loaded.storage().as_cpu().unwrap().len();
        assert_eq!(reader.payload_bytes_copied(), total);
    }
    assert_eq!(reader.tensor_metadata("bf16").unwrap().dtype, DType::BF16);
    assert_eq!(
        reader.tensor_metadata("packed").unwrap().dtype,
        DType::E2M1Pair
    );
}

#[test]
fn only_requested_tensor_is_materialized_and_result_outlives_reader() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let unused = vec![9; 1024 * 1024];
    checkpoint(
        &path,
        &[
            ("selected", "BF16", &[2], &[0x80, 0x3f, 0, 0x40]),
            ("unselected", "U8", &[unused.len()], &unused),
        ],
    );
    let mut reader = NativeSafetensorsReader::open_path(&path).unwrap();
    assert_eq!(reader.payload_bytes_copied(), 0);
    assert_eq!(
        reader.tensor_metadata("unselected").unwrap().byte_len,
        unused.len()
    );
    assert_eq!(reader.payload_bytes_copied(), 0);
    let first = reader.load_tensor("selected").unwrap();
    assert_eq!(reader.payload_bytes_copied(), 4);
    assert!(reader.tensor_metadata("missing").is_none());
    assert!(reader.load_tensor("missing").is_err());
    assert_eq!(reader.payload_bytes_copied(), 4);
    let second = reader.load_tensor("selected").unwrap();
    assert_eq!(reader.payload_bytes_copied(), 8);
    drop(reader);
    drop(dir);
    assert_eq!(first.storage().as_cpu().unwrap(), [0x80, 0x3f, 0, 0x40]);
    assert_eq!(
        first.storage().as_cpu().unwrap(),
        second.storage().as_cpu().unwrap()
    );
}

#[test]
fn sharded_index_directory_and_explicit_index_match_eager_loading() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(
        &dir.path().join("a.safetensors"),
        &[
            ("a", "F32", &[1], &[0, 0, 0x80, 0x3f]),
            ("not_indexed", "U8", &[1], &[8]),
        ],
    );
    checkpoint(
        &dir.path().join("b.safetensors"),
        &[("b", "BF16", &[1], &[0, 0x40])],
    );
    index(dir.path(), json!({"b":"b.safetensors","a":"a.safetensors"}));
    let (eager, meta) = load_native_path(dir.path()).unwrap();
    assert_eq!(eager.len(), 2);
    for path in [
        dir.path().to_path_buf(),
        dir.path().join("model.safetensors.index.json"),
    ] {
        let mut reader = NativeSafetensorsReader::open_path(&path).unwrap();
        assert_eq!(reader.metadata(), &meta);
        assert_eq!(reader.payload_bytes_copied(), 0);
        assert!(reader.tensor_metadata("not_indexed").is_none());
        for name in ["a", "b"] {
            assert_eq!(
                reader
                    .load_tensor(name)
                    .unwrap()
                    .storage()
                    .as_cpu()
                    .unwrap(),
                eager[name].storage().as_cpu().unwrap()
            );
        }
        assert_eq!(reader.payload_bytes_copied(), 6);
    }
}

#[test]
fn scalar_and_zero_element_shapes_are_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    checkpoint(
        &path,
        &[
            ("scalar", "F32", &[], &[0, 0, 0x80, 0x3f]),
            ("empty", "BF16", &[2, 0, 3], &[]),
        ],
    );
    let mut reader = NativeSafetensorsReader::open_path(&path).unwrap();
    assert_eq!(
        reader.load_tensor("scalar").unwrap().shape().dims(),
        &[] as &[usize]
    );
    assert_eq!(
        reader.load_tensor("empty").unwrap().shape().dims(),
        &[2, 0, 3]
    );
    assert_eq!(reader.payload_bytes_copied(), 4);
}

#[test]
fn directory_resolution_matches_the_existing_loader() {
    let dir = tempfile::tempdir().unwrap();
    assert!(error(dir.path()).contains("no SafeTensors"));
    let one = dir.path().join("weights.safetensors");
    checkpoint(&one, &[("a", "U8", &[1], &[7])]);
    assert_eq!(
        NativeSafetensorsReader::open_path(dir.path())
            .unwrap()
            .tensor_names()
            .count(),
        1
    );
    checkpoint(
        &dir.path().join("other.safetensors"),
        &[("b", "U8", &[1], &[8])],
    );
    assert!(error(dir.path()).contains("multiple SafeTensors"));
    fs::rename(one, dir.path().join("model.safetensors")).unwrap();
    assert!(NativeSafetensorsReader::open_path(dir.path())
        .unwrap()
        .tensor_metadata("a")
        .is_some());
    fs::write(dir.path().join("model.safetensors.index.json"), b"invalid").unwrap();
    assert!(error(dir.path()).contains("index"));
}

#[test]
fn indexed_shards_reject_missing_and_misassigned_tensors() {
    let dir = tempfile::tempdir().unwrap();
    checkpoint(
        &dir.path().join("a.safetensors"),
        &[("a", "U8", &[1], &[7])],
    );
    index(dir.path(), json!({}));
    assert!(error(dir.path()).contains("empty weight_map"));
    index(dir.path(), json!({"a":"missing.safetensors"}));
    assert!(error(dir.path()).contains("missing.safetensors"));
    index(dir.path(), json!({"missing":"a.safetensors"}));
    assert!(error(dir.path()).contains("missing indexed tensors"));
    checkpoint(
        &dir.path().join("b.safetensors"),
        &[("b", "U8", &[1], &[8])],
    );
    index(dir.path(), json!({"a":"b.safetensors","b":"a.safetensors"}));
    assert!(error(dir.path()).contains("index assigns"));
}

#[test]
fn indexed_shards_reject_duplicate_copies_and_unsafe_paths() {
    let dir = tempfile::tempdir().unwrap();
    for file in ["a.safetensors", "b.safetensors"] {
        checkpoint(
            &dir.path().join(file),
            &[("duplicate", "U8", &[1], &[7]), ("b", "U8", &[1], &[8])],
        );
    }
    index(
        dir.path(),
        json!({"duplicate":"a.safetensors","b":"b.safetensors"}),
    );
    assert!(error(dir.path()).contains("index assigns"));
    for shard in [
        "../escape.safetensors".to_owned(),
        dir.path()
            .join("a.safetensors")
            .to_str()
            .unwrap()
            .to_owned(),
    ] {
        index(dir.path(), json!({"a":shard}));
        assert!(error(dir.path()).contains("unsafe shard path"));
    }
}

#[test]
fn invalid_headers_and_ranges_are_rejected_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.safetensors");
    for data in [
        vec![],
        vec![0; 7],
        u64::MAX.to_le_bytes().to_vec(),
        vec![1, 0, 0, 0, 0, 0, 0, 0, 0xff],
    ] {
        fs::write(&path, data).unwrap();
        assert!(NativeSafetensorsReader::open_path(&path).is_err());
    }
    for (description, entry, payload) in [
        (
            "reversed",
            json!({"dtype":"U8","shape":[1],"data_offsets":[1,0]}),
            vec![0],
        ),
        (
            "out_of_bounds",
            json!({"dtype":"F32","shape":[1],"data_offsets":[0,8]}),
            vec![0; 4],
        ),
        (
            "length_mismatch",
            json!({"dtype":"BF16","shape":[2],"data_offsets":[0,2]}),
            vec![0; 2],
        ),
        (
            "shape_overflow",
            json!({"dtype":"F32","shape":[usize::MAX,2],"data_offsets":[0,4]}),
            vec![0; 4],
        ),
        (
            "offset_overflow",
            json!({"dtype":"U8","shape":[0],"data_offsets":[0,usize::MAX]}),
            vec![],
        ),
        (
            "unsupported_dtype",
            json!({"dtype":"I64","shape":[1],"data_offsets":[0,8]}),
            vec![0; 8],
        ),
    ] {
        fs::write(&path, bytes(json!({"weight":entry}), &payload)).unwrap();
        assert!(
            NativeSafetensorsReader::open_path(&path).is_err(),
            "{description}"
        );
    }
}

#[test]
#[ignore = "set APXINF_TEST_SAFETENSORS to an existing real checkpoint"]
fn real_checkpoint_selected_payloads_match_independent_file_reads() {
    let path =
        PathBuf::from(std::env::var_os("APXINF_TEST_SAFETENSORS").expect("set checkpoint path"));
    let files = safetensors_files(&path).unwrap();
    // Independently parse headers using ordinary file reads, then select the
    // smallest nonempty payload of each native dtype. Never load the full model.
    let mut selected = BTreeMap::<String, (String, PathBuf, u64, usize)>::new();
    let mut checkpoint_bytes = 0u64;
    for file in &files {
        let mut input = fs::File::open(file).unwrap();
        checkpoint_bytes += input.metadata().unwrap().len();
        let mut length = [0; 8];
        input.read_exact(&mut length).unwrap();
        let header_len = u64::from_le_bytes(length);
        let mut header = vec![0; header_len as usize];
        input.read_exact(&mut header).unwrap();
        let header: Value = serde_json::from_slice(&header).unwrap();
        for (name, info) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let dtype = info["dtype"].as_str().unwrap();
            let start = info["data_offsets"][0].as_u64().unwrap();
            let end = info["data_offsets"][1].as_u64().unwrap();
            let size = (end - start) as usize;
            if size == 0 || size > 1024 * 1024 {
                continue;
            }
            if selected.get(dtype).is_none_or(|x| size < x.3) {
                selected.insert(
                    dtype.into(),
                    (name.clone(), file.clone(), 8 + header_len + start, size),
                );
            }
        }
    }
    assert!(!selected.is_empty());
    let mut reader = NativeSafetensorsReader::open_path(&path).unwrap();
    let tensor_count = reader.tensor_names().count();
    assert_eq!(reader.payload_bytes_copied(), 0);
    let mut copied = 0;
    let mut verified = Vec::new();
    for (dtype, (name, file, offset, size)) in selected {
        let info = reader.tensor_metadata(&name).unwrap();
        assert_eq!(info.byte_len, size);
        let dims = info.shape.to_vec();
        let mut reference = vec![0; size];
        let mut input = fs::File::open(file).unwrap();
        input.seek(SeekFrom::Start(offset)).unwrap();
        input.read_exact(&mut reference).unwrap();
        let loaded = reader.load_tensor(&name).unwrap();
        assert_eq!(loaded.storage().as_cpu().unwrap(), reference);
        assert_eq!(loaded.shape().dims(), dims);
        copied += size;
        assert_eq!(reader.payload_bytes_copied(), copied);
        verified.push(json!({"tensor":name,"file_dtype":dtype,"native_dtype":format!("{:?}",loaded.dtype()),"shape":dims,"bytes":size}));
    }
    println!(
        "{}",
        json!({"checkpoint":path,"file_bytes":checkpoint_bytes,"shards":files.len(),"tensors":tensor_count,"payload_bytes_after_open":0,"payload_bytes_copied":copied,"selected":verified})
    );
}

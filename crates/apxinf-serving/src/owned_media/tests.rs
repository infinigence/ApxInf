use super::*;

fn corpus() -> Value {
    serde_json::from_str(include_str!(
        "../../../../tests/fixtures/serving/owned-media-v0.1.json"
    ))
    .unwrap()
}

fn chunk(kind: &str, data: &[u8]) -> Vec<u8> {
    let mut result = (data.len() as u32).to_be_bytes().to_vec();
    result.extend_from_slice(kind.as_bytes());
    result.extend_from_slice(data);
    result.extend_from_slice(&crc32fast::hash(&result[4..]).to_be_bytes());
    result
}

fn hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn descriptors(case: &Value, corpus: &Value) -> Vec<Value> {
    case["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| {
            let mut value = corpus["images"][name.as_str().unwrap()].clone();
            let mut data = STANDARD
                .decode(value["data_base64"].as_str().unwrap())
                .unwrap();
            if let Some(payload) = case["replace_idat_hex"].as_str() {
                let length = u32::from_be_bytes(data[33..37].try_into().unwrap()) as usize;
                data.splice(33..45 + length, chunk("IDAT", &hex(payload)));
            }
            if let Some(target) = case["pad_to_bytes"].as_u64() {
                let mut payload = hex("49492a0008000000000000000000");
                payload.resize(target as usize - data.len() - 12, 0);
                data.splice(33..33, chunk("eXIf", &payload));
            }
            if case.get("insert_chunk").is_some() {
                let extra = &case["insert_chunk"];
                let mut insert_at = 33;
                if extra["after_idat"] == true {
                    insert_at += 12 + u32::from_be_bytes(data[33..37].try_into().unwrap()) as usize;
                }
                let payload = chunk(
                    extra["type"].as_str().unwrap(),
                    &hex(extra["data_hex"].as_str().unwrap()),
                );
                for _ in 0..extra["repeat"].as_u64().unwrap_or(1) {
                    data.splice(insert_at..insert_at, payload.clone());
                }
            }
            for _ in 0..case["empty_idats"].as_u64().unwrap_or(0) {
                data.splice(33..33, chunk("IDAT", &[]));
            }
            if let Some(offset) = case["xor_offset"].as_u64() {
                data[offset as usize] ^= 1;
            }
            if let Some(edit) = case["set_byte"].as_array() {
                data[edit[0].as_u64().unwrap() as usize] = edit[1].as_u64().unwrap() as u8;
            }
            if let Some(offset) = case["repair_crc_offset"].as_u64() {
                let offset = offset as usize;
                let length =
                    u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                let end = offset + 8 + length;
                let crc = crc32fast::hash(&data[offset + 4..end]).to_be_bytes();
                data[end..end + 4].copy_from_slice(&crc);
            }
            if let Some(count) = case["truncate"].as_u64() {
                data.truncate(data.len() - count as usize);
            }
            data.extend(hex(case["append_hex"].as_str().unwrap_or("")));
            value["data_base64"] = json!(STANDARD.encode(&data));
            value["byte_count"] = json!(data.len());
            value["sha256"] = json!(format!("{:x}", Sha256::digest(&data)));
            value
        })
        .collect()
}

#[test]
fn shared_original_corpus() {
    let corpus = corpus();
    for case in corpus["cases"].as_array().unwrap() {
        let expected = descriptors(case, &corpus);
        let operation = case["operation"].as_str().unwrap();
        let mut value = if operation == "batch" {
            json!(expected)
        } else if operation == "source" {
            let descriptor = &expected[0];
            let mut encoded = descriptor["data_base64"].as_str().unwrap().to_owned();
            encoded.push_str(case["base64_suffix"].as_str().unwrap_or(""));
            if case["strip_padding"] == true {
                encoded = encoded.trim_end_matches('=').into();
            }
            if let Some(tail) = case["base64_tail"].as_str() {
                encoded.truncate(encoded.len() - 4);
                encoded.push_str(tail);
            }
            json!({"type":"base64", "media_type":descriptor["media_type"], "data":encoded})
        } else {
            expected[0].clone()
        };
        if let Some(edits) = case["set"].as_object() {
            for (key, edit) in edits {
                value[key] = edit.clone();
            }
        }
        if let Some(key) = case["remove"].as_str() {
            value.as_object_mut().unwrap().remove(key);
        }
        let before = value.clone();
        let result = match operation {
            "batch" => MediaBatch::from_descriptors(&value).map(|batch| batch.images),
            "source" => OwnedPng::from_source(&value).map(|image| vec![image]),
            "descriptor" => OwnedPng::from_descriptor(&value).map(|image| vec![image]),
            _ => panic!("Unknown corpus operation"),
        };
        if case["valid"] == true {
            let images = result.unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
            assert_eq!(
                images.iter().map(OwnedPng::descriptor).collect::<Vec<_>>(),
                expected,
                "{}",
                case["name"]
            );
            for (image, descriptor) in images.iter().zip(&expected) {
                assert_eq!(
                    image.bytes(),
                    STANDARD
                        .decode(descriptor["data_base64"].as_str().unwrap())
                        .unwrap()
                );
                assert!(OwnedPng::from_descriptor(&image.descriptor()).unwrap() == *image);
            }
        } else {
            let error = result
                .err()
                .unwrap_or_else(|| panic!("{} unexpectedly passed", case["name"]));
            assert_eq!(
                error.code,
                case["error"].as_str().unwrap(),
                "{}",
                case["name"]
            );
        }
        assert_eq!(value, before);
    }
}

#[test]
fn literal_identity_vectors() {
    for value in corpus()["images"].as_object().unwrap().values() {
        assert_eq!(
            OwnedPng::from_descriptor(value).unwrap().descriptor(),
            *value
        );
    }
}

#[test]
fn nonobjects_and_nonarray_batches() {
    for value in [
        Value::Null,
        json!(true),
        json!(1),
        json!("image"),
        json!([]),
    ] {
        assert!(OwnedPng::from_source(&value).is_err());
    }
    for value in [
        Value::Null,
        json!(true),
        json!(1),
        json!("images"),
        json!({}),
    ] {
        assert!(MediaBatch::from_descriptors(&value).is_err());
    }
}

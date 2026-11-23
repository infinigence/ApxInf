use super::*;
use image::{Rgb, RgbImage};
fn message(content: Value) -> ChatCompletionMessage {
    serde_json::from_value(json!({"role":"user","content":content})).unwrap()
}
fn encoded(format: ImageFormat, width: u32, height: u32) -> Vec<u8> {
    let image = RgbImage::from_fn(width, height, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, 50])
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, format)
        .unwrap();
    bytes.into_inner()
}
fn part(bytes: &[u8], mime: &str) -> Value {
    json!({"type":"image_url","image_url":{"url":format!("data:{mime};base64,{}",base64::encode(bytes))}})
}
#[test]
fn decoded_images_preserve_message_and_block_order_in_memory() {
    let image = part(&encoded(ImageFormat::Png, 12, 8), "image/png");
    let (messages, images) = decode_messages(
        vec![
            message(json!([{"type":"text","text":"before"},image,{"type":"text","text":"after"}])),
            message(json!([{"type":"text","text":"next"}])),
        ],
        MediaLimits::default(),
    )
    .unwrap();
    assert_eq!(images.len(), 1);
    assert_eq!((images[0].width, images[0].height), (12, 8));
    assert_eq!(
        messages[0].content.as_ref().unwrap()[1],
        json!({"type":"image"})
    );
    assert_eq!(messages[1].content, Some(json!("next")));
    assert!(!serde_json::to_string(&messages).unwrap().contains("base64"));
}
#[test]
fn exif_orientation_is_applied_before_rgb_admission() {
    let jpeg = encoded(ImageFormat::Jpeg, 12, 8);
    let original = image::load_from_memory(&jpeg).unwrap();
    for orientation in [2u8, 6, 8] {
        let mut exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
        exif.extend([orientation, 0, 0, 0, 0, 0, 0, 0]);
        let mut bytes = jpeg[..2].to_vec();
        bytes.extend([0xff, 0xe1]);
        bytes.extend(((exif.len() + 2) as u16).to_be_bytes());
        bytes.extend(exif);
        bytes.extend(&jpeg[2..]);
        let (_, images) = decode_messages(
            vec![message(json!([part(&bytes, "image/jpeg")]))],
            MediaLimits::default(),
        )
        .unwrap();
        let expected = match orientation {
            2 => original.fliph(),
            6 => original.rotate90(),
            _ => original.rotate270(),
        }
        .into_rgb8();
        assert_eq!(images[0].rgb, expected.as_raw().as_slice());
        assert_eq!(
            (images[0].width, images[0].height),
            (expected.width() as usize, expected.height() as usize)
        );
    }
}
#[test]
fn bad_containers_urls_roles_and_configured_limits_fail_before_admission() {
    let good = part(&encoded(ImageFormat::Png, 12, 8), "image/png");
    for url in [
        "https://example.com/x.png",
        "file:///x.png",
        "data:image/png;base64,!",
        "data:image/png;base64,iVBORw0KGgo=",
    ] {
        assert!(decode_messages(
            vec![message(
                json!([{"type":"image_url","image_url":{"url":url}}])
            )],
            MediaLimits::default()
        )
        .is_err());
    }
    for limits in [
        MediaLimits {
            image_bytes: 8,
            ..MediaLimits::default()
        },
        MediaLimits {
            image_dimension: 8,
            ..MediaLimits::default()
        },
        MediaLimits {
            decoded_pixels: 32,
            ..MediaLimits::default()
        },
    ] {
        assert_eq!(
            decode_messages(vec![message(json!([good.clone()]))], limits)
                .unwrap_err()
                .status,
            axum::http::StatusCode::BAD_REQUEST
        );
    }
    let mut tool = message(json!([good]));
    tool.role = "tool".into();
    assert!(decode_messages(vec![tool], MediaLimits::default()).is_err());
}

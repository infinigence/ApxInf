//! Transport-only image decoding. Checkpoint transforms belong to the model.
use crate::{protocol::ChatCompletionMessage, ApiError};
use apxinf_model::vision_preprocess::DecodedImage;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use serde_json::{json, Value};
use std::io::Cursor;

#[derive(Clone, Copy, Debug)]
pub struct MediaLimits {
    pub body_bytes: usize,
    pub image_bytes: usize,
    pub image_dimension: u32,
    pub decoded_pixels: usize,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            body_bytes: 12 * 1024 * 1024,
            image_bytes: 4 * 1024 * 1024,
            image_dimension: 4096,
            decoded_pixels: 16 * 1024 * 1024,
        }
    }
}

impl MediaLimits {
    pub(crate) fn validate(self) -> Result<(), String> {
        if self.body_bytes == 0
            || self.image_bytes == 0
            || self.image_dimension == 0
            || self.decoded_pixels == 0
            || self.decoded_pixels.checked_mul(8).is_none()
        {
            return Err("media limits must be positive and fit in memory address space".into());
        }
        Ok(())
    }
}

/// Normalize image parts in place before template guidance can insert turns.
/// Only decoded RGB is sent to the inference queue; encoded request data never
/// leaves memory. The caller runs this on its bounded blocking decoder pool.
pub(crate) fn decode_messages(
    mut messages: Vec<ChatCompletionMessage>,
    limits: MediaLimits,
) -> Result<(Vec<ChatCompletionMessage>, Vec<DecodedImage>), ApiError> {
    let mut images = Vec::new();
    let mut total_pixels = 0usize;
    for (index, message) in messages.iter_mut().enumerate() {
        let Some(Value::Array(parts)) = message.content.as_mut() else {
            continue;
        };
        if message.role == "tool" {
            if parts.iter().any(|p| {
                matches!(
                    p.get("type").and_then(Value::as_str),
                    Some("image" | "image_url")
                )
            }) {
                return Err(invalid("images are supported only in user messages", index));
            }
            continue;
        }
        if message.role != "user" {
            return Err(invalid(
                "content arrays are supported only for user messages",
                index,
            ));
        }
        if parts.is_empty() {
            return Err(invalid("content must not be empty", index));
        }
        let mut normalized = Vec::new();
        let mut text = String::new();
        let mut has_image = false;
        for part in parts.iter() {
            match part.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let value = part
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("text block needs a text string", index))?;
                    text.push_str(value);
                    normalized.push(json!({"type":"text","text":value}));
                }
                Some("image_url") => {
                    let source = part
                        .get("image_url")
                        .ok_or_else(|| invalid("image_url block needs image_url", index))?;
                    if source
                        .get("detail")
                        .is_some_and(|v| v.as_str() != Some("auto"))
                    {
                        return Err(invalid("only image detail=auto is supported", index));
                    }
                    let url = source
                        .get("url")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("image_url.url must be a base64 data URL", index))?;
                    let image = decode_image(url, limits, &mut total_pixels)
                        .map_err(|e| invalid(e, index))?;
                    images.push(image);
                    normalized.push(json!({"type":"image"}));
                    has_image = true;
                }
                _ => {
                    return Err(invalid(
                        "supported content blocks are text and image_url",
                        index,
                    ))
                }
            }
        }
        message.content = Some(if has_image {
            Value::Array(normalized)
        } else {
            Value::String(text)
        });
    }
    Ok((messages, images))
}

fn invalid(message: impl Into<String>, index: usize) -> ApiError {
    ApiError::invalid(message, Some(format!("messages[{index}].content")))
}

fn decode_image(
    url: &str,
    limits: MediaLimits,
    total_pixels: &mut usize,
) -> Result<DecodedImage, String> {
    let (header, data) = url
        .split_once(',')
        .ok_or("images must be base64 data URLs")?;
    let format = match header {
        "data:image/png;base64" => ImageFormat::Png,
        "data:image/jpeg;base64" => ImageFormat::Jpeg,
        _ => return Err("supported image media types are inline image/png and image/jpeg".into()),
    };
    if data.len().saturating_mul(3) / 4 > limits.image_bytes.saturating_add(2) {
        return Err(format!(
            "encoded image exceeds image_bytes={}",
            limits.image_bytes
        ));
    }
    let bytes = base64::decode(data).map_err(|e| format!("invalid image base64: {e}"))?;
    if bytes.is_empty() || bytes.len() > limits.image_bytes {
        return Err(format!(
            "encoded image must contain 1..={} bytes",
            limits.image_bytes
        ));
    }
    if image::guess_format(&bytes).ok() != Some(format) {
        return Err("image bytes do not match their media type".into());
    }
    if format == ImageFormat::Png {
        let mut chunks = &bytes[8..];
        while chunks.len() >= 12 {
            let size = u32::from_be_bytes(chunks[..4].try_into().unwrap()) as usize;
            if &chunks[4..8] == b"acTL" {
                return Err("animated images are unsupported".into());
            }
            let step = size
                .checked_add(12)
                .filter(|&s| s <= chunks.len())
                .ok_or("invalid PNG chunk length")?;
            chunks = &chunks[step..];
        }
    }
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut bounds = image::Limits::default();
    bounds.max_image_width = Some(limits.image_dimension);
    bounds.max_image_height = Some(limits.image_dimension);
    bounds.max_alloc = Some((limits.decoded_pixels as u64).saturating_mul(8));
    reader.limits(bounds);
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| format!("invalid image: {e}"))?;
    if !matches!(
        decoder.color_type(),
        image::ColorType::L8
            | image::ColorType::La8
            | image::ColorType::Rgb8
            | image::ColorType::Rgba8
    ) {
        return Err("only 8-bit PNG/JPEG images are supported".into());
    }
    let (w, h) = decoder.dimensions();
    let pixels = (w as usize)
        .checked_mul(h as usize)
        .ok_or("decoded pixel count overflow")?;
    *total_pixels = total_pixels
        .checked_add(pixels)
        .filter(|&n| n <= limits.decoded_pixels)
        .ok_or_else(|| {
            format!(
                "decoded images exceed decoded_pixels={}",
                limits.decoded_pixels
            )
        })?;
    let orientation = decoder
        .orientation()
        .map_err(|e| format!("invalid image orientation: {e}"))?;
    let mut decoded =
        DynamicImage::from_decoder(decoder).map_err(|e| format!("invalid image: {e}"))?;
    decoded.apply_orientation(orientation);
    let rgb = decoded.into_rgb8();
    Ok(DecodedImage {
        width: rgb.width() as usize,
        height: rgb.height() as usize,
        rgb: rgb.into_raw(),
    })
}

#[cfg(test)]
mod tests;

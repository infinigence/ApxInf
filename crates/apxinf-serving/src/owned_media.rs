use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fmt, io::Cursor};

pub const MAX_IMAGE_BYTES: usize = 131_072;
pub const MAX_BATCH_BYTES: usize = 262_144;
pub const MAX_IMAGES: usize = 4;
const MAX_DIMENSION: u32 = 1024;
const MAX_CHUNKS: usize = 1024;
const MAX_FRAME_STORAGE: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaError {
    pub code: &'static str,
    pub message: &'static str,
}

impl fmt::Display for MediaError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(self.message)
    }
}
impl std::error::Error for MediaError {}
type Result<T> = std::result::Result<T, MediaError>;

fn invalid() -> MediaError {
    MediaError {
        code: "invalid_request",
        message: "The PNG source or descriptor is invalid.",
    }
}
fn unsupported() -> MediaError {
    MediaError {
        code: "unsupported_feature",
        message: "The media form or limit is unsupported.",
    }
}
fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let object = value.as_object().ok_or_else(invalid)?;
    if object.len() != names.len() || names.iter().any(|name| !object.contains_key(*name)) {
        return Err(invalid());
    }
    Ok(())
}

#[derive(Clone, PartialEq, Eq)]
pub struct OwnedPng {
    bytes: Box<[u8]>,
    width: u32,
    height: u32,
    sha256: String,
}

impl OwnedPng {
    pub fn from_source(value: &Value) -> Result<Self> {
        fields(value, &["type", "media_type", "data"])?;
        let kind = value["type"].as_str().ok_or_else(invalid)?;
        let media_type = value["media_type"].as_str().ok_or_else(invalid)?;
        if kind != "base64" || media_type != "image/png" {
            return Err(unsupported());
        }
        Self::from_base64(value["data"].as_str().ok_or_else(invalid)?)
    }

    pub fn from_descriptor(value: &Value) -> Result<Self> {
        fields(
            value,
            &[
                "media_type",
                "data_base64",
                "sha256",
                "byte_count",
                "width",
                "height",
            ],
        )?;
        let media_type = value["media_type"].as_str().ok_or_else(invalid)?;
        if media_type != "image/png" {
            return Err(unsupported());
        }
        let image = Self::from_base64(value["data_base64"].as_str().ok_or_else(invalid)?)?;
        if value["sha256"].as_str() != Some(&image.sha256)
            || value["byte_count"].as_u64() != Some(image.bytes.len() as u64)
            || value["width"].as_u64() != Some(image.width as u64)
            || value["height"].as_u64() != Some(image.height as u64)
        {
            return Err(invalid());
        }
        Ok(image)
    }

    fn from_base64(encoded: &str) -> Result<Self> {
        if encoded.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
            return Err(unsupported());
        }
        let bytes = STANDARD.decode(encoded).map_err(|_| invalid())?;
        if STANDARD.encode(&bytes) != encoded {
            return Err(invalid());
        }
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(unsupported());
        }
        let (width, height) = png_shape(&bytes)?;
        let mut options = png::DecodeOptions::default();
        options.set_ignore_checksums(false);
        options.set_skip_ancillary_crc_failures(false);
        let mut decoder = png::Decoder::new_with_options(Cursor::new(&bytes), options);
        decoder.set_limits(png::Limits {
            bytes: 8 * 1024 * 1024,
        });
        let mut reader = decoder.read_info().map_err(|_| invalid())?;
        let length = reader.output_buffer_size().ok_or_else(invalid)?;
        if length > MAX_FRAME_STORAGE {
            return Err(unsupported());
        }
        let mut pixels = vec![0; length];
        reader.next_frame(&mut pixels).map_err(|_| invalid())?;
        reader.finish().map_err(|_| invalid())?;
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        drop(reader);
        Ok(Self {
            bytes: bytes.into_boxed_slice(),
            width,
            height,
            sha256,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn descriptor(&self) -> Value {
        json!({"media_type":"image/png", "data_base64":STANDARD.encode(&self.bytes),
            "sha256":self.sha256, "byte_count":self.bytes.len(), "width":self.width, "height":self.height})
    }
}

fn png_shape(bytes: &[u8]) -> Result<(u32, u32)> {
    if bytes.len() < 33 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[8..16] != b"\0\0\0\rIHDR" {
        return Err(invalid());
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    if width == 0 || height == 0 {
        return Err(invalid());
    }
    if width > MAX_DIMENSION
        || height > MAX_DIMENSION
        || bytes[24] != 8
        || ![2, 6].contains(&bytes[25])
        || bytes[26..29] != [0, 0, 0]
    {
        return Err(unsupported());
    }
    let mut offset = 8;
    let mut data_bytes = 0;
    let mut data_seen = false;
    let mut compressed = Vec::new();
    let mut ancillary_seen = Vec::new();
    for index in 0..MAX_CHUNKS {
        if bytes.len().saturating_sub(offset) < 12 {
            return Err(invalid());
        }
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        if length > bytes.len() - offset - 12 {
            return Err(invalid());
        }
        let kind = &bytes[offset + 4..offset + 8];
        if !kind.iter().all(u8::is_ascii_alphabetic) || !kind[2].is_ascii_uppercase() {
            return Err(invalid());
        }
        let crc_offset = offset + 8 + length;
        let expected_crc =
            u32::from_be_bytes(bytes[crc_offset..crc_offset + 4].try_into().unwrap());
        if crc32fast::hash(&bytes[offset + 4..crc_offset]) != expected_crc {
            return Err(invalid());
        }
        let payload = &bytes[offset + 8..crc_offset];
        offset += length + 12;
        match kind {
            b"IHDR" if index == 0 => {}
            b"IHDR" => return Err(invalid()),
            b"IDAT" => {
                data_seen = true;
                data_bytes += length;
                compressed.extend_from_slice(payload);
            }
            b"IEND" => {
                if length != 0 || data_bytes == 0 || offset != bytes.len() {
                    return Err(invalid());
                }
                let channels = if bytes[25] == 2 { 3 } else { 4 };
                check_pixel_stream(
                    &compressed,
                    height as usize * (1 + width as usize * channels),
                )?;
                return Ok((width, height));
            }
            b"sRGB" | b"eXIf" | b"pHYs" | b"gAMA" | b"cHRM" | b"tIME" => {
                if data_seen || ancillary_seen.contains(&kind) {
                    return Err(invalid());
                }
                check_ancillary(kind, payload)?;
                ancillary_seen.push(kind);
            }
            _ => return Err(unsupported()),
        }
    }
    Err(unsupported())
}

fn check_ancillary(kind: &[u8], payload: &[u8]) -> Result<()> {
    let valid = match kind {
        b"sRGB" => payload.len() == 1 && payload[0] <= 3,
        b"gAMA" => payload.len() == 4 && payload != [0, 0, 0, 0],
        b"cHRM" => payload.len() == 32,
        b"pHYs" => payload.len() == 9 && payload[8] <= 1,
        b"tIME" => {
            payload.len() == 7
                && (1..=12).contains(&payload[2])
                && (1..=31).contains(&payload[3])
                && payload[4] <= 23
                && payload[5] <= 59
                && payload[6] <= 60
        }
        b"eXIf" => payload.len() >= 8 && (&payload[..4] == b"II*\0" || &payload[..4] == b"MM\0*"),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid())
    }
}

fn check_pixel_stream(compressed: &[u8], expected_bytes: usize) -> Result<()> {
    let mut decoder = flate2::Decompress::new(true);
    let mut output = vec![0; expected_bytes + 1];
    loop {
        let before = (decoder.total_in(), decoder.total_out());
        let status = decoder
            .decompress(
                &compressed[before.0 as usize..],
                &mut output[before.1 as usize..],
                flate2::FlushDecompress::Finish,
            )
            .map_err(|_| invalid())?;
        let after = (decoder.total_in(), decoder.total_out());
        if status == flate2::Status::StreamEnd {
            return if after == (compressed.len() as u64, expected_bytes as u64) {
                Ok(())
            } else {
                Err(invalid())
            };
        }
        if before == after || after.1 > expected_bytes as u64 {
            return Err(invalid());
        }
    }
}

pub struct MediaBatch {
    images: Vec<OwnedPng>,
}

impl MediaBatch {
    pub fn from_descriptors(value: &Value) -> Result<Self> {
        let values = value.as_array().ok_or_else(invalid)?;
        if values.len() > MAX_IMAGES {
            return Err(unsupported());
        }
        let mut images = Vec::with_capacity(values.len());
        let mut total = 0;
        for value in values {
            let image = OwnedPng::from_descriptor(value)?;
            total += image.bytes.len();
            if total > MAX_BATCH_BYTES {
                return Err(unsupported());
            }
            images.push(image);
        }
        Ok(Self { images })
    }
    pub fn images(&self) -> &[OwnedPng] {
        &self.images
    }
}

#[cfg(test)]
mod tests;

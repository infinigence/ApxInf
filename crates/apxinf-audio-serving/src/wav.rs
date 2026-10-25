use crate::contracts::{MAX_OUTPUT, MAX_UPLOAD};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub frames: u64,
    pub data_offset: usize,
    pub data_bytes: usize,
}

pub fn inspect(bytes: &[u8], asr: bool) -> Result<WavInfo, String> {
    let cap = if asr { MAX_UPLOAD } else { MAX_OUTPUT };
    if bytes.len() < 44 || bytes.len() > cap || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("invalid WAV container".into());
    }
    let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    if u32_at(4) as usize != bytes.len() - 8 {
        return Err("WAV length mismatch".into());
    }
    let mut offset = 12usize;
    let mut format = None;
    let mut data = None;
    let mut metadata = 0usize;
    let mut chunks = 0;
    while offset < bytes.len() {
        chunks += 1;
        if chunks > 32 || offset + 8 > bytes.len() {
            return Err("WAV chunk limit".into());
        }
        let id = &bytes[offset..offset + 4];
        let size = u32_at(offset + 4) as usize;
        let start = offset + 8;
        let end = start.checked_add(size).ok_or("WAV size overflow")?;
        let padded = end.checked_add(size % 2).ok_or("WAV padding overflow")?;
        if padded > bytes.len() || (size % 2 == 1 && bytes[end] != 0) {
            return Err("invalid WAV chunk length or padding".into());
        }
        if id == b"fmt " {
            if format.is_some() || size != 16 {
                return Err("WAV fmt must occur once with 16 bytes".into());
            }
            let u16_at = |i| u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap());
            let rate = u32_at(start + 4);
            if u16_at(start) != 1
                || u16_at(start + 2) != 1
                || u16_at(start + 14) != 16
                || u16_at(start + 12) != 2
                || !(8000..=48000).contains(&rate)
                || u32_at(start + 8) != rate * 2
                || (asr && rate != 16000)
            {
                return Err("unsupported WAV format".into());
            }
            format = Some(rate);
        } else if id == b"data" {
            if data.is_some() || size == 0 || size % 2 != 0 {
                return Err("invalid WAV data".into());
            }
            data = Some((start, size));
        } else {
            metadata = metadata
                .checked_add(size + 8 + size % 2)
                .ok_or("WAV metadata overflow")?;
            if metadata > 65536 {
                return Err("WAV metadata limit".into());
            }
        }
        offset = padded;
    }
    let rate = format.ok_or("missing WAV fmt")?;
    let (start, size) = data.ok_or("missing WAV data")?;
    let frames = size as u64 / 2;
    if frames > rate as u64 * if asr { 30 } else { 60 } {
        return Err("WAV duration limit".into());
    }
    Ok(WavInfo {
        sample_rate: rate,
        frames,
        data_offset: start,
        data_bytes: size,
    })
}

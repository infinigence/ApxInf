//! MILBlob storage version 2, as coremltools 9.0 `StorageWriter` writes it
//! (BSD-3-Clause; see NOTICE): a 64-byte header, then for each blob 64 bytes
//! of metadata at a 64-byte boundary followed by the data. A program's
//! `blobFileValue.offset` is the metadata offset.

const ALIGNMENT: usize = 64;
const SENTINEL: u32 = 0xDEAD_BEEF;

/// `MILBlob::Blob::BlobDataType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum BlobDataType {
    Float16 = 1,
    Int8 = 4,
}

#[derive(Default)]
pub struct BlobWriter {
    bytes: Vec<u8>,
    count: u32,
}

impl BlobWriter {
    pub fn new() -> Self {
        Self {
            bytes: vec![0; ALIGNMENT],
            count: 0,
        }
    }

    /// Appends one blob and returns its metadata offset.
    pub fn append(&mut self, dtype: BlobDataType, data: &[u8]) -> u64 {
        let metadata = self.bytes.len().next_multiple_of(ALIGNMENT);
        self.bytes.resize(metadata, 0);
        let mut record = [0u8; ALIGNMENT];
        record[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        record[4..8].copy_from_slice(&(dtype as u32).to_le_bytes());
        record[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
        record[16..24].copy_from_slice(&((metadata + ALIGNMENT) as u64).to_le_bytes());
        self.bytes.extend_from_slice(&record);
        self.bytes.extend_from_slice(data);
        self.count += 1;
        metadata as u64
    }

    /// The file: the header with the blob count and version 2, then the blobs.
    pub fn finish(mut self) -> Vec<u8> {
        self.bytes[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        self.bytes
    }
}

/// Reads a blob written by [`BlobWriter`] or coremltools: its dtype and data.
pub fn read(file: &[u8], metadata: u64) -> Option<(u32, &[u8])> {
    let at = usize::try_from(metadata).ok()?;
    let record = file.get(at..at.checked_add(ALIGNMENT)?)?;
    let word = |i: usize| u64::from_le_bytes(record[i..i + 8].try_into().unwrap());
    if u32::from_le_bytes(record[0..4].try_into().unwrap()) != SENTINEL {
        return None;
    }
    let dtype = u32::from_le_bytes(record[4..8].try_into().unwrap());
    let (size, offset) = (
        usize::try_from(word(8)).ok()?,
        usize::try_from(word(16)).ok()?,
    );
    Some((dtype, file.get(offset..offset.checked_add(size)?)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blobs_are_aligned_and_readable() {
        let mut writer = BlobWriter::new();
        let first = writer.append(BlobDataType::Int8, &[1, 2, 3]);
        let second = writer.append(BlobDataType::Float16, &[0, 0x3c]);
        let file = writer.finish();
        assert_eq!((first, second), (64, 192));
        assert_eq!(&file[0..8], &[2, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(file.len(), 192 + 64 + 2);
        assert_eq!(read(&file, first), Some((4, &[1u8, 2, 3][..])));
        assert_eq!(read(&file, second), Some((1, &[0u8, 0x3c][..])));
        assert_eq!(read(&file, 0), None);
        assert_eq!(read(&file, 1 << 40), None);
    }
}

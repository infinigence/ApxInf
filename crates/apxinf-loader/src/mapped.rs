//! A read-only memory map of one file, for callers that stream large checkpoints without copying them into the heap.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

/// A whole file mapped read-only. The bytes are the page cache itself: nothing is copied or read until a page is touched.
///
/// As with every memory map, the file must not be truncated or rewritten while the map is alive; checkpoints are
/// treated as immutable inputs, the same assumption the safetensors loader makes.
#[derive(Debug)]
pub struct MappedFile {
    map: Mmap,
}

impl MappedFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file =
            File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        // SAFETY: read-only map of an input file that is not modified while mapped (see the type documentation).
        let map = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("failed to map {}: {e}", path.display()))?;
        Ok(Self { map })
    }
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        &self.map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_the_file_bytes_and_reports_a_missing_file() {
        let path = std::env::temp_dir().join(format!("apxinf-mapped-{}", std::process::id()));
        std::fs::write(&path, b"abcdef").unwrap();
        let mapped = MappedFile::open(&path).unwrap();
        assert_eq!(mapped.as_ref(), b"abcdef");
        std::fs::remove_file(&path).unwrap();
        assert!(MappedFile::open(&path)
            .err()
            .unwrap()
            .contains("failed to open"));
    }
}

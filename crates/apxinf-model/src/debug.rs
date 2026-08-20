//! Debug utilities for capturing activations during model forward pass.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

/// Configuration for activation capture during debugging.
#[derive(Clone, Debug, Default)]
pub struct DebugConfig {
    /// Enable debug mode
    pub enabled: bool,
    /// Output file path for .npz
    pub output_path: Option<PathBuf>,
    /// Filter patterns (glob-style). Empty = capture all.
    pub filters: Vec<String>,
    /// Capture positions (empty = all positions)
    pub positions: Vec<usize>,
}

impl DebugConfig {
    /// Create a new debug config.
    pub fn new(output_path: PathBuf) -> Self {
        Self {
            enabled: true,
            output_path: Some(output_path),
            filters: Vec::new(),
            positions: Vec::new(),
        }
    }

    /// Returns true if this activation should be captured.
    pub fn should_capture(&self, name: &str, pos: usize) -> bool {
        if !self.enabled {
            return false;
        }

        // Check position filter
        if !self.positions.is_empty() && !self.positions.contains(&pos) {
            return false;
        }

        // Check name filter (empty = match all)
        if self.filters.is_empty() {
            return true;
        }

        self.filters.iter().any(|pattern| glob_match(pattern, name))
    }

    /// Returns list of all possible activation names for a model with n_layers.
    pub fn list_all_activations(n_layers: usize) -> Vec<String> {
        let mut names = Vec::new();

        // Embedding
        names.push("embed.token".to_string());

        // Per-layer activations
        for i in 0..n_layers {
            let prefix = format!("layer.{}", i);
            names.push(format!("{}.norm_attn.input", prefix));
            names.push(format!("{}.norm_attn.output", prefix));
            names.push(format!("{}.attn.q", prefix));
            names.push(format!("{}.attn.k", prefix));
            names.push(format!("{}.attn.v", prefix));
            names.push(format!("{}.attn.q_rope", prefix));
            names.push(format!("{}.attn.k_rope", prefix));
            names.push(format!("{}.attn.output", prefix));
            names.push(format!("{}.attn.proj_output", prefix));
            names.push(format!("{}.residual_attn", prefix));
            names.push(format!("{}.norm_ffn.input", prefix));
            names.push(format!("{}.norm_ffn.output", prefix));
            names.push(format!("{}.ffn.gate", prefix));
            names.push(format!("{}.ffn.up", prefix));
            names.push(format!("{}.ffn.gated", prefix));
            names.push(format!("{}.ffn.output", prefix));
            names.push(format!("{}.residual_ffn", prefix));
        }

        // Final
        names.push("final.norm.input".to_string());
        names.push("final.norm.output".to_string());
        names.push("final.logits".to_string());

        names
    }
}

/// Captures and stores activation tensors during forward pass.
pub struct DebugCapture {
    config: DebugConfig,
    /// Captured activations: name -> (data, shape)
    activations: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// Current position being processed
    current_position: usize,
}

impl DebugCapture {
    /// Create a new debug capture.
    pub fn new(config: DebugConfig) -> Self {
        Self {
            config,
            activations: HashMap::new(),
            current_position: 0,
        }
    }

    /// Set the current position (called at start of forward pass).
    pub fn set_position(&mut self, pos: usize) {
        self.current_position = pos;
    }

    /// Capture an activation tensor if it passes the filter.
    pub fn capture(&mut self, name: &str, data: &[f32], shape: &[usize]) {
        if self.config.should_capture(name, self.current_position) {
            let full_name = if self.config.positions.len() > 1 {
                format!("{}.pos{}", name, self.current_position)
            } else {
                name.to_string()
            };

            self.activations
                .insert(full_name, (data.to_vec(), shape.to_vec()));
        }
    }

    /// Save captured activations to an NPZ file.
    ///
    /// NPZ is a ZIP file containing NPY files. NPY format:
    /// - Magic: "\x93NUMPY"
    /// - Version: 1.0 (two bytes)
    /// - Header length (2 bytes, little endian)
    /// - Header: Python dict with 'descr' and 'shape'
    /// - Data: raw bytes
    pub fn save(&self, path: &PathBuf) -> std::io::Result<()> {
        // Create a simple ZIP file manually
        let mut file = std::fs::File::create(path)?;

        // Local file header signature
        const LOCAL_FILE_HEADER_SIG: u32 = 0x04034b50;
        // Central directory header signature
        const CENTRAL_DIR_HEADER_SIG: u32 = 0x02014b50;
        // End of central directory signature
        const END_OF_CENTRAL_DIR_SIG: u32 = 0x06054b50;

        let mut central_dir = Vec::new();
        let mut offset = 0u32;

        for (name, (data, shape)) in &self.activations {
            // Build NPY content
            let npy_content = build_npy(data, shape);

            // Local file header
            let filename = format!("{}.npy", name);
            let filename_bytes = filename.as_bytes();

            // Write local file header
            file.write_all(&LOCAL_FILE_HEADER_SIG.to_le_bytes())?; // signature
            file.write_all(&20u16.to_le_bytes())?; // version needed (2.0)
            file.write_all(&0u16.to_le_bytes())?; // general purpose bit flag
            file.write_all(&0u16.to_le_bytes())?; // compression method (stored)
            file.write_all(&0u16.to_le_bytes())?; // last mod time
            file.write_all(&0u16.to_le_bytes())?; // last mod date
            file.write_all(&crc32(&npy_content).to_le_bytes())?; // CRC-32
            file.write_all(&(npy_content.len() as u32).to_le_bytes())?; // compressed size
            file.write_all(&(npy_content.len() as u32).to_le_bytes())?; // uncompressed size
            file.write_all(&(filename_bytes.len() as u16).to_le_bytes())?; // filename length
            file.write_all(&0u16.to_le_bytes())?; // extra field length
            file.write_all(filename_bytes)?; // filename
            file.write_all(&npy_content)?; // file data

            // Save central dir entry info
            let local_header_size = 30 + filename_bytes.len() + npy_content.len();
            central_dir.extend_from_slice(&CENTRAL_DIR_HEADER_SIG.to_le_bytes());
            central_dir.extend_from_slice(&20u16.to_le_bytes()); // version made by
            central_dir.extend_from_slice(&20u16.to_le_bytes()); // version needed
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // bit flag
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // compression
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // mod time
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // mod date
            central_dir.extend_from_slice(&crc32(&npy_content).to_le_bytes()); // CRC-32
            central_dir.extend_from_slice(&(npy_content.len() as u32).to_le_bytes()); // compressed size
            central_dir.extend_from_slice(&(npy_content.len() as u32).to_le_bytes()); // uncompressed size
            central_dir.extend_from_slice(&(filename_bytes.len() as u16).to_le_bytes()); // filename length
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // extra field length
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // comment length
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // disk number start
            central_dir.extend_from_slice(&0u16.to_le_bytes()); // internal file attributes
            central_dir.extend_from_slice(&0u32.to_le_bytes()); // external file attributes
            central_dir.extend_from_slice(&offset.to_le_bytes()); // relative offset of local header
            central_dir.extend_from_slice(filename_bytes);

            offset += local_header_size as u32;
        }

        let central_dir_offset = offset;
        let central_dir_size = central_dir.len() as u32;

        // Write central directory
        file.write_all(&central_dir)?;

        // End of central directory record
        file.write_all(&END_OF_CENTRAL_DIR_SIG.to_le_bytes())?;
        file.write_all(&0u16.to_le_bytes())?; // disk number
        file.write_all(&0u16.to_le_bytes())?; // disk with central dir
        file.write_all(&(self.activations.len() as u16).to_le_bytes())?; // entries on disk
        file.write_all(&(self.activations.len() as u16).to_le_bytes())?; // total entries
        file.write_all(&central_dir_size.to_le_bytes())?; // central dir size
        file.write_all(&central_dir_offset.to_le_bytes())?; // central dir offset
        file.write_all(&0u16.to_le_bytes())?; // comment length

        Ok(())
    }

    /// Get number of captured activations.
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Check if any activations were captured.
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }
}

/// Build NPY file content for f32 array.
fn build_npy(data: &[f32], shape: &[usize]) -> Vec<u8> {
    let mut out = Vec::new();

    // Magic number
    out.extend_from_slice(b"\x93NUMPY");

    // Version 1.0
    out.push(1);
    out.push(0);

    // Build header dict
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );

    // Header must be padded to make total length divisible by 64 for alignment
    // Header length includes the padding and terminating newline
    // Total prefix: 10 bytes (magic + version + header_len)
    // We need: (10 + header_len) % 64 == 0
    let base_header_len = header.len() + 1; // +1 for newline
    let total_prefix = 10 + base_header_len;
    let padding_needed = (64 - (total_prefix % 64)) % 64;
    let padded_header_len = base_header_len + padding_needed;

    // Write header length (little endian u16)
    out.extend_from_slice(&(padded_header_len as u16).to_le_bytes());

    // Write header with padding
    out.extend_from_slice(header.as_bytes());
    // Padding spaces
    for _ in 0..padding_needed {
        out.push(' ' as u8);
    }
    // Newline terminator
    out.push('\n' as u8);

    // Write data as f32 (little endian)
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }

    out
}

/// Simple CRC-32 implementation.
fn crc32(data: &[u8]) -> u32 {
    // CRC-32 table (IEEE polynomial)
    const CRC_TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };

    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc = CRC_TABLE[(crc as usize ^ byte as usize) & 0xFF] ^ (crc >> 8);
    }
    crc ^ 0xFFFFFFFF
}

/// Simple glob-style pattern matching.
/// Supports:
/// - `*` matches any sequence of characters
/// - `?` matches any single character
/// - literal characters match themselves
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();

    fn match_helper(pattern: &[char], text: &[char]) -> bool {
        match (pattern.first(), text.first()) {
            (None, None) => true,
            (Some('*'), _) => {
                // Try matching * with zero chars, or with one+ chars
                match_helper(&pattern[1..], text)
                    || (!text.is_empty() && match_helper(pattern, &text[1..]))
            }
            (Some('?'), Some(_)) => match_helper(&pattern[1..], &text[1..]),
            (Some(p), Some(t)) if *p == *t => match_helper(&pattern[1..], &text[1..]),
            (Some(_), None) => {
                // Pattern has more chars but text is done - only ok if remaining is all *
                pattern.iter().all(|c| *c == '*')
            }
            _ => false,
        }
    }

    match_helper(&pattern, &text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(glob_match("layer.*.attn.q", "layer.0.attn.q"));
        assert!(glob_match("layer.*.attn.q", "layer.21.attn.q"));
        assert!(glob_match("layer.0.*", "layer.0.attn.q"));
        assert!(glob_match("*.attn.*", "layer.5.attn.q"));
        assert!(!glob_match("layer.0.*", "layer.1.attn.q"));
        assert!(glob_match("final.*", "final.logits"));
    }

    #[test]
    fn test_should_capture() {
        let config = DebugConfig {
            enabled: true,
            output_path: Some(PathBuf::from("test.npz")),
            filters: vec!["layer.*.attn.q".to_string()],
            positions: vec![0],
        };

        assert!(config.should_capture("layer.0.attn.q", 0));
        assert!(config.should_capture("layer.21.attn.q", 0));
        assert!(!config.should_capture("layer.0.attn.k", 0));
        assert!(!config.should_capture("layer.0.attn.q", 1)); // wrong position
    }

    #[test]
    fn test_list_all_activations() {
        let names = DebugConfig::list_all_activations(2);
        assert!(names.contains(&"embed.token".to_string()));
        assert!(names.contains(&"layer.0.attn.q".to_string()));
        assert!(names.contains(&"layer.1.attn.q".to_string()));
        assert!(names.contains(&"final.logits".to_string()));
        assert_eq!(names.len(), 1 + 2 * 17 + 3); // 1 embed + 17 per layer + 3 final
    }

    #[test]
    fn test_crc32() {
        // Known test vector
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }
}

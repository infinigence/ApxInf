//! Helpers shared by the device-gated operator tests.
//!
//! Compiled only under `cfg(test)`. These exist because every native operator
//! test needs the same four things — encode an f32 to bf16 bits, read it back,
//! put bytes on the device, and take them off — and copying them per module
//! would mean four places to fix when the encoding changes.
//!
//! Encoding is round-to-nearest-even, and the decoders used by the references
//! below consume the same bits, so a test measures the operator and not the
//! conversion. The alternative — `apxinf_core::bf16` — would work for building
//! a CPU tensor, but the device path here goes through raw bytes either way.

use apxinf_core::{DType, Shape, Tensor};

use crate::buffer::MusaBuffer;
use crate::context::MusaContext;

/// Round-to-nearest-even to bf16, returned as the 16 bits.
pub fn to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

pub fn from_bf16_bits(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Deterministic values in [-1, 1), so a failing shape can be re-run.
pub fn values(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xffff) as f32 / 32768.0 - 1.0
        })
        .collect()
}

pub fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| to_bf16_bits(*v).to_le_bytes())
        .collect()
}

/// Put `values` on the device as a BF16 tensor of `shape`.
pub fn upload(ctx: &MusaContext, values: &[f32], shape: Shape) -> Tensor {
    let bytes = bf16_bytes(values);
    assert_eq!(bytes.len(), shape.numel() * 2);
    let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate operand");
    buffer.copy_from_host(&bytes).expect("upload operand");
    buffer.as_tensor(shape, DType::BF16).expect("view as bf16")
}

/// A same-shaped tensor of another dtype, for the tests that check a wrong
/// operand is *refused* rather than reinterpreted.
///
/// Zeroed on purpose: the call under test must not reach a kernel, so a test
/// that starts passing because something ran is not a test that passed.
pub fn zeros_as(ctx: &MusaContext, shape: Shape, dtype: DType) -> Tensor {
    let bytes = vec![0u8; shape.numel() * dtype.size_in_bytes()];
    let buffer = MusaBuffer::alloc_on(ctx, bytes.len()).expect("allocate operand");
    buffer.copy_from_host(&bytes).expect("upload operand");
    buffer
        .as_tensor(shape, dtype)
        .expect("view as the requested dtype")
}

pub fn read_back(tensor: &Tensor) -> Vec<f32> {
    let buffer = MusaBuffer::from_tensor(tensor).expect("device tensor");
    let mut bytes = vec![0u8; tensor.size_in_bytes()];
    buffer.copy_to_host(&mut bytes).expect("read back");
    bytes
        .chunks_exact(2)
        .map(|pair| from_bf16_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect()
}

/// The gate every operator test here uses: the largest absolute error must stay
/// under 1% of the output's own largest magnitude.
///
/// Absolute rather than relative, because these outputs are sums of many terms
/// that cancel — a relative gate reports the f32 accumulation floor as if it
/// were kernel error on the elements that land near zero. Same criterion as
/// the replay harness uses, and the muBLAS heuristic probe before it.
pub const GATE_FRACTION: f32 = 1e-2;

pub fn gate_for(output: &[f32]) -> f32 {
    output
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        * GATE_FRACTION
}

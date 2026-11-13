//! The concrete compute blocks of the GR00T HIP port: vision tower, Qwen3-VL
//! text stack, VLLN + VL self-attention, and the DiT action head. BF16 is the
//! only precision, so the BF16 blocks are the blocks.
//!
//! These modules copy, never share, the reference implementation's math; they
//! import only the backend's semantic operator layer.

pub mod action_head;
pub mod backbone;
pub mod dit_source;
pub mod qkv;
pub mod vision;

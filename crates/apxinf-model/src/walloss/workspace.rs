//! Shape/config-derived resource budget, independent of execution policy.

use apxinf_core::{Error, Result};

use super::WallossConfig;

fn product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |total, &value| total.checked_mul(value))
        .ok_or_else(overflow)
}

fn sum(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(0usize, |total, &value| total.checked_add(value))
        .ok_or_else(overflow)
}

fn overflow() -> Error {
    Error::Other("walloss workspace size overflows usize".into())
}

/// Conservative allocation budget for the BF16/FP8 forward implementations.
/// Lease-backed workspace slices are reused after the last tensor owner drops,
/// so scratch scales with simultaneously live blocks rather than every layer
/// or denoising step. KV persists across layers and steps. Keep 50% headroom
/// for alignment, allocator fragmentation, and shape-dependent overlap.
pub(super) fn workspace_bytes(config: &WallossConfig, rows: usize, prefix: usize) -> Result<usize> {
    let text = &config.text;
    let vision = &config.vision;
    let action = &config.action;
    let q = product(&[text.num_attention_heads, text.head_dim])?;
    let kv = product(&[text.num_kv_heads, text.head_dim])?;
    let cache = product(&[
        2,
        2,
        text.num_layers,
        kv,
        sum(&[prefix, action.action_horizon])?,
    ])?;
    // Residuals, normalized/projected hidden, QKV and attention coexist with
    // the BF16 gate/up (2*inner) and activated (inner) tensors in residual_mlp.
    let language = product(&[
        2,
        prefix,
        sum(&[
            product(&[8, text.hidden_size])?,
            product(&[3, text.intermediate_size])?,
            q,
            product(&[2, kv])?,
        ])?,
    ])?;
    let padded_inner = product(&[sum(&[vision.intermediate_size, 63])? / 64, 64])?;
    let patch_width = product(&[
        3,
        vision.temporal_patch_size,
        vision.patch_size,
        vision.patch_size,
    ])?;
    let vision_scratch = product(&[
        2,
        rows,
        sum(&[
            product(&[16, vision.hidden_size])?,
            product(&[6, padded_inner])?,
            patch_width,
            product(&[4, vision.out_hidden_size])?,
        ])?,
    ])?;
    let action_scratch = product(&[
        2,
        action.action_horizon,
        sum(&[
            product(&[12, action.hidden_size])?,
            product(&[4, action.intermediate_size])?,
            product(&[4, q])?,
            product(&[4, kv])?,
            product(&[4, action.action_dim])?,
            action.state_hidden_size,
        ])?,
    ])?;
    // Summing phase budgets retains vision outputs during language and prefix
    // KV during actions. Fixed reserve covers alignment and small scratch.
    let bytes = sum(&[16 << 20, cache, language, vision_scratch, action_scratch])?;
    sum(&[bytes, bytes / 2])
}

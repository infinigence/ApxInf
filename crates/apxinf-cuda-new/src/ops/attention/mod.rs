mod attention;
pub(crate) mod contracts;
pub(crate) mod execution;
mod kv_cache_attention;
mod segmented_attention;

pub use attention::attention;
pub use contracts::{AttentionArgs, AttentionMask, AttentionPolicy};
pub use kv_cache_attention::{kv_cache_attention, KvCacheAttentionArgs, KvCacheDecodeMeta};
pub use segmented_attention::{segmented_attention, SegmentedAttentionArgs};

#[cfg(test)]
pub(crate) use kv_cache_attention::normalize as normalize_kv_cache_attention;
#[cfg(test)]
pub(crate) use segmented_attention::normalize as normalize_segmented_attention;

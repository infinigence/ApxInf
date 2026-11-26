#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
use apxinf_core::Result;
use apxinf_mlx::{
    clear_cache, memory_stats, set_cache_limit, Array, MlxDType, Stream, DEFAULT_CACHE_LIMIT_BYTES,
};

const MIB: usize = 1 << 20;

/// Evaluates and drops one array per size, then returns the cache size. Sizes
/// differ by more than MLX's reuse tolerance, so each freed buffer is a new
/// cache entry. MLX trims the cache to its limit on the next allocation, so
/// the stats are read while one more array is alive.
fn cache_after_distinct_shapes(s: &Stream, total_bytes: usize) -> Result<usize> {
    clear_cache()?;
    let mut allocated = 0;
    let mut bytes = 16 * MIB;
    while allocated < total_bytes {
        Array::zeros(s, &[bytes / 4], MlxDType::F32)?.eval()?;
        allocated += bytes;
        bytes += MIB;
    }
    let live = Array::zeros(s, &[1], MlxDType::F32)?;
    live.eval()?;
    s.synchronize()?;
    Ok(memory_stats()?.cache_bytes)
}

/// Separate integration executable: the allocator cache limit is process-wide.
#[test]
#[ignore = "requires shared Metal qualification lock and isolated process"]
fn metal_stream_bounds_the_cache_unless_the_caller_set_a_limit() -> Result<()> {
    let s = Stream::metal(0)?;
    let cache = cache_after_distinct_shapes(&s, 2 * DEFAULT_CACHE_LIMIT_BYTES)?;
    assert!(
        cache <= DEFAULT_CACHE_LIMIT_BYTES,
        "default limit not applied: {cache} cached bytes"
    );

    let caller_limit = 64 * MIB;
    set_cache_limit(caller_limit)?;
    let _later = Stream::metal(0)?;
    let cache = cache_after_distinct_shapes(&s, 2 * DEFAULT_CACHE_LIMIT_BYTES)?;
    assert!(
        cache <= caller_limit,
        "a later stream replaced the caller's limit: {cache} cached bytes"
    );
    Ok(())
}

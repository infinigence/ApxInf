#![cfg(feature = "native")]
use apxinf_core::Result;
use apxinf_mlx::{clear_cache, memory_stats, reset_peak_memory, Array, Compiled, Stream};

/// Separate integration executable: allocator observations are process-wide.
#[test]
#[ignore = "requires shared Metal qualification lock and isolated process"]
fn compiled_instance_drop_releases_active_weights_and_bounds_pool() -> Result<()> {
    let s = Stream::metal(0)?;
    let run = |value: f32| -> Result<()> {
        let weight = Array::from_f32(&s, &[128, 128], &vec![value; 128 * 128])?;
        let compiled = Compiled::new(&s, 1, move |a| Ok(vec![a[0].matmul(&weight)?]))?;
        let x = Array::from_f32(&s, &[1, 128], &[1.; 128])?;
        let output = compiled.call_and_eval(&[x])?;
        assert_eq!(output[0].slice_axis(1, 0, 1)?.to_f32_vec()?, [128. * value]);
        Ok(())
    };
    run(1.)?;
    s.synchronize()?;
    clear_cache()?;
    let baseline = memory_stats()?;
    reset_peak_memory()?;
    let mut pool = Vec::new();
    for i in 1..=24 {
        run(i as f32)?;
        s.synchronize()?;
        let stats = memory_stats()?;
        assert_eq!(
            stats.active_bytes, baseline.active_bytes,
            "retained arrays after instance {i}: {stats:?}"
        );
        pool.push(stats.cache_bytes);
    }
    // A fixed shape repeats allocations; the second half must fit in the
    // first-half observed pool, rather than growing with compiled instances.
    assert!(
        pool[12..].iter().max() <= pool[..12].iter().max(),
        "pool grows: {pool:?}"
    );
    let before_clear = memory_stats()?;
    clear_cache()?;
    let after_clear = memory_stats()?;
    assert_eq!(after_clear.active_bytes, baseline.active_bytes);
    assert_eq!(after_clear.cache_bytes, 0);
    assert!(before_clear.peak_bytes >= before_clear.active_bytes);
    eprintln!("baseline={baseline:?} before_clear={before_clear:?} after_clear={after_clear:?} pool={pool:?}");
    Ok(())
}

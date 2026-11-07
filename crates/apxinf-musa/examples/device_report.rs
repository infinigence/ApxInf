//! Print the MUSA device facts this build targets.
//!
//! The backend's own smoke test: it proves the toolkit was found at build
//! time, that the crate links `musart`, and that device queries answer.
//!
//! `cargo run -p apxinf-musa --example device_report -- [device_index]`
//!
//! The index is the same one `--device musa:N` takes, so `device_report 1`
//! answers "would this build serve card 1?" without loading a model. The
//! binary serves one architecture, so a card of another generation is refused
//! by name here too.

use apxinf_musa::{MusaBuffer, MusaContext, D256_ROUTE, KERNEL_BUILD_ID, MUSA_ARCH, MUSA_AVAILABLE};

fn main() -> Result<(), String> {
    println!(
        "apxinf-musa: toolkit={} arch={MUSA_ARCH} kernel_build_id={KERNEL_BUILD_ID}",
        if MUSA_AVAILABLE { "found" } else { "absent" }
    );
    if !MUSA_AVAILABLE {
        return Err("built without a MUSA toolkit — nothing to report".into());
    }

    let device_id: usize = match std::env::args().nth(1) {
        Some(value) => value
            .parse()
            .map_err(|_| format!("device index `{value}` is not a number"))?,
        None => 0,
    };
    let ctx = MusaContext::new(device_id).map_err(|error| error.to_string())?;
    let caps = ctx.caps();
    println!("device {}: {}", ctx.device_id(), caps.device_name);
    println!(
        "  compute capability  {}.{}   (mp_{}, --offload-arch={})",
        caps.compute_major,
        caps.compute_minor,
        caps.mp,
        caps.offload_arch()
    );
    println!("  arch family         {}", caps.arch_family);
    println!("  multiprocessors     {}", caps.multiprocessor_count);
    println!(
        "  hardware warp size  {}   (shuffle groups are 32 lanes regardless)",
        caps.hardware_warp_size
    );
    println!("  D256 attention      {D256_ROUTE}");

    // Not just queryable: usable. One allocation, a host round trip through the
    // context's stream, and a byte-for-byte comparison. Capture itself has its
    // own entry point (`graph::capture`); this example stops at the round trip.
    let pattern: Vec<u8> = (0..=255).collect();
    let buffer = MusaBuffer::alloc_on(&ctx, pattern.len())?;
    buffer.copy_from_host(&pattern)?;
    ctx.synchronize().map_err(|error| error.to_string())?;
    let mut readback = vec![0u8; pattern.len()];
    buffer.copy_to_host(&mut readback)?;
    if readback != pattern {
        return Err("alloc/copy round trip returned different bytes".into());
    }
    println!("  alloc/copy          ok ({} bytes round-tripped)", pattern.len());
    Ok(())
}

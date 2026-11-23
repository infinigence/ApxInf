//! Pins the D256 routing rule to what the build script published.
//!
//! There is one function that decides which D256 attention kernel a build runs:
//! `build_support::musa_arch::d256_route`. `build.rs` asks it what to compile
//! and publishes the answer as `APXINF_MUSA_D256_ROUTE`; `kernels::attention`
//! reads that string. There is no second reading of the rule to hold in step —
//! what is left to check is that the published value is the one the rule
//! produces for *this* build's inputs.
//!
//! The failure this prevents is not a wrong message. `Mutlass` is matched by
//! string, so a drift would make it match nothing, and mp_31's only attention
//! would become a named gap with nothing to point at.
//!
//! `build_support/musa_arch.rs` is included by path rather than depended on:
//! it is not part of the crate's public API, and the CUDA crate's
//! `tests/cuda_arch.rs` sets the same precedent. Its own `use crate::mcc` is
//! satisfied by including `mcc` beside it, exactly as `build.rs` does.

// The included modules are the build script's, so most of what they carry is
// for a caller this test is not: toolchain discovery, the arch probe, the
// scratch-compile. Only the routing rule and the two constants are asserted on,
// and dead-code warnings for the rest would be a permanent floor of noise.
#![allow(dead_code)]

#[path = "../build_support/mcc.rs"]
mod mcc;

#[path = "../build_support/musa_arch.rs"]
mod build_side;

use build_side::D256Route;

/// The generations the build script gates on are the ones the library compares
/// against. These are read from `build.rs` through `rustc-env`, so a build that
/// compiled one generation's kernel while the launcher asked about another's is
/// the state this rules out.
#[test]
fn the_published_arches_are_the_build_scripts() {
    assert_eq!(build_side::ASSEMBLY_ARCH, apxinf_musa::ASSEMBLY_ARCH);
    assert_eq!(build_side::VENDOR_FMHA_ARCH, apxinf_musa::VENDOR_FMHA_ARCH);
}

/// The route the library reads is the one the rule produces for this build.
///
/// This is the whole of the coupling now: `build.rs` ran `d256_route` over the
/// arch it selected and the vendor gate it computed, and `kernels::attention`
/// branches on whatever came back. Recomputing it here from the two published
/// facts the route is made of — the arch and whether the vendored tree was
/// compiled — closes the loop, and is why the vendor gate crosses the crate
/// boundary as a `const` beside the route.
#[test]
fn the_published_route_is_the_rules_answer_for_this_build() {
    let expected = build_side::d256_route(apxinf_musa::MUSA_ARCH, apxinf_musa::VENDOR_FMHA_BUILT);
    assert_eq!(
        expected.as_str(),
        apxinf_musa::D256_ROUTE,
        "build.rs published a D256 route ({}) that its own rule does not produce for \
         arch={} vendor_fmha_built={}",
        apxinf_musa::D256_ROUTE,
        apxinf_musa::MUSA_ARCH,
        apxinf_musa::VENDOR_FMHA_BUILT
    );
}

/// The build-side rule, over the inputs that matter.
///
/// The `false` column is the point of the vendor argument: mp_31 without the
/// vendored tree is not a route, and a build that forgot to pass the gate would
/// claim one it cannot launch.
#[test]
fn the_build_side_rule_routes_by_arch_and_vendor_gate() {
    let assembly = build_side::ASSEMBLY_ARCH;
    let vendor = build_side::VENDOR_FMHA_ARCH;

    assert_eq!(build_side::d256_route(assembly, false), D256Route::Assembly);
    // The assembly is not gated on the vendored tree, so the argument cannot
    // change its answer.
    assert_eq!(build_side::d256_route(assembly, true), D256Route::Assembly);
    assert_eq!(build_side::d256_route(vendor, true), D256Route::Mutlass);
    assert_eq!(build_side::d256_route(vendor, false), D256Route::Unsupported);
    // A generation neither kernel was built for has neither, whatever the gate.
    assert_eq!(build_side::d256_route("mp_50", true), D256Route::Unsupported);
    assert_eq!(
        build_side::d256_route("mp_50", false),
        D256Route::Unsupported
    );
}

/// The names the build script publishes are the ones the library parses.
///
/// `kernels::attention` maps the string to an enum whose catch-all is
/// `Unsupported`, and `lib.rs` asserts at compile time that only these three can
/// arrive — so an added variant that nobody renamed here would be a `build.rs`
/// that cannot compile, rather than a route that silently became a gap.
#[test]
fn the_routes_the_library_understands_are_the_build_scripts_names() {
    for route in [D256Route::Assembly, D256Route::Mutlass, D256Route::Unsupported] {
        assert!(
            matches!(route.as_str(), "assembly" | "mutlass" | "unsupported"),
            "{route:?} publishes a name lib.rs would reject"
        );
    }
}

/// The family that answers "does this device have muDNN flash MHA" agrees with
/// the pair the dispatch site used to spell out.
#[test]
fn the_family_capability_matches_the_generations_that_have_the_kernel() {
    use apxinf_musa::MusaArchFamily;

    assert!(MusaArchFamily::Mp22.has_vision_flash_mha());
    assert!(MusaArchFamily::Mp31.has_vision_flash_mha());
    // `Other` answering `true` would hide "not measured yet" behind the same
    // answer as "measured and supported", which is the one thing this method
    // exists to keep apart.
    assert!(!MusaArchFamily::Other(50).has_vision_flash_mha());
    assert_eq!(MusaArchFamily::from_mp(22), MusaArchFamily::Mp22);
    assert_eq!(MusaArchFamily::from_mp(31), MusaArchFamily::Mp31);
    assert_eq!(MusaArchFamily::from_mp(50), MusaArchFamily::Other(50));
}

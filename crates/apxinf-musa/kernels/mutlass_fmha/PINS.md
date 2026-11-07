# Pinned public sources for the vendored FMHA tree

The 218 headers under `mate/data/include` are not a private snapshot. They are,
byte for byte, two public trees:

| | repository | revision | subtree | vendored at |
|---|---|---|---|---|
| MATE | `https://github.com/MooreThreads/mate` | tag `v0.2.7` = `e5d73e914fcb30aab915a12fae146d9d26357908` | `include/mate/{attention/fmha,common}` | `mate/data/include/mate/` (22 files) |
| MUTLASS | `https://github.com/MooreThreads/mutlass` | commit `78b349514831ed54a797b717f2fc0534646acff2` | `include/{mutlass,mute}` | `mate/data/mutlass/include/` (196 files) |

`78b34951` is not an arbitrary choice: it is the `3rdparty/mutlass` gitlink that
MATE `v0.2.7` itself pins, so the two revisions are the pair MATE shipped.

Fingerprints over `<sha256>  <path>` lines, sorted:

```
mate/data/include           14666ce707db1b3b5ef14da4c1379c881920fc50dd5f5b9cd60c1d3dcafc4b78  (22 files)
mate/data/mutlass/include   92f22b7e72e03596d08df41ddc7d7a884c4eb9beadc4d2f90bac48d3039bf50d  (196 files)
```

## Checking it

```bash
# two public clones, any checkout state — the script reads objects, never the worktree
git clone https://github.com/MooreThreads/mate.git    <path-to-mate>
git clone https://github.com/MooreThreads/mutlass.git <path-to-mutlass>

MATE_SRC=<path-to-mate> MUTLASS_SRC=<path-to-mutlass> \
  crates/apxinf-musa/kernels/mutlass_fmha/tools/verify_pins.sh
```

It prints per-file differences, the compared/differing/missing counts, the
upstream count for the pinned subtrees and both fingerprints, and exits non-zero
on any mismatch. The counts make the check two-sided: a re-vendor that drops a
file fails, instead of passing as a byte-identical subset. It needs no network
beyond the clones, no toolkit and no device.

**Run the reverse control too**, or the check proves nothing:

```bash
MATE_SRC=… MUTLASS_SRC=… MATE_REV=v0.2.8 MUTLASS_REV=435707c2 \
  crates/apxinf-musa/kernels/mutlass_fmha/tools/verify_pins.sh
```

That must fail, naming exactly `epilogue_fwd.hpp`, `fmha_combine.hpp`,
`fmha_fwd_kernel.hpp`, `metadata.hpp`, `mp31_fmha_fwd_tme_warpspecialized.hpp`
and `mute/pointer_swizzle.hpp`. Pin the tag, not a branch: those six are what a
branch head costs, and the FMHA epilogue contract the two frozen `.mu` files
were rendered against is one of them.

## Re-vendoring

The vendored tree keeps the wheel's layout (`mate/data/include/...`), not the
source layout, and the two are the same content. To re-vendor from public
sources, take `include/` from each repository at the pinned revision into the
matching vendored root, then:

1. `git add -f` the `builders/` directories — a global `**/build*/` ignore rule
   silently drops seven MUTLASS headers; see the parent `README.md`.
2. Re-run `tools/verify_pins.sh` against the revision just taken.
3. `cargo test -p apxinf-musa --test vendor_tree`.

## What is not covered

The two `fmha_fwd/mp31_*.mu` renders are JIT output and appear in no checkout.
They come from the MATE 0.2.7 release wheel — published from the same
tag, and the parent `README.md` records which public template
(`csrc/templates/attention/fmha/fwd_kern.j2`) produced them.

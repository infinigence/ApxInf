#!/usr/bin/env python3
"""Freeze one MATE-rendered mp_31 FMHA kernel into a checkable-in file.

MATE does not ship a binary for this kernel and there is no way to fetch one:
its published mubin map carries head dimensions 128 and 192x128 only, and every
d256 attention on mp_31 is JIT-compiled from a C++ template at first call. So
the port vendors the render and the headers it includes.

This script is the whole of the transform. Given a `.mu` from MATE's cache it
emits the file that is checked in, and the only thing it changes is the host
entry: the includes lose `tvm_ffi` and `op_utils.hpp` (the entry was a tvm_ffi
function and those are the only consumers of the closure), `using namespace tvm`
goes with them, and the render's own `constexpr` config block is copied through
untouched. Everything from `// FMHA config` to `TileScheduler>;` is byte-for-byte
the render's, which is what keeps the device kernel the same one MATE ran.

    tools/freeze_kernel.py <rendered.mu> <launcher-name> <output.mu>

The output is compiled by `crates/apxinf-musa/build.rs` with MATE's own device
flags; see `kernels/mutlass_fmha/README.md` for why they are not the repo's.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

# The template that supplies the host entry, next to this script.
TEMPLATE = pathlib.Path(__file__).with_name("fmha_fwd_host.cpp.in")

# Dropped from the render's include block, and why:
#   <tvm/...>        the render's host entry was a tvm_ffi function
#   "op_utils.hpp"   its only content is tvm_ffi/device-guard glue
DROP_INCLUDE_MARKERS = ("<tvm/", '"op_utils.hpp"')

CONFIG_BEGIN = "  // FMHA config"
CONFIG_END = "    TileScheduler>;"
USING_MUTE = "using namespace mute;"
USING_TVM = "using namespace tvm;"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(rendered: pathlib.Path, launcher: str, out_path: pathlib.Path) -> None:
    text = rendered.read_text()
    lines = text.splitlines()

    def find(needle: str) -> int:
        for index, line in enumerate(lines):
            if line == needle:
                return index
        raise SystemExit(f"{rendered}: no line equal to {needle!r}")

    mute_at = find(USING_MUTE)
    tvm_at = find(USING_TVM)
    config_at = find(CONFIG_BEGIN)

    includes = [
        line
        for line in lines[:mute_at]
        if not any(marker in line for marker in DROP_INCLUDE_MARKERS)
    ]
    # Collapse the runs of blank lines the drops leave behind, and the leading one.
    includes = [line for line in includes if line.strip()]

    if tvm_at != mute_at + 1:
        # The render puts the two `using` directives on consecutive lines. If a
        # later MATE moves them, the include block this script lifts would span
        # the wrong range and the frozen file would quietly lose its includes.
        raise SystemExit(
            f"{rendered}: expected {USING_TVM!r} on the line after {USING_MUTE!r}"
        )

    config_block = lines[config_at : find(CONFIG_END) + 1]

    host = TEMPLATE.read_text().replace("@LAUNCHER@", launcher)
    if host.count("@CONFIG_BLOCK@") != 1:
        # Prose in the template that mentions the marker would be substituted
        # too, and the config block would land in a comment.
        raise SystemExit(
            f"{TEMPLATE}: expected exactly one @CONFIG_BLOCK@, found {host.count('@CONFIG_BLOCK@')}"
        )
    host = host.replace("@CONFIG_BLOCK@", "\n".join(config_block))

    provenance = f"""// Frozen mp_31 D256 FMHA kernel, vendored from MATE.
//
// Do not edit by hand: regenerate with `tools/freeze_kernel.py`. The config
// block below `// FMHA config` is the render's, verbatim; the host entry is
// ours and `tools/fmha_fwd_host.cpp.in` is where it is written.
//
//   rendered from  fmha_fwd_{rendered.stem.split('fmha_fwd_')[-1]}.mu
//   render sha256  {sha256(rendered)}
//   MATE           0.2.7
//
// See kernels/mutlass_fmha/README.md for provenance, the edit list, and the
// launch contract.
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        provenance
        + "\n"
        + "\n".join(includes)
        + "\n\n"
        # The render opens with this and the config block below uses `Shape`,
        # `Int` and `_1` unqualified, so dropping it takes the instantiation with
        # it. `using namespace tvm;`, which the render has next, does not come.
        + USING_MUTE
        + "\n\n"
        + host
    )
    print(f"{out_path}  {sha256(out_path)[:16]}  ({len(out_path.read_text().splitlines())} lines)")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    rendered, launcher, out = pathlib.Path(argv[1]), argv[2], pathlib.Path(argv[3])
    freeze(rendered, launcher, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

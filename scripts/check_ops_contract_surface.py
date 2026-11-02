#!/usr/bin/env python3
"""Check that the typed-operator surface covers what the model tree imports.

The MUSA backend mirrors `apxinf-cuda-new`'s `ops` public surface, but only the
part PI0.5's BF16 path actually names. This script is the check on that claim:
it collects every `backend::ops` name the model tree mentions and compares the
set against what `crates/apxinf-musa/src/ops/` exports.

It deliberately scans the model tree, not `cuda-new/src/ops/mod.rs`: the latter
is the full surface, most of which nothing here reaches, and a comparison
against it would stay quiet while a name the model imports went missing.

`--tree` may be repeated. Once the MUSA device copy exists it has to be
scanned too, because that copy is what the backend actually compiles against.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TREES = (
    "crates/apxinf-model/src/pi05",
    # On a MUSA build the port is what the backend actually compiles against.
    "crates/apxinf-model/src/musa_pi05",
)
# Files whose imports do not describe this backend's surface, keyed by tree.
# `pi05/` is not compiled on MUSA, so its FP8 and INT8 files describe cuda-new's
# contracts, not this crate's; the port's own INT8 files do count.
TREE_EXCLUSIONS = {
    "crates/apxinf-model/src/pi05": ("fp8_static", "int8_dynamic"),
    "crates/apxinf-model/src/musa_pi05": ("fp8_static",),
}
DEFAULT_EXCLUSIONS = ("fp8_static", "int8_dynamic")

# `ops::{ a, b as c }` — the block form PI0.5 uses.
OPS_BLOCK = re.compile(r"ops::\s*\{([^}]*)\}", re.S)
# `ops::Name` anywhere, including the fully-qualified call sites.
OPS_NAME = re.compile(r"\bops::([A-Za-z_][A-Za-z0-9_]*)")
# A `pub use path::{ ... }` block, and bare `pub use path::Name;`.
PUB_USE_BLOCK = re.compile(r"pub use [A-Za-z0-9_:]+::\s*\{([^}]*)\}", re.S)
PUB_USE_NAME = re.compile(r"pub use [A-Za-z0-9_:]+::([A-Za-z_][A-Za-z0-9_]*);")
PUB_FN = re.compile(r"pub fn ([A-Za-z_][A-Za-z0-9_]*)")
# A submodule the model names as a path segment (`ops::w8a8::gemm`). The model
# tree's `ops::Name` scan sees the module, not its contents, so the module
# itself has to be an exported name.
PUB_MOD = re.compile(r"pub mod ([A-Za-z_][A-Za-z0-9_]*);")

RESERVED = {
    # Not names this crate exports; `ops` is also a path segment in the model.
    "mod",
}


def exclusions_for(tree: Path):
    relative = tree.relative_to(ROOT).as_posix()
    return TREE_EXCLUSIONS.get(relative, DEFAULT_EXCLUSIONS)


def rust_files(tree: Path):
    excluded = exclusions_for(tree)
    for path in sorted(tree.rglob("*.rs")):
        if any(name in path.name for name in excluded):
            continue
        yield path


def implied_names(tree: Path):
    names = set()
    for path in rust_files(tree):
        text = path.read_text(encoding="utf-8")
        for block in OPS_BLOCK.findall(text):
            for entry in block.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                name = entry.split(" as ")[0].strip()
                if name and name not in RESERVED:
                    names.add(name)
        for name in OPS_NAME.findall(text):
            if name not in RESERVED:
                names.add(name)
    return names


def exported_names():
    source = (ROOT / "crates/apxinf-musa/src/ops/mod.rs").read_text(encoding="utf-8")
    names = (
        set(PUB_FN.findall(source))
        | set(PUB_USE_NAME.findall(source))
        | set(PUB_MOD.findall(source))
    )
    for block in PUB_USE_BLOCK.findall(source):
        for entry in block.split(","):
            entry = entry.strip()
            if entry:
                # On the export side the alias is the name callers see.
                names.add(entry.split(" as ")[-1].strip())
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree",
        action="append",
        default=None,
        help=f"tree to scan, relative to the repository root (default: {', '.join(DEFAULT_TREES)})",
    )
    args = parser.parse_args()
    trees = [ROOT / tree for tree in (args.tree or DEFAULT_TREES)]
    for tree in trees:
        if not tree.is_dir():
            print(f"missing tree: {tree}", file=sys.stderr)
            return 2

    imported = set()
    for tree in trees:
        imported |= implied_names(tree)
    exported = exported_names()

    missing = sorted(imported - exported)
    if missing:
        print("MUSA ops surface is missing names the model tree imports:")
        for name in missing:
            print(f"  {name}")
        return 1

    print(
        f"ops surface covers {len(imported)} imported names "
        f"({len(exported)} exported)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

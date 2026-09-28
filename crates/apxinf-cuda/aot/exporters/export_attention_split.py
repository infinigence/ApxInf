#!/usr/bin/env python3
"""Export the one tested batch-2 packed-GQA A shape to a native SM110 C ABI.

Static-check imports no CUDA packages. Actual export requires a GPU flock.
"""

import argparse
import hashlib
import json
import os
import site
import subprocess
import sys
import time
import types
from pathlib import Path

REVISION = "d15f1531a460ba456f41b01a774f33ab2db8febf"
OFFICIAL_SHA = "144a3dd6f72f955e43834500808c7d47b3b4a76fdcd0b7188f9b459d85007cab"
GENERAL_SHA = "9d43194751128963a701f0d47b04f68cb6bfd2c8e87e4734d49b33a4cc24d1f2"
SYMBOL = "apxinf_fa4_d256_a_splitbatch_sm110"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fa4-src", type=Path, required=True)
    parser.add_argument("--cutlass-deps", type=Path, required=True)
    parser.add_argument("--extra-deps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--static-check", action="store_true")
    args = parser.parse_args()
    source = args.fa4_src.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checks = {
        "official_interface": (source / "flash_attn/cute/interface.py", OFFICIAL_SHA),
        "general_sm100_forward": (source / "flash_attn/cute/flash_fwd_sm100.py", GENERAL_SHA),
    }
    for label, (path, expected) in checks.items():
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{label} SHA mismatch: {actual} != {expected}")
    # Thor3 stores the pinned source as a copy without .git. Its two kernel
    # file hashes above are still mandatory and the origin revision is pinned.
    revision = (
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if (source / ".git").exists() else REVISION
    )
    if revision != REVISION:
        raise SystemExit(f"unexpected FA4 revision: {revision}")
    metadata = {
        "source_revision": revision,
        "source_has_git_metadata": (source / ".git").exists(),
        "source_sha256": {label: digest for label, (_, digest) in checks.items()},
        "symbol": SYMBOL,
        "shape": {"q": [2, 50, 16, 256], "kv": [2, 1718, 4, 256],
                  "lse": [2, 16, 50], "seqused_k": [2]},
        "config": {
            "dtype": "bf16", "scale": 0.0625, "causal": False,
            "pack_gqa": True, "q_stage": 1, "m_block": 128,
            "n_block": 128, "cta_group": 1, "persistent": False,
            "num_splits": 1,
        },
    }
    if args.static_check:
        print(json.dumps({**metadata, "gate": "CPU source and syntax only"}, indent=2))
        return

    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    site.addsitedir(str(args.cutlass_deps.resolve()))
    site.addsitedir(str(args.extra_deps.resolve()))
    sys.path.insert(0, str(source))

    # Bypass the unrelated legacy FA2 extension imported by flash_attn/__init__.
    package = types.ModuleType("flash_attn")
    package.__path__ = [str(source / "flash_attn")]
    sys.modules["flash_attn"] = package
    import torch
    import cutlass
    if cutlass.__version__ != "4.7.0":
        raise RuntimeError(f"CuTe DSL 4.7.0 required, got {cutlass.__version__}")
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
    from flash_attn.cute.utils import AuxData

    if torch.cuda.get_device_capability() != (11, 0):
        raise SystemExit("SM110 is required")
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    # This is the maintained action configuration, constructed directly rather
    # than selected through or patched into the upstream inference interface.
    forward = FlashAttentionForwardSm100(
        head_dim=256, head_dim_v=256, qhead_per_kvhead=4,
        is_causal=False, is_local=False, is_split_kv=False, pack_gqa=True,
        m_block_size=128, n_block_size=128, q_stage=1,
        is_static_persistent=False,
    )

    @cute.jit
    def native_forward(
        q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
        out_tensor: cute.Tensor, lse: cute.Tensor,
        seqused_k: cute.Tensor, scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        forward(q, k, v, out_tensor, lse, scale, mSeqUsedK=seqused_k,
                aux_data=AuxData(), stream=stream)

    q = torch.empty((2, 50, 16, 256), dtype=torch.bfloat16, device="cuda")
    # Preserve the accepted 1718-row compiler specialization. Runtime tensor
    # descriptors admit 1720-row views with a 1718-row batch stride and masks.
    k = torch.empty((2, 1718, 4, 256), dtype=torch.bfloat16, device="cuda")
    v = torch.empty_like(k)
    output = torch.empty_like(q)
    lse = torch.empty((2, 16, 50), dtype=torch.float32, device="cuda")
    seqused_k = torch.tensor([1718, 1717], dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    options = "--gpu-arch sm_110a --host-target linux-aarch64"
    started = time.monotonic()
    compiled = cute.compile(
        native_forward, *[to_cute_tensor(t) for t in (q, k, v, output)],
        to_cute_tensor(lse, assumed_align=4),
        to_cute_tensor(seqused_k, assumed_align=4, leading_dim=0),
        0.0625, stream, options=options,
    )
    compiled.export_to_c(file_path=str(out), file_name=SYMBOL, function_prefix=SYMBOL)
    export_record = {
        "compile_options_native": options,
        "compile_seconds": time.monotonic() - started,
        "header_sha256": sha256(out / f"{SYMBOL}.h"),
        "object_sha256": sha256(out / f"{SYMBOL}.o"),
    }
    metadata.update(export_record)
    (out / "export-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

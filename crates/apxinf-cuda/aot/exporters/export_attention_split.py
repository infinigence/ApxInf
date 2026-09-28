#!/usr/bin/env python3
"""Export the one tested batch-2 packed-GQA A shape to a native SM110 C ABI.

Static-check imports no CUDA packages. Actual export requires a GPU flock.
"""

import argparse
import hashlib
import importlib.util
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
OVERLAY_SHA = "a98db9434170a72277777eb1f1a8b20f7c2ba3a29b6d9fe3b9843f449202659a"
SYMBOL = "apxinf_fa4_d256_a_splitbatch_sm110"


class ExportComplete(Exception):
    pass


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
    overlay = out / "interface_split_batch.py"
    subprocess.run([
        "patch", "--batch", "--output", str(overlay),
        str(source / "flash_attn/cute/interface.py"),
        str(Path(__file__).resolve().parents[1] / "patches" / "fa4-split-batch.patch"),
    ], check=True)
    checks = {
        "official_interface": (source / "flash_attn/cute/interface.py", OFFICIAL_SHA),
        "general_sm100_forward": (source / "flash_attn/cute/flash_fwd_sm100.py", GENERAL_SHA),
        "isolated_interface_overlay": (overlay, OVERLAY_SHA),
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
        compile(overlay.read_text(), str(overlay), "exec")
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
    spec = importlib.util.spec_from_file_location("flash_attn.cute.interface_split_batch", overlay)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load isolated FA4 interface")
    splitbatch = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = splitbatch
    spec.loader.exec_module(splitbatch)

    original_compile = cute.compile
    export_record = {}

    def compile_and_export(*compile_args, **kwargs):
        if not compile_args or not isinstance(compile_args[0], FlashAttentionForwardSm100):
            return original_compile(*compile_args, **kwargs)
        if export_record:
            raise RuntimeError("more than one specialization requested")
        original_options = kwargs.pop("options", "")
        if original_options.strip() != "--enable-tvm-ffi":
            raise RuntimeError(f"unexpected FA4 compile options: {original_options!r}")
        forward = compile_args[0]
        expected = (
            forward.head_dim_padded == 256 and forward.head_dim_v_padded == 256
            and forward.qhead_per_kvhead == 4 and forward.pack_gqa
            and forward.q_stage == 1 and forward.m_block_size == 128
            and forward.n_block_size == 128 and forward.cta_group_size == 1
            and not forward.is_static_persistent and not forward.is_split_kv
        )
        if not expected:
            raise RuntimeError("unexpected generic FA4 configuration")

        @cute.jit
        def native_forward(
            q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
            out_tensor: cute.Tensor, lse: cute.Tensor,
            seqused_k: cute.Tensor, scale: cutlass.Float32,
            stream: cuda.CUstream,
        ):
            forward(q, k, v, out_tensor, lse, scale, mSeqUsedK=seqused_k,
                    aux_data=AuxData(), stream=stream)

        explicit_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        native_args = (
            native_forward, compile_args[1], compile_args[2], compile_args[3],
            compile_args[4], compile_args[5], compile_args[10],
            compile_args[6], explicit_stream,
        )
        options = "--gpu-arch sm_110a --host-target linux-aarch64"
        started = time.monotonic()
        compiled = original_compile(*native_args, options=options, **kwargs)
        compiled.export_to_c(file_path=str(out), file_name=SYMBOL, function_prefix=SYMBOL)
        export_record.update({
            "compile_options_original": original_options,
            "compile_options_native": options,
            "compile_seconds": time.monotonic() - started,
            "header_sha256": sha256(out / f"{SYMBOL}.h"),
            "object_sha256": sha256(out / f"{SYMBOL}.o"),
        })
        raise ExportComplete

    cute.compile = compile_and_export
    try:
        q = torch.empty((2, 50, 16, 256), dtype=torch.bfloat16, device="cuda")
        # Preserve the accepted compiler specialization. The generated ABI has
        # dynamic extents/strides, so the adapter can pass 1720-row views with
        # a 1718-row batch stride and seqused masks. Compiling an example with
        # 1720 rows changes rounding on real action inputs despite matching
        # random-input checks.
        k = torch.empty((2, 1718, 4, 256), dtype=torch.bfloat16, device="cuda")
        v = torch.empty_like(k)
        seqused_k = torch.tensor([1718, 1717], dtype=torch.int32, device="cuda")
        splitbatch.flash_attn_varlen_func(
            q, k, v, seqused_k=seqused_k, max_seqlen_q=50,
            max_seqlen_k=1718, softmax_scale=0.0625, causal=False,
            num_splits=1, pack_gqa=True, return_lse=True,
        )
    except ExportComplete:
        pass
    finally:
        cute.compile = original_compile
    if not export_record:
        raise RuntimeError("packed generic FA4 compile was not reached")
    metadata.update(export_record)
    (out / "export-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

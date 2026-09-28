#!/usr/bin/env python3
"""Export one pinned official FA4 D256 2CTA dense-forward specialization.

The public interface selects all kernel settings. Only the CuTe compilation
provider is replaced so the resulting object has a direct C ABI. Export uses
the GPU to specialize/compile; --probe-only checks CPU-side dependencies.
"""

import argparse
import hashlib
import json
import os
import site
import subprocess
import sys
import time
from pathlib import Path

CASES = {
    "L": {"nq": 3387, "nk": 3387, "causal": True},
}
PINNED_REVISION = "d15f1531a460ba456f41b01a774f33ab2db8febf"
PINNED_INTERFACE_SHA256 = "144a3dd6f72f955e43834500808c7d47b3b4a76fdcd0b7188f9b459d85007cab"
PINNED_D256_SHA256 = "8834e4c80d8cbff8eadd4719707016f58cc7440d620331e7d34c2ec8e49f1b73"


class ExportComplete(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=CASES, default="L")
    parser.add_argument("--fa4-src", type=Path, required=True)
    parser.add_argument("--cutlass-deps", type=Path, required=True)
    parser.add_argument("--extra-deps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    source = args.fa4_src.resolve()
    interface_file = source / "flash_attn/cute/interface.py"
    d256_file = source / "flash_attn/cute/sm100_hd256_2cta_fmha_forward.py"
    if (sha256(interface_file), sha256(d256_file)) != (
        PINNED_INTERFACE_SHA256, PINNED_D256_SHA256
    ):
        raise SystemExit("FA4 interface or D256 2CTA forward source is not pinned")
    revision = (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if (source / ".git").exists() else PINNED_REVISION
    )
    if revision != PINNED_REVISION:
        raise SystemExit(f"unexpected FA4 revision: {revision}")

    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    site.addsitedir(str(args.cutlass_deps.resolve()))
    site.addsitedir(str(args.extra_deps.resolve()))
    sys.path.insert(0, str(source))
    import torch
    import cutlass
    if cutlass.__version__ != "4.7.0":
        raise RuntimeError(f"CuTe DSL 4.7.0 required, got {cutlass.__version__}")
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from flash_attn.cute import interface
    from flash_attn.cute.sm100_hd256_2cta_fmha_forward import (
        BlackwellFusedMultiHeadAttentionForward,
    )
    from flash_attn.cute.utils import AuxData

    case = CASES[args.scope]
    symbol = f"apxinf_fa4_d256_{args.scope.lower()}_sm110"
    info = {
        "scope": args.scope, "q_shape": [1, case["nq"], 16, 256],
        "kv_shape": [1, case["nk"], 4, 256],
        "causal": case["causal"], "scale": 0.0625,
        "source": str(source), "revision": revision,
        "interface_sha256": sha256(interface_file), "d256_sha256": sha256(d256_file),
        "cutlass_version": cutlass.__version__, "torch_version": torch.__version__,
        "symbol": symbol,
    }
    if args.probe_only:
        print(json.dumps(info, indent=2))
        return

    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("FA4 D256 export requires SM110")
    args.out.mkdir(parents=True, exist_ok=True)
    original_compile = cute.compile
    export_record = {}

    def compile_and_export(*compile_args, **kwargs):
        if not compile_args or not isinstance(
            compile_args[0], BlackwellFusedMultiHeadAttentionForward
        ):
            return original_compile(*compile_args, **kwargs)
        if export_record:
            raise RuntimeError("more than one FA4 D256 specialization requested")
        original_options = kwargs.pop("options", "")
        if original_options.strip() != "--enable-tvm-ffi":
            raise RuntimeError(f"unexpected FA4 compile options: {original_options!r}")
        forward = compile_args[0]

        @cute.jit
        def native_forward(
            q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
            out: cute.Tensor, scale: cutlass.Float32,
            stream: cuda.CUstream,
        ):
            forward(q, k, v, out, None, scale, aux_data=AuxData(), stream=stream)

        explicit_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        native_args = (
            native_forward, compile_args[1], compile_args[2], compile_args[3],
            compile_args[4], compile_args[6], explicit_stream,
        )
        options = "--gpu-arch sm_110a --host-target linux-aarch64"
        started = time.monotonic()
        compiled = original_compile(*native_args, options=options, **kwargs)
        compiled.export_to_c(
            file_path=str(args.out), file_name=symbol, function_prefix=symbol
        )
        export_record.update({
            "compile_options_original": original_options,
            "compile_options_native": options,
            "compile_seconds": time.monotonic() - started,
            "header_sha256": sha256(args.out / f"{symbol}.h"),
            "object_sha256": sha256(args.out / f"{symbol}.o"),
        })
        raise ExportComplete

    cute.compile = compile_and_export
    try:
        q = torch.empty((1, case["nq"], 16, 256), dtype=torch.bfloat16, device="cuda")
        k = torch.empty((1, case["nk"], 4, 256), dtype=torch.bfloat16, device="cuda")
        v = torch.empty_like(k)
        interface.flash_attn_func(
            q, k, v, softmax_scale=0.0625, causal=case["causal"],
            return_lse=False,
        )
    except ExportComplete:
        pass
    finally:
        cute.compile = original_compile
    if not export_record:
        raise RuntimeError("official D256 forward compile was not reached")
    info.update(export_record)
    (args.out / "export-metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

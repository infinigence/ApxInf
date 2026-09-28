#!/usr/bin/env python3
"""Export the BF16 M256N256 SwiGLU implementation for SM110."""
import argparse
import hashlib
import json
import logging
import os
import site
from pathlib import Path

M, K, I = 3387, 2560, 9216
SYMBOL = "apxinf_quack_swiglu_bf16_m256n256_sm110"
PINNED = {
    "gemm_interface.py": "6215031bf8a870d16a1be224ecad21042076d62096f4621f477d63371eab92d3",
    "gemm_config.py": "608a9488f24f11a43dcfe8a92235728debe83611d37587ec6a5495e5255b4009",
    "gemm_sm100.py": "a60ed28ddd2bb45ee4b054bcec9bbc50bbc7a3beeb86a12798037bfa85b13215",
    "epilogue/library.py": "374c95060989c11d20476bec2e60ab54e7faccd92021367f87db28832238f65f",
}


class ExportComplete(Exception):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quack-deps", type=Path, required=True)
    p.add_argument("--cutlass-deps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    quack_src = args.quack_deps / "quack"
    got = {name: sha(quack_src / name) for name in PINNED}
    if got != PINNED:
        raise RuntimeError("Quack source differs from measured 0.6.5 source")
    if args.probe_only:
        # Do not import Torch, Quack, Cutlass or CUDA: this option is genuinely
        # CPU-only even if an imported package eagerly creates a CUDA context.
        print(json.dumps({"source_check": "ok", "quack_sha256": got,
                          "symbol": SYMBOL, "cuda_imports": False}))
        return
    os.environ["QUACK_CACHE_ENABLED"] = "0"
    site.addsitedir(str(args.cutlass_deps))
    site.addsitedir(str(args.quack_deps))
    logging.Logger.__quack_semantic_key__ = lambda self: (self.name, self.level)
    import torch
    import cutlass
    if cutlass.__version__ != "4.7.0":
        raise RuntimeError(f"CuTe DSL 4.7.0 required, got {cutlass.__version__}")
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from quack.gemm_config import GemmConfig
    from quack.gemm_interface import gemm_gated
    from quack.tile_scheduler import TileSchedulerOptions
    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("AOT target requires Thor SM110")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_compile = cute.compile
    captured = {}

    def export_compile(*compile_args, **kwargs):
        if len(compile_args) != 11:
            raise RuntimeError(f"unexpected Quack compile arity {len(compile_args)}")
        options = kwargs.pop("options", "")
        if options.strip() != "--enable-tvm-ffi" or kwargs:
            raise RuntimeError(f"unexpected Quack compile options {options!r}/{kwargs!r}")
        epi_type = type(compile_args[5])
        epi_fake = compile_args[5]
        gemm_obj = compile_args[0]
        captured.update({
            "gemm_class": type(compile_args[0]).__name__,
            "arg_types": [type(v).__name__ for v in compile_args],
            "epi_fields": list(getattr(compile_args[5], "_fields", ())),
            "scheduler_fields": list(getattr(compile_args[6], "_fields", ())),
            "options_original": options,
        })
        if tuple(epi_fake._fields) != ("mAuxOut", "split_k_semaphore", "split_k_workspace"):
            raise RuntimeError(f"unexpected epilogue fields {epi_fake._fields}")
        if compile_args[3] is not None or compile_args[4] is not None or compile_args[7] is not None:
            raise RuntimeError("unexpected D/C/varlen in measured no-preact route")

        @cute.jit
        def native_forward(
            a: cute.Tensor, b: cute.Tensor, y: cute.Tensor,
            max_active_clusters: cutlass.Int32, stream: cuda.CUstream,
        ):
            epilogue = epi_type(mAuxOut=y)
            scheduler = TileSchedulerOptions(max_active_clusters=max_active_clusters)
            gemm_obj(a, b, None, None, epilogue, scheduler,
                     None, stream, None, None)

        # CuTe 4.7 C-header generation does not flatten Quack's nested
        # EpilogueArguments. A thin JIT wrapper exposes only physical tensors,
        # one scheduler scalar and explicit stream, then reconstructs structs.
        native_args = (
            native_forward, compile_args[1], compile_args[2], epi_fake.mAuxOut,
            cutlass.Int32(1), cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        try:
            compiled = original_compile(
                *native_args, options="--gpu-arch sm_110a --host-target linux-aarch64"
            )
            compiled.export_to_c(file_path=str(out), file_name=SYMBOL,
                                 function_prefix=SYMBOL)
        except Exception as exc:
            captured["error"] = repr(exc)
            (out / "export-error.json").write_text(json.dumps(captured, indent=2) + "\n")
            raise
        captured["header_sha256"] = sha(out / f"{SYMBOL}.h")
        captured["object_sha256"] = sha(out / f"{SYMBOL}.o")
        (out / "export.json").write_text(json.dumps(captured, indent=2) + "\n")
        raise ExportComplete

    cute.compile = export_compile
    try:
        x = torch.empty((M, K), dtype=torch.bfloat16, device="cuda")
        # Physical B is [2I,K] row-major gate-half then up-half. The torch
        # front end sees its [K,2I] transpose view; Quack relabels B inside.
        b_nk = torch.empty((2 * I, K), dtype=torch.bfloat16, device="cuda")
        b_view = b_nk.T
        y = torch.empty((M, I), dtype=torch.bfloat16, device="cuda")
        cfg = GemmConfig(tile_m=256, tile_n=256, cluster_m=2, cluster_n=1,
                         pingpong=False, is_dynamic_persistent=True,
                         swap_ab=False, device_capacity=11)
        gemm_gated(x, b_view, activation="swiglu", store_preact=False,
                   postact_out=y, tuned=False, config=cfg,
                   concat_layout=("B",))
    except ExportComplete:
        print(json.dumps({"export": "ok", **captured}, indent=2))
    finally:
        cute.compile = original_compile


if __name__ == "__main__":
    main()

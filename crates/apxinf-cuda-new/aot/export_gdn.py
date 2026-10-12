#!/usr/bin/env python3
"""Export the pinned BF16 T64 recurrent attention implementation for SM110."""

import argparse
import types
import hashlib
import json
import site
import sys
import time
from pathlib import Path


EXPECTED = {
    "gated_delta_net_chunked.py": "94f3f975e9e46e6fe3222dd46ee5a10537cc7d655fe096e75c638933caf05fb7",
    "gated_delta_net_tile_scheduler.py": "96654e5e496907907f211b0768a993144b1793c094d2dd9919a5bc78502ed4a9",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdn-src", type=Path, required=True)
    parser.add_argument("--cutlass-deps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    cli = parser.parse_args()
    source = cli.gdn_src.resolve()
    for filename, expected in EXPECTED.items():
        if sha(source / filename) != expected:
            raise RuntimeError(f"pinned FlashInfer source changed: {filename}")
    site.addsitedir(str(cli.cutlass_deps.resolve()))
    package = types.ModuleType("apxinf_flashinfer_aot")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    import torch
    import cutlass
    if cutlass.__version__ != "4.7.0":
        raise RuntimeError(f"CuTe DSL 4.7.0 required, got {cutlass.__version__}")
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import from_dlpack
    from apxinf_flashinfer_aot.gated_delta_net_chunked import GatedDeltaNetChunkedKernel

    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("one-chunk export requires Thor SM110")
    num_sm = torch.cuda.get_device_properties(0).multi_processor_count
    bf = torch.bfloat16
    seq_pad = 3392
    q = torch.zeros((seq_pad, 16, 128), dtype=bf, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros((seq_pad, 32, 128), dtype=bf, device="cuda")
    alpha = torch.ones((seq_pad, 32), dtype=torch.float32, device="cuda")
    beta = torch.zeros_like(alpha)
    o = torch.empty_like(v)
    cu = torch.tensor([0, 3387], dtype=torch.int32, device="cuda")
    state = torch.empty((1, 32, 128, 128), dtype=torch.float32, device="cuda")
    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(num_sm, 1, 16, 32, True)
    workspace = torch.empty(workspace_size, dtype=torch.int8, device="cuda")
    gdn = GatedDeltaNetChunkedKernel(
        io_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        state_dtype=cutlass.Float32,
        mma_tiler_qk=(64, 64, 128),
        mma_tiler_qs=(128, 64, 128),
        mma_tiler_qkv=(128, 64, 64),
        mma_tiler_kv=(128, 128, 64),
        max_active_clusters=num_sm,
        num_sm=num_sm,
        is_GQA=False,
        use_initial_state=False,
        store_final_state=True,
        enable_checkpoints=False,
        is_persistent=True,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(alpha, assumed_align=16),
        from_dlpack(beta, assumed_align=16),
        from_dlpack(o, assumed_align=16),
        from_dlpack(cu, assumed_align=4),
        None,
        from_dlpack(state, assumed_align=16),
        None,
        None,
        cutlass.Int32(0),
        cutlass.Float32(1.0 / 128**0.5),
        from_dlpack(workspace, assumed_align=16),
        stream,
    )
    out = cli.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    symbol = "flashinfer_gdn_bf16_t64_sm110_real3392"
    options = "--gpu-arch sm_110a --opt-level 2"
    start = time.monotonic()
    compiled = cute.compile(gdn, *args, options=options)
    elapsed = time.monotonic() - start
    compiled.export_to_c(file_path=str(out), file_name=symbol, function_prefix=symbol)
    record = {
        "source_tag": "flashinfer/v0.6.14",
        "source_commit": "19f1a41e6b21f0c422d775e377b6fdf9a1fc9d23",
        "kernel_sha256": sha(source / "gated_delta_net_chunked.py"),
        "scheduler_sha256": sha(source / "gated_delta_net_tile_scheduler.py"),
        "cutlass_version": cutlass.__version__,
        "torch_version": torch.__version__,
        "options": options,
        "num_sm": num_sm,
        "workspace_bytes": workspace_size,
        "seq": 3387,
        "seq_pad": seq_pad,
        "scale": 1.0 / 128**0.5,
        "compile_seconds": elapsed,
        "header_sha256": sha(out / f"{symbol}.h"),
        "object_sha256": sha(out / f"{symbol}.o"),
    }
    (out / "export-metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

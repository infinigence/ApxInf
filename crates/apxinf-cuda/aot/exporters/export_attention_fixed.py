#!/usr/bin/env python3
"""Export exactly one pinned SM110 BF16 FA4 fixed-batch geometry."""
import argparse
import hashlib
import json
import os
import site
import sys
import time
import types
from pathlib import Path


INTERFACE_SHA = "144a3dd6f72f955e43834500808c7d47b3b4a76fdcd0b7188f9b459d85007cab"
FORWARD_SHA = "9d43194751128963a701f0d47b04f68cb6bfd2c8e87e4734d49b33a4cc24d1f2"


class ExportComplete(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, choices=(624, 2200), required=True)
    ap.add_argument("--fa4-src", type=Path, required=True)
    ap.add_argument("--cutlass-deps", type=Path, required=True)
    ap.add_argument("--extra-deps", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src = args.fa4_src.resolve()
    assert digest(src / "flash_attn/cute/interface.py") == INTERFACE_SHA
    assert digest(src / "flash_attn/cute/flash_fwd_sm100.py") == FORWARD_SHA
    deps = args.cutlass_deps.resolve()
    extra = args.extra_deps.resolve()
    site.addsitedir(str(deps))
    site.addsitedir(str(extra))
    sys.path.insert(0, str(src))
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"

    import torch
    import cutlass
    if cutlass.__version__ != "4.7.0":
        raise RuntimeError(f"CuTe DSL 4.7.0 required, got {cutlass.__version__}")
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    pkg = types.ModuleType("flash_attn")
    pkg.__path__ = [str(src / "flash_attn")]
    sys.modules["flash_attn"] = pkg
    from flash_attn.cute.interface import _flash_attn_fwd
    from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
    from flash_attn.cute.utils import AuxData

    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("SM110 required")
    L = args.length
    symbol = f"apxinf_fa4_vfixed_b3_l{L}_sm110"
    args.out.mkdir(parents=True, exist_ok=True)
    original_compile = cute.compile
    record = {}

    def export(*compile_args, **kwargs):
        if not compile_args or not isinstance(compile_args[0], FlashAttentionForwardSm100):
            return original_compile(*compile_args, **kwargs)
        if record:
            raise RuntimeError("unexpected second forward compile")
        original_options = kwargs.pop("options", "")
        if original_options.strip() != "--enable-tvm-ffi":
            raise RuntimeError(f"unexpected options {original_options!r}")
        forward = compile_args[0]
        if compile_args[5] is not None or compile_args[7] is not None or compile_args[8] is not None:
            raise RuntimeError("fixed path unexpectedly has LSE or varlen offsets")
        if any(compile_args[i] is not None for i in list(range(9,17))+list(range(18,25))):
            raise RuntimeError("unexpected optional fixed forward argument")
        if not isinstance(compile_args[17], AuxData):
            raise RuntimeError("unexpected auxiliary forward argument")
        if compile_args[-2] != L:
            raise RuntimeError(f"unexpected max_seqlen_q: {compile_args[-2]!r}")

        @cute.jit
        def native_forward(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
                           out: cute.Tensor, scale: cutlass.Float32,
                           stream: cuda.CUstream):
            forward(q, k, v, out, None, scale, aux_data=AuxData(),
                    max_seqlen_q=L, stream=stream)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        options = "--gpu-arch sm_110a --host-target linux-aarch64"
        began = time.monotonic()
        compiled = original_compile(native_forward, *compile_args[1:5],
                                    compile_args[6], stream, options=options)
        compiled.export_to_c(file_path=str(args.out), file_name=symbol,
                             function_prefix=symbol)
        record.update(symbol=symbol, length=L, compile_seconds=time.monotonic()-began,
                      source_interface_sha=INTERFACE_SHA, source_forward_sha=FORWARD_SHA,
                      options=options, cutlass_version=cutlass.__version__,
                      object_sha=digest(args.out / f"{symbol}.o"),
                      header_sha=digest(args.out / f"{symbol}.h"))
        raise ExportComplete

    cute.compile = export
    try:
        tokens = 3*4072
        q = torch.empty((tokens,16,64), dtype=torch.bfloat16, device="cuda")
        k = torch.empty_like(q)
        packed = torch.empty((tokens,3,16,64), dtype=torch.bfloat16, device="cuda")
        v = packed[:,2]
        out = torch.empty_like(q)
        def view(t):
            return torch.as_strided(t, (3,L,16,64),
                                    (4072*t.stride(0),t.stride(0),64,1))
        _flash_attn_fwd(view(q),view(k),view(v),softmax_scale=.125,
                        causal=False,num_splits=1,tile_mn=(128,128),out=view(out))
    except ExportComplete:
        pass
    finally:
        cute.compile = original_compile
    if not record:
        raise RuntimeError("forward specialization was not intercepted")
    (args.out / "export.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()

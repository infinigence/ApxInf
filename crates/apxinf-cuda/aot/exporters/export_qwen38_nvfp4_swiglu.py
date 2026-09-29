#!/usr/bin/env python3
"""Export the Qwen3.8 dense NVFP4 FC1 + SwiGLU + requant kernel."""

import argparse
import hashlib
import json
import os
import shutil
import site
import sys
from pathlib import Path


SYMBOL = "dense_swiglu"
M, K, I = 2048, 5120, 17408
SF_VEC_SIZE = 16
M_TILE_SIZE = 128
PINNED = {
    "blockscaled_contiguous_gather_grouped_gemm_act_fusion.py":
        "9a1c08088b1614870c6e411eca2107875cdf35b1e54663e5011cff3ccc8f815d",
    "custom_pipeline.py":
        "6c15e7f4473a3e33c5b93f55e1a185f214ab15602b08f18f071e9d8bd1d46b39",
    "cute_utils.py":
        "4d9909d3ad2ea160515ca0448b3ec67e08b904cb9f63e7f69669e9f3d2e776c1",
    "export_common.py":
        "5d71e0a87ec64341127e7f2ccd372a79de7cfe7f7923529e1feff57caf665f1e",
    "moe_compat.py":
        "71c4319cfd7c68aa4c332127bec42b923548864449b55b8242c0813d30b0e6c0",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_kernel(source: Path, destination: Path) -> None:
    text = source.read_text()
    anchor = "from moe_compat import ActivationType, is_gated_activation\n"
    replacement = anchor + "from apxinf_qwen38_precision import quant_inverse\n"
    if text.count(anchor) != 1:
        raise RuntimeError("TensorRT-Edge-LLM quantization import anchor changed")
    text = text.replace(anchor, replacement)

    vectorized = """                                acc_scale = cute.arch.mul_packed_f32x2(
                                    (
                                        cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi]),
                                        cute.arch.rcp_approx(tCrSFC_qpvscale_up[vi + 1]),
                                    ),
                                    (norm_const, norm_const),
                                )"""
    vectorized_exact = """                                acc_scale = (
                                    quant_inverse(tCrSFC_qpvscale_up[vi], down_input_scale_tensor[expert_idx]),
                                    quant_inverse(tCrSFC_qpvscale_up[vi + 1], down_input_scale_tensor[expert_idx]),
                                )"""
    scalar = """                                acc_scale = norm_const * cute.arch.rcp_approx(
                                    tCrSFC_qpvscale_up[vi]
                                )"""
    scalar_exact = """                                acc_scale = quant_inverse(
                                    tCrSFC_qpvscale_up[vi], down_input_scale_tensor[expert_idx]
                                )"""
    if text.count(vectorized) != 1 or text.count(scalar) != 1:
        raise RuntimeError("TensorRT-Edge-LLM NVFP4 scale epilogue changed")
    destination.write_text(
        text.replace(vectorized, vectorized_exact).replace(scalar, scalar_exact)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-src", type=Path, required=True)
    parser.add_argument("--cutlass-deps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.edge_src.resolve()
    got = {name: sha(source / name) for name in PINNED}
    if got != PINNED:
        raise RuntimeError("TensorRT-Edge-LLM NVFP4 source differs from pinned revision")

    out = args.out.resolve()
    staged = out / "staged-source"
    staged.mkdir(parents=True, exist_ok=True)
    for name in PINNED:
        if name != "blockscaled_contiguous_gather_grouped_gemm_act_fusion.py":
            shutil.copy2(source / name, staged / name)
    patch_kernel(
        source / "blockscaled_contiguous_gather_grouped_gemm_act_fusion.py",
        staged / "apxinf_qwen38_fc1_kernel.py",
    )
    if args.probe_only:
        print(json.dumps({"source_check": "ok", "source_sha256": got,
                          "symbol": SYMBOL, "cuda_imports": False}))
        return

    site.addsitedir(str(args.cutlass_deps.resolve()))
    sys.path.insert(0, str(staged))
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import torch
    from cutlass import Float32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import dsl_user_op

    if cutlass.__version__ != "4.7.0":
        raise RuntimeError(f"CuTe DSL 4.7.0 required, got {cutlass.__version__}")

    @dsl_user_op
    def divide_rn(numerator, denominator, *, loc=None, ip=None):
        return Float32(llvm.inline_asm(
            Float32.mlir_type,
            [Float32(numerator).ir_value(loc=loc, ip=ip),
             Float32(denominator).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False,
            loc=loc, ip=ip,
        ))

    @cute.jit
    def quant_inverse(scale, global_scale):
        effective = scale * global_scale
        result = Float32(0.0)
        if effective > Float32(0.0):
            result = divide_rn(Float32(1.0), effective)
        return result

    # The staged upstream module imports this helper by name. Keeping the
    # precision patch here makes it part of the reviewed exporter digest.
    import types
    precision = types.ModuleType("apxinf_qwen38_precision")
    precision.quant_inverse = quant_inverse
    sys.modules[precision.__name__] = precision

    from apxinf_qwen38_fc1_kernel import (
        BlockScaledContiguousGatherGroupedGemmKernel as BaseKernel,
    )
    from export_common import atom_scale_bytes, make_ptr
    from moe_compat import ActivationType

    class DenseSwiGLUKernel(BaseKernel):
        @cute.jit
        def _apply_swiglu_epilogue(
            self, acc_vec_up, acc_vec_gate, alpha_val, tCompute
        ):
            for index in cutlass.range_constexpr(cute.size(acc_vec_up.shape)):
                up = (acc_vec_up[index] * Float32(alpha_val)).to(
                    cutlass.BFloat16
                ).to(Float32)
                gate = (acc_vec_gate[index] * Float32(alpha_val)).to(
                    cutlass.BFloat16
                ).to(Float32)
                activated = divide_rn(
                    gate, Float32(1.0) + cute.math.exp(-gate, fastmath=True)
                )
                activated = activated.to(cutlass.BFloat16).to(Float32)
                tCompute[index] = (activated * up).to(
                    cutlass.BFloat16
                ).to(Float32)

    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("Qwen3.8 fused FC1 export requires Thor SM110")
    buffers = []

    def allocate(shape, dtype):
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        buffers.append(tensor)
        return tensor

    n = 2 * I
    a = allocate((M, K // 2), torch.uint8)
    b = allocate((1, n, K // 2), torch.uint8)
    a_sf = allocate((M, K // SF_VEC_SIZE), torch.uint8)
    b_sf = allocate((atom_scale_bytes(n, K, 1),), torch.uint8)
    c = allocate((M, I // 2), torch.uint8)
    c_sf = allocate((atom_scale_bytes(M, I),), torch.uint8)
    alpha = allocate((1,), torch.float32)
    input_global_scale = allocate((1,), torch.float32)
    down_input_scale = allocate((1,), torch.float32)
    tile_group = allocate((M // M_TILE_SIZE,), torch.int32)
    tile_limit = allocate((M // M_TILE_SIZE,), torch.int32)
    token_map = allocate((M,), torch.int32)
    num_tiles = allocate((1,), torch.int32)
    ptrs = (
        make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), assumed_align=32),
        make_ptr(cutlass.Float4E2M1FN, b.data_ptr(), assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, a_sf.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, b_sf.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float4E2M1FN, c.data_ptr(), assumed_align=32),
        make_ptr(cutlass.Float8E4M3FN, c_sf.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float32, alpha.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float32, input_global_scale.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Float32, down_input_scale.data_ptr(), assumed_align=16),
        make_ptr(cutlass.Int32, tile_group.data_ptr()),
        make_ptr(cutlass.Int32, tile_limit.data_ptr()),
        make_ptr(cutlass.Int32, token_map.data_ptr()),
        make_ptr(cutlass.Int32, num_tiles.data_ptr()),
    )
    kernel = DenseSwiGLUKernel(
        sf_vec_size=SF_VEC_SIZE,
        mma_tiler_mn=(256, 256),
        cluster_shape_mn=(2, 1),
        vectorized_f32=True,
        topk=1,
        raster_along_m=True,
        b_tensor_l_sizes=(1,),
        activation_type=ActivationType.Swiglu,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    @cute.jit
    def single_b_wrapper(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        input_global_scale_ptr: cute.Pointer,
        down_input_scale_ptr: cute.Pointer,
        tile_group_ptr: cute.Pointer,
        tile_limit_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        num_tiles_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64, n_: cutlass.Int64, k_: cutlass.Int64,
        groups: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
        activation_type: cutlass.Constexpr,
    ):
        return kernel.wrapper(
            a_ptr, (b_ptr,), a_sf_ptr, (b_sf_ptr,), c_ptr, c_sf_ptr,
            (alpha_ptr,), input_global_scale_ptr, down_input_scale_ptr,
            tile_group_ptr, tile_limit_ptr, token_map_ptr, num_tiles_ptr,
            orig_m, m, n_, k_, groups, tile_size, scaling_vector_size,
            max_active_clusters, stream, activation_type=activation_type,
        )

    compiled = cute.compile(
        single_b_wrapper, *ptrs, M, M_TILE_SIZE, n, K, 1,
        tile_size=M_TILE_SIZE,
        scaling_vector_size=SF_VEC_SIZE,
        max_active_clusters=cutlass.Int32(10),
        stream=stream,
        activation_type=ActivationType.Swiglu,
        options="--gpu-arch sm_110a --host-target linux-aarch64",
    )
    compiled.export_to_c(
        file_path=str(out), file_name=SYMBOL, function_prefix=SYMBOL
    )
    metadata = {
        "source_revision": "e8b29522938901f6df19ebeedd4b69bc8edbcd97",
        "source_sha256": got,
        "cutlass_version": cutlass.__version__,
        "shape": [M, I, K],
        "tile": [256, 256],
        "cluster": [2, 1],
        "header_sha256": sha(out / f"{SYMBOL}.h"),
        "object_sha256": sha(out / f"{SYMBOL}.o"),
    }
    (out / "export-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

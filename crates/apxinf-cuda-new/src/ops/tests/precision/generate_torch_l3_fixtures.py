#!/usr/bin/env python3
"""Regenerate the checked-in L3 operator precision fixtures with PyTorch.

The reference starts from the exact BF16/FP8 tensors consumed by the L3 API.
It therefore measures GEMM implementation error, not FP32-to-low-precision
quantization error.

Maintenance rule: every new L3 operator must extend this generator, regenerate
torch_l3_fixtures.rs, and add its all-candidates test to precision.rs.
"""

from pathlib import Path
import struct

import torch


M, K, N = 8, 16, 16
BF16_ALPHA, BF16_OUTPUT_SCALE = 0.75, 1.25
FP8_UNIT_ALPHA, FP8_UNIT_OUTPUT_SCALE = 1.0, 1.0
FP8_SCALED_ALPHA, FP8_SCALED_OUTPUT_SCALE = 0.75, 1.25


def words(tensor: torch.Tensor) -> list[int]:
    return tensor.contiguous().view(torch.uint16).flatten().tolist()


def bytes_(tensor: torch.Tensor) -> list[int]:
    return tensor.contiguous().view(torch.uint8).flatten().tolist()


def f32_bits(tensor: torch.Tensor) -> list[int]:
    return [struct.unpack("<I", struct.pack("<f", float(x)))[0] for x in tensor.flatten()]


def rust_array(name: str, ty: str, values: list[int], width: int) -> str:
    rows = []
    for offset in range(0, len(values), width):
        chunk = values[offset : offset + width]
        if ty == "u8":
            rendered = ", ".join(f"0x{x:02x}" for x in chunk)
        elif ty == "u16":
            rendered = ", ".join(f"0x{x:04x}" for x in chunk)
        else:
            rendered = ", ".join(f"f32::from_bits(0x{x:08x})" for x in chunk)
        rows.append(f"    {rendered},")
    return f"pub(crate) const {name}: &[{ty}] = &[\n" + "\n".join(rows) + "\n];\n"


def finish(projection: torch.Tensor, semantic: str, bias: torch.Tensor | None,
           alpha: float, output_scale: float) -> torch.Tensor:
    if semantic == "gemm":
        result = alpha * projection
    elif semantic == "gemm_bias":
        result = alpha * projection + bias
    elif semantic == "gemm_bias_gelu":
        result = torch.nn.functional.gelu(alpha * projection + bias, approximate="tanh")
    else:
        raise ValueError(f"unsupported L3 semantic: {semantic}")
    return (result / output_scale).to(torch.float32)


def geglu(a: torch.Tensor, b: torch.Tensor, alpha: float,
          output_scale: float) -> torch.Tensor:
    """Evaluate the public B=[B_gate, B_up] contract as two GEMMs."""
    width = b.shape[1] // 2
    gate = alpha * (a @ b[:, :width])
    up = alpha * (a @ b[:, width:])
    return (torch.nn.functional.gelu(gate, approximate="tanh") * up /
            output_scale).to(torch.float32)


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator(device="cpu").manual_seed(0x5EED_C0DE)
    raw_a = torch.randn((M, K), generator=generator) * 0.75
    raw_b = torch.randn((K, N), generator=generator) * 0.50
    raw_geglu_b = torch.randn((K, 2 * N), generator=generator) * 0.50
    raw_bias = torch.randn((N,), generator=generator) * 0.25

    bf16_a = raw_a.to(torch.bfloat16)
    bf16_b = raw_b.to(torch.bfloat16)
    bf16_geglu_b = raw_geglu_b.to(torch.bfloat16)
    bf16_bias = raw_bias.to(torch.bfloat16)

    fp8_a = raw_a.to(torch.float8_e4m3fn)
    fp8_b = raw_b.to(torch.float8_e4m3fn)
    fp8_geglu_b = raw_geglu_b.to(torch.float8_e4m3fn)
    fp8_bias = raw_bias.to(torch.float32)
    row_scales = torch.linspace(0.25, 1.125, M, dtype=torch.float32)
    channel_scales = torch.linspace(0.50, 1.4375, N, dtype=torch.float32)

    bf16_projection = bf16_a.float() @ bf16_b.float()
    fp8_projection = fp8_a.float() @ fp8_b.float()
    scaled_projection = (
        fp8_a.float() * row_scales[:, None]
    ) @ (
        fp8_b.float() * channel_scales[None, :]
    )

    arrays: list[str] = []
    arrays.append(rust_array("BF16_A", "u16", words(bf16_a), K))
    arrays.append(rust_array("BF16_B", "u16", words(bf16_b), N))
    arrays.append(rust_array("BF16_GEGLU_B", "u16", words(bf16_geglu_b), 2 * N))
    arrays.append(rust_array("BF16_BIAS", "u16", words(bf16_bias), N))
    arrays.append(rust_array("FP8_A", "u8", bytes_(fp8_a), K))
    arrays.append(rust_array("FP8_B", "u8", bytes_(fp8_b), N))
    arrays.append(rust_array("FP8_GEGLU_B", "u8", bytes_(fp8_geglu_b), 2 * N))
    arrays.append(rust_array("FP8_BIAS", "f32", f32_bits(fp8_bias), 4))
    arrays.append(rust_array("ROW_SCALES", "f32", f32_bits(row_scales), 4))
    arrays.append(rust_array("CHANNEL_SCALES", "f32", f32_bits(channel_scales), 4))

    for prefix, projection, bias, alpha, output_scale, semantics in [
        ("BF16", bf16_projection, bf16_bias.float(), BF16_ALPHA, BF16_OUTPUT_SCALE,
         ["gemm", "gemm_bias", "gemm_bias_gelu"]),
        ("FP8_UNIT", fp8_projection, fp8_bias, FP8_UNIT_ALPHA, FP8_UNIT_OUTPUT_SCALE,
         ["gemm"]),
        ("FP8_SCALED", scaled_projection, fp8_bias, FP8_SCALED_ALPHA,
         FP8_SCALED_OUTPUT_SCALE, ["gemm", "gemm_bias", "gemm_bias_gelu"]),
    ]:
        for semantic in semantics:
            expected = finish(projection, semantic, bias, alpha, output_scale)
            arrays.append(rust_array(f"{prefix}_{semantic.upper()}", "f32", f32_bits(expected), 4))

    for prefix, a, b, alpha, output_scale in [
        ("BF16", bf16_a.float(), bf16_geglu_b.float(), BF16_ALPHA, BF16_OUTPUT_SCALE),
        ("FP8_UNIT", fp8_a.float(), fp8_geglu_b.float(), FP8_UNIT_ALPHA,
         FP8_UNIT_OUTPUT_SCALE),
    ]:
        expected = geglu(a, b, alpha, output_scale)
        arrays.append(rust_array(f"{prefix}_GEMM_GEGLU", "f32", f32_bits(expected), 4))

    destination = Path(__file__).with_name("torch_l3_fixtures.rs")
    destination.write_text(
        "// @generated by generate_torch_l3_fixtures.py\n"
        "// Every new L3 operator must add its fixed inputs and final Torch outputs here\n"
        "// by extending and rerunning the generator; do not hand-edit numeric fixtures.\n"
        f"// PyTorch {torch.__version__}; seed=0x5eed_c0de.\n\n"
        f"pub(crate) const M: usize = {M};\n"
        f"pub(crate) const K: usize = {K};\n"
        f"pub(crate) const N: usize = {N};\n"
        f"pub(crate) const BF16_ALPHA: f32 = {BF16_ALPHA};\n"
        f"pub(crate) const BF16_OUTPUT_SCALE: f32 = {BF16_OUTPUT_SCALE};\n"
        f"pub(crate) const FP8_UNIT_ALPHA: f32 = {FP8_UNIT_ALPHA};\n"
        f"pub(crate) const FP8_UNIT_OUTPUT_SCALE: f32 = {FP8_UNIT_OUTPUT_SCALE};\n"
        f"pub(crate) const FP8_SCALED_ALPHA: f32 = {FP8_SCALED_ALPHA};\n"
        f"pub(crate) const FP8_SCALED_OUTPUT_SCALE: f32 = {FP8_SCALED_OUTPUT_SCALE};\n\n"
        + "\n".join(arrays)
    )


if __name__ == "__main__":
    main()

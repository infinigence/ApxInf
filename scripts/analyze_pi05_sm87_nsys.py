#!/usr/bin/env python3
"""Attribute native-BF16 or W8A8-INT8 PI0.5 SM87 CUDA graph traces.

The two precision paths share a mathematical schedule but not a GEMM pipeline:

* BF16 GEMMs launch one main kernel and may launch a split-K reduction.
* INT8 GEMMs launch row quantization followed by a fused CUTLASS GEMM; the
  unaligned K=588 patch projection additionally materializes and dequantizes
  INT32 output.

This analyzer reconstructs that exact schedule for the two-view PI0.5 graph,
validates every kernel and prefix K/V copy in launch order, and emits stage,
flow-step, logical-operation, kernel-category, and kernel-family breakdowns.
It fails closed if a graph changes instead of silently applying stale labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


STAGE_ORDER = (
    "raw_preprocess",
    "vision_encoder",
    "language_prefix",
    "action_denoise_10_steps",
)
EXPECTED_LOGICAL_KERNELS = 2_131
EXPECTED_GEMMS = 919
EXPECTED_BF16_SPLIT_K = 360
EXPECTED_INT8_FUSED_GEMMS = 918
EXPECTED_PREFIX_COPIES = 36


@dataclass(frozen=True)
class OperationSpec:
    name: str
    stage: str
    kind: str
    expected_short_name: str | None = None
    gemm: dict[str, Any] | None = None
    attention: dict[str, Any] | None = None
    traffic_bytes_per_call: int | None = None
    flow_step: int | None = None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export")
    parser.add_argument("--report", type=Path, help="source .nsys-rep")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--precision", required=True, choices=("bf16", "int8"))
    parser.add_argument("--token-count", type=int, default=21)
    parser.add_argument("--graph-p50-ms", required=True, type=float)
    parser.add_argument("--update-plus-graph-p50-ms", required=True, type=float)
    parser.add_argument("--protocol-p50-ms", required=True, type=float)
    parser.add_argument("--benchmark-binary-sha256")
    parser.add_argument("--remote-report")
    parser.add_argument("--cuda-version", default="13.2")
    parser.add_argument("--nsys-version", default="2026.3.1")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def merge_busy_time(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def duration_summary(durations_ns: list[int], graph_span_ns: int) -> dict[str, Any]:
    total = sum(durations_ns)
    return {
        "count": len(durations_ns),
        "total_ms": round(total / 1e6, 6),
        "graph_span_percent": round(100.0 * total / graph_span_ns, 4),
        "mean_us": round(total / len(durations_ns) / 1e3, 4),
        "median_us": round(median(durations_ns) / 1e3, 4),
        "p95_us": round(percentile(durations_ns, 0.95) / 1e3, 4),
        "max_us": round(max(durations_ns) / 1e3, 4),
    }


def bf16_tensor_bytes(*dimensions: int) -> int:
    elements = 1
    for dimension in dimensions:
        elements *= dimension
    return 2 * elements


def gemm_metadata(m: int, n: int, k: int) -> dict[str, int]:
    return {"m": m, "n": n, "k": k}


def attention_metadata(
    *,
    batch: int = 1,
    query_tokens: int,
    key_tokens: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, int]:
    return {
        "batch": batch,
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }


def attention_traffic(metadata: dict[str, int]) -> int:
    batch = metadata["batch"]
    q = metadata["query_tokens"]
    k = metadata["key_tokens"]
    q_heads = metadata["query_heads"]
    kv_heads = metadata["kv_heads"]
    d = metadata["head_dim"]
    elements = q * q_heads * d + 2 * k * kv_heads * d + q * q_heads * d
    return 2 * batch * elements


def build_schedule(token_count: int) -> list[OperationSpec]:
    if not 0 < token_count <= 200:
        raise ValueError("token count must be in 1..=200")
    prefix_tokens = 512 + token_count
    action_kv_tokens = prefix_tokens + 10
    schedule: list[OperationSpec] = []

    def add(
        name: str,
        stage: str,
        kind: str,
        *,
        short: str | None = None,
        gemm: dict[str, Any] | None = None,
        attention: dict[str, Any] | None = None,
        traffic: int | None = None,
        step: int | None = None,
    ) -> None:
        schedule.append(
            OperationSpec(
                name=name,
                stage=stage,
                kind=kind,
                expected_short_name=short,
                gemm=gemm,
                attention=attention,
                traffic_bytes_per_call=traffic,
                flow_step=step,
            )
        )

    add(
        "raw.rgb_to_patches_bf16",
        "raw_preprocess",
        "custom",
        short="rgb_u8_to_patches_bf16_kernel",
        traffic=2 * 224 * 224 * 3 + bf16_tensor_bytes(512, 588),
    )

    stage = "vision_encoder"
    add("vision.patch_gemm", stage, "gemm", gemm=gemm_metadata(512, 1_152, 588))
    add(
        "vision.bias_position",
        stage,
        "custom",
        short="bias_position_bf16_kernel",
        traffic=(
            2 * bf16_tensor_bytes(512, 1_152)
            + bf16_tensor_bytes(1_152)
            + bf16_tensor_bytes(256, 1_152)
        ),
    )
    vision_attention = attention_metadata(
        batch=2,
        query_tokens=256,
        key_tokens=256,
        query_heads=16,
        kv_heads=16,
        head_dim=72,
    )
    for _ in range(27):
        add(
            "vision.layer_norm",
            stage,
            "custom",
            short="layer_norm_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(512, 1_152) + 2 * bf16_tensor_bytes(1_152),
        )
        add("vision.qkv_gemm", stage, "gemm", gemm=gemm_metadata(512, 3_456, 1_152))
        add(
            "vision.qkv_split_bias",
            stage,
            "custom",
            short="qkv_split_bias_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(512, 3_456) + bf16_tensor_bytes(3_456),
        )
        add(
            "vision.mha_fa2",
            stage,
            "attention",
            short="flash_fwd_kernel",
            attention=vision_attention,
            traffic=attention_traffic(vision_attention),
        )
        add(
            "vision.attention_out_gemm",
            stage,
            "gemm",
            gemm=gemm_metadata(512, 1_152, 1_152),
        )
        add(
            "vision.bias_residual_layer_norm",
            stage,
            "custom",
            short="bias_residual_layer_norm_bf16_kernel",
            traffic=(4 * bf16_tensor_bytes(512, 1_152) + 3 * bf16_tensor_bytes(1_152)),
        )
        add("vision.fc1_gemm", stage, "gemm", gemm=gemm_metadata(512, 4_304, 1_152))
        add(
            "vision.bias_gelu",
            stage,
            "custom",
            short="bias_activation_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(512, 4_304) + bf16_tensor_bytes(4_304),
        )
        add("vision.fc2_gemm", stage, "gemm", gemm=gemm_metadata(512, 1_152, 4_304))
        add(
            "vision.bias_residual",
            stage,
            "custom",
            short="bias_residual_bf16_kernel",
            traffic=3 * bf16_tensor_bytes(512, 1_152) + bf16_tensor_bytes(1_152),
        )
    add(
        "vision.post_layer_norm",
        stage,
        "custom",
        short="layer_norm_bf16_kernel",
        traffic=2 * bf16_tensor_bytes(512, 1_152) + 2 * bf16_tensor_bytes(1_152),
    )
    add("vision.projector_gemm", stage, "gemm", gemm=gemm_metadata(512, 2_048, 1_152))
    add(
        "vision.projector_bias",
        stage,
        "custom",
        short="bias_activation_bf16_kernel",
        traffic=2 * bf16_tensor_bytes(512, 2_048) + bf16_tensor_bytes(2_048),
    )

    stage = "language_prefix"
    add(
        "language.embedding",
        stage,
        "custom",
        short="embedding_bf16_kernel",
        traffic=bf16_tensor_bytes(token_count, 2_048),
    )
    add(
        "language.concat_prefix",
        stage,
        "custom",
        short="concat_rows_bf16_kernel",
        traffic=2 * bf16_tensor_bytes(prefix_tokens, 2_048),
    )
    language_attention = attention_metadata(
        query_tokens=prefix_tokens,
        key_tokens=prefix_tokens,
        query_heads=8,
        kv_heads=1,
        head_dim=256,
    )
    for layer in range(18):
        add(
            "language.rms_norm",
            stage,
            "custom",
            short="rms_norm_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(prefix_tokens, 2_048)
            + bf16_tensor_bytes(2_048),
        )
        add(
            "language.qkv_gemm",
            stage,
            "gemm",
            gemm=gemm_metadata(prefix_tokens, 2_560, 2_048),
        )
        add(
            "language.qkv_rope",
            stage,
            "custom",
            short="qkv_rope_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(prefix_tokens, 2_560)
            + bf16_tensor_bytes(2_560),
        )
        if layer == 17:
            continue
        add(
            "language.mqa_fa2",
            stage,
            "attention",
            short="flash_fwd_kernel",
            attention=language_attention,
            traffic=attention_traffic(language_attention),
        )
        add(
            "language.attention_out_gemm",
            stage,
            "gemm",
            gemm=gemm_metadata(prefix_tokens, 2_048, 2_048),
        )
        add(
            "language.bias_residual_rms_norm",
            stage,
            "custom",
            short="bias_residual_rms_norm_bf16_kernel",
            traffic=(
                4 * bf16_tensor_bytes(prefix_tokens, 2_048)
                + 2 * bf16_tensor_bytes(2_048)
            ),
        )
        add(
            "language.gate_up_gemm",
            stage,
            "gemm",
            gemm=gemm_metadata(prefix_tokens, 32_768, 2_048),
        )
        add(
            "language.geglu",
            stage,
            "custom",
            short="geglu_bf16_kernel",
            traffic=bf16_tensor_bytes(prefix_tokens, 32_768)
            + bf16_tensor_bytes(prefix_tokens, 16_384),
        )
        add(
            "language.down_gemm",
            stage,
            "gemm",
            gemm=gemm_metadata(prefix_tokens, 2_048, 16_384),
        )
        add(
            "language.bias_residual",
            stage,
            "custom",
            short="bias_residual_bf16_kernel",
            traffic=3 * bf16_tensor_bytes(prefix_tokens, 2_048)
            + bf16_tensor_bytes(2_048),
        )

    stage = "action_denoise_10_steps"
    action_attention = attention_metadata(
        query_tokens=10,
        key_tokens=action_kv_tokens,
        query_heads=8,
        kv_heads=1,
        head_dim=256,
    )
    for step in range(1, 11):
        add(
            "action.input_gemm",
            stage,
            "gemm",
            gemm=gemm_metadata(10, 1_024, 32),
            step=step,
        )
        add(
            "action.input_bias",
            stage,
            "custom",
            short="bias_activation_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(10, 1_024) + bf16_tensor_bytes(1_024),
            step=step,
        )
        add(
            "action.initial_ada_rms_norm",
            stage,
            "custom",
            short="ada_rms_norm_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(10, 1_024) + bf16_tensor_bytes(1_024),
            step=step,
        )
        for _ in range(18):
            add(
                "action.qkv_gemm",
                stage,
                "gemm",
                gemm=gemm_metadata(10, 2_560, 1_024),
                step=step,
            )
            add(
                "action.qkv_rope_cache_write",
                stage,
                "custom",
                short="qkv_rope_bf16_kernel",
                traffic=2 * bf16_tensor_bytes(10, 2_560) + bf16_tensor_bytes(2_560),
                step=step,
            )
            add(
                "action.mqa_fa2",
                stage,
                "attention",
                short="flash_fwd_kernel",
                attention=action_attention,
                traffic=attention_traffic(action_attention),
                step=step,
            )
            add(
                "action.attention_out_gemm",
                stage,
                "gemm",
                gemm=gemm_metadata(10, 1_024, 2_048),
                step=step,
            )
            add(
                "action.attention_gate_residual_norm",
                stage,
                "custom",
                short="ada_gate_residual_rms_norm_bf16_kernel",
                traffic=4 * bf16_tensor_bytes(10, 1_024) + 2 * bf16_tensor_bytes(1_024),
                step=step,
            )
            add(
                "action.gate_up_gemm",
                stage,
                "gemm",
                gemm=gemm_metadata(10, 8_192, 1_024),
                step=step,
            )
            add(
                "action.geglu",
                stage,
                "custom",
                short="geglu_bf16_kernel",
                traffic=bf16_tensor_bytes(10, 8_192) + bf16_tensor_bytes(10, 4_096),
                step=step,
            )
            add(
                "action.down_gemm",
                stage,
                "gemm",
                gemm=gemm_metadata(10, 1_024, 4_096),
                step=step,
            )
            add(
                "action.mlp_gate_residual_next_norm",
                stage,
                "custom",
                short="ada_gate_residual_rms_norm_bf16_kernel",
                traffic=4 * bf16_tensor_bytes(10, 1_024) + 2 * bf16_tensor_bytes(1_024),
                step=step,
            )
        add(
            "action.output_gemm",
            stage,
            "gemm",
            gemm=gemm_metadata(10, 32, 1_024),
            step=step,
        )
        add(
            "action.output_bias",
            stage,
            "custom",
            short="bias_activation_bf16_kernel",
            traffic=2 * bf16_tensor_bytes(10, 32) + bf16_tensor_bytes(32),
            step=step,
        )
        add(
            "action.euler_update",
            stage,
            "custom",
            short="euler_update_bf16_kernel",
            traffic=3 * bf16_tensor_bytes(10, 32),
            step=step,
        )

    if len(schedule) != EXPECTED_LOGICAL_KERNELS:
        raise AssertionError(
            f"internal schedule has {len(schedule)} operations, expected {EXPECTED_LOGICAL_KERNELS}"
        )
    gemm_count = sum(operation.kind == "gemm" for operation in schedule)
    if gemm_count != EXPECTED_GEMMS:
        raise AssertionError(
            f"internal schedule has {gemm_count} GEMMs, expected {EXPECTED_GEMMS}"
        )
    return schedule


def is_bf16_gemm(kernel: sqlite3.Row) -> bool:
    short = kernel["short_name"]
    return short.startswith("ampere_bf16_") or (
        short == "Kernel2" and "bf16" in kernel["demangled_name"]
    )


def parse_schedule(
    kernels: list[sqlite3.Row], schedule: list[OperationSpec], precision: str
) -> list[dict[str, Any]]:
    cursor = 0
    parsed = []
    for logical_index, operation in enumerate(schedule, 1):
        selected = []
        if cursor >= len(kernels):
            raise SystemExit(
                f"trace ended before logical operation {logical_index} {operation.name}"
            )
        if operation.kind == "gemm" and precision == "bf16":
            if not is_bf16_gemm(kernels[cursor]):
                raise SystemExit(
                    f"expected BF16 GEMM for {operation.name} at kernel {cursor + 1}, "
                    f"found {kernels[cursor]['short_name']}"
                )
            selected.append(kernels[cursor])
            cursor += 1
            if (
                cursor < len(kernels)
                and kernels[cursor]["short_name"] == "splitKreduce_kernel"
            ):
                selected.append(kernels[cursor])
                cursor += 1
        elif operation.kind == "gemm":
            if kernels[cursor]["short_name"] != "quantize_rows_bf16_int8_kernel":
                raise SystemExit(
                    f"expected INT8 row quantization for {operation.name} at kernel {cursor + 1}, "
                    f"found {kernels[cursor]['short_name']}"
                )
            selected.append(kernels[cursor])
            cursor += 1
            if operation.name == "vision.patch_gemm":
                expected = ("Kernel2", "dequantize_int32_bf16_kernel")
                for short_name in expected:
                    if (
                        cursor >= len(kernels)
                        or kernels[cursor]["short_name"] != short_name
                    ):
                        found = (
                            kernels[cursor]["short_name"]
                            if cursor < len(kernels)
                            else "EOF"
                        )
                        raise SystemExit(
                            f"expected {short_name} in unaligned patch GEMM, found {found}"
                        )
                    selected.append(kernels[cursor])
                    cursor += 1
            else:
                if cursor >= len(kernels) or kernels[cursor]["short_name"] != "Kernel":
                    found = (
                        kernels[cursor]["short_name"]
                        if cursor < len(kernels)
                        else "EOF"
                    )
                    raise SystemExit(
                        f"expected fused INT8 GEMM for {operation.name}, found {found}"
                    )
                selected.append(kernels[cursor])
                cursor += 1
        else:
            if kernels[cursor]["short_name"] != operation.expected_short_name:
                raise SystemExit(
                    f"expected {operation.expected_short_name} for {operation.name} at kernel "
                    f"{cursor + 1}, found {kernels[cursor]['short_name']}"
                )
            if operation.kind == "attention":
                padded_head_dim = 96 if operation.attention["head_dim"] == 72 else 256
                if f"(int){padded_head_dim}" not in kernels[cursor]["demangled_name"]:
                    raise SystemExit(
                        f"expected FA2 head dimension {padded_head_dim} for {operation.name} "
                        f"at kernel {cursor + 1}"
                    )
            selected.append(kernels[cursor])
            cursor += 1

        parsed.append(
            {
                "spec": operation,
                "kernels": selected,
                "start": selected[0]["start"],
                "end": selected[-1]["end"],
                "duration": sum(kernel["end"] - kernel["start"] for kernel in selected),
            }
        )

    if cursor != len(kernels):
        remaining = [kernel["short_name"] for kernel in kernels[cursor : cursor + 8]]
        raise SystemExit(
            f"{len(kernels) - cursor} unparsed kernels remain: {remaining}"
        )
    return parsed


def kernel_category(short_name: str, precision: str) -> str:
    if short_name == "rgb_u8_to_patches_bf16_kernel":
        return "raw_image_preprocess"
    if short_name == "flash_fwd_kernel":
        return "flash_attention_2"
    if precision == "bf16" and (
        short_name.startswith("ampere_bf16_") or short_name == "Kernel2"
    ):
        return "bf16_gemm_main"
    if short_name == "splitKreduce_kernel":
        return "bf16_split_k_reduction"
    if short_name == "quantize_rows_bf16_int8_kernel":
        return "int8_activation_row_quantization"
    if short_name == "Kernel":
        return "int8_fused_cutlass_gemm_epilogue"
    if precision == "int8" and short_name == "Kernel2":
        return "int8_unaligned_patch_cublas_gemm"
    if short_name == "dequantize_int32_bf16_kernel":
        return "int8_unaligned_patch_dequantization"
    return "bf16_elementwise_or_layout"


def kernel_family(kernel: sqlite3.Row, precision: str) -> str:
    short = kernel["short_name"]
    demangled = kernel["demangled_name"]
    if short == "flash_fwd_kernel":
        return "fa2.bf16.head96" if "(int)96" in demangled else "fa2.bf16.head256"
    if short == "Kernel" and precision == "int8":
        match = re.search(
            r"GemmShape<\(int\)(\d+), \(int\)(\d+), \(int\)(\d+)>", demangled
        )
        shape = "x".join(match.groups()) if match else "unknown"
        return f"cutlass.int8_w8a8_to_bf16.{shape}"
    if short == "Kernel2":
        if precision == "int8":
            return "cublas.int8_unaligned_patch"
        if "gemm_relu" in demangled:
            return "cutlass.bf16.256x64_relu_named_epilogue"
        return "cutlass.bf16.128x128"
    if short.startswith("ampere_bf16_"):
        return f"cublas.{short}"
    return f"custom.{short}"


def gemm_minimum_bytes(
    metadata: dict[str, int], precision: str, unaligned: bool
) -> int:
    m, n, k = metadata["m"], metadata["n"], metadata["k"]
    if precision == "bf16":
        return 2 * m * k + 2 * k * n + 2 * m * n
    if unaligned:
        return 4 * m * k + k * n + 10 * m * n + 8 * m + 4 * n
    return 4 * m * k + k * n + 2 * m * n + 8 * m + 4 * n


def main() -> None:
    arguments = parse_arguments()
    schedule = build_schedule(arguments.token_count)
    database_uri = f"file:{arguments.sqlite.resolve()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    kernels = list(
        connection.execute(
            """
            SELECT k.start, k.end, k.graphNodeId, k.gridX, k.gridY,
                   short.value AS short_name, demangled.value AS demangled_name
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS short ON short.id = k.shortName
            JOIN StringIds AS demangled ON demangled.id = k.demangledName
            ORDER BY k.start
            """
        )
    )
    copies = list(
        connection.execute(
            """
            SELECT copies.start, copies.end, copies.bytes, kinds.label AS copy_kind
            FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copies
            JOIN ENUM_CUDA_MEMCPY_OPER AS kinds ON kinds.id = copies.copyKind
            ORDER BY copies.start
            """
        )
    )
    runtime_rows = list(
        connection.execute(
            """
            SELECT strings.value AS name, runtime.end - runtime.start AS duration
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
            JOIN StringIds AS strings ON strings.id = runtime.nameId
            ORDER BY runtime.start
            """
        )
    )
    connection.close()

    parsed = parse_schedule(kernels, schedule, arguments.precision)
    prefix_copy_bytes = (512 + arguments.token_count) * 256 * 2
    invalid_copies = [
        copy
        for copy in copies
        if copy["copy_kind"] != "Device-to-Device" or copy["bytes"] != prefix_copy_bytes
    ]
    if len(copies) != EXPECTED_PREFIX_COPIES or invalid_copies:
        raise SystemExit(
            f"expected {EXPECTED_PREFIX_COPIES} {prefix_copy_bytes}-byte D2D prefix copies; "
            f"found {len(copies)}, invalid={len(invalid_copies)}"
        )
    if not all(
        parsed[0]["start"] <= copy["start"] <= parsed[-1]["end"] for copy in copies
    ):
        raise SystemExit("a prefix K/V copy lies outside the graph kernel interval")

    short_counts = Counter(kernel["short_name"] for kernel in kernels)
    if arguments.precision == "bf16":
        if short_counts["splitKreduce_kernel"] != EXPECTED_BF16_SPLIT_K:
            raise SystemExit(
                f"expected {EXPECTED_BF16_SPLIT_K} split-K reductions, "
                f"found {short_counts['splitKreduce_kernel']}"
            )
    else:
        expected = {
            "quantize_rows_bf16_int8_kernel": EXPECTED_GEMMS,
            "Kernel": EXPECTED_INT8_FUSED_GEMMS,
            "Kernel2": 1,
            "dequantize_int32_bf16_kernel": 1,
        }
        for short_name, count in expected.items():
            if short_counts[short_name] != count:
                raise SystemExit(
                    f"expected {count} {short_name} kernels, found {short_counts[short_name]}"
                )

    all_intervals = [(kernel["start"], kernel["end"]) for kernel in kernels] + [
        (copy["start"], copy["end"]) for copy in copies
    ]
    graph_start = min(start for start, _ in all_intervals)
    graph_end = max(end for _, end in all_intervals)
    graph_span = graph_end - graph_start
    busy_time = merge_busy_time(all_intervals)

    first_stage_start = {
        stage: next(item["start"] for item in parsed if item["spec"].stage == stage)
        for stage in STAGE_ORDER
    }
    stage_boundaries = {}
    for index, stage in enumerate(STAGE_ORDER):
        start = first_stage_start[stage]
        end = (
            first_stage_start[STAGE_ORDER[index + 1]]
            if index + 1 < len(STAGE_ORDER)
            else graph_end
        )
        stage_boundaries[stage] = (start, end)
    language_start, language_end = stage_boundaries["language_prefix"]
    if not all(language_start <= copy["start"] < language_end for copy in copies):
        raise SystemExit("a prefix K/V copy lies outside the language-prefix stage")

    stages = []
    for stage in STAGE_ORDER:
        selected = [item for item in parsed if item["spec"].stage == stage]
        start, end = stage_boundaries[stage]
        stage_copies = [copy for copy in copies if start <= copy["start"] < end]
        span = end - start
        stages.append(
            {
                "stage": stage,
                "critical_span_ms": round(span / 1e6, 6),
                "graph_span_percent": round(100.0 * span / graph_span, 4),
                "normalized_to_graph_p50_ms": round(
                    span / graph_span * arguments.graph_p50_ms, 6
                ),
                "logical_operation_count": len(selected),
                "kernel_count": sum(len(item["kernels"]) for item in selected),
                "kernel_time_ms": round(
                    sum(item["duration"] for item in selected) / 1e6, 6
                ),
                "memcpy_count": len(stage_copies),
                "memcpy_time_ms": round(
                    sum(copy["end"] - copy["start"] for copy in stage_copies) / 1e6, 6
                ),
            }
        )

    flow_steps = []
    for step in range(1, 11):
        selected = [item for item in parsed if item["spec"].flow_step == step]
        start, end = selected[0]["start"], selected[-1]["end"]
        flow_steps.append(
            {
                "step": step,
                "logical_operation_count": len(selected),
                "kernel_count": sum(len(item["kernels"]) for item in selected),
                "kernel_time_ms": round(
                    sum(item["duration"] for item in selected) / 1e6, 6
                ),
                "critical_span_ms": round((end - start) / 1e6, 6),
            }
        )

    grouped_operations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed:
        grouped_operations[item["spec"].name].append(item)
    operations = []
    for name, items in grouped_operations.items():
        spec: OperationSpec = items[0]["spec"]
        durations = [item["duration"] for item in items]
        total_duration = sum(durations)
        entry: dict[str, Any] = {
            "operation": name,
            "stage": spec.stage,
            "kind": spec.kind,
            **duration_summary(durations, graph_span),
            "normalized_to_graph_p50_ms": round(
                total_duration / graph_span * arguments.graph_p50_ms, 6
            ),
            "cuda_kernel_count": sum(len(item["kernels"]) for item in items),
        }
        if spec.gemm is not None:
            calls = len(items)
            component_ns: defaultdict[str, int] = defaultdict(int)
            for item in items:
                for kernel in item["kernels"]:
                    component_ns[
                        kernel_category(kernel["short_name"], arguments.precision)
                    ] += (kernel["end"] - kernel["start"])
            operations_count = (
                2 * spec.gemm["m"] * spec.gemm["n"] * spec.gemm["k"] * calls
            )
            traffic = (
                gemm_minimum_bytes(
                    spec.gemm, arguments.precision, name == "vision.patch_gemm"
                )
                * calls
            )
            entry["gemm"] = {
                **spec.gemm,
                "math": "2 operations per multiply-accumulate",
                "total_tera_operations": round(operations_count / 1e12, 6),
                "effective_tops": round(operations_count / (total_duration * 1e3), 3),
                "minimum_algorithmic_bytes": traffic,
                "effective_minimum_bandwidth_gbps": round(traffic / total_duration, 3),
                "pipeline_components_ms": {
                    name: round(duration / 1e6, 6)
                    for name, duration in sorted(component_ns.items())
                },
            }
        elif spec.attention is not None:
            calls = len(items)
            q = spec.attention["query_tokens"]
            k = spec.attention["key_tokens"]
            heads = spec.attention["query_heads"]
            d = spec.attention["head_dim"]
            batch = spec.attention["batch"]
            operations_count = 4 * batch * q * k * heads * d * calls
            traffic = (spec.traffic_bytes_per_call or 0) * calls
            entry["attention"] = {
                **spec.attention,
                "total_tera_operations": round(operations_count / 1e12, 6),
                "effective_tops": round(operations_count / (total_duration * 1e3), 3),
                "minimum_algorithmic_bytes": traffic,
                "effective_minimum_bandwidth_gbps": round(traffic / total_duration, 3),
            }
        elif spec.traffic_bytes_per_call is not None:
            traffic = spec.traffic_bytes_per_call * len(items)
            entry["minimum_algorithmic_bytes"] = traffic
            entry["effective_minimum_bandwidth_gbps"] = round(
                traffic / total_duration, 3
            )
        operations.append(entry)
    operations.sort(key=lambda item: item["total_ms"], reverse=True)

    category_durations: defaultdict[str, list[int]] = defaultdict(list)
    family_durations: defaultdict[str, list[int]] = defaultdict(list)
    for kernel in kernels:
        duration = kernel["end"] - kernel["start"]
        category_durations[
            kernel_category(kernel["short_name"], arguments.precision)
        ].append(duration)
        family_durations[kernel_family(kernel, arguments.precision)].append(duration)
    category_durations["prefix_kv_device_copy"].extend(
        copy["end"] - copy["start"] for copy in copies
    )
    categories = [
        {"category": name, **duration_summary(durations, graph_span)}
        for name, durations in category_durations.items()
    ]
    categories.sort(key=lambda item: item["total_ms"], reverse=True)
    families = [
        {"kernel_family": name, **duration_summary(durations, graph_span)}
        for name, durations in family_durations.items()
    ]
    families.sort(key=lambda item: item["total_ms"], reverse=True)

    provenance: dict[str, Any] = {
        "analyzer": Path(__file__).name,
        "analyzer_sha256": sha256(Path(__file__)),
        "sqlite": arguments.sqlite.name,
        "sqlite_sha256": sha256(arguments.sqlite),
    }
    if arguments.report is not None:
        provenance.update(
            {
                "nsys_report": arguments.report.name,
                "nsys_report_sha256": sha256(arguments.report),
            }
        )
    if arguments.remote_report is not None:
        provenance["remote_report"] = arguments.remote_report
    if arguments.benchmark_binary_sha256 is not None:
        provenance["benchmark_binary_sha256"] = arguments.benchmark_binary_sha256

    output = {
        "schema": "apxinf.pi05.sm87-nsys-breakdown.v1",
        "precision": arguments.precision,
        "provenance": provenance,
        "environment": {
            "device": "NVIDIA Jetson AGX Orin Developer Kit",
            "cuda_arch": "sm_87",
            "cuda_version": arguments.cuda_version,
            "nsight_systems_version": arguments.nsys_version,
            "power_mode": "MAXN",
            "gpu_clock_hz": 1_300_500_000,
            "emc_clock_hz": 3_199_000_000,
            "active_gpu_tpcs": 8,
        },
        "workload": {
            "image_input": "uint8_nhwc_2x224x224x3",
            "prompt_tokens": arguments.token_count,
            "vision_tokens": 512,
            "prefix_tokens": 512 + arguments.token_count,
            "action_kv_tokens": 522 + arguments.token_count,
            "action_horizon": 10,
            "action_dim": 32,
            "flow_steps": 10,
            "vision_layers": 27,
            "language_layers": 18,
            "action_layers_per_step": 18,
        },
        "production_baseline_ms": {
            "graph_p50": arguments.graph_p50_ms,
            "raw_input_update_plus_graph_p50": arguments.update_plus_graph_p50_ms,
            "full_stdio_request_p50": arguments.protocol_p50_ms,
        },
        "trace": {
            "logical_operation_count": len(parsed),
            "kernel_count": len(kernels),
            "memcpy_count": len(copies),
            "gpu_critical_span_ms": round(graph_span / 1e6, 6),
            "kernel_time_ms": round(
                sum(kernel["end"] - kernel["start"] for kernel in kernels) / 1e6, 6
            ),
            "memcpy_time_ms": round(
                sum(copy["end"] - copy["start"] for copy in copies) / 1e6, 6
            ),
            "gpu_busy_union_ms": round(busy_time / 1e6, 6),
            "gpu_inter_node_gap_ms": round((graph_span - busy_time) / 1e6, 6),
            "runtime_api_ms": {
                row["name"]: round(row["duration"] / 1e6, 6) for row in runtime_rows
            },
            "note": (
                "CUDA graph node tracing perturbs timing; production P50 values are the "
                "performance baselines and this one-replay trace supplies attribution."
            ),
        },
        "stages": stages,
        "flow_steps": flow_steps,
        "categories": categories,
        "operations": operations,
        "kernel_families": families,
        "validation": {
            "static_mapping_passed": True,
            "expected_logical_operation_count": EXPECTED_LOGICAL_KERNELS,
            "expected_gemm_count": EXPECTED_GEMMS,
            "expected_prefix_kv_copy_count": EXPECTED_PREFIX_COPIES,
            "expected_prefix_kv_copy_bytes": prefix_copy_bytes,
            "actual_short_name_counts": dict(sorted(short_counts.items())),
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=False) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(arguments.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

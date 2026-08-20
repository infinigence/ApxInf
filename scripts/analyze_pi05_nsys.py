#!/usr/bin/env python3
"""Attribute a token-10, two-view π0.5 Nsight Systems trace.

The optimized CUDA graph has a deliberately static launch sequence.  This
script validates that sequence and converts an Nsight Systems SQLite export
into a compact JSON breakdown by stage, flow step, logical operation, exact
GEMM shape, and kernel family.

Export a captured report with:

    nsys export --type sqlite --output=pi05.sqlite pi05.nsys-rep
    python3 scripts/analyze_pi05_nsys.py pi05.sqlite --report pi05.nsys-rep

The mapping matches `Pi05Config::thor_two_view()` with 10 prompt tokens.  It
fails closed if the kernel or K/V-copy count changes so a modified graph is
never silently attributed using a stale schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


EXPECTED_KERNELS = 2_688
EXPECTED_MEMCPY = 36
EXPECTED_PREFIX_KV_COPY_BYTES = 522 * 256 * 2

VISION_LAYER_OPERATIONS = (
    "vision.layer_norm_quant",
    "vision.qkv_gemm",
    "vision.qkv_split_bias",
    "vision.fmha",
    "vision.attention_quant",
    "vision.attention_out_gemm",
    "vision.attention_residual_norm_quant",
    "vision.fc1_gemm_bias_gelu_quant",
    "vision.fc2_gemm_bias_residual",
)

LANGUAGE_LAYER_OPERATIONS = (
    "language.rms_norm_quant",
    "language.qkv_gemm",
    "language.qkv_rope",
    "language.attention_qk_fp16",
    "language.attention_softmax",
    "language.attention_pv_fp16",
    "language.attention_quant",
    "language.attention_out_gemm",
    "language.attention_residual_norm_quant",
    "language.gate_up_gemm",
    "language.geglu_quant",
    "language.down_gemm_bias_residual",
)

ACTION_LAYER_OPERATIONS = (
    "action.qkv_gemm",
    "action.qkv_rope_cache_write",
    "action.attention_qk_fp16",
    "action.attention_softmax",
    "action.attention_pv_fp16",
    "action.attention_quant",
    "action.attention_out_gemm",
    "action.attention_gate_residual_norm_quant",
    "action.gate_up_gemm",
    "action.geglu_quant",
    "action.down_gemm",
    "action.mlp_gate_residual_next_norm_quant",
)


GEMM_METADATA: dict[str, dict[str, Any]] = {
    "vision.patch_gemm": dict(m=512, n=1_152, k=588, dtype="fp8", backend="cublaslt"),
    "vision.qkv_gemm": dict(m=512, n=3_456, k=1_152, dtype="fp8", backend="cublaslt"),
    "vision.attention_out_gemm": dict(
        m=512, n=1_152, k=1_152, dtype="fp8", backend="cublaslt"
    ),
    "vision.fc1_gemm_bias_gelu_quant": dict(
        m=512,
        n=4_304,
        k=1_152,
        dtype="fp8",
        backend="cublaslt_fused_epilogue",
    ),
    "vision.fc2_gemm_bias_residual": dict(
        m=512,
        n=1_152,
        k=4_304,
        dtype="fp8",
        backend="cublaslt_fused_epilogue",
    ),
    "vision.projector_gemm": dict(
        m=512, n=2_048, k=1_152, dtype="fp8", backend="cutlass_tactic_4"
    ),
    "language.qkv_gemm": dict(
        m=522, n=2_560, k=2_048, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "language.attention_qk_fp16": dict(
        batch=8, m=522, n=522, k=256, dtype="fp16", backend="cublas"
    ),
    "language.attention_pv_fp16": dict(
        batch=8, m=522, n=256, k=522, dtype="fp16", backend="cublas"
    ),
    "language.attention_out_gemm": dict(
        m=522, n=2_048, k=2_048, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "language.gate_up_gemm": dict(
        m=522, n=32_768, k=2_048, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "language.down_gemm_bias_residual": dict(
        m=522,
        n=2_048,
        k=16_384,
        dtype="fp8",
        backend="cublaslt_fused_residual",
    ),
    "action.input_gemm": dict(
        m=10, n=1_024, k=32, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "action.qkv_gemm": dict(
        m=10, n=2_560, k=1_024, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "action.attention_qk_fp16": dict(
        batch=8, m=10, n=532, k=256, dtype="fp16", backend="cublas"
    ),
    "action.attention_pv_fp16": dict(
        batch=8, m=10, n=256, k=532, dtype="fp16", backend="cublas"
    ),
    "action.attention_out_gemm": dict(
        m=10, n=1_024, k=2_048, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "action.gate_up_gemm": dict(
        m=10, n=8_192, k=1_024, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "action.down_gemm": dict(
        m=10, n=1_024, k=4_096, dtype="fp8", backend="cublaslt_rank_0"
    ),
    "action.output_gemm": dict(
        m=10, n=32, k=1_024, dtype="fp8", backend="cublaslt"
    ),
}


EXPECTED_OPERATION_COUNTS = {
    "raw.rgb_to_patches_quant": 1,
    "vision.patch_gemm": 1,
    "vision.bias_position": 1,
    **{operation: 27 for operation in VISION_LAYER_OPERATIONS},
    "vision.post_norm_quant": 1,
    "vision.projector_gemm": 1,
    "vision.projector_bias": 1,
    "language.embedding": 1,
    "language.concat_prefix": 1,
    "language.rms_norm_quant": 18,
    "language.qkv_gemm": 18,
    "language.qkv_rope": 18,
    **{operation: 17 for operation in LANGUAGE_LAYER_OPERATIONS[3:]},
    "action.input_quant": 10,
    "action.input_gemm": 10,
    "action.input_bias": 10,
    "action.initial_ada_rms_norm_quant": 10,
    **{operation: 180 for operation in ACTION_LAYER_OPERATIONS},
    "action.output_gemm": 10,
    "action.output_bias": 10,
    "action.euler_update": 10,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="Nsight Systems SQLite export")
    parser.add_argument(
        "--report", type=Path, help="optional source .nsys-rep used for provenance hashing"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[int], fraction: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def logical_operation(sequence: int) -> str:
    if sequence == 1:
        return "raw.rgb_to_patches_quant"
    if sequence == 2:
        return "vision.patch_gemm"
    if sequence == 3:
        return "vision.bias_position"
    if 4 <= sequence <= 246:
        return VISION_LAYER_OPERATIONS[(sequence - 4) % len(VISION_LAYER_OPERATIONS)]
    if sequence == 247:
        return "vision.post_norm_quant"
    if sequence == 248:
        return "vision.projector_gemm"
    if sequence == 249:
        return "vision.projector_bias"
    if sequence == 250:
        return "language.embedding"
    if sequence == 251:
        return "language.concat_prefix"
    if 252 <= sequence <= 455:
        return LANGUAGE_LAYER_OPERATIONS[(sequence - 252) % len(LANGUAGE_LAYER_OPERATIONS)]
    if sequence == 456:
        return "language.rms_norm_quant"
    if sequence == 457:
        return "language.qkv_gemm"
    if sequence == 458:
        return "language.qkv_rope"

    step_position = (sequence - 459) % 223
    if step_position == 0:
        return "action.input_quant"
    if step_position == 1:
        return "action.input_gemm"
    if step_position == 2:
        return "action.input_bias"
    if step_position == 3:
        return "action.initial_ada_rms_norm_quant"
    if 4 <= step_position <= 219:
        return ACTION_LAYER_OPERATIONS[(step_position - 4) % len(ACTION_LAYER_OPERATIONS)]
    if step_position == 220:
        return "action.output_gemm"
    if step_position == 221:
        return "action.output_bias"
    if step_position == 222:
        return "action.euler_update"
    raise AssertionError(f"unmapped kernel sequence {sequence}")


def stage_for_sequence(sequence: int) -> str:
    if sequence == 1:
        return "raw_preprocess"
    if sequence <= 249:
        return "vision_encoder"
    if sequence <= 458:
        return "language_prefix"
    return "action_denoise_10_steps"


def kernel_family(short_name: str, demangled_name: str) -> str:
    if short_name.startswith("nvjet_"):
        return f"cublaslt.{short_name}"
    if short_name.startswith("cutlass3x_"):
        if "gelu" in short_name:
            return "cublaslt.siglip_fc1_bias_gelu_fp8"
        return f"cublaslt.{short_name}"
    if "Sm100Fmha" in demangled_name:
        return "cutlass.sm100_vision_fmha"
    if "GemmUniversal" in demangled_name:
        return "cutlass.sm100_fp8_gemm"
    if short_name == "Kernel2":
        match = re.search(r"cutlass_80_([^>]+)", demangled_name)
        suffix = match.group(1) if match else "unknown"
        return f"cublas.fp16.{suffix}"
    return f"custom.{short_name}"


def operation_category(operation: str) -> str:
    metadata = GEMM_METADATA.get(operation)
    if metadata is not None:
        return "fp8_gemm" if metadata["dtype"] == "fp8" else "fp16_attention_gemm"
    if operation == "vision.fmha":
        return "fused_vision_attention"
    if operation == "raw.rgb_to_patches_quant":
        return "raw_image_preprocess"
    if operation == "language.reserve_kv_d2d":
        return "device_memory_copy"
    return "fused_elementwise_or_layout"


def merge_busy_time(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
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


def main() -> None:
    arguments = parse_arguments()
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
    if len(kernels) != EXPECTED_KERNELS:
        raise SystemExit(
            f"expected {EXPECTED_KERNELS} graph kernels, found {len(kernels)}; "
            "the static mapping must be updated"
        )
    if len(copies) != EXPECTED_MEMCPY:
        raise SystemExit(
            f"expected {EXPECTED_MEMCPY} graph memcopies, found {len(copies)}; "
            "the static mapping must be updated"
        )
    invalid_copies = [
        copy
        for copy in copies
        if copy["copy_kind"] != "Device-to-Device"
        or copy["bytes"] != EXPECTED_PREFIX_KV_COPY_BYTES
    ]
    if invalid_copies:
        raise SystemExit(
            "expected every graph copy to be a 267264-byte device-to-device prefix K/V copy"
        )

    operations: defaultdict[str, list[int]] = defaultdict(list)
    families: defaultdict[str, list[int]] = defaultdict(list)
    stages: defaultdict[str, list[int]] = defaultdict(list)
    for sequence, kernel in enumerate(kernels, start=1):
        duration = kernel["end"] - kernel["start"]
        operation = logical_operation(sequence)
        operations[operation].append(duration)
        stages[stage_for_sequence(sequence)].append(duration)
        families[kernel_family(kernel["short_name"], kernel["demangled_name"])].append(duration)

    actual_counts = Counter({name: len(values) for name, values in operations.items()})
    expected_counts = Counter(EXPECTED_OPERATION_COUNTS)
    if actual_counts != expected_counts:
        missing = expected_counts - actual_counts
        extra = actual_counts - expected_counts
        raise SystemExit(f"logical operation count mismatch; missing={dict(missing)}, extra={dict(extra)}")

    graph_start = kernels[0]["start"]
    graph_end = kernels[-1]["end"]
    graph_span = graph_end - graph_start
    kernel_time = sum(kernel["end"] - kernel["start"] for kernel in kernels)
    memcpy_time = sum(copy["end"] - copy["start"] for copy in copies)
    busy_time = merge_busy_time(
        [(kernel["start"], kernel["end"]) for kernel in kernels]
        + [(copy["start"], copy["end"]) for copy in copies]
    )

    boundaries = {
        "raw_preprocess": (kernels[0]["start"], kernels[1]["start"]),
        "vision_encoder": (kernels[1]["start"], kernels[249]["start"]),
        "language_prefix": (kernels[249]["start"], kernels[458]["start"]),
        "action_denoise_10_steps": (kernels[458]["start"], kernels[-1]["end"]),
    }

    stage_output = []
    for name in (
        "raw_preprocess",
        "vision_encoder",
        "language_prefix",
        "action_denoise_10_steps",
    ):
        start, end = boundaries[name]
        span = end - start
        copy_ns = (
            sum(copy["end"] - copy["start"] for copy in copies)
            if name == "language_prefix"
            else 0
        )
        stage_output.append(
            {
                "stage": name,
                "critical_span_ms": round(span / 1e6, 6),
                "graph_span_percent": round(100.0 * span / graph_span, 4),
                "kernel_count": len(stages[name]),
                "kernel_time_ms": round(sum(stages[name]) / 1e6, 6),
                "memcpy_count": len(copies) if name == "language_prefix" else 0,
                "memcpy_time_ms": round(copy_ns / 1e6, 6),
            }
        )

    step_output = []
    for step in range(10):
        first = 458 + step * 223
        selected = kernels[first : first + 223]
        start = selected[0]["start"]
        end = selected[-1]["end"]
        step_output.append(
            {
                "step": step + 1,
                "kernel_count": len(selected),
                "kernel_time_ms": round(
                    sum(kernel["end"] - kernel["start"] for kernel in selected) / 1e6,
                    6,
                ),
                "critical_span_ms": round((end - start) / 1e6, 6),
            }
        )

    operation_output = []
    for name, durations in operations.items():
        entry = {
            "operation": name,
            "category": operation_category(name),
            **duration_summary(durations, graph_span),
        }
        metadata = GEMM_METADATA.get(name)
        if metadata is not None:
            batch = metadata.get("batch", 1)
            flops = 2 * batch * metadata["m"] * metadata["n"] * metadata["k"] * len(durations)
            entry["gemm"] = {
                **metadata,
                "total_tflop": round(flops / 1e12, 6),
                "effective_tflops": round(flops / (sum(durations) * 1e3), 3),
            }
        operation_output.append(entry)
    operation_output.sort(key=lambda item: item["total_ms"], reverse=True)

    category_durations: defaultdict[str, list[int]] = defaultdict(list)
    for operation, durations in operations.items():
        category_durations[operation_category(operation)].extend(durations)
    category_durations["device_memory_copy"].extend(
        copy["end"] - copy["start"] for copy in copies
    )
    category_output = [
        {"category": name, **duration_summary(durations, graph_span)}
        for name, durations in category_durations.items()
    ]
    category_output.sort(key=lambda item: item["total_ms"], reverse=True)

    family_output = [
        {"kernel_family": name, **duration_summary(durations, graph_span)}
        for name, durations in families.items()
    ]
    family_output.sort(key=lambda item: item["total_ms"], reverse=True)

    nvtx_row = connection.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = 'pi05.graph_replay' LIMIT 1"
    ).fetchone()
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

    provenance: dict[str, Any] = {
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

    output = {
        "schema": "apxinf.pi05.nsys-breakdown.v1",
        "provenance": provenance,
        "workload": {
            "device": "NVIDIA Jetson AGX Thor Developer Kit",
            "cuda_arch": "sm_110",
            "image_input": "uint8_nhwc_2x224x224x3",
            "prompt_tokens": 10,
            "vision_tokens": 512,
            "prefix_tokens": 522,
            "action_horizon": 10,
            "action_dim": 32,
            "flow_steps": 10,
            "vision_layers": 27,
            "language_layers": 18,
            "action_layers_per_step": 18,
        },
        "trace": {
            "kernel_count": len(kernels),
            "memcpy_count": len(copies),
            "gpu_critical_span_ms": round(graph_span / 1e6, 6),
            "kernel_time_ms": round(kernel_time / 1e6, 6),
            "memcpy_time_ms": round(memcpy_time / 1e6, 6),
            "gpu_busy_union_ms": round(busy_time / 1e6, 6),
            "gpu_inter_node_gap_ms": round((graph_span - busy_time) / 1e6, 6),
            "nvtx_host_range_ms": (
                round((nvtx_row["end"] - nvtx_row["start"]) / 1e6, 6)
                if nvtx_row is not None
                else None
            ),
            "runtime_api_ms": {
                row["name"]: round(row["duration"] / 1e6, 6) for row in runtime_rows
            },
            "note": (
                "CUDA graph node tracing perturbs launch and kernel timing; use the adjacent "
                "unprofiled fixed-clock P50 as the end-to-end latency baseline and this trace "
                "for attribution."
            ),
        },
        "stages": stage_output,
        "flow_steps": step_output,
        "categories": category_output,
        "operations": operation_output,
        "kernel_families": family_output,
        "validation": {
            "static_mapping_passed": True,
            "expected_kernel_count": EXPECTED_KERNELS,
            "expected_memcpy_count": EXPECTED_MEMCPY,
            "expected_prefix_kv_copy_bytes": EXPECTED_PREFIX_KV_COPY_BYTES,
            "expected_operation_counts": EXPECTED_OPERATION_COUNTS,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()

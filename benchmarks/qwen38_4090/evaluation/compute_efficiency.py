#!/usr/bin/env python3
"""Compute frozen proxy and optional counter-based efficiency metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _profile_metrics(phase: Any, peak_bandwidth_gb_s: float) -> dict[str, float | None] | None:
    if not isinstance(phase, dict):
        return None
    elapsed = _positive(phase.get("kernel_elapsed_s"))
    read_bytes = phase.get("dram_read_bytes")
    write_bytes = phase.get("dram_write_bytes")
    if elapsed is None or not isinstance(read_bytes, (int, float)) or not isinstance(
        write_bytes, (int, float)
    ):
        return None
    if isinstance(read_bytes, bool) or isinstance(write_bytes, bool) or read_bytes < 0 or write_bytes < 0:
        return None
    bandwidth = (float(read_bytes) + float(write_bytes)) / elapsed / 1e9
    tensor_active = phase.get("tensor_pipe_active_pct")
    return {
        "dram_bandwidth_gb_s": bandwidth,
        "profiled_bwu_pct": 100.0 * bandwidth / peak_bandwidth_gb_s,
        "tensor_pipe_active_pct": float(tensor_active)
        if isinstance(tensor_active, (int, float)) and not isinstance(tensor_active, bool)
        else None,
    }


def compute_efficiency(
    contract: dict[str, Any], submission: dict[str, Any]
) -> dict[str, Any]:
    model = contract["model"]
    hardware = contract["hardware"]
    flops_per_token = float(model["dense_equivalent_flops_per_token"])
    attention_layers = float(model["full_attention_layers"])
    query_width = float(model["attention_query_width"])
    weight_bytes = float(model["minimum_weight_bytes"])
    peak_tflops = float(hardware["peak_bf16_tensor_tflops_dense_fp32_accumulate"])
    peak_bandwidth = float(hardware["peak_memory_bandwidth_gb_s"])

    cell_results: dict[str, Any] = {}
    for cell_id, cell in submission.get("cells", {}).items():
        if not isinstance(cell, dict):
            continue
        prompt_tokens = cell.get("actual_prompt_tokens")
        completion_tokens = cell.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            continue
        if not isinstance(completion_tokens, int) or completion_tokens <= 0:
            continue
        profile = cell.get("profile") if isinstance(cell.get("profile"), dict) else {}
        phases: dict[str, Any] = {}

        ttft = _positive(cell.get("ttft_s"))
        if ttft is not None:
            causal_attention_flops = (
                2.0 * query_width * attention_layers * prompt_tokens * (prompt_tokens + 1)
            )
            prefill_flops = flops_per_token * prompt_tokens + causal_attention_flops
            achieved_tflops = prefill_flops / ttft / 1e12
            minimum_bandwidth = weight_bytes / ttft / 1e9
            phases["prefill"] = {
                "duration_s": ttft,
                "estimated_flops": prefill_flops,
                "estimated_tflops": achieved_tflops,
                "estimated_mfu_bf16_equivalent_pct": 100.0 * achieved_tflops / peak_tflops,
                "minimum_model_bandwidth_gb_s": minimum_bandwidth,
                "minimum_model_bwu_pct": 100.0 * minimum_bandwidth / peak_bandwidth,
                "profiled": _profile_metrics(profile.get("prefill"), peak_bandwidth),
            }

        tpot = _positive(cell.get("tpot_s"))
        decode_steps = max(0, completion_tokens - 1)
        if tpot is not None and decode_steps:
            decode_duration = tpot * decode_steps
            attention_flops = 4.0 * query_width * attention_layers * (
                decode_steps * prompt_tokens + decode_steps * (decode_steps - 1) / 2.0
            )
            decode_flops = flops_per_token * decode_steps + attention_flops
            achieved_tflops = decode_flops / decode_duration / 1e12
            minimum_weight_traffic = weight_bytes * decode_steps
            minimum_bandwidth = minimum_weight_traffic / decode_duration / 1e9
            phases["decode"] = {
                "steps": decode_steps,
                "duration_s": decode_duration,
                "estimated_flops": decode_flops,
                "estimated_tflops": achieved_tflops,
                "estimated_mfu_bf16_equivalent_pct": 100.0 * achieved_tflops / peak_tflops,
                "minimum_model_weight_traffic_bytes": minimum_weight_traffic,
                "minimum_model_bandwidth_gb_s": minimum_bandwidth,
                "minimum_model_bwu_pct": 100.0 * minimum_bandwidth / peak_bandwidth,
                "profiled": _profile_metrics(profile.get("decode"), peak_bandwidth),
            }
        cell_results[cell_id] = phases

    return {
        "schema": "apxinf.qwen38_27b.efficiency_report.v1",
        "implementation": submission.get("implementation"),
        "declared_peaks": {
            "bf16_tensor_tflops_dense_fp32_accumulate": peak_tflops,
            "int4_tensor_tops_dense_reference_only": float(hardware["peak_int4_tensor_tops_dense"]),
            "memory_bandwidth_gb_s": peak_bandwidth,
        },
        "method": {
            "mfu": "Frozen BF16-equivalent model FLOP proxy divided by wall time and dense BF16 Tensor Core peak. This is not a measured tensor-pipe utilization.",
            "minimum_model_bwu": "Frozen minimum model/checkpoint byte proxy divided by wall time and peak DRAM bandwidth. It is a lower-bound model-byte proxy, not measured HBM traffic.",
            "profiled_bwu": "When phase-scoped profiler counters are supplied: (dram_read_bytes + dram_write_bytes) / kernel_elapsed_s / peak DRAM bandwidth.",
            "int4_note": "INT4 TOPS is reported only as a hardware reference; it is not used as the MFU denominator for mixed W4A16 execution.",
        },
        "cells": cell_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--contract", type=Path, default=here / "contract-v1.json")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = compute_efficiency(_load_json(args.contract), _load_json(args.submission))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

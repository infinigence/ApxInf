#!/usr/bin/env python3
"""Reproducible lower-bound MFU/BWU estimate for one-token BF16 decode."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecodeShape:
    layers: int = 36
    hidden: int = 2048
    intermediate: int = 11008
    kv_heads: int = 2
    head_dim: int = 128
    vocab: int = 151936
    dtype_bytes: int = 2

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    def linear_weight_elements(self) -> int:
        per_layer = (
            2 * self.hidden * self.hidden
            + 2 * self.hidden * self.kv_width
            + 3 * self.hidden * self.intermediate
        )
        return self.layers * per_layer + self.hidden * self.vocab

    def minimum_weight_elements(self) -> int:
        per_layer = (
            2 * self.hidden
            + 2 * self.hidden * self.hidden
            + 2 * self.hidden * self.kv_width
            + 3 * self.hidden * self.intermediate
            + self.hidden
            + 2 * self.kv_width
        )
        # Layer weights/biases/norms, output norm, LM head and one embedding row.
        return self.layers * per_layer + self.hidden + self.hidden * self.vocab + self.hidden


def estimate(
    shape: DecodeShape,
    tpot_ms: float,
    kv_len: int,
    peak_bandwidth_gbps: float,
    peak_tflops: float | None,
) -> dict[str, object]:
    if tpot_ms <= 0 or kv_len < 0 or peak_bandwidth_gbps <= 0:
        raise ValueError("TPOT and peak bandwidth must be positive; KV length must be nonnegative")
    if peak_tflops is not None and peak_tflops <= 0:
        raise ValueError("peak TFLOPS must be positive when provided")

    seconds = tpot_ms / 1000.0
    weight_bytes = shape.minimum_weight_elements() * shape.dtype_bytes
    kv_read_bytes = 2 * shape.layers * shape.kv_width * kv_len * shape.dtype_bytes
    minimum_bytes = weight_bytes + kv_read_bytes
    linear_flops = 2 * shape.linear_weight_elements()
    effective_weight_gbps = weight_bytes / seconds / 1e9
    effective_minimum_gbps = minimum_bytes / seconds / 1e9
    effective_tflops = linear_flops / seconds / 1e12

    result: dict[str, object] = {
        "schema": "apxinf.qwen25_omni.decode_roofline.v1",
        "shape": asdict(shape),
        "tpot_ms": tpot_ms,
        "kv_len": kv_len,
        "peak_bandwidth_gbps": peak_bandwidth_gbps,
        "weight_bytes_lower_bound": weight_bytes,
        "kv_read_bytes_lower_bound": kv_read_bytes,
        "minimum_bytes_per_token": minimum_bytes,
        "effective_weight_bandwidth_gbps": effective_weight_gbps,
        "effective_minimum_bandwidth_gbps": effective_minimum_gbps,
        "weight_bwu_pct": 100.0 * effective_weight_gbps / peak_bandwidth_gbps,
        "minimum_bwu_pct": 100.0 * effective_minimum_gbps / peak_bandwidth_gbps,
        "linear_flops_per_token": linear_flops,
        "effective_linear_tflops": effective_tflops,
        "peak_tflops": peak_tflops,
        "linear_mfu_pct": (
            None if peak_tflops is None else 100.0 * effective_tflops / peak_tflops
        ),
        "limitations": [
            "Weight and KV bytes are algorithmic lower bounds, not measured HBM transactions",
            "Activation, allocator, cache-line, protocol and replay traffic are excluded",
            "Linear FLOPS count multiply-add as two operations and excludes elementwise work",
            "MFU is omitted unless the caller declares the dense peak convention to use"
        ],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpot-ms", type=float, default=8.255564196850394)
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--peak-bandwidth-gbps", type=float, default=1008.0)
    parser.add_argument("--peak-tflops", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = estimate(
        DecodeShape(),
        args.tpot_ms,
        args.kv_len,
        args.peak_bandwidth_gbps,
        args.peak_tflops,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a deterministic, offline inventory from Qwen metadata fixtures.

This tool reads only config.json and model.safetensors.index.json.  It never
opens a safetensors shard and never downloads weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

MODEL_REVISION = "63768c10df38c0395e12ef49edac1bd539eaeeea"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_prefix(name: str) -> str:
    if name == "lm_head.weight":
        return "lm_head"
    if name.startswith("model.language_model.embed_tokens."):
        return "embed_tokens"
    if name.startswith("model.language_model.layers."):
        return "language_model.layers"
    if name.startswith("model.visual."):
        return "visual"
    if name.startswith("mtp."):
        return "mtp"
    return "other"


def classify_quantization(name: str) -> str:
    if name.endswith(".weight_packed"):
        return "packed_weight"
    if name.endswith(".weight_scale"):
        return "weight_scale"
    if name.endswith(".weight_zero_point"):
        return "weight_zero_point"
    if name.endswith(".weight_shape"):
        return "weight_shape_metadata"
    return "not_quantization_metadata"


def layer_inventory(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config") or config
    types = list(text.get("layer_types") or [])
    counts = Counter(types)
    return {
        "count": int(text.get("num_hidden_layers", len(types))),
        "layer_types_present": sorted(counts),
        "layer_type_counts": dict(sorted(counts.items())),
        "full_attention_interval": text.get("full_attention_interval"),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "vocab_size": text.get("vocab_size"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "linear_key_head_dim": text.get("linear_key_head_dim"),
        "linear_value_head_dim": text.get("linear_value_head_dim"),
        "linear_num_key_heads": text.get("linear_num_key_heads"),
        "linear_num_value_heads": text.get("linear_num_value_heads"),
        "mamba_ssm_dtype": text.get("mamba_ssm_dtype"),
    }


def memory_ledger(config: dict[str, Any], total_size: int | None) -> dict[str, Any]:
    text = config.get("text_config") or config
    types = list(text.get("layer_types") or [])
    full_layers = types.count("full_attention")
    linear_layers = types.count("linear_attention")
    kv_heads = int(text.get("num_key_value_heads", 0))
    head_dim = int(text.get("head_dim", 0))
    kv_bytes_per_token = full_layers * 2 * kv_heads * head_dim * 2
    value_heads = int(text.get("linear_num_value_heads", 0))
    value_dim = int(text.get("linear_value_head_dim", 0))
    key_dim = int(text.get("linear_key_head_dim", 0))
    gdn_bytes_per_layer = value_heads * value_dim * key_dim * 4
    gdn_bytes_per_request = linear_layers * gdn_bytes_per_layer
    contexts = (32768, 65536, 131072, 196608, 262016)
    return {
        "model_files": {
            "exact_total_bytes": total_size,
            "exact_total_gib": round(total_size / (1 << 30), 6) if total_size is not None else None,
            "source": "model.safetensors.index.json metadata.total_size",
        },
        "full_attention_kv": {
            "derived_bytes_per_token_per_request": kv_bytes_per_token,
            "assumptions": {
                "dtype_bytes": 2,
                "full_attention_layers": full_layers,
                "k_and_v": 2,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
            },
            "context_bytes_per_request": {
                str(context): context * kv_bytes_per_token for context in contexts
            },
        },
        "linear_attention_recurrent_state": {
            "derived_bytes_per_layer_per_request": gdn_bytes_per_layer,
            "derived_bytes_per_request": gdn_bytes_per_request,
            "assumptions": {
                "dtype_bytes": 4,
                "linear_attention_layers": linear_layers,
                "value_heads": value_heads,
                "value_head_dim": value_dim,
                "key_head_dim": key_dim,
            },
        },
        "concurrency_state_bytes": {
            "C1_at_1024": gdn_bytes_per_request + 1024 * kv_bytes_per_token,
            "C4_at_1024": 4 * (gdn_bytes_per_request + 1024 * kv_bytes_per_token),
            "C8_at_1024": 8 * (gdn_bytes_per_request + 1024 * kv_bytes_per_token),
        },
        "exclusions": [
            "allocator and fragmentation overhead",
            "activations and workspaces",
            "CUDA Graph pools",
            "vision and MTP temporary buffers",
            "per-tensor category byte totals absent from committed SafeTensors headers",
        ],
    }


def build_inventory(config_path: Path, index_path: Path, revision: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("index JSON has no object-valued weight_map")
    names = sorted(str(name) for name in weight_map)
    prefix_counts = Counter(classify_prefix(name) for name in names)
    quant_counts = Counter(classify_quantization(name) for name in names)
    shard_counts = Counter(str(shard) for shard in weight_map.values())
    packed_names = [name for name in names if name.endswith(".weight_packed")]
    language_layers = sorted(
        {
            int(match.group(1))
            for name in names
            if (match := re.search(r"model\.language_model\.layers\.(\d+)", name))
        }
    )
    configured_layers = int((config.get("text_config") or config).get("num_hidden_layers", 0))
    if language_layers != list(range(configured_layers)):
        raise ValueError(
            "language layer indices in the index do not match config num_hidden_layers"
        )
    directional_examples = {
        "weight_packed": ["[1024, 640] I32; 8 int4 along K for hidden_size=5120"],
        "weight_scale": ["[1024, 160] BF16; group_size=32 along K"],
        "weight_zero_point": ["[128, 160] I32; 8 int4 along N for output_size=1024"],
    }
    quant_config = config.get("quantization_config") or {}
    groups = quant_config.get("config_groups") or {}
    first_group = next(iter(groups.values()), {})
    weights = first_group.get("weights") or {}
    total_size = (index.get("metadata") or {}).get("total_size")
    return {
        "schema": "apxinf.member3.shape-inventory.v1",
        "model_revision": revision,
        "fixtures": {
            "config_path": config_path.as_posix(),
            "config_sha256": sha256(config_path),
            "index_path": index_path.as_posix(),
            "index_sha256": sha256(index_path),
        },
        "model": {
            "architecture": (config.get("architectures") or [None])[0],
            "model_type": config.get("model_type"),
            "image_token_id": config.get("image_token_id"),
            "text": layer_inventory(config),
        },
        "tensors": {
            "count": len(names),
            "shard_count": len(shard_counts),
            "shards": dict(sorted(shard_counts.items())),
            "prefix_counts": dict(sorted(prefix_counts.items())),
            "quantization_name_counts": dict(sorted(quant_counts.items())),
            "packed_weight_count": len(packed_names),
            "language_layer_indices": language_layers,
        },
        "quantization": {
            "format": quant_config.get("format"),
            "bits": weights.get("num_bits"),
            "group_size": weights.get("group_size"),
            "symmetric": weights.get("symmetric"),
            "packing": "asymmetric pack-quantized; direction differs by tensor kind",
            "directional_examples": directional_examples,
        },
        "memory_ledger": memory_ledger(config, total_size),
        "warnings": [
            "No model weights were read or downloaded.",
            "Per-tensor shapes and dtypes require SafeTensors headers; do not infer byte totals from names alone.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    try:
        result = build_inventory(args.config, args.index, args.model_revision)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"shape_inventory: error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

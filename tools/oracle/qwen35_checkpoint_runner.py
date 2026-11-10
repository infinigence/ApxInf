#!/usr/bin/env python3
"""GPU-only Qwen3.5 checkpoint runner for the M2-O0 oracle bundle.

This executable is intentionally separate from production.  The generator
passes a frozen job manifest through environment variables; this runner writes
only the declared binary/json artifacts and a report consumed by the generator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any


EOS_TOKEN_IDS = (248_046, 248_044)
MODEL_VOCAB_SIZE = 248_320


def decompress_module(module: Any) -> None:
    """Expand one compressed-tensors module for its current GPU invocation."""
    from compressed_tensors.compressors.base import decompress_module as _decompress

    _decompress(module)


def snapshot_compressed_module(module: Any) -> dict[str, Any]:
    """Retain one module's packed tensors so they can be restored bit-for-bit."""
    from compressed_tensors.utils import get_direct_state_dict

    return {
        "state": dict(get_direct_state_dict(module)),
        "quantization_status": getattr(module, "quantization_status", None),
    }


def restore_compressed_module(module: Any, snapshot: dict[str, Any]) -> None:
    """Restore the packed state captured before a temporary expansion."""
    from compressed_tensors.utils import replace_direct_state_dict

    replace_direct_state_dict(module, snapshot["state"])
    if snapshot["quantization_status"] is None:
        try:
            delattr(module, "quantization_status")
        except AttributeError:
            pass
    else:
        module.quantization_status = snapshot["quantization_status"]


def install_ephemeral_decompression_hooks(model: Any) -> list[Any]:
    """Expand and restore each packed linear only during its forward call.

    Transformers' compressed-tensors loader otherwise registers one model-level
    hook that expands all 399 quantized modules at once. The per-module hooks
    keep the checkpoint packed between calls and retain the original tensors.
    """
    handles: list[Any] = []
    snapshots: dict[int, dict[str, Any]] = {}

    def before(module: Any, _inputs: tuple[Any, ...]) -> None:
        key = id(module)
        if key in snapshots:
            raise RuntimeError("quantized module entered twice before restoration")
        snapshot = snapshot_compressed_module(module)
        snapshots[key] = snapshot
        try:
            decompress_module(module)
        except BaseException:
            snapshots.pop(key, None)
            raise

    def after(module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
        snapshot = snapshots.pop(id(module), None)
        if snapshot is None:
            raise RuntimeError("quantized module completed without a packed snapshot")
        restore_compressed_module(module, snapshot)

    for _name, module in model.named_modules():
        if getattr(module, "quantization_scheme", None) is None:
            continue
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after, always_call=True))
    return handles


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def safe_artifact_path(root: Path, file_name: str) -> Path:
    if not file_name or Path(file_name).is_absolute():
        raise ValueError(f"artifact path must be relative: {file_name!r}")
    candidate = (root / file_name).resolve()
    root_resolved = root.resolve()
    if candidate.parent != root_resolved or candidate.name != file_name:
        raise ValueError(f"artifact path escapes output directory: {file_name!r}")
    return candidate


def write_f32(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for index, value in enumerate(values):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"non-finite F32 value at index {index}")
            handle.write(struct.pack("<f", numeric))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def greedy_token_ids(candidates: list[int], max_new_tokens: int) -> tuple[list[int], str]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    output: list[int] = []
    for token in candidates[:max_new_tokens]:
        if token < 0 or token >= MODEL_VOCAB_SIZE:
            raise ValueError(f"generated token outside model vocab: {token}")
        output.append(int(token))
        if token in EOS_TOKEN_IDS:
            return output, "eos"
    if len(output) != max_new_tokens:
        raise ValueError("candidate sequence ended before the requested budget")
    return output, "budget"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _tensor_values(tensor: Any) -> list[float]:
    import torch

    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    return [float(value) for value in flat.tolist()]


def _tensor_shape(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.shape]


def gdn_required_dimensions(model_meta: dict[str, Any]) -> dict[str, int]:
    names = (
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
    )
    dimensions = {}
    for name in names:
        value = model_meta.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"job model is missing positive GDN dimension {name}")
        dimensions[name] = value
    return dimensions


def _capture_cache_tensor(cache: Any, layer_index: int, field: str) -> Any:
    layers = getattr(cache, "layers", None)
    if layers is None or layer_index >= len(layers):
        raise ValueError(f"cache has no layer {layer_index}")
    value = getattr(layers[layer_index], field, None)
    if value is None or not hasattr(value, "shape") or value.numel() == 0:
        raise ValueError(f"cache layer {layer_index} has no {field}")
    return value


def _capture_gdn_state(cache: Any, layer_index: int) -> tuple[list[float], list[int]]:
    layers = getattr(cache, "layers", None)
    if layers is None or layer_index >= len(layers):
        raise ValueError(f"cache has no layer {layer_index}")
    layer = layers[layer_index]
    chunks = []
    for field in ("conv_states", "recurrent_states"):
        states = getattr(layer, field, None)
        if not isinstance(states, dict):
            raise ValueError(f"cache layer {layer_index} has no {field}")
        for state_index in sorted(states):
            value = states[state_index]
            if value is None or not hasattr(value, "shape") or value.numel() == 0:
                raise ValueError(f"cache layer {layer_index} has incomplete {field}")
            chunks.append(value)
    if not chunks:
        raise ValueError(f"cache layer {layer_index} has no GDN state")
    import torch

    flattened = torch.cat([chunk.detach().to(dtype=torch.float32).reshape(-1) for chunk in chunks])
    return _tensor_values(flattened), [int(flattened.numel())]


def _register_hidden_hooks(model: Any, layer_indices: list[int]) -> tuple[dict[int, list[Any]], list[Any]]:
    captures: dict[int, list[Any]] = {index: [] for index in layer_indices}
    handles = []

    def make_hook(index: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            import torch

            tensor = output[0] if isinstance(output, tuple) else output
            if not hasattr(tensor, "shape"):
                raise ValueError(f"layer {index} output is not a tensor")
            captures[index].append(tensor.detach().to(dtype=torch.float32).cpu())

        return hook

    layers = model.model.language_model.layers
    for index in layer_indices:
        if index < 0 or index >= len(layers):
            raise ValueError(f"selected layer is outside model: {index}")
        handles.append(layers[index].register_forward_hook(make_hook(index)))
    return captures, handles


def run_job() -> None:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    manifest_path = Path(os.environ["APXINF_ORACLE_JOB_MANIFEST"]).resolve()
    output_dir = Path(os.environ["APXINF_ORACLE_OUTPUT_DIR"]).resolve()
    job = _read_json(manifest_path)
    model_meta = job["model"]
    gdn_dimensions = gdn_required_dimensions(model_meta)
    model_dir = Path(model_meta["model_dir"]).resolve()
    generation = _read_json(manifest_path.parent / "generation.json")
    selection = _read_json(manifest_path.parent / "selection.json")
    input_manifest = _read_json(manifest_path.parent / "input-manifest.json")
    input_ids_list = input_manifest["input_ids"]
    max_new_tokens = int(generation["max_new_tokens"])
    layers = [int(value) for value in selection["layers"]]
    stages = set(selection["stages"])

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "GPU-343bc895-b011-22fa-4449-97207aa2bdec":
        raise RuntimeError("oracle runner requires the fixed GPU1 UUID")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the checkpoint oracle")
    device = torch.device("cuda:0")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    # Transformers' default compressed-tensors hook expands every quantized
    # module on the first model call. Replace it with per-module hooks so only
    # the active linear temporarily occupies dense BF16 memory.
    model_decompress_hook = getattr(model, "ct_decompress_hook", None)
    if model_decompress_hook is not None:
        model_decompress_hook.remove()
        delattr(model, "ct_decompress_hook")
    ephemeral_handles = install_ephemeral_decompression_hooks(model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)

    hidden_layers = sorted(index for index in layers if "layer_hidden" in stages)
    captures, hidden_handles = _register_hidden_hooks(model, hidden_layers)
    logits = []
    top_margins = []
    generated: list[int] = []
    past = None
    try:
        with torch.inference_mode():
            embedding = model.model.language_model.embed_tokens(input_ids)
            outputs = model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past = outputs.past_key_values
            for _ in range(max_new_tokens):
                step_logits = outputs.logits[:, -1, :].float()
                logits.append(step_logits[0].detach().cpu())
                values, indices = torch.topk(step_logits, k=2, dim=-1)
                top_margins.append(float((values[0, 0] - values[0, 1]).item()))
                token = int(indices[0, 0].item())
                generated.append(token)
                next_input = torch.tensor([[token]], dtype=torch.long, device=device)
                outputs = model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
                past = outputs.past_key_values
                if token in EOS_TOKEN_IDS:
                    stop_reason = "eos"
                    break
            else:
                stop_reason = "budget"
    finally:
        for handle in hidden_handles:
            handle.remove()
        for handle in ephemeral_handles:
            handle.remove()

    if not generated:
        raise RuntimeError("checkpoint produced no output token")
    if stop_reason == "eos" and len(generated) > max_new_tokens:
        raise RuntimeError("checkpoint exceeded generation budget")

    artifacts: dict[str, tuple[list[float], list[int], str]] = {}
    if "embedding" in stages:
        artifacts["embedding.f32.bin"] = (_tensor_values(embedding), _tensor_shape(embedding[0]), "embedding")
    for index, chunks in captures.items():
        if not chunks:
            raise RuntimeError(f"selected layer {index} produced no hidden state")
        hidden = torch.cat(chunks, dim=1)
        artifacts[f"layer-{index:03d}-hidden.f32.bin"] = (
            _tensor_values(hidden[0]),
            _tensor_shape(hidden[0]),
            "layer_hidden",
        )
    for index in layers:
        if "gdn_state" in stages and model.config.text_config.layer_types[index] == "linear_attention":
            values, shape = _capture_gdn_state(past, index)
            artifacts[f"layer-{index:03d}-gdn-state.f32.bin"] = (values, shape, "gdn_state")
        if "kv_state" in stages and model.config.text_config.layer_types[index] == "full_attention":
            key = _capture_cache_tensor(past, index, "keys")
            value = _capture_cache_tensor(past, index, "values")
            artifacts[f"layer-{index:03d}-kv-key.f32.bin"] = (_tensor_values(key), _tensor_shape(key), "kv_key")
            artifacts[f"layer-{index:03d}-kv-value.f32.bin"] = (_tensor_values(value), _tensor_shape(value), "kv_value")
    if "logits" in stages:
        logits_tensor = torch.stack(logits).to(dtype=torch.float32)
        artifacts["logits.f32.bin"] = (_tensor_values(logits_tensor), _tensor_shape(logits_tensor), "logits")
    if "tokens" in stages:
        token_path = safe_artifact_path(output_dir, "output-tokens.json")
        token_path.write_bytes(canonical_bytes({
            "schema": "apxinf.oracle-tokens.v1",
            "output_token_ids": generated,
            "decoded_text": tokenizer.decode(generated, skip_special_tokens=True),
        }))

    records = []
    for name, (values, shape, schema_ref) in sorted(artifacts.items()):
        path = safe_artifact_path(output_dir, name)
        write_f32(path, values)
        record = {
            "file": name,
            "schema_ref": schema_ref,
            "dtype": "F32",
            "shape": shape,
            "sha256": sha256_file(path),
        }
        if schema_ref == "gdn_state":
            record["required_dimensions"] = gdn_dimensions
        records.append(record)
    if "tokens" in stages:
        token_path = safe_artifact_path(output_dir, "output-tokens.json")
        records.append({
            "file": "output-tokens.json",
            "schema_ref": "tokens",
            "dtype": "json",
            "shape": [len(generated)],
            "sha256": sha256_file(token_path),
        })
    report = {
        "schema": "apxinf.oracle-artifact-report.v1",
        "generation": {
            "completion_tokens": len(generated),
            "stop_reason": stop_reason,
            "top1_top2_margin": top_margins,
        },
        "artifacts": records,
    }
    safe_artifact_path(output_dir, "artifact-report.json").write_bytes(canonical_bytes(report))


if __name__ == "__main__":
    run_job()

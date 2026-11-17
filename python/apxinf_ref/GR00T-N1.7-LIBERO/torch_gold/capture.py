"""Run upstream get_action without ApxInf kernels or a rewritten flow loop."""

import importlib.metadata
import platform
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .artifact import sha256


def file_identity(root, patterns):
    root = Path(root).resolve()
    paths = sorted({path for pattern in patterns for path in root.rglob(pattern) if path.is_file()})
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in paths}


def capture(args, inputs):
    import torch

    if args.device.startswith("musa"):
        import torch_musa
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    device = torch.device(args.device)
    if device.type not in ("cuda", "musa"):
        raise ValueError("gold capture requires an explicit CUDA or MUSA device")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    stack = getattr(torch, device.type)
    backbone_config = AutoConfig.from_pretrained(args.backbone, local_files_only=True)
    backbone_config._attn_implementation = "sdpa"

    def local_backbone(*unused_args, **unused_kwargs):
        return Qwen3VLForConditionalGeneration(backbone_config)

    with patch.object(Qwen3VLForConditionalGeneration, "from_pretrained", side_effect=local_backbone), \
            patch.object(processing_gr00t_n1d7, "Gr00tN1d7DataCollator", return_value=None):
        model, loading = Gr00tN1d7.from_pretrained(
            args.checkpoint, local_files_only=True, output_loading_info=True,
        )
    problems = {key: loading[key] for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if problems:
        raise RuntimeError(f"checkpoint did not load strictly: {problems}")
    model.eval().to(device=device, dtype=torch.bfloat16)

    def bf16(name):
        return torch.from_numpy(inputs[name].copy()).view(torch.bfloat16).to(device)

    noise = bf16("noise")
    tensors = {
        "pixel_values": bf16("pixel_values"),
        "state": bf16("state"),
        "input_ids": torch.from_numpy(inputs["token_ids"].astype(np.int64))[None].to(device),
        "attention_mask": torch.from_numpy(inputs["attention_mask"].astype(np.int64))[None].to(device),
        "image_grid_thw": torch.from_numpy(inputs["image_grid_thw"].astype(np.int64)).to(device),
        "embodiment_id": torch.from_numpy(inputs["embodiment_id"].astype(np.int64)).to(device),
    }
    expected_shape = (1, model.config.action_horizon, model.action_head.action_dim)
    if tuple(noise.shape) != expected_shape:
        raise ValueError(f"noise does not match model action shape {expected_shape}")
    calls = 0

    def fixed_randn(*positional, **keywords):
        nonlocal calls
        shape = keywords.get("size", positional[0] if positional else ())
        if isinstance(shape, (tuple, list, torch.Size)) and tuple(shape) == tuple(noise.shape):
            calls += 1
            return noise.to(device=keywords.get("device", device),
                            dtype=keywords.get("dtype", torch.bfloat16)).clone()
        raise RuntimeError("unexpected random call in fixed-noise reference")

    def infer():
        before = calls
        with patch.object(torch, "randn", side_effect=fixed_randn), torch.inference_mode():
            result = model.get_action(inputs=dict(tensors))["action_pred"]
        stack.synchronize(device)
        if calls - before != 1:
            raise RuntimeError("expected exactly one fixed-noise draw per inference")
        output = result.detach().float().cpu().numpy()[0].copy()
        if not np.isfinite(output).all():
            raise RuntimeError("non-finite model output")
        return output

    for unused in range(args.warmup_runs):
        infer()
    outputs = [infer() for unused in range(args.capture_runs)]
    if any(not np.array_equal(outputs[0], output) for output in outputs[1:]):
        raise RuntimeError("repeat captures differ; refusing to publish gold")
    weight_files = file_identity(args.checkpoint, ("*.safetensors", "*.json"))
    if not any(name.endswith(".safetensors") for name in weight_files):
        raise RuntimeError("no safetensors checkpoint identity found")
    provenance = {
        "checkpoint": {"path": str(args.checkpoint.resolve()), "files": weight_files},
        "backbone": {"path": str(args.backbone.resolve()),
                     "files": file_identity(args.backbone, ("config.json",))},
        "source": {"path": str(args.source.resolve()),
                   "files": file_identity(args.source / "gr00t", ("*.py",))},
        "generator": file_identity(Path(__file__).resolve().parents[1], ("*.py",)),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "numpy": np.__version__,
            "device": str(device),
            "device_type": device.type,
            "device_name": stack.get_device_name(device),
            "precision": "bfloat16",
            "attention": "sdpa",
            "cuda": torch.version.cuda,
            "seed": 0,
            "cudnn_enabled": getattr(getattr(torch.backends, "cudnn", None), "enabled", None),
            "cudnn_benchmark": getattr(getattr(torch.backends, "cudnn", None), "benchmark", None),
            "cudnn_deterministic": getattr(getattr(torch.backends, "cudnn", None), "deterministic", None),
            "tf32_matmul": getattr(
                getattr(getattr(torch.backends, "cuda", None), "matmul", None),
                "allow_tf32",
                None,
            ),
            "tf32_cudnn": getattr(getattr(torch.backends, "cudnn", None), "allow_tf32", None),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "model_config": model.config.to_dict(),
        "num_inference_timesteps": model.action_head.num_inference_timesteps,
        "checks": {"strict_weights": True, "fixed_noise": True, "finite": True,
                   "repeat_exact": True, "warmup_runs": args.warmup_runs,
                   "capture_runs": args.capture_runs},
        "boundary": "preprocessed fixture tensors to normalized actions; no processor encode/decode",
    }
    return {"normalized_actions": outputs[0]}, provenance

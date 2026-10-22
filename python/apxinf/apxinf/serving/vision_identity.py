from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
from pathlib import Path
import platform
import sys
from typing import Any

if __package__:
    from . import contracts
    from .text_worker import WorkerFailure, _file_digest
else:
    import contracts
    from text_worker import WorkerFailure, _file_digest

CHECKPOINT = "openbmb/MiniCPM-V-4.7-1B"
CHECKPOINT_REVISION = "379f120c2ff3588e6e06559ec6b9ec15cc05114e"
ADAPTER_REVISION = "apxinf-mlx-minicpmv47-vision-v1"
PROCESSOR_CLASS = "mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor"
MLX_LIBRARY_SHA256 = "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6"
PINNED_RUNTIME = {"python": "3.14.3", "mlx": "0.32.4", "mlx_vlm": "0.7.10", "transformers": "5.18.0"}
PINNED_PACKAGES = {
    "mlx": "0.32.4", "mlx-metal": "0.32.4", "mlx-vlm": "0.7.10", "transformers": "5.18.0",
    "numpy": "2.5.4", "Pillow": "13.0.0", "torch": "2.12.1", "torchvision": "0.27.1",
    "tokenizers": "0.23.3", "safetensors": "0.8.0", "huggingface-hub": "2.2.0",
}
PINNED_ARTIFACTS = {
    "chat_template.jinja": (7249, "49c37bcc86c7f7f20ed2ebf79e9f872be41ba2b5f9d0d51e6712cc038ef51d6c"),
    "config.json": (2575, "20ed1fe71b174719b88ed1ce37c840ec45e3657f1120391d45f33505be7638a4"),
    "generation_config.json": (132, "4dc190baaab3bc4bccb796ee73deb2c3dd776cd4feae3587b29bc22083e03415"),
    "model.safetensors": (3109054480, "8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002"),
    "preprocessor_config.json": (148, "c64289d04f36e48d9d059e3a0069f54062b2ef6044a807adb448b98449cb54de"),
    "processor_config.json": (2, "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
    "tokenizer.json": (19992481, "33861e37bb955af1e3f3061182b820f347eba2b9c2c1011c82794bf0d6e77b54"),
    "tokenizer_config.json": (1436, "900ecc6ba690a07270dc23935d17bbb7d2ed69a073a90a9cf731e5313e86ebba"),
}
SOURCE_FILES = ("vision_worker.py", "vision_runtime.py", "vision_identity.py", "vision_preparation.py",
                "text_worker.py", "contracts.py", "owned_media.py")
STATE_MODULES = {
    "module_state": ("mlx.nn.layers.base", "cefa99a579f3c2016b70e15c183ae5e8bbef103ce9a0d89cb3789d9aedb23503"),
    "minicpm_model": ("mlx_vlm.models.minicpmv4_7.minicpmv4_7", "d52b40283d9dfc62bb0e06df94814b21263d7ff35d90b1bf59741c681e35fe48"),
    "minicpm_language": ("mlx_vlm.models.minicpmv4_7.language", "790769ce0dc543e5cc481cde1ac3084b434684fbbc4d948203d4efbfeb417fb5"),
    "minicpm_language_base": ("mlx_vlm.models.minicpmv4_6.language", "999aeceece5de4b6bfa386f6a719be898584803af46bf5ac07af68ad75f928ad"),
    "qwen_language": ("mlx_vlm.models.qwen3_5.language", "f5821a9d850eef2868e048977b56bf5798a57d36bf2141272661a5f59725000b"),
    "rotary_state": ("mlx_vlm.models.rope_utils", "6944f69a03c41cbb9613afbd2f1b54b71c504840aaeb0eade39c6109dbfdd25b"),
}


def state_modules() -> dict[str, Any]:
    modules = {}
    for name, (module_name, digest) in STATE_MODULES.items():
        module = importlib.import_module(module_name)
        if _file_digest(Path(module.__file__).resolve(strict=True)) != digest:
            raise WorkerFailure("model_unavailable", "The request-state dependency sources differ from the qualified profile.")
        modules[name] = module
    return modules


def execution(max_context: int, max_output_tokens: int, memory_limit_bytes: int) -> dict[str, Any]:
    return {
        "provider": "mlx-vlm-minicpmv47", "precision": "bf16", "enable_thinking": False,
        "kv_reuse": False, "max_context": max_context, "max_output_tokens": max_output_tokens,
        "prefill_step_size": 256, "output_batch_tokens": 1, "memory_limit_bytes": memory_limit_bytes,
        "selection": {"algorithm": "mlx-vlm-fixed-controls-v1", "top_k": 100, "min_p": "0",
                      "repetition_penalty": "1.03", "repetition_context_size": 8192,
                      "presence_penalty": "0", "frequency_penalty": "0",
                      "seed_policy": "request-id-sha256-v1"},
    }


def check_packages() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}
    if actual != PINNED_PACKAGES or platform.python_version() != PINNED_RUNTIME["python"]:
        raise WorkerFailure("model_unavailable", "The vision runtime versions differ from the pinned profile.")
    if platform.system() != "Darwin":
        raise WorkerFailure("model_unavailable", "This vision profile requires macOS Metal.")
    return actual


def runtime_files(modules: dict[str, Any]) -> dict[str, Any]:
    environment = Path(sys.prefix).resolve(strict=True)
    files = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve(strict=True)
        if not path.is_relative_to(environment):
            raise WorkerFailure("model_unavailable", "A vision module is outside the selected environment.")
        files[name] = {"path": str(path), "sha256": _file_digest(path)}
    loader = ctypes.CDLL(None)
    loader._dyld_image_count.restype = ctypes.c_uint32
    loader._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    loader._dyld_get_image_name.restype = ctypes.c_char_p
    paths = []
    for index in range(loader._dyld_image_count()):
        value = loader._dyld_get_image_name(index)
        if value is not None:
            path = Path(value.decode())
            if "libmlx" in path.name and path.suffix == ".dylib":
                paths.append(path.resolve(strict=True))
    libraries = []
    for path in sorted(set(paths)):
        digest = _file_digest(path)
        if not path.is_relative_to(environment) or digest != MLX_LIBRARY_SHA256:
            raise WorkerFailure("model_unavailable", "The loaded MLX library differs from the vision profile.")
        libraries.append({"path": str(path), "sha256": digest})
    if len(libraries) != 1:
        raise WorkerFailure("model_unavailable", "The vision profile requires one checked MLX library.")
    return {"modules": files, "libraries": libraries}


def complete_manifest(manifest: dict[str, Any], files: dict[str, Any], packages: dict[str, str]) -> dict[str, Any]:
    actual = {item["path"]: (item["size_bytes"], item["sha256"]) for item in manifest["artifacts"]}
    if actual != PINNED_ARTIFACTS:
        raise WorkerFailure("model_unavailable", "The checkpoint artifacts differ from the pinned vision revision.")
    processor = {
        "class": PROCESSOR_CLASS, "runtime": PINNED_RUNTIME,
        "artifacts": [item for item in manifest["artifacts"] if item["path"] != "model.safetensors"],
        "options": {"return_tensors": "np", "downsample_mode": "16x", "max_slice_nums": 1,
                    "padding": True, "add_special_tokens": False},
        "template_options": {"enable_thinking": False, "tokenize": False, "add_generation_prompt": True},
    }
    root = Path(__file__).resolve().parent
    return {
        **manifest, "checkpoint": CHECKPOINT, "checkpoint_revision": CHECKPOINT_REVISION,
        "adapter_revision": ADAPTER_REVISION, "platform": "macos-metal", "packages": packages,
        "adapter_sha256": _file_digest(root / "vision_worker.py"),
        "runtime_files": files, "sources": {name: _file_digest(root / name) for name in SOURCE_FILES},
        "processor": processor,
        "processor_revision": contracts.canonical_identity_digest("apxinf-vision-processor-v1", processor),
    }

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
from typing import Any

if __package__:
    from . import contracts
    from . import clef_contracts as schema
else:
    import contracts
    import clef_contracts as schema


class ClefError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, preserve_path: bool = False) -> dict:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ClefError("model_unavailable", "A selected identity path is not a file.")
    return {"path": str(path.absolute() if preserve_path else resolved), "sha256": file_digest(resolved)}


def _reject_remote_code(document: Any) -> None:
    if type(document) is dict:
        if "auto_map" in document:
            raise ClefError("model_unavailable", "This profile does not permit remote-code mappings.")
        for value in document.values():
            _reject_remote_code(value)
    elif type(document) is list:
        for value in document:
            _reject_remote_code(value)


def model_artifacts(model_path: Path) -> tuple[list[dict], int]:
    if not model_path.is_absolute() or not model_path.is_dir():
        raise ClefError("model_unavailable", "The Clef model directory must exist locally at an absolute path.")
    artifacts = []
    documents = {}
    for path in sorted(model_path.iterdir()):
        if path.suffix not in {".json", ".safetensors", ".jinja"} or path.is_symlink() or not path.is_file():
            continue
        size = path.stat().st_size
        digest = file_digest(path)
        if path.stat().st_size != size:
            raise ClefError("model_unavailable", "A Clef artifact changed while hashing.")
        artifacts.append({"path": path.name, "size_bytes": size, "sha256": digest})
        if path.suffix == ".json":
            if size > 64 * 1024 * 1024:
                raise ClefError("model_unavailable", "A Clef JSON artifact exceeds the identity limit.")
            document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=contracts._pairs,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite artifact number")))
            _reject_remote_code(document)
            documents[path.name] = document
    names = {item["path"] for item in artifacts}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "joint_head_config.json", "joint_head.safetensors", "model.safetensors.index.json"}
    if not required <= names:
        raise ClefError("model_unavailable", "The Clef artifact set is incomplete.")
    index = documents["model.safetensors.index.json"]
    shards = index.get("weight_map") if type(index) is dict else None
    if type(shards) is not dict or not shards:
        raise ClefError("model_unavailable", "The Clef weight index has no shard map.")
    if any(type(name) is not str or Path(name).name != name or not name.endswith(".safetensors") or name not in names
           for name in shards.values()):
        raise ClefError("model_unavailable", "The Clef weight index references an unchecked artifact.")
    config = documents["config.json"]
    if type(config) is not dict or type(config.get("text_config")) is not dict:
        raise ClefError("model_unavailable", "The Clef text configuration is missing.")
    quantization = config.get("quantization")
    if type(quantization) is not dict or quantization.get("bits") != 4:
        raise ClefError("model_unavailable", "The Clef backbone must use its four-bit profile.")
    vocab_size = config["text_config"].get("vocab_size")
    contracts._int(vocab_size, 1, 2147483648)
    return artifacts, vocab_size


def check_runtime() -> tuple[dict[str, str], dict[str, Any]]:
    if platform.system() != "Darwin" or platform.python_version() != schema.RUNTIME["python"]:
        raise ClefError("model_unavailable", "The Clef profile requires its pinned macOS Python runtime.")
    distributions = {"mlx": "mlx", "mlx_lm": "mlx-lm", "transformers": "transformers", "numpy": "numpy", "pillow": "Pillow"}
    versions = {"python": platform.python_version(), **{name: importlib.metadata.version(package)
                for name, package in distributions.items()}}
    if versions != schema.RUNTIME:
        raise ClefError("model_unavailable", "The Clef runtime versions differ from the pinned profile.")
    modules = {name: importlib.import_module(module) for name, module in
               {"mlx_core": "mlx.core", "mlx_lm": "mlx_lm", "transformers": "transformers",
                "numpy": "numpy", "pillow": "PIL", "tokenizers": "tokenizers"}.items()}
    return versions, modules


def runtime_files(modules: dict[str, Any]) -> dict:
    environment = Path(sys.prefix).resolve(strict=True)
    records = {}
    for name, module in modules.items():
        record = file_record(Path(module.__file__))
        if not Path(record["path"]).is_relative_to(environment):
            raise ClefError("model_unavailable", "A runtime module is outside the selected environment.")
        records[name] = record
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    loaded = set()
    for index in range(dyld._dyld_image_count()):
        raw = dyld._dyld_get_image_name(index)
        if raw is not None:
            path = Path(raw.decode()).resolve(strict=True)
            if path.suffix == ".dylib" and "libmlx" in path.name:
                loaded.add(path)
    if len(loaded) != 1 or not all(path.is_relative_to(environment) for path in loaded):
        raise ClefError("model_unavailable", "The Clef profile requires one MLX library from the selected environment.")
    return {"modules": records, "libraries": [file_record(path) for path in sorted(loaded)]}


def load_dependency(loader_path: Path) -> Any:
    if not loader_path.is_absolute() or not loader_path.is_file() or file_digest(loader_path) != schema.LOADER_SHA256:
        raise ClefError("model_unavailable", "The external Clef loader differs from the pinned dependency.")
    spec = importlib.util.spec_from_file_location("apxinf_selected_clef_dependency", loader_path)
    if spec is None or spec.loader is None:
        raise ClefError("model_unavailable", "The selected Clef loader cannot be imported.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    if file_digest(loader_path) != schema.LOADER_SHA256:
        raise ClefError("model_unavailable", "The external Clef loader changed during import.")
    return module


def manifest(loader_path: Path, artifacts: list[dict], runtime: dict,
             files: dict, max_context: int, memory_limit_bytes: int) -> dict:
    source_root = Path(__file__).resolve().parent
    return {"checkpoint": schema.CHECKPOINT, "checkpoint_revision": schema.CHECKPOINT_REVISION,
            "adapter_revision": schema.ADAPTER_REVISION, "platform": "macos-metal", "artifacts": artifacts,
            "loader": file_record(loader_path), "runtime": runtime,
            "sources": {name: file_record(source_root / name) for name in schema.SOURCE_FILES},
            "python": file_record(Path(sys.executable), preserve_path=True), "runtime_files": files,
            "execution": schema.execution(max_context, memory_limit_bytes)}

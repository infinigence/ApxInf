"""Artifact identities shared by maintained model benchmarks."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def artifact(path):
    if path is None:
        return None
    path = Path(path).resolve()
    if path.is_dir():
        manifest = [(item.relative_to(path).as_posix(), artifact(item)['sha256'])
                    for item in sorted(path.rglob('*')) if item.is_file()]
        encoded = json.dumps(manifest, ensure_ascii=True, separators=(',', ':')).encode()
        return {'path': str(path), 'kind': 'directory', 'file_count': len(manifest),
                'sha256': hashlib.sha256(encoded).hexdigest()}
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def recipe_artifact(path):
    """Mirror AutoModel's cuda_recipe_options for an explicit --tactics path.

    cuda-new consumes recipe directories, not legacy tactic JSON contents.
    Source-tree hardware tactic aliases defer to the native default location;
    we cannot identify that resolved directory from the Python option alone.
    """
    if path is None:
        return None
    path = Path(path)
    if (path.name == 'tactics.json' and len(path.parents) >= 4
            and path.parents[1].name == 'nvidia'
            and path.parents[2].name == 'tuning'
            and path.parents[3].name == 'configs'):
        return {'requested_path': str(path.resolve()), 'selection': 'native_default'}
    if not path.is_dir() and (path.is_file() or path.suffix.lower() == '.json'):
        path = path.with_suffix('.recipes')
    if not path.exists():
        return {'path': str(path.resolve()), 'exists': False}
    return artifact(path)


def provenance(root, model_dir, *, binary=None, tactics=None, calibration=None):
    def command(args):
        try:
            return subprocess.check_output(args, cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    if binary is None:
        binary = next((getattr(module, '__file__', None)
                       for name, module in tuple(sys.modules.items())
                       if name.startswith('apxinf_py')
                       and str(getattr(module, '__file__', '')).endswith(('.so', '.pyd'))), None)
    config = Path(model_dir) / 'config.json'
    index = Path(model_dir) / 'model.safetensors.index.json'
    packages = {}
    for name in ('numpy', 'pillow', 'tokenizers'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(engine_commit=command(['git','rev-parse','HEAD']),
                engine_status=command(['git','status','--porcelain']),
                python=platform.python_version(), packages=packages,
                gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,clocks.current.graphics','--format=csv,noheader']),
                binary=artifact(binary), tactics=recipe_artifact(tactics), calibration=artifact(calibration),
                model_config=artifact(config), weight_index=artifact(index) if index.exists() else None)

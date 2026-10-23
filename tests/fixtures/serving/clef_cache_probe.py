"""Original CPU probe for normal Python bytecode lookup at worker launch."""
import hashlib
import json
import os
from pathlib import Path
import sys


if sys.argv[1:2] == ["--seed-bytecode"]:
    import py_compile

    root = Path(sys.argv[2])
    sources = {
        "clef_contracts.py": (b"apxinf-python-clef-text-v1", b"apxinf-python-clef-text-v0"),
        "probe_package/__init__.py": (b"current", b"stale__"),
        "probe_package/member.py": (b"current", b"stale__"),
    }
    caches = []
    for name, (current, stale) in sources.items():
        path = root / name
        original = path.read_bytes()
        metadata = path.stat()
        assert original.count(current) == 1
        obsolete = original.replace(current, stale)
        assert len(obsolete) == len(original)
        path.write_bytes(obsolete)
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        cache = py_compile.compile(
            str(path), doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
        path.write_bytes(original)
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        caches.append(cache)
    print(json.dumps(caches))
    sys.exit(0)

import clef_contracts
import probe_package
from probe_package import member

root = Path(__file__).parent
model = Path(sys.argv[sys.argv.index("--model") + 1])
observation = {
    "revision": clef_contracts.ADAPTER_REVISION,
    "initializer": probe_package.MARKER,
    "member": member.MARKER,
    "source_sha256": hashlib.sha256((root / "clef_contracts.py").read_bytes()).hexdigest(),
    "python": sys.executable,
    "argv": sys.argv,
    "orig_argv": sys.orig_argv,
    "device_imports": sorted(
        name for name in sys.modules
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_lm"
    ),
}
(model / "observation.json").write_text(json.dumps(observation))

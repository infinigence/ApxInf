"""Check an installed runtime wheel without importing the source checkout."""

from importlib.metadata import distribution
from pathlib import Path

import sys

import apxinf

assert "apxinf_py" not in sys.modules, "import apxinf must keep the backend lazy"
import apxinf_py
from apxinf import AutoPolicy, ComposablePolicy, ModelRunner, Pi05Policy
from apxinf.processors import Normalizer, Unnormalizer

runtime = distribution("apxinf")
assert runtime.version == apxinf.__version__ == apxinf_py.__version__
owned = {Path(runtime.locate_file(file)).resolve() for file in runtime.files}
assert Path(apxinf.__file__).resolve() in owned
assert Path(apxinf_py.__file__).resolve() in owned
assert Path(apxinf_py.__file__).with_name("apxinf_py.pyi").is_file()
assert ModelRunner is apxinf_py.ModelRunner
assert callable(ModelRunner.load)
assert callable(AutoPolicy.from_pretrained)
assert callable(ComposablePolicy.with_adapter)
assert callable(Pi05Policy.default_pipelines)
assert callable(Normalizer) and callable(Unnormalizer)
print(f"Installed apxinf {runtime.version}: Python policies and native backend available")

"""Install the pinned Transformers replacements required by vendored Pi0.5."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TypedDict

# What the vendored replace found when it was installed, for whoever records the
# environment a reference run happened in (`pi05_gold/capture.py` writes the
# manifest). `ok` is the vendored check's own verdict, not a copy of it.
REPLACE_STATUS: dict[str, object] = {"ok": None, "installed": None}


def install_transformers_overlay() -> None:
    import transformers

    if not hasattr(transformers.utils, "LossKwargs"):
        transformers.utils.LossKwargs = TypedDict("LossKwargs", {}, total=False)

    # Upstream nests these one level deeper, under ``models/``.  This repository
    # ignores every directory called ``models/`` (.gitignore), which would drop
    # any replacement added there without a word, so the family directories sit
    # directly under ``transformers_replace/``.  Nothing in the layout below
    # carries meaning to Python: a module's parent comes from the dotted name
    # handed to ``spec_from_file_location``, never from where the file lives.
    root = Path(__file__).parent / "transformers_replace"
    # Order is load-bearing.  ``modeling_paligemma`` imports GemmaModel and
    # SiglipVisionModel through ``..gemma``/``..siglip``, which resolve out of
    # ``sys.modules``.  Load paligemma first and it would bind to the *unpatched*
    # transformers modules instead of failing -- gemma and siglip have to be in
    # place before it runs.
    modules = (
        ("transformers.models.gemma.configuration_gemma", "gemma/configuration_gemma.py"),
        ("transformers.models.gemma.modeling_gemma", "gemma/modeling_gemma.py"),
        ("transformers.models.siglip.modeling_siglip", "siglip/modeling_siglip.py"),
        ("transformers.models.paligemma.modeling_paligemma", "paligemma/modeling_paligemma.py"),
        ("transformers.models.siglip.check", "siglip/check.py"),
    )
    for module_name, relative_path in modules:
        path = root / relative_path
        if not path.is_file():
            raise ImportError(f"vendored Transformers replacement is missing: {path}")
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load vendored Transformers replacement: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(parent, attribute, module)

    from transformers.models.siglip import check

    # The vendored replace's own check, wrapped rather than replaced.
    #
    # It asks `transformers.__version__ == "4.53.2"` (see
    # `transformers_replace/siglip/check.py`), and this overlay used to answer
    # `lambda: True` on its behalf — which turned `pi0_pytorch.py:123-127`'s
    # version gate into dead code. Nothing recorded the other half either: the
    # goldens in `results/` were captured on hosts whose Transformers version
    # their manifests do not name, and this module is the only thing that knows
    # whether the replace matched. The reference still runs on a mismatch — it
    # is the reference for numbers that already exist, and the replace itself
    # needs 4.53.x (`transformers.masking_utils`), so a mismatch means "this
    # environment is not the pinned one", not "these numbers are wrong". What
    # changes is that it is *said*, once, and recorded: `REPLACE_STATUS` is what
    # the manifest quotes.
    installed_check = check.check_whether_transformers_replace_is_installed_correctly
    REPLACE_STATUS["installed"] = transformers.__version__
    reported = False

    def _checked_and_reported() -> bool:
        nonlocal reported
        ok = installed_check()
        REPLACE_STATUS["ok"] = ok
        if not ok and not reported:
            reported = True
            print(
                f"[apxinf_ref] the vendored Transformers replace does not report its pinned "
                f"version (installed transformers {transformers.__version__}); the reference "
                f"still runs, and the manifest records this",
                file=sys.stderr,
            )
        return True

    check.check_whether_transformers_replace_is_installed_correctly = _checked_and_reported

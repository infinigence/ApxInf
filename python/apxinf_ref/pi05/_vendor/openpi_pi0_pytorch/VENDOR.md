# Vendor: openpi PyTorch pi0/pi0.5 model

## Source

Files in this directory are a snapshot of the upstream openpi PyTorch
modeling code:

| Local file | Source |
|---|---|
| `pi0_pytorch.py` | `openpi_src/src/openpi/models_pytorch/pi0_pytorch.py` |
| `gemma_pytorch.py` | `openpi_src/src/openpi/models_pytorch/gemma_pytorch.py` |
| `preprocessing_pytorch.py` | `openpi_src/src/openpi/models_pytorch/preprocessing_pytorch.py` |
| `image_tools.py` | `openpi_src/src/openpi/shared/image_tools.py` (PyTorch subset, JAX paths stripped) |
| `gemma_config.py` | shim replacing `openpi.models.gemma.get_config` (JAX-free) |
| `pi0_config.py` | shim replacing `openpi.models.pi0_config.Pi0Config` (JAX-free) |
| `transformers_replace/` | pinned Gemma, PaliGemma and SigLIP replacements required by the PyTorch port |
| `transformers_overlay.py` | installs the replacements into Transformers' normal module names |

Source repository: openpi (PyTorch port subtree).
Capture date: 2026-04-25.

## Why vendor

The upstream `openpi` package transitively depends on JAX, Flax, and
nnx — pulling those into the training container would inflate the
runtime footprint and risk version drift against the inference
framework. Only the PyTorch subset is needed by `training.trainers`,
so we snapshot it locally and rewrite the cross-package imports as
relative.

## Modifications from upstream

1. `from openpi.models.gemma import …` → `from . import gemma_config as _gemma`
2. `from openpi.models_pytorch.gemma_pytorch import …` → `from .gemma_pytorch import …`
3. `from openpi.shared import image_tools` → `from . import image_tools`
4. `image_tools.py` JAX functions removed; only `resize_with_pad_torch` retained.
5. `gemma_config.py`, `pi0_config.py`: thin JAX-free dataclass shims.
6. `transformers_replace/models/<family>/` → `transformers_replace/<family>/`: the
   family directories were lifted one level. Upstream (and the copy procedure
   quoted in `pi0_pytorch.py`) nests them under `models/`, but this repository
   ignores every directory named `models/`, so a replacement added there would
   never be committed and a fresh clone would be missing it. The overlay loads
   each file by explicit path with an explicit dotted module name, so the
   nesting it sits in has no meaning to Python; the family directories are what
   map a file onto the `transformers.models.<family>` module it replaces.

No semantic changes — model architecture, forward path, RMSNorm/RoPE
math, attention masking, and flow-matching loss are byte-identical to
upstream. The HF `transformers_replace/` patch is vendored as part of this
snapshot because a stock `transformers==4.53.2` installation does not expose
the `use_adarms` Gemma/PaliGemma interfaces used by this model. The overlay
loads only these package-local files and does not modify the installed package
on disk.

One behavioural difference, and it is in the overlay rather than in the
vendored code: `transformers_replace/siglip/check.py`'s version gate
(`transformers.__version__ == "4.53.2"`) is **wrapped, not enforced**. The
overlay calls it, records its verdict in `REPLACE_STATUS`, and warns once on
stderr when it fails, but still answers `True`. The replace itself needs 4.53.x
(it imports `transformers.masking_utils`), so a mismatch means "this environment
is not the pinned pair" rather than "the numbers are wrong" — and it is a fact
worth recording rather than a run worth refusing. `pi05_gold/capture.py` writes
the installed version and that verdict into the manifest's `runtime` block. The
goldens already in `results/` predate the field, so which Transformers produced
them is not recorded anywhere.

## Extending the vendor

If `training/` adds dependencies on additional openpi modules,
snapshot them here and document the source in this table. Do **not**
import directly from an external openpi checkout — keep the vendor
tree as the single source of truth so the training stack remains
self-contained.

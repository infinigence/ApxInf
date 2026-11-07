"""Vendored snapshot of openpi PyTorch pi0/pi0.5 model.

See ``VENDOR.md`` for source path and capture date. The training stack
imports ``PI0Pytorch`` and ``PaliGemmaWithExpertModel`` from this module
to avoid taking a hard dependency on the upstream ``openpi`` package
tree.
"""

from .gemma_config import Config as GemmaConfig
from .gemma_config import LORA_DEFAULTS, Variant, get_config


def __getattr__(name):
    # Keep the package import side-effect free.  The Pi0.5 runner installs its
    # pinned Transformers overlay before importing the model modules; eager
    # imports here would load stock Transformers first and silently bypass the
    # replacement Gemma/PaliGemma implementations.
    if name == "PaliGemmaWithExpertModel":
        from .gemma_pytorch import PaliGemmaWithExpertModel

        return PaliGemmaWithExpertModel
    if name == "Pi0Config":
        from .pi0_config import Pi0Config

        return Pi0Config
    if name == "PI0Pytorch":
        from .pi0_pytorch import PI0Pytorch

        return PI0Pytorch
    raise AttributeError(name)

__all__ = [
    "LORA_DEFAULTS",
    "GemmaConfig",
    "PI0Pytorch",
    "PaliGemmaWithExpertModel",
    "Pi0Config",
    "Variant",
    "get_config",
]

"""Pi0/Pi0.5 model config (JAX-free).

Replacement shim for the subset of ``openpi.models.pi0_config.Pi0Config``
that ``PI0Pytorch.__init__`` and ``forward`` actually read. Field
defaults match the upstream PyTorch port @ 2026-04-25.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from .gemma_config import Variant


@dataclasses.dataclass(frozen=True)
class Pi0Config:
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    paligemma_variant: Variant = "gemma_2b"
    action_expert_variant: Variant = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int | None = None
    pi05: bool = False
    discrete_state_input: bool | None = None
    # ``None`` disables compilation for portable/eager runtimes.
    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

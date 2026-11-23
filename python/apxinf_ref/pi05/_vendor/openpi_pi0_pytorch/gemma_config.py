"""Gemma variant dimension configs (JAX-free).

Replacement shim for ``openpi.models.gemma.get_config`` used by the
vendored ``pi0_pytorch.PI0Pytorch``. The dimensions match
``openpi.models.gemma`` (PyTorch port subtree) @ 2026-04-25 verbatim. LoRA is
applied externally by ``training.lora.inject``, so we drop the JAX
``lora_configs`` field — the official rank/alpha values are documented
in ``LORA_DEFAULTS`` for reference.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

Variant = Literal[
    "dummy",
    "gemma_300m",
    "gemma_300m_lora",
    "gemma_2b",
    "gemma_2b_lora",
]


@dataclasses.dataclass(frozen=True)
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


# Official LoRA defaults from openpi.models.gemma (used by the JAX
# baseline ``train_jax_lora_recap.py`` via ``gemma_2b_lora`` /
# ``gemma_300m_lora`` variants). Init: nn.initializers.normal(stddev=0.01)
# for both A and B; rslora=False.
LORA_DEFAULTS: dict[str, dict[str, float]] = {
    "gemma_2b_lora": {"rank": 16, "alpha": 16.0},
    "gemma_300m_lora": {"rank": 32, "alpha": 32.0},
}


def get_config(variant: Variant) -> Config:
    """Returns dim config for the named gemma variant."""
    if variant == "dummy":
        return Config(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant in ("gemma_300m", "gemma_300m_lora"):
        return Config(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant in ("gemma_2b", "gemma_2b_lora"):
        return Config(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
    raise ValueError(f"Unknown variant: {variant}")

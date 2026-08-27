"""Type stubs for the ``apxinf_py`` PyO3 extension (L0/L1 bare-model infer)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__version__: str

class Model:
    """A loaded VLA model handle exposing L0/L1 bare-model inference."""

    @staticmethod
    def load(
        model: str,
        path: str,
        device: str = ...,
        precision: str = ...,
        calibration: str | None = ...,
        tactics: str | None = ...,
        action_horizon: int | None = ...,
        num_views: int | None = ...,
        sampling_seed: int = ...,
    ) -> "Model":
        """Load a checkpoint through the unified ``AutoModel`` frontend.

        ``device`` is ``cuda:N`` (default) or ``cpu``.
        ``precision`` is ``auto`` (default), ``fp8``, ``bf16``, or ``int8``.
        ``action_horizon`` overrides the checkpoint's chunk length (a sequence
        length, not a weight dimension).
        ``num_views`` serves fewer cameras than the checkpoint declares (1..=its
        own count), which is numerically equivalent to openpi padding + masking
        the absent views but skips their patch tokens.
        ``sampling_seed`` seeds the implicit device-side noise stream.
        """
        ...

    @staticmethod
    def random(
        model: str,
        device: str = ...,
        precision: str = ...,
        num_views: int = ...,
        image_size: int = ...,
        action_horizon: int = ...,
        action_dim: int = ...,
        num_flow_steps: int = ...,
        max_token_len: int = ...,
        calibration: str | None = ...,
        tactics: str | None = ...,
        seed: int = ...,
        sampling_seed: int = ...,
    ) -> "Model":
        """Build a checkpoint-free PI0.5 runtime with deterministic weights."""
        ...

    def _infer_patches(
        self,
        patches: npt.NDArray[np.float32],
        token_ids: npt.NDArray[np.uint32],
        noise: npt.NDArray[np.float32] | None = ...,
    ) -> npt.NDArray[np.float32]:
        """Private L0 path for consistency tests."""
        ...

    def _infer_patches_seeded(
        self,
        patches: npt.NDArray[np.float32],
        token_ids: npt.NDArray[np.uint32],
        seed: int,
        sequence: int = ...,
        draw: int = ...,
    ) -> npt.NDArray[np.float32]:
        """Private seeded L0 path for consistency tests."""
        ...

    def infer_rgb(
        self,
        rgb_u8: npt.NDArray[np.uint8],
        layout: str,
        token_ids: npt.NDArray[np.uint32],
        noise: npt.NDArray[np.float32] | None = ...,
    ) -> npt.NDArray[np.float32]:
        """L1: infer from resized RGB uint8. Returns normalized-domain action."""
        ...

    def infer_rgb_seeded(
        self,
        rgb_u8: npt.NDArray[np.uint8],
        layout: str,
        token_ids: npt.NDArray[np.uint32],
        seed: int,
        sequence: int = ...,
        draw: int = ...,
    ) -> npt.NDArray[np.float32]:
        """Seeded L1 path using the runtime's device-side normal generator."""
        ...

    def reset_sampling(self, seed: int | None = ...) -> None: ...

    @property
    def device(self) -> str: ...
    @property
    def action_dim(self) -> int: ...
    @property
    def action_horizon(self) -> int: ...
    @property
    def num_views(self) -> int: ...
    @property
    def image_size(self) -> int: ...
    @property
    def patch_size(self) -> int: ...
    @property
    def patches_per_view(self) -> int: ...
    @property
    def max_token_len(self) -> int: ...

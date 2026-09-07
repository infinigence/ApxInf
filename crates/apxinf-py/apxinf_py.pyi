"""Type stubs for the ``apxinf_py`` PyO3 extension (L0/L1 bare-model infer)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__version__: str

class Model:
    """A loaded VLA model handle exposing its bare-model inference contract."""

    @staticmethod
    def load(
        model: str,
        path: str,
        device: str = ...,
        precision: str = ...,
        calibration: str | None = ...,
        tactics: str | None = ...,
        autotune: bool = ...,
        config_json: str | None = ...,
        action_horizon: int | None = ...,
        num_views: int | None = ...,
        num_flow_steps: int | None = ...,
        flow_start_time: float | None = ...,
        sampling_seed: int = ...,
    ) -> "Model":
        """Load a checkpoint through the unified ``AutoModel`` frontend.

        ``device`` is ``cuda:N`` (default) or ``cpu``.
        ``precision`` is ``auto`` (default), ``fp8``, ``bf16``, or ``int8``.
        ``config_json`` supplies architecture JSON when it is stored outside the
        checkpoint's ``config.json``. ``None`` leaves loading to AutoModel.
        ``action_horizon`` overrides the checkpoint's chunk length (a sequence
        length, not a weight dimension).
        ``num_views`` serves fewer cameras than the checkpoint declares (1..=its
        own count), which is numerically equivalent to openpi padding + masking
        the absent views but skips their patch tokens.
        ``num_flow_steps`` and ``flow_start_time`` are PI0.5 deployment overrides.
        ``sampling_seed`` seeds the implicit device-side noise stream.
        """
        ...

    def _infer_patches(
        self,
        patches: npt.NDArray[np.float32],
        token_ids: npt.NDArray[np.uint32],
        noise: npt.NDArray[np.float32] | None = ...,
        action_mask: npt.NDArray[np.float32] | None = ...,
    ) -> npt.NDArray[np.float32]:
        """Private L0 path used by model-specific policies and parity tests."""
        ...

    def infer_rgb(
        self,
        rgb_u8: npt.NDArray[np.uint8],
        layout: str,
        token_ids: npt.NDArray[np.uint32],
        noise: npt.NDArray[np.float32] | None = ...,
        action_mask: npt.NDArray[np.float32] | None = ...,
    ) -> npt.NDArray[np.float32]:
        """L1: infer from resized RGB uint8. Returns normalized-domain action."""
        ...

    def _calibrate_rgb(
        self,
        rgb_u8: npt.NDArray[np.uint8],
        layout: str,
        token_ids: npt.NDArray[np.uint32],
        noise: npt.NDArray[np.float32],
    ) -> dict[str, float]: ...

    def _calibration_plan(self) -> list[str]: ...

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
    @property
    def accepts_rgb_u8(self) -> bool: ...

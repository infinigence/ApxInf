"""OpenPI websocket adapter and the wire contract required by LIBERO."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np


LIBERO_ACTION_DIM = 7
EXPECTED_SERVER_METADATA = {
    "protocol": "openpi.websocket_policy",
    "robot": "franka_libero",
    "image_keys": ["observation/image", "observation/wrist_image"],
    "state_key": "observation/state",
    "action_dim": LIBERO_ACTION_DIM,
}


def validate_libero_server_metadata(metadata: dict) -> None:
    """Reject a published server contract that cannot run LIBERO correctly.

    Third-party OpenPI servers do not necessarily publish every ApxInf field, so
    absent fields remain compatible. Published fields, however, must match.
    """
    mismatches = []
    for key, expected_value in EXPECTED_SERVER_METADATA.items():
        if key not in metadata:
            continue
        actual_value = metadata[key]
        if key == "image_keys" and not isinstance(actual_value, list):
            try:
                actual_value = list(actual_value)
            except TypeError:
                pass
        if actual_value != expected_value:
            mismatches.append(f"{key}={actual_value!r} (expected {expected_value!r})")
    if mismatches:
        raise RuntimeError(
            "server wire contract is not LIBERO-compatible: " + "; ".join(mismatches)
        )


def _observation(base, wrist, state, prompt) -> dict:
    return {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": prompt,
    }


class LiberoWebsocketClient:
    """Reach an OpenPI-compatible server through the official client adapter."""

    def __init__(self, host: str, port: int, expected_precision: str) -> None:
        from openpi_client import websocket_client_policy

        for variable in ("NO_PROXY", "no_proxy"):
            entries = [item for item in os.environ.get(variable, "").split(",") if item]
            if host not in entries:
                entries.append(host)
            os.environ[variable] = ",".join(entries)
        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.metadata = self._client.get_server_metadata()
        actual_precision = self.metadata.get("precision")
        if actual_precision != expected_precision:
            self.close()
            raise RuntimeError(
                f"server precision is {actual_precision!r}, "
                f"expected {expected_precision!r}"
            )
        try:
            validate_libero_server_metadata(self.metadata)
        except Exception:
            self.close()
            raise

    def infer(self, base, wrist, state, prompt) -> Tuple[np.ndarray, dict]:
        response = self._client.infer(_observation(base, wrist, state, prompt))
        actions = np.asarray(response["actions"], dtype=np.float32)
        policy_timing = response.get("policy_timing", {}) or {}
        server_timing = response.get("server_timing", {}) or {}
        model_ms = float(policy_timing.get("infer_ms", 0.0))
        policy_ms = float(policy_timing.get("policy_ms", model_ms))
        server_compute_ms = float(server_timing.get("infer_ms", policy_ms))
        return actions, {
            "model_seconds": model_ms / 1000.0,
            "server_processor_seconds": max(0.0, server_compute_ms - model_ms)
            / 1000.0,
        }

    def close(self) -> None:
        connection = getattr(self._client, "_ws", None)
        if connection is not None:
            connection.close()

#!/usr/bin/env python3
"""Client for an ApxInf (OpenPI-compatible) websocket policy server.

This is the robot-side half of a split deployment: the machine that renders and
drives the arm calls a policy that runs on another host. It exists because the
serving side deliberately ships no client -- ``apxinf.serving`` is transport for
*any* policy and pairs with the upstream ``openpi_client`` -- and that upstream
package is one more thing to have on a robot host, with its own proxy handling
and its own release cadence.

So this file is **self-contained**: numpy + msgpack + websockets, nothing else,
no import of ``apxinf`` and no checkout needed. ``scp`` it to the robot host and
run it. Its MessagePack codec is a byte-for-byte mirror of
``python/apxinf/apxinf/serving/msgpack_numpy.py``, pinned by
``tests/test_pi05_remote_client.py`` -- that copy is legal *because* it is
provably identical, so it may be brought into line but never "improved".

Wire protocol (``apxinf/serving/websocket.py``): the server sends one msgpack
frame of metadata the moment you connect, then answers one binary observation
frame with one result frame. ``GET /healthz`` answers ``OK``. There is no key
negotiation -- **the metadata is the contract**, so this client asserts what it
sends against what the server published instead of hardcoding a database's keys.

    python pi05_remote_client.py --host 127.0.0.1 --port 8000
    python pi05_remote_client.py --host 127.0.0.1 --requests 5 --noise-seed 0

In a robot loop::

    from pi05_remote_client import RemotePolicy

    policy = RemotePolicy("127.0.0.1", 8000)      # connects, reads metadata
    result = policy.act([base_rgb, wrist_rgb], state=state, prompt=task)
    result.actions                                    # float32 [H, action_dim]

Two facts about the server worth knowing before you build on it:

* **It serves one request at a time.** The model handle is bound to the thread
  that created it, so the server calls the policy on its event loop rather than
  in a worker; concurrent clients queue. One arm per server is the intended shape.
* **Consecutive identical requests do not give identical actions**, because the
  flow-matching latent is re-drawn per request from the engine's own RNG stream.
  That is the stochastic policy behaving normally. To make a call reproducible --
  to replay it, or to compare two paths -- supply ``noise`` (see
  :meth:`RemotePolicy.infer`); the engine reads it out of the observation, so it
  needs no protocol extension.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Mapping, NamedTuple, Sequence

import msgpack
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

__all__ = [
    "RemotePolicy",
    "RemotePolicyError",
    "RemotePolicyTimeout",
    "RemoteDialectError",
    "InferenceResult",
    "pack_numpy",
    "unpack_numpy",
    "packer",
    "unpackb",
    "timing_segments",
]

logger = logging.getLogger("pi05_remote_client")

#: The observation key a caller-supplied latent rides under. Not an OpenPI wire
#: key -- it is the key ``Pi05Policy`` already reads out of the observation
#: (``python/apxinf/apxinf/policies/impls/pi05.py``), which is what lets this
#: client be reproducible without the server knowing about it. Reserved: an
#: observation with a real ``noise`` field of its own would collide.
NOISE_KEY = "noise"

#: Distinguishes "the caller said nothing, use whatever the server publishes"
#: from "the caller said None", which for ``state_key`` is a real value meaning
#: the observation carries no state under any key.
_UNSET = object()


# ── OpenPI-compatible MessagePack + NumPy codec ──────────────────────────────
#
# A copy of ``python/apxinf/apxinf/serving/msgpack_numpy.py``, which itself
# mirrors ``openpi_client.msgpack_numpy``. The spelling below is deliberately
# the same one, defaults included: ``use_bin_type`` (str vs raw), ``raw=False``
# and ``strict_map_key`` are all inherited from msgpack >= 1.0 on both sides of
# the wire, and naming them here would let this file drift from the library
# while still looking deliberate. Tests pin encoded frames byte-for-byte
# against the library, and against upstream when it is installed.


def pack_numpy(value: Any) -> Any:
    """Encode arrays exactly like ``openpi_client.msgpack_numpy``."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"unsupported NumPy dtype {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def unpack_numpy(value: dict) -> Any:
    """Decode arrays exactly like ``openpi_client.msgpack_numpy``.

    The arrays come back **read-only** (their buffer is the frame's ``bytes``).
    Everything this module hands a caller is rebuilt writable; anything you
    decode by hand is not.
    """
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def packer() -> "msgpack.Packer":
    """A fresh ``Packer`` wired to :func:`pack_numpy` (one per connection)."""
    return msgpack.Packer(default=pack_numpy)


def unpackb(payload: bytes) -> Any:
    """Decode one MessagePack frame, rebuilding NumPy arrays via :func:`unpack_numpy`."""
    return msgpack.unpackb(payload, object_hook=unpack_numpy)


# ── Errors ───────────────────────────────────────────────────────────────────


class RemotePolicyError(RuntimeError):
    """The server could not serve a request, or the connection to it failed."""


class RemotePolicyTimeout(RemotePolicyError, TimeoutError):
    """No response arrived in time.

    The connection is closed before this is raised: the server answers exactly
    one frame per request, so a late response would otherwise be read as the
    *next* request's answer -- wrong actions that look entirely normal.
    """


class RemoteDialectError(RemotePolicyError):
    """The server's published contract is not the one this client was told to send."""


# ── Timing ───────────────────────────────────────────────────────────────────


#: Display labels for the four segments, which nest as
#: ``e2e = model + pre/post + net``. Display-only: the dict keys below freeze
#: the ``--json`` report and the eval ledger, so a reworded label never moves them.
SEGMENT_LABELS = {
    "round_trip": "e2e",  # client-measured end to end
    "model": "model",  # server: pure model forward
    "server_processor": "pre/post",  # server: the policy around the model
    "transport": "net",  # link + both ends' codec
}


def timing_segments(response: Mapping[str, Any], round_trip_ms: float) -> dict:
    """Split one round trip into the four segments both benches report.

    The arithmetic is ``scripts/bench_pi05.py``'s L3 shell, line for line, and
    the four nest: ``e2e = model + pre/post + net``.

    * ``policy_timing.infer_ms``  -- pure model (bare-model ``infer_rgb``)
    * ``policy_timing.policy_ms`` -- whole policy (pre + model + post)
    * ``server_timing.infer_ms``  -- event-loop wall clock of ``policy.infer``

    ``server_processor`` is what the server's own pipeline costs beyond the
    model, and ``transport`` is everything the client can see that the server
    does not count: socket, codec, and -- on a real deployment -- the link.
    Missing keys degrade to zero or to the coarser field rather than raising;
    a server that reports nothing is still served.
    """
    policy_timing = response.get("policy_timing") or {}
    server_timing = response.get("server_timing") or {}
    model_ms = float(policy_timing.get("infer_ms", 0.0))
    server_total_ms = float(policy_timing.get("policy_ms", model_ms))
    server_compute_ms = float(server_timing.get("infer_ms", server_total_ms))
    return {
        "round_trip": float(round_trip_ms),
        "model": model_ms,
        "server_processor": max(0.0, server_total_ms - model_ms),
        "transport": max(0.0, round_trip_ms - server_compute_ms),
    }


def _shape_label(value: Any) -> str:
    """One value as one token: ``256x256x3u8``, ``7f32``, ``text[61]``."""
    if isinstance(value, np.ndarray):
        dims = "x".join(str(axis) for axis in value.shape)
        if value.dtype.kind in "fiu":
            return f"{dims}{value.dtype.kind}{value.dtype.itemsize * 8}"
        return f"{dims}{value.dtype}"
    if isinstance(value, str):
        return f"text[{len(value)}]"
    return type(value).__name__


def _observation_summary(observation: Mapping[str, Any]) -> str:
    """Describe request shapes without logging camera contents."""
    return ", ".join(f"{key}={_shape_label(value)}" for key, value in observation.items())


def _stats(samples_ms: Sequence[float]) -> dict:
    """Nearest-rank percentiles; byte-for-byte ``scripts/bench_pi05.py``'s contract."""
    ordered = sorted(samples_ms)
    n = len(ordered)
    return {
        "p50": ordered[int(0.50 * (n - 1))],
        "p95": ordered[int(0.95 * (n - 1))],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if n > 1 else 0.0,
        "samples": n,
    }


class InferenceResult(NamedTuple):
    """One served chunk: the actions, how the call split up, and the raw response."""

    actions: np.ndarray
    segments: dict
    round_trip_ms: float
    raw: Mapping[str, Any]


# ── The client ───────────────────────────────────────────────────────────────


class RemotePolicy:
    """A policy that lives on another host, shaped like an ``apxinf.Policy``.

    ``infer`` is the ``openpi_client``-shaped entry point (an observation dict
    in, a result dict out) so a caller can move off ``WebsocketClientPolicy`` by
    changing where the object comes from. ``act`` is the robot-loop one: camera
    frames positionally, in the server's own view-slot order.

    Everything the client needs is read off the server's metadata at connect
    time. Passing ``image_keys`` / ``state_key`` / ``prompt_key`` here does not
    *tell* the server anything -- it declares what this client intends to send,
    and a disagreement raises :class:`RemoteDialectError` at connect instead of
    degrading accuracy in the field. Omit them and the server's vocabulary is
    adopted wholesale.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        uri: str | None = None,
        image_keys: Sequence[str] | None = None,
        state_key: Any = _UNSET,
        prompt_key: str | None = None,
        expect_model_variant: str | None = None,
        noise_seed: int | None = None,
        connect_timeout_s: float = 10.0,
        request_timeout_s: float | None = 120.0,
        max_reconnects: int | None = None,
        require_consistent_metadata: bool = True,
    ) -> None:
        self._uri = uri or f"ws://{host}:{port}"
        # The host and port come from the URI, because the URI is the address
        # this object actually connects to. The keyword arguments are still at
        # their defaults when `uri` is given, so reading them instead would make
        # `RemotePolicy(uri="ws://127.0.0.1:8000").health()` probe 127.0.0.1 and
        # report UNREACHABLE about a server it is talking to.
        parsed = urllib.parse.urlsplit(self._uri)
        self._host = parsed.hostname or host
        self._port = parsed.port or port
        self._declared_image_keys = (
            None if image_keys is None else tuple(str(key) for key in image_keys)
        )
        self._declared_state_key = state_key
        self._declared_prompt_key = prompt_key
        self._expect_model_variant = expect_model_variant
        self._noise_seed = noise_seed
        self._connect_timeout_s = connect_timeout_s
        # A generous default on purpose: the first request to a freshly started
        # server pays weight upload and engine prepare, and on MUSA the torch
        # bridge can be imported lazily inside it. A short timeout there looks
        # like a hang. ``None`` waits forever.
        self._request_timeout_s = request_timeout_s
        self._max_reconnects = max_reconnects
        self._require_consistent_metadata = require_consistent_metadata

        self._connection = None
        self._frame_packer = packer()
        self._reconnects = 0
        self._requests = 0
        self._noise_rng = (
            None if noise_seed is None else np.random.default_rng(noise_seed)
        )
        self._latent: np.ndarray | None = None
        self.metadata: dict = {}
        self._connect()

    # -- connection / contract ------------------------------------------------

    def _connect(self) -> None:
        connection = connect(
            self._uri,
            # Both of these have to be named. ``compression`` defaults to
            # permessage-deflate while the server serves with compression off,
            # and ``proxy`` defaults to True, which would route a LAN address
            # through whatever proxy the robot host has configured -- the
            # NO_PROXY dance the upstream client needs, done at the transport
            # layer instead.
            compression=None,
            proxy=None,
            # A frame of two 720p cameras is ~8 MB; the 1 MiB default would cut
            # it off mid-observation. The server makes the same choice.
            max_size=None,
            open_timeout=self._connect_timeout_s,
            ping_interval=20,
            close_timeout=5,
        )
        try:
            frame = connection.recv(timeout=self._connect_timeout_s)
            if isinstance(frame, str):
                # A text frame this early is the server's traceback, sent before
                # it closed the connection.
                raise RemotePolicyError(
                    f"{self._uri} refused the connection:\n{frame}"
                )
            metadata = unpackb(frame)
            if not isinstance(metadata, Mapping):
                raise RemotePolicyError(
                    f"{self._uri} sent a {type(metadata).__name__} where the metadata "
                    "frame belongs. Is that address an apxinf / OpenPI policy server?"
                )
            metadata = dict(metadata)
        except BaseException:
            connection.close()
            raise

        previous = self.metadata
        self.metadata = metadata
        self._connection = connection
        try:
            self._assert_contract(previous)
        except BaseException:
            self.close()
            raise
        logger.info(
            "connected to %s: model_type=%s model_variant=%s H=%s D=%s views=%s keys=%s state=%s",
            self._uri,
            metadata.get("model_type"),
            metadata.get("model_variant") or metadata.get("precision"),
            metadata.get("action_horizon"),
            metadata.get("action_dim"),
            metadata.get("num_views"),
            list(metadata.get("image_keys") or ()),
            "dropped" if metadata.get("state_key") is None else metadata.get("state_key"),
        )

    def _assert_contract(self, previous: Mapping[str, Any]) -> None:
        """Check the declared dialect against what the server published.

        Run at connect *and* after every reconnect, because the reconnect is
        where a server can quietly become a different one -- reloaded with
        another checkpoint, another variant, another view count -- and a run
        that mixes two of those into one set of numbers is worse than a run
        that stops.
        """
        metadata = self.metadata
        for key in ("image_keys", "action_horizon", "action_dim"):
            if metadata.get(key) is None:
                raise RemotePolicyError(
                    f"{self._uri} published no {key!r}; this client will not guess "
                    f"what the server reads. Metadata: {sorted(metadata)}"
                )

        served_images = tuple(str(key) for key in metadata["image_keys"])
        if self._declared_image_keys is not None and self._declared_image_keys != served_images:
            raise RemoteDialectError(
                f"{self._uri} serves image_keys={list(served_images)} but this client "
                f"was told to send {list(self._declared_image_keys)}; names and order "
                "both matter (key i fills model view slot i)"
            )

        served_state = metadata.get("state_key")
        if self._declared_state_key is not _UNSET:
            if served_state is not None and served_state != self._declared_state_key:
                raise RemoteDialectError(
                    f"{self._uri} serves state_key={served_state!r} but this client "
                    f"was told to send {self._declared_state_key!r}"
                )
            if served_state is None and self._declared_state_key is not None:
                logger.warning(
                    "%s serves no state key: the policy drops state, so %r will be "
                    "ignored. Proprioception is off for this deployment.",
                    self._uri,
                    self._declared_state_key,
                )

        served_prompt = metadata.get("prompt_key")
        if (
            self._declared_prompt_key is not None
            and served_prompt is not None
            and served_prompt != self._declared_prompt_key
        ):
            raise RemoteDialectError(
                f"{self._uri} serves prompt_key={served_prompt!r} but this client was "
                f"told to send {self._declared_prompt_key!r}"
            )

        variant = metadata.get("model_variant") or metadata.get("precision")
        if self._expect_model_variant is not None and variant != self._expect_model_variant:
            raise RemoteDialectError(
                f"{self._uri} serves model_variant={variant!r}, expected "
                f"{self._expect_model_variant!r}"
            )

        if previous:
            drift = {
                key: (previous.get(key), metadata.get(key))
                for key in set(previous) | set(metadata)
                if previous.get(key) != metadata.get(key)
            }
            if drift and self._require_consistent_metadata:
                raise RemoteDialectError(
                    f"{self._uri} published different metadata after reconnecting: "
                    f"{drift}. A server that restarted as another policy must not be "
                    "mixed into one run; reconnect deliberately if that is what you "
                    "want (require_consistent_metadata=False)."
                )

    # -- published contract ---------------------------------------------------

    @property
    def image_keys(self) -> tuple:
        """The camera keys the server reads, in model view-slot order."""
        declared = self._declared_image_keys
        return declared if declared is not None else tuple(self.metadata["image_keys"])

    @property
    def state_key(self):
        """The state key the server reads, or ``None`` if it drops state.

        ``None`` wins over anything declared here: the server publishing no state
        key is it saying it reads none, and a client that answered with its own
        key would be promising a channel that does not exist. The declaration is
        still checked against the server's at connect (a mismatch raises; a
        server that drops state warns once).
        """
        served = self.metadata.get("state_key")
        if served is None:
            return None
        if self._declared_state_key is _UNSET:
            return served
        return self._declared_state_key

    @property
    def prompt_key(self) -> str:
        """The key the language instruction rides under."""
        declared = self._declared_prompt_key
        return declared or self.metadata.get("prompt_key") or "prompt"

    @property
    def action_dim(self) -> int:
        """Width of one deployable action vector (post-processing output)."""
        return int(self.metadata["action_dim"])

    @property
    def action_horizon(self) -> int:
        """Number of actions in one predicted chunk."""
        return int(self.metadata["action_horizon"])

    @property
    def model_action_dim(self) -> int:
        """Width of the *latent* the model consumes -- the shape ``noise`` takes.

        Not the deployable width: a checkpoint that trims 32 channels down to 7
        still runs its flow in 32.
        """
        return int(self.metadata.get("model_action_dim") or self.metadata["action_dim"])

    def __repr__(self) -> str:
        return (
            f"<RemotePolicy {self._uri} model_type={self.metadata.get('model_type')} "
            f"H={self.metadata.get('action_horizon')} D={self.metadata.get('action_dim')} "
            f"keys={list(self.metadata.get('image_keys') or ())}>"
        )

    # -- requests -------------------------------------------------------------

    def build_observation(
        self,
        images: Sequence[np.ndarray] | Mapping[str, np.ndarray],
        *,
        state: np.ndarray | None = None,
        prompt: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict:
        """Assemble one observation in the server's own vocabulary.

        ``images`` is either a mapping keyed by wire key, or a sequence in
        ``image_keys`` order. Camera frames are sent raw: the server owns
        resizing, so any resolution works -- but they must be **RGB**, both
        here and in openpi, neither of which converts colour. A client reading
        frames with OpenCV has to ``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)``
        first, or it will score badly with everything looking fine.
        """
        observation: dict = {}
        if isinstance(images, Mapping):
            observation.update(images)
        else:
            keys = self.image_keys
            if len(images) != len(keys):
                raise RemotePolicyError(
                    f"{len(images)} image(s) for {len(keys)} served view(s) "
                    f"{list(keys)}; the server was loaded for that many cameras "
                    "and cannot read a different number"
                )
            observation.update(zip(keys, images))
        if state is not None and self.state_key is not None:
            observation[self.state_key] = state
        observation[self.prompt_key] = prompt
        if extra:
            observation.update(extra)
        return observation

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        noise: np.ndarray | None = None,
        timeout_s: float | None = None,
    ) -> dict:
        """Serve one observation. Returns ``actions`` plus timing and the raw response.

        ``noise`` optionally fixes the flow-matching latent, which is what makes
        a call reproducible: the same observation and the same latent give
        bit-identical actions, while without it the engine samples a fresh
        latent every request. It rides in the observation under ``"noise"``,
        which is a key the policy already reads -- no protocol extension, and
        an upstream client that sends none is unaffected.

        The returned dict carries the two keys every ``apxinf.Policy`` promises
        -- ``actions`` (writable ``float32 [H, action_dim]``) and ``timing`` --
        plus ``policy_timing`` / ``server_timing`` as the server sent them and
        ``round_trip_ms``.
        """
        raw = self._request(observation, noise=noise, timeout_s=timeout_s)
        actions = raw.get("actions")
        if actions is None:
            raise RemotePolicyError(
                f"{self._uri} answered without an `actions` key: {sorted(raw)}"
            )
        # ``copy=True`` is load-bearing: a decoded array views the frame's bytes
        # and is read-only, and every in-place edit downstream would raise
        # somewhere far from here.
        actions = np.array(actions, dtype=np.float32, copy=True)
        expected = (
            self.metadata.get("action_horizon"),
            self.metadata.get("action_dim"),
        )
        if expected[0] is not None and actions.shape != expected:
            raise RemotePolicyError(
                f"{self._uri} returned actions{tuple(actions.shape)}, but its metadata "
                f"declares {expected}"
            )
        if not np.isfinite(actions).all():
            raise RemotePolicyError(
                f"{self._uri} returned non-finite actions; the model produced NaN/Inf"
            )
        policy_timing = raw.get("policy_timing") or {}
        return {
            "actions": actions,
            "timing": {
                "model_ms": float(policy_timing.get("infer_ms", 0.0)),
                "total_ms": float(
                    policy_timing.get("policy_ms", policy_timing.get("infer_ms", 0.0))
                ),
                "round_trip_ms": float(raw["round_trip_ms"]),
            },
            "policy_timing": policy_timing,
            "server_timing": raw.get("server_timing") or {},
            "round_trip_ms": float(raw["round_trip_ms"]),
        }

    def act(
        self,
        images: Sequence[np.ndarray] | Mapping[str, np.ndarray],
        *,
        state: np.ndarray | None = None,
        prompt: str,
        noise: np.ndarray | None = None,
        timeout_s: float | None = None,
    ) -> InferenceResult:
        """The robot-loop entry point: frames in, one chunk out.

        Same request as :meth:`infer`, but it also returns the four timing
        segments, which is what a control loop wants to log next to the action
        it acted on.
        """
        observation = self.build_observation(images, state=state, prompt=prompt)
        result = self.infer(observation, noise=noise, timeout_s=timeout_s)
        return InferenceResult(
            actions=result["actions"],
            segments=timing_segments(result, result["round_trip_ms"]),
            round_trip_ms=result["round_trip_ms"],
            raw=result,
        )

    def _request(
        self,
        observation: Mapping[str, Any],
        *,
        noise: np.ndarray | None,
        timeout_s: float | None,
    ) -> dict:
        required = list(self.image_keys) + [self.prompt_key]
        if self.state_key is not None and self.metadata.get("discrete_state"):
            # State is only mandatory where it is injected into the prompt;
            # otherwise the policy ignores it entirely.
            required.append(self.state_key)
        missing = [key for key in required if key not in observation]
        if missing:
            raise RemotePolicyError(
                f"observation is missing {missing}; this server reads "
                f"image_keys={list(self.image_keys)}, prompt_key={self.prompt_key!r}"
                + (
                    f", state_key={self.state_key!r}"
                    if self.metadata.get("discrete_state")
                    else ""
                )
                + f" and got {sorted(observation)}"
            )

        payload: Mapping[str, Any] = observation
        latent = self._resolve_noise(noise)
        if latent is not None:
            # Copied rather than mutated: the caller's observation is theirs, and
            # a control loop may well be reusing one dict across calls.
            payload = dict(observation)
            payload[NOISE_KEY] = latent

        self._ensure_connected()
        request_number = self._requests + 1
        logger.debug(
            "req=%d sending to %s keys=%s in[%s]",
            request_number,
            self._uri,
            list(self.image_keys),
            _observation_summary(observation),
        )
        started = time.perf_counter()
        try:
            self._connection.send(self._frame_packer.pack(dict(payload)))
            frame = self._connection.recv(timeout=self._timeout_for(timeout_s))
        except TimeoutError as error:
            self._drop()
            raise RemotePolicyTimeout(
                f"{self._uri} did not answer within {self._timeout_for(timeout_s):.1f} s. "
                "The connection was closed: a late response would be read as the next "
                "request's answer. If this was the first request, the server may still "
                "be loading -- the torch bridge on MUSA costs tens of seconds on first use."
            ) from error
        except (ConnectionClosed, OSError) as error:
            self._drop()
            raise RemotePolicyError(
                f"connection to {self._uri} failed: {error}. The request was NOT "
                "retried -- a resend would draw a different latent and answer with "
                "different actions. Retry at your level if that is acceptable."
            ) from error
        round_trip_ms = (time.perf_counter() - started) * 1000.0

        if isinstance(frame, str):
            # The server sends its traceback as a text frame, then closes with
            # 1011. That is a server-side failure, not a transport one.
            self._drop()
            raise RemotePolicyError(f"{self._uri} raised while serving:\n{frame}")
        response = unpackb(frame)
        if not isinstance(response, Mapping):
            raise RemotePolicyError(
                f"{self._uri} sent a {type(response).__name__}, expected a response mapping"
            )
        response = dict(response)
        response["round_trip_ms"] = round_trip_ms
        self._requests = request_number
        actions = response.get("actions")
        # The same numbers the summary table prints, from the same function. The
        # shapes only ride along at DEBUG: this client *is* the contract.
        logger.debug(
            "req=%d %s in[%s] out%s",
            request_number,
            self._uri,
            _observation_summary(observation),
            tuple(actions.shape) if isinstance(actions, np.ndarray) else type(actions).__name__,
        )
        segments = timing_segments(response, round_trip_ms)
        logger.info(
            "req=%d | e2e=%.1f model=%.1f pre/post=%.1f net=%.1f ms",
            request_number,
            segments["round_trip"],
            segments["model"],
            segments["server_processor"],
            segments["transport"],
        )
        return response

    def _resolve_noise(self, noise: np.ndarray | None) -> np.ndarray | None:
        """The latent for this request: the caller's, the fixed one, or none.

        ``noise_seed`` draws **one** latent and reuses it for every request. A
        seeded *stream* would also be reproducible at the level of a whole
        sequence, but it would not make two calls comparable on their own, and
        comparing two calls is the reason to fix a latent at all -- an A/B
        between two checkpoints or two hosts has to see the same input.
        """
        if noise is None:
            if self._noise_seed is None:
                return None
            if self._latent is None:
                self._latent = np.ascontiguousarray(
                    self._noise_rng.standard_normal(
                        (self.action_horizon, self.model_action_dim)
                    ),
                    dtype=np.float32,
                )
            return self._latent
        latent = np.ascontiguousarray(noise, dtype=np.float32)
        expected = (self.action_horizon, self.model_action_dim)
        if latent.shape != expected:
            # Checked here because the *server* raises on a bad latent and drops
            # the connection: the cost of a wrong shape would be a reconnect and
            # a traceback, not a message naming the shape.
            raise ValueError(
                f"noise shape {tuple(np.shape(noise))}, expected {expected} "
                f"(action_horizon x model_action_dim, not action_dim="
                f"{self.metadata.get('action_dim')})"
            )
        if not np.isfinite(latent).all():
            raise ValueError("noise must contain only finite values")
        return latent

    def _timeout_for(self, timeout_s: float | None) -> float | None:
        return self._request_timeout_s if timeout_s is None else timeout_s

    def _ensure_connected(self) -> None:
        if self._connection is not None:
            return
        if self._max_reconnects is not None and self._reconnects >= self._max_reconnects:
            raise RemotePolicyError(
                f"{self._uri} is not connected and this client allows "
                f"{self._max_reconnects} transparent reconnect(s)"
            )
        self._reconnects += 1
        logger.warning("reconnecting to %s (reconnect %d)", self._uri, self._reconnects)
        self._connect()

    def _drop(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - the socket is going away regardless
                pass

    # -- lifecycle ------------------------------------------------------------

    def health(self, timeout_s: float = 5.0) -> bool:
        """``GET /healthz``. The server answers without touching the model."""
        scheme = "https" if self._uri.startswith("wss") else "http"
        url = f"{scheme}://{self._host}:{self._port}/healthz"
        # An empty ProxyHandler: a loopback or LAN server is reached directly,
        # whatever proxy the environment names.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(url, timeout=timeout_s) as response:
                return response.status == 200
        except Exception as error:  # noqa: BLE001 - unreachable is the answer, not a crash
            logger.debug("health check on %s failed: %s", url, error)
            return False

    def close(self) -> None:
        """Drop the connection. The object can still reconnect on the next call."""
        self._drop()

    def __enter__(self) -> "RemotePolicy":
        return self

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────


def _requantize_to_u8(images: np.ndarray) -> np.ndarray:
    """``(views, 3, H, W)`` float32 in [-1, 1] -> ``(views, H, W, 3)`` uint8.

    ``<path-to-inputs.npz>`` holds frames already resized and normalized by
    the reference side, so raw bytes can only be recovered by rounding back --
    lossy, and measurable (image-domain relative L2 ~3.9e-3). Same arithmetic
    as ``scripts/bench_pi05_musa.py``, so a run here and the timed layers there
    consume the same frames.
    """
    interleaved = np.ascontiguousarray(images).transpose(0, 2, 3, 1).astype(np.float32)
    return np.ascontiguousarray(
        np.round((interleaved + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    )


def _synthetic_frames(
    count: int, image_size: int, seed: int
) -> list[np.ndarray]:
    """Random uint8 HWC frames. The actions they yield are meaningless."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
        for _ in range(count)
    ]


def _fixture_frames(path: pathlib.Path) -> list[np.ndarray]:
    """Frames out of ``<path-to-inputs.npz>``, one per view, in slot order."""
    with np.load(path) as data:
        if "images" not in data.files:
            raise SystemExit(f"{path} has no `images`; keys are {data.files}")
        images = np.asarray(data["images"], dtype=np.float32)
    if images.ndim != 4 or images.shape[1] != 3:
        raise SystemExit(f"{path}: images{images.shape}, expected (views, 3, H, W)")
    frames = _requantize_to_u8(images)
    return [np.ascontiguousarray(frame) for frame in frames]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="inference server host")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--prompt",
        default="pick up the block",
        help="language instruction; the actions of a synthetic run depend on it "
        "only in that the server tokenizes it",
    )
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--fixture",
        type=pathlib.Path,
        help="send frames from a saved bundle (<path-to-inputs.npz>), float32 "
        "NCHW in [-1, 1], requantized to uint8 exactly as the bench does; "
        "omitted, seeded random uint8 frames are sent",
    )
    parser.add_argument("--seed", type=int, default=0, help="synthetic frame RNG seed")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--state-dim",
        type=int,
        default=8,
        help="width of the zero state vector to send; --no-state omits it entirely",
    )
    parser.add_argument("--no-state", action="store_true", help="send no state key")
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="fix the flow-matching latent from this seed, making every request "
        "reproducible; omitted, the server draws its own latent per request",
    )
    parser.add_argument(
        "--determinism-check",
        action="store_true",
        help="send the same observation twice and require the two answers to agree "
        "exactly when a latent is fixed, and to differ when it is not",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="connect, print the served contract, assert it, and exit without inferring",
    )
    parser.add_argument("--expect-model-variant", default=None)
    parser.add_argument("--expect-action-horizon", type=int, default=None)
    parser.add_argument("--expect-action-dim", type=int, default=None)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request seconds; the first request to a fresh server can cost "
        "tens of seconds (weight upload, engine prepare, and on MUSA a lazily "
        "imported torch bridge)",
    )
    parser.add_argument(
        "--max-reconnects",
        type=int,
        default=None,
        help="transparent reconnects allowed before a request fails; omitted means "
        "unlimited, 0 makes any dropped connection fatal. An in-flight request is "
        "never resent either way",
    )
    parser.add_argument("--dump-actions", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, help="write this run's report as JSON")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _contract_line(metadata: Mapping[str, Any]) -> str:
    normalization = metadata.get("normalization") or {}
    state_key = metadata.get("state_key")
    return (
        f"model_type={metadata.get('model_type')} "
        f"model_variant={metadata.get('model_variant') or metadata.get('precision')} "
        f"H={metadata.get('action_horizon')} D={metadata.get('action_dim')} "
        f"model_D={metadata.get('model_action_dim')} views={metadata.get('num_views')} "
        f"normalization.action={normalization.get('action')} "
        f"state={'dropped' if state_key is None else state_key}"
    )


def _assert_expectations(args: argparse.Namespace, metadata: Mapping[str, Any]) -> None:
    """Everything asserted here is read off the server, so a mismatch is loud."""
    checks = (
        ("--expect-model-variant", args.expect_model_variant,
         metadata.get("model_variant") or metadata.get("precision")),
        ("--expect-action-horizon", args.expect_action_horizon,
         metadata.get("action_horizon")),
        ("--expect-action-dim", args.expect_action_dim, metadata.get("action_dim")),
    )
    for flag, expected, served in checks:
        if expected is not None and expected != served:
            raise SystemExit(f"{flag} {expected!r} but the server serves {served!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    started_wall = time.perf_counter()

    policy = RemotePolicy(
        args.host,
        args.port,
        expect_model_variant=args.expect_model_variant,
        noise_seed=args.noise_seed,
        connect_timeout_s=args.connect_timeout,
        request_timeout_s=args.timeout,
        max_reconnects=args.max_reconnects,
    )
    try:
        metadata = policy.metadata
        _assert_expectations(args, metadata)
        print(f"server {args.host}:{args.port}  {_contract_line(metadata)}")
        print(f"image_keys={list(policy.image_keys)} prompt_key={policy.prompt_key!r}")
        healthy = policy.health()
        print(f"healthz: {'OK' if healthy else 'UNREACHABLE'}")
        if not healthy:
            # The server answers /healthz without touching the model, so a
            # failure here is a transport or address problem, not a busy GPU.
            return 2
        if args.metadata_only:
            return 0

        if args.fixture is not None:
            images = _fixture_frames(args.fixture)
            if len(images) != len(policy.image_keys):
                raise SystemExit(
                    f"--fixture holds {len(images)} view(s) for {len(policy.image_keys)} "
                    f"served slot(s) {list(policy.image_keys)}"
                )
        else:
            images = _synthetic_frames(len(policy.image_keys), args.image_size, args.seed)
        state = None
        if not args.no_state and policy.state_key is not None:
            state = np.zeros(args.state_dim, dtype=np.float32)
        observation = policy.build_observation(images, state=state, prompt=args.prompt)

        for index in range(args.warmup):
            result = policy.infer(observation)
            logger.info(
                "warmup %d/%d: e2e=%.1f model=%.1f ms",
                index + 1,
                args.warmup,
                result["round_trip_ms"],
                result["timing"]["model_ms"],
            )

        segments: dict[str, list[float]] = {name: [] for name in SEGMENT_LABELS}
        last_actions = None
        for _ in range(args.requests):
            result = policy.infer(observation)
            last_actions = result["actions"]
            for name, value in timing_segments(result, result["round_trip_ms"]).items():
                segments[name].append(value)

        determinism = None
        if args.determinism_check:
            determinism = _determinism_check(
                policy, observation, latent_fixed=args.noise_seed is not None
            )
            print(
                "determinism: latent={latent} identical={identical} "
                "max|delta|={max_abs_delta:.3e}".format(**determinism)
            )

        if last_actions is None:
            # A `known = True` feature for a single call. An `assert` here would
            # vanish under `python -O`, leaving the prints below to raise an
            # AttributeError with no explanation — and this is a diagnostic
            # script whose whole job is to say what is wrong.
            raise SystemExit(
                "the request returned no actions, so there is nothing to report. This "
                "is a bug in this script rather than a server response: `last_actions` "
                "is set from a response whose `actions` key was already checked."
            )
        print(
            f"actions: shape={last_actions.shape} dtype={last_actions.dtype} "
            f"finite={bool(np.isfinite(last_actions).all())} "
            f"|a|max={float(np.abs(last_actions).max()):.4g}"
        )
        print(f"first action vector: {last_actions[0]}")
        stats = {name: _stats(values) for name, values in segments.items()}
        print(
            f"\nsegments ms over {args.requests} request(s), p50 (p95) "
            "[e2e = model + pre/post + net]:"
        )
        for name, label in SEGMENT_LABELS.items():
            entry = stats[name]
            print(f"  {label:<18}{entry['p50']:>9.2f} ({entry['p95']:.2f})")

        report = {
            "schema": "apxinf.pi05.remote-client.v1",
            "host": args.host,
            "port": args.port,
            "metadata": metadata,
            "requests": args.requests,
            "warmup": args.warmup,
            "prompt": args.prompt,
            "frames": "fixture" if args.fixture is not None else f"synthetic:seed={args.seed}",
            "noise": (
                f"seeded:{args.noise_seed}" if args.noise_seed is not None else "server-sampled"
            ),
            "segments_ms": stats,
            "determinism": determinism,
            "total_wall_ms": (time.perf_counter() - started_wall) * 1000.0,
        }
        if args.dump_actions is not None:
            args.dump_actions.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.dump_actions, actions=last_actions)
            print(f"wrote {args.dump_actions}")
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"wrote {args.out}")
        if args.json:
            print(json.dumps(report, indent=2))
        return 0
    finally:
        policy.close()


def _determinism_check(
    policy: RemotePolicy, observation: Mapping[str, Any], *, latent_fixed: bool
) -> dict:
    """Two identical requests, once with a fixed latent and once without.

    Both halves matter and they prove different things. With the latent fixed,
    the two answers must be **bit-identical** -- that is the wire carrying the
    observation and the result intact. Without it, they must **differ**, because
    the engine draws a fresh latent per request; if they matched, something
    downstream was caching a result and calling it an inference.
    """
    first = policy.infer(observation)["actions"]
    second = policy.infer(observation)["actions"]
    delta = float(np.abs(first - second).max())
    identical = bool(np.array_equal(first, second))
    if latent_fixed and not identical:
        raise SystemExit(
            f"determinism check failed: the latent was fixed but two identical "
            f"requests differ by {delta:.3e}; the response is not a pure function "
            "of the request"
        )
    if not latent_fixed and identical:
        raise SystemExit(
            "determinism check failed: no latent was sent, yet two identical "
            "requests returned identical actions -- a fresh latent is drawn per "
            "request, so something is caching a result and calling it an inference"
        )
    return {
        "latent": "fixed" if latent_fixed else "server-sampled",
        "identical": identical,
        "max_abs_delta": delta,
    }


if __name__ == "__main__":
    sys.exit(main())

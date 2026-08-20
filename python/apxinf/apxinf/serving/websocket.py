"""The project's websocket policy server — a thin transport shell.

Model-agnostic: it holds any object satisfying the :class:`apxinf.Policy` contract
(``obs dict -> result dict``) and does only protocol translation, so the same
server serves :class:`~apxinf.policies.impls.pi05.Pi05Policy` today and a future
``GrootPolicy`` unchanged. Its wire protocol is compatible with the unmodified
``openpi_client.WebsocketClientPolicy`` (metadata-on-connect, msgpack-numpy
frames, ``/healthz``), so existing robot clients connect without changes.

The policy is called **in-process** (no subprocess / stdio hop). The library's
rich result keys (``normalized_actions`` / ``token_ids`` / ``noise`` /
``metadata``) stay in-process; only ``actions`` + ``policy_timing`` are put on the
wire — see :func:`wire_response`.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Any

import numpy as np
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from .msgpack_numpy import packer, unpackb

__all__ = ["WebsocketPolicyServer", "wire_response", "health_check"]

logger = logging.getLogger(__name__)


def wire_response(result: dict) -> dict:
    """Project a ``apxinf`` policy result onto the wire response.

    Clients read ``actions`` and ``policy_timing``; every other key the library
    returns is in-process detail and is deliberately *not* serialized. The shape
    matches what ``openpi_client`` expects.
    """
    actions = np.ascontiguousarray(result["actions"], dtype=np.float32)
    timing = result.get("timing", {}) or {}
    policy_timing = {"infer_ms": float(timing.get("model_ms", 0.0))}
    if "total_ms" in timing:
        policy_timing["policy_ms"] = float(timing["total_ms"])
    return {"actions": actions, "policy_timing": policy_timing}


class WebsocketPolicyServer:
    """Serve any ``apxinf.Policy`` over the websocket protocol (openpi-compatible wire)."""

    def __init__(
        self, policy: Any, host: str, port: int, *, metadata: dict | None = None
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = (
            dict(metadata)
            if metadata is not None
            else dict(getattr(policy, "metadata", {}))
        )

    async def handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info("connection from %s opened", websocket.remote_address)
        pack = packer()
        await websocket.send(pack.pack(self._metadata))
        previous_total_seconds = None
        while True:
            try:
                request_started = time.monotonic()
                payload = await websocket.recv()
                if isinstance(payload, str):
                    raise TypeError(
                        "inference requests must be binary MessagePack frames"
                    )
                observation = unpackb(payload)
                infer_started = time.monotonic()
                # Call the policy directly on the event-loop thread rather than
                # offloading to a worker (``asyncio.to_thread``): the L1 handle
                # ``apxinf_py.Model`` is *unsendable* — its CUDA context is bound
                # to the thread that created it (the main thread, where the
                # policy was constructed), and touching it from a thread-pool
                # thread panics. Inference is one-at-a-time per GPU anyway, so
                # briefly blocking the loop here is the correct, simplest shape.
                result = self._policy.infer(observation)
                infer_seconds = time.monotonic() - infer_started
                response = wire_response(result)
                response["server_timing"] = {"infer_ms": infer_seconds * 1000}
                if previous_total_seconds is not None:
                    response["server_timing"]["prev_total_ms"] = (
                        previous_total_seconds * 1000
                    )
                await websocket.send(pack.pack(response))
                previous_total_seconds = time.monotonic() - request_started
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.exception("websocket inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                break

    async def run(self) -> None:
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        async with websocket_server.serve(
            self.handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=health_check,
        ) as server:
            logger.info("websocket policy server listening on %s", server.sockets)
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())


def health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None

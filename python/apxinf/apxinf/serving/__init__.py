"""Network serving for ApxInf policies (OpenPI websocket and DM05 HTTP).

Kept out of the top-level ``apxinf`` namespace and imported only on demand, so
``import apxinf`` (processor / offline use) never pulls in ``msgpack`` /
``websockets``. Import explicitly:

    from apxinf.serving import Dm05HttpService, WebsocketPolicyServer

The websocket server is model-agnostic and compatible with the unmodified
``openpi_client.WebsocketClientPolicy``. The stdlib HTTP service owns the fixed
DM05 ``POST /v1/infer`` wire. This package intentionally ships no client.
"""

from __future__ import annotations

import importlib

from .http import Dm05HttpService, serve_dm05_http

__all__ = [
    "WebsocketPolicyServer",
    "wire_response",
    "health_check",
    "Dm05HttpService",
    "serve_dm05_http",
]


def __getattr__(name: str):
    # DM05's stdlib HTTP service must not require the optional websocket stack.
    if name in {"WebsocketPolicyServer", "wire_response", "health_check"}:
        websocket = importlib.import_module(f"{__name__}.websocket")
        return getattr(websocket, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

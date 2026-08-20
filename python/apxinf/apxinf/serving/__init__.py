"""Network serving for apxinf policies (websocket, openpi-compatible wire).

Kept out of the top-level ``apxinf`` namespace and imported only on demand, so
``import apxinf`` (processor / offline use) never pulls in ``msgpack`` /
``websockets``. Import explicitly:

    from apxinf.serving import WebsocketPolicyServer

The server is model-agnostic — it serves any :class:`apxinf.Policy`. Its wire
protocol is compatible with the unmodified ``openpi_client.WebsocketClientPolicy``,
so robot-side clients connect without changes; this package intentionally ships
no client.
"""

from __future__ import annotations

from .websocket import WebsocketPolicyServer, health_check, wire_response

__all__ = ["WebsocketPolicyServer", "wire_response", "health_check"]

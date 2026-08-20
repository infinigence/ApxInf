"""Adapters that expose an :class:`apxinf.Policy` through a *foreign* API.

Like :mod:`apxinf.serving`, this package sits *downstream* of the policy layer: it
consumes the ``Policy`` contract (``obs dict -> result dict``) and translates it
into somebody else's calling convention. Nothing in ``policies`` / ``processors``
depends back on it, and it is kept out of the top-level ``apxinf`` namespace and
imported only on demand, so ``import apxinf`` never pulls in an adapter's heavy
optional dependencies (``torch`` for the lerobot adapter, as ``msgpack`` /
``websockets`` for ``serving``). Import explicitly::

    from apxinf.adapters.lerobot import ApxInfPolicy

An adapter is *not* a place for new inference logic. If an adapter needs to change
numerics, that belongs in a processor step or a policy instead.
"""

from __future__ import annotations

__all__: list[str] = []

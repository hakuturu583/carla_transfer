"""World-model backends. Each one is a client for a model server in Docker."""

from __future__ import annotations

from carla_transfer.backends.base import WorldModelBackend
from carla_transfer.config import BackendKind, TransferConfig


def make_backend(config: TransferConfig) -> WorldModelBackend:
    if config.backend is BackendKind.OMNIDREAMS:
        from carla_transfer.backends.omnidreams import OmniDreamsBackend

        return OmniDreamsBackend(config)
    if config.backend is BackendKind.COSMOS3:
        from carla_transfer.backends.cosmos3 import Cosmos3Backend

        return Cosmos3Backend(config)
    if config.backend is BackendKind.MOCK:
        from carla_transfer.backends.mock import MockBackend

        return MockBackend(config)
    raise ValueError(f"unknown backend {config.backend!r}")


__all__ = ["WorldModelBackend", "make_backend"]

"""carla_transfer as a carla_canvas renderer backend.

``renderer.backend: carla_transfer.carla_canvas:CarlaTransferRenderer`` in a
carla_canvas configuration renders a typesafe_carla program's RGB camera with
carla_transfer's world model (see :mod:`carla_transfer.carla_canvas.renderer`
and docs/carla_canvas.md).
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "CarlaTransferRenderer":  # imports carla_canvas
        from carla_transfer.carla_canvas.renderer import CarlaTransferRenderer

        return CarlaTransferRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CarlaTransferRenderer"]

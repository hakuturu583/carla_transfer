"""CARLA side: attach an RGB camera to a vehicle and turn its images into Frames.

``carla`` is imported lazily so the rest of the package works without the
CARLA egg installed (tests, container manager, remote viewers).
"""

from __future__ import annotations

import logging
import math
import queue
from typing import Any

import numpy as np

from carla_transfer.frames import Frame, Pose, bgra_to_rgb

log = logging.getLogger(__name__)


class CarlaCameraSource:
    """Spawns a ``sensor.camera.rgb`` on ``vehicle`` and exposes frames via
    :meth:`get`. Designed for synchronous mode: call ``world.tick()`` and then
    ``get()`` once per tick."""

    def __init__(
        self,
        world: Any,
        vehicle: Any,
        width: int,
        height: int,
        fov_deg: float = 90.0,
        transform: Any | None = None,
        sensor_tick: float = 0.0,
    ):
        import carla

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov_deg))
        bp.set_attribute("sensor_tick", str(sensor_tick))
        transform = transform or carla.Transform(carla.Location(x=1.5, z=1.6))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.vehicle = vehicle
        self._q: queue.Queue = queue.Queue(maxsize=4)
        self.sensor.listen(self._on_image)

    def _on_image(self, image: Any) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        frame = Frame(
            rgb=bgra_to_rgb(arr),
            timestamp_us=int(image.timestamp * 1e6),
            pose=pose_from_transform(self.vehicle.get_transform()),
            meta={"carla_frame": image.frame},
        )
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self._q.get_nowait()
            self._q.put_nowait(frame)

    def get(self, timeout: float = 2.0) -> Frame:
        return self._q.get(timeout=timeout)

    def destroy(self) -> None:
        self.sensor.stop()
        self.sensor.destroy()


def transform_from_rig_to_world(m: np.ndarray, z_offset: float = 0.3) -> Any:
    """Inverse of :func:`pose_from_transform` for a 4x4 rig-to-world (FLU,
    OpenDRIVE frame): a ``carla.Transform`` to spawn the ego at. ``z_offset``
    lifts the vehicle so it does not spawn intersecting the road."""
    import carla

    yaw = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    return carla.Transform(
        carla.Location(x=float(m[0, 3]), y=-float(m[1, 3]), z=float(m[2, 3]) + z_offset),
        carla.Rotation(yaw=-yaw),
    )


def pose_from_transform(t: Any) -> Pose:
    """CARLA (UE4, left-handed, x-fwd y-right z-up, degrees) → right-handed
    x-fwd y-left z-up pose with an (x, y, z, w) quaternion."""
    loc, rot = t.location, t.rotation
    roll, pitch, yaw = (math.radians(rot.roll), math.radians(-rot.pitch), math.radians(-rot.yaw))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return Pose(xyz=(loc.x, -loc.y, loc.z), quat_xyzw=(qx, qy, qz, qw))

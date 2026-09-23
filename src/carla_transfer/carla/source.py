"""CARLA side: attach an RGB camera to a vehicle and turn its images into Frames.

``carla`` is imported lazily so the rest of the package works without the
CARLA egg installed (tests, container manager, remote viewers).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from typing import Any

import numpy as np

from carla_transfer.frames import Frame, Pose, bgra_to_rgb

log = logging.getLogger(__name__)


class FThetaRemap:
    """Resample a wide pinhole render into an f-theta camera (``r = f * theta``,
    principal point at the image centre, ``f = (width / 2) / (fov / 2)``).

    That is the camera the scene archives declare (``camera_front_wide_120fov``:
    ``polynomial-type: angle-to-pixeldistance``) and the model was trained on.
    CARLA only renders pinhole images, and a 120 degree pinhole has 1.65x less
    focal length at the centre (``320 / tan 60 deg = 185`` vs ``320 / (pi / 3)
    = 306`` px at 640 wide), so everything ahead looked 1.65x smaller in CARLA
    than in the HD-map condition. The pinhole render covers every f-theta ray
    (the corners reach ~68 degrees off axis at 16:9) at ``scale`` times the
    f-theta centre resolution; the sampling map is built once.
    """

    def __init__(self, width: int, height: int, fov_deg: float, scale: float = 0.8):
        self.width, self.height = width, height
        f = (width / 2) / math.radians(fov_deg / 2)
        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        dx, dy = u - width / 2, v - height / 2
        r = np.hypot(dx, dy)
        theta = r / f
        if theta.max() >= math.pi / 2 * 0.98:
            raise ValueError(f"f-theta camera reaches {math.degrees(theta.max()):.0f} deg off axis")
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(r > 0, np.tan(theta) / r, 1.0 / f)  # pinhole tan / pixel radius
        tx, ty = dx * k, dy * k  # pinhole image-plane coordinates (focal length 1)
        fp = f * scale
        half_w = math.ceil(fp * np.abs(tx).max()) + 2
        half_h = math.ceil(fp * np.abs(ty).max()) + 2
        self.pinhole_width, self.pinhole_height = 2 * half_w, 2 * half_h
        self.pinhole_fov_deg = math.degrees(2 * math.atan(half_w / fp))
        # Bilinear taps into the pinhole image (pixel centres at integers).
        mx, my = half_w + fp * tx, half_h + fp * ty
        self._map_x, self._map_y = mx.astype(np.float32), my.astype(np.float32)
        x0, y0 = np.floor(mx).astype(np.int64), np.floor(my).astype(np.int64)
        self._x0 = np.clip(x0, 0, self.pinhole_width - 2)
        self._y0 = np.clip(y0, 0, self.pinhole_height - 2)
        self._wx = (mx - self._x0)[..., None].astype(np.float32)
        self._wy = (my - self._y0)[..., None].astype(np.float32)

    def __call__(self, pinhole_rgb: np.ndarray) -> np.ndarray:
        if pinhole_rgb.shape[:2] != (self.pinhole_height, self.pinhole_width):
            raise ValueError(f"expected a {self.pinhole_width}x{self.pinhole_height} render")
        try:  # ~1 ms; the numpy path below is ~20 ms at 640x352
            import cv2

            return cv2.remap(pinhole_rgb, self._map_x, self._map_y, cv2.INTER_LINEAR)
        except ImportError:
            pass
        img = pinhole_rgb.astype(np.float32)
        x0, y0, wx, wy = self._x0, self._y0, self._wx, self._wy
        top = img[y0, x0] * (1 - wx) + img[y0, x0 + 1] * wx
        bottom = img[y0 + 1, x0] * (1 - wx) + img[y0 + 1, x0 + 1] * wx
        return np.clip(top * (1 - wy) + bottom * wy + 0.5, 0, 255).astype(np.uint8)


class CarlaCameraSource:
    """Spawns a ``sensor.camera.rgb`` on ``vehicle`` and exposes frames via
    :meth:`get`. Designed for synchronous mode: call ``world.tick()`` and then
    ``get()`` once per tick.

    With ``ftheta=True`` the frames are in the archives' f-theta camera model
    (:class:`FThetaRemap`) instead of CARLA's pinhole one, so they line up with
    the HD-map condition and the generated video."""

    def __init__(
        self,
        world: Any,
        vehicle: Any,
        width: int,
        height: int,
        fov_deg: float = 90.0,
        transform: Any | None = None,
        sensor_tick: float = 0.0,
        ftheta: bool = False,
        max_queue: int = 4,
    ):
        import carla

        self._remap = FThetaRemap(width, height, fov_deg) if ftheta else None
        if self._remap is not None:
            width, height = self._remap.pinhole_width, self._remap.pinhole_height
            fov_deg = self._remap.pinhole_fov_deg
            log.info("f-theta camera: pinhole render %dx%d at %.1f deg", width, height, fov_deg)
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov_deg))
        bp.set_attribute("sensor_tick", str(sensor_tick))
        transform = transform or carla.Transform(carla.Location(x=1.5, z=1.6))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.vehicle = vehicle
        self.world = world
        # The ego pose of a frame is the vehicle's transform in the world
        # snapshot of the tick the image was captured on. Neither
        # vehicle.get_transform() in the sensor callback nor image.transform
        # is: with another client moving the vehicle between ticks (a replay),
        # both read the next tick's pose about half the time.
        self._poses: dict[int, Any] = {}
        self._poses_cv = threading.Condition()
        self._tick_id = world.on_tick(self._on_tick)
        # max_queue=0: keep every frame (a follower must not skip ticks, or
        # its frames stop being consecutive); otherwise the oldest is dropped.
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.sensor.listen(self._on_image)

    def _on_tick(self, snapshot: Any) -> None:
        actor = snapshot.find(self.vehicle.id)
        if actor is None:
            return
        with self._poses_cv:
            self._poses[snapshot.frame] = actor.get_transform()
            if len(self._poses) > 2000:
                for key in sorted(self._poses)[:1000]:
                    del self._poses[key]
            self._poses_cv.notify_all()

    def _on_image(self, image: Any) -> None:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        rgb = bgra_to_rgb(arr)
        if self._remap is not None:
            rgb = self._remap(rgb)
        # Pose is filled in by get(), from the tick's snapshot.
        frame = Frame(
            rgb=rgb,
            timestamp_us=int(image.timestamp * 1e6),
            meta={"carla_frame": image.frame},
        )
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self._q.get_nowait()
            self._q.put_nowait(frame)

    def get(self, timeout: float = 2.0) -> Frame:
        frame = self._q.get(timeout=timeout)
        frame_id = frame.meta["carla_frame"]
        with self._poses_cv:
            self._poses_cv.wait_for(lambda: frame_id in self._poses, timeout=timeout)
            vehicle_tf = self._poses.get(frame_id)
        if vehicle_tf is None:
            log.warning("no world snapshot for frame %d; using the current vehicle pose", frame_id)
            vehicle_tf = self.vehicle.get_transform()
        frame.pose = pose_from_transform(vehicle_tf)
        frame.meta["vehicle_transform"] = vehicle_tf
        return frame

    def destroy(self) -> None:
        self.world.remove_on_tick(self._tick_id)
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

"""CARLA poses → OmniDreams rig poses, and the model's camera → CARLA's.

numpy only (no carla_canvas, no gRPC), unit-tested in tests/test_carla_canvas.py.

* carla_canvas hands the renderer CARLA poses: Unreal's left-handed frame (x
  forward, y right, z up), a 4x4 whose rotation columns are the body's
  forward, right and up axes. OmniDreams takes rig-to-world poses in a
  right-handed x forward / y left / z up frame. A scene archive made by
  ``carla-transfer prepare-scene`` is in the map's OpenDRIVE frame, which is
  CARLA's with y negated (as :func:`carla_transfer.carla.source.pose_from_transform`).
* The model renders the archive's camera: f-theta (``r = f * theta``,
  ``f = (width / 2) / (fov / 2)``), mounted on the rig at the clipgt
  calibration. :class:`FThetaToPinhole` resamples it into CARLA's pinhole
  camera (horizontal fov, square pixels, principal point at the centre).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_FLIP_Y = np.diag([1.0, -1.0, 1.0])


def flu_from_carla(carla_pose: Array) -> Array:
    """4x4 body-to-world in the OpenDRIVE frame (FLU body, right-handed) of a
    4x4 CARLA pose (Unreal: columns forward, right, up)."""
    m = np.asarray(carla_pose, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = _FLIP_Y @ m[:3, :3] @ _FLIP_Y
    out[:3, 3] = _FLIP_Y @ m[:3, 3]
    return out


def rotation_rpy(roll_pitch_yaw_deg) -> Array:
    """3x3 FLU rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (degrees)."""
    roll, pitch, yaw = (math.radians(float(a)) for a in roll_pitch_yaw_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class ArchiveCamera:
    """The scene archive's camera (clipgt calibration, ``hdmap.clipgt.CameraSpec``):
    its mount on the rig (FLU, metres / degrees) and its f-theta fov."""

    position: tuple[float, float, float] = (1.7, 0.0, 1.45)
    roll_pitch_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_deg: float = 120.0

    @staticmethod
    def from_options(spec: dict | None) -> ArchiveCamera:
        spec = dict(spec or {})
        base = ArchiveCamera()
        return ArchiveCamera(
            tuple(float(v) for v in spec.get("position", base.position)),
            tuple(float(v) for v in spec.get("roll_pitch_yaw", base.roll_pitch_yaw)),
            float(spec.get("fov", base.fov_deg)),
        )

    def camera_to_rig(self) -> Array:
        m = np.eye(4)
        m[:3, :3] = rotation_rpy(self.roll_pitch_yaw)
        m[:3, 3] = self.position
        return m

    def rig_from_camera(self, camera_flu: Array) -> Array:
        """The rig pose that puts this camera at ``camera_flu`` (FLU, x along
        the optical axis): the model then renders from the CARLA camera's pose."""
        return np.asarray(camera_flu, dtype=np.float64) @ np.linalg.inv(self.camera_to_rig())


def quat_xyzw(rotation: Array) -> tuple[float, float, float, float]:
    """``(x, y, z, w)`` of a proper rotation matrix, w >= 0."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x, y, z = (
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w, x, y, z = (
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        )
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w, x, y, z = (
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        )
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w, x, y, z = (
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            (0.25 * s),
        )
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return float(x), float(y), float(z), float(w)


class FThetaToPinhole:
    """Resample the model's f-theta image (``src_width`` x ``src_height``,
    ``src_fov_deg`` across the width) into a CARLA pinhole camera (``width`` x
    ``height``, horizontal ``fov_deg``) looking along the same axis. Pixels
    outside the f-theta image are black. The sampling map is built once."""

    def __init__(
        self,
        src_width: int,
        src_height: int,
        src_fov_deg: float,
        width: int,
        height: int,
        fov_deg: float,
    ):
        self.src_shape = (int(src_height), int(src_width))
        self.shape = (int(height), int(width))
        f_src = (src_width / 2.0) / math.radians(src_fov_deg / 2.0)
        f_dst = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        u = np.arange(width, dtype=np.float64) + 0.5 - width / 2.0
        v = np.arange(height, dtype=np.float64) + 0.5 - height / 2.0
        dx, dy = np.meshgrid(u, v)
        r = np.hypot(dx, dy)
        theta = np.arctan(r / f_dst)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(r > 0, f_src * theta / r, f_src / f_dst)
        # Source coordinates with pixel centres at integers.
        mx = src_width / 2.0 + dx * k - 0.5
        my = src_height / 2.0 + dy * k - 0.5
        self._map_x, self._map_y = mx.astype(np.float32), my.astype(np.float32)
        self._inside = (
            (mx >= -0.5) & (mx <= src_width - 0.5) & (my >= -0.5) & (my <= src_height - 0.5)
        )
        x0 = np.clip(np.floor(mx).astype(np.int64), 0, src_width - 2)
        y0 = np.clip(np.floor(my).astype(np.int64), 0, src_height - 2)
        self._x0, self._y0 = x0, y0
        self._wx = np.clip(mx - x0, 0.0, 1.0)[..., None].astype(np.float32)
        self._wy = np.clip(my - y0, 0.0, 1.0)[..., None].astype(np.float32)

    @property
    def coverage(self) -> float:
        """Fraction of the pinhole image the f-theta image covers."""
        return float(self._inside.mean())

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[:2] != self.src_shape:
            raise ValueError(f"expected a {self.src_shape[1]}x{self.src_shape[0]} image")
        img = rgb.astype(np.float32)
        x0, y0, wx, wy = self._x0, self._y0, self._wx, self._wy
        top = img[y0, x0] * (1 - wx) + img[y0, x0 + 1] * wx
        bottom = img[y0 + 1, x0] * (1 - wx) + img[y0 + 1, x0 + 1] * wx
        out = np.clip(top * (1 - wy) + bottom * wy + 0.5, 0, 255).astype(np.uint8)
        out[~self._inside] = 0
        return out


def resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize (PIL) of an RGB uint8 image."""
    if rgb.shape[:2] == (height, width):
        return rgb
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR))


def delivery_delay(first_chunk: int, chunk: int, pipelined_chunks: int = 0) -> int:
    """Frames between a camera tick and the generated frame it delivers.

    The model needs a whole chunk of poses before it renders any of them, so
    tick ``k`` can only deliver frame ``k - D``. With chunks of ``first_chunk``
    then ``chunk`` frames, frame ``k - (chunk - 1)`` belongs to a chunk that
    has ended by tick ``k`` (a smaller first chunk only ends earlier); each
    pipelined chunk lets generation run ``chunk`` ticks behind instead of
    blocking the tick that completes it."""
    if first_chunk < 1 or chunk < 1 or pipelined_chunks < 0:
        raise ValueError("chunk sizes must be positive, pipelined_chunks >= 0")
    return max(first_chunk, chunk) - 1 + chunk * int(pipelined_chunks)

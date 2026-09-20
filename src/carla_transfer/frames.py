"""Frame container plus the image codecs the backends need."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class Pose:
    """Ego pose in the world frame: translation (m) + unit quaternion (x, y, z, w)."""

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class Frame:
    """One RGB frame with its simulation timestamp and (optional) ego pose."""

    rgb: np.ndarray  # HxWx3 uint8
    timestamp_us: int
    pose: Pose | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be HxWx3 uint8, got {self.rgb.shape} {self.rgb.dtype}")

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    def resized(self, width: int, height: int) -> Frame:
        if (self.width, self.height) == (width, height):
            return self
        img = Image.fromarray(self.rgb).resize((width, height), Image.BILINEAR)
        return Frame(np.asarray(img), self.timestamp_us, self.pose, dict(self.meta))


def encode_image(rgb: np.ndarray, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def bgra_to_rgb(bgra: np.ndarray) -> np.ndarray:
    """CARLA camera sensors deliver BGRA; the models want RGB."""
    return np.ascontiguousarray(bgra[:, :, [2, 1, 0]])

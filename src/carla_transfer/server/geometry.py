"""Pure-numpy geometry shared by the server (and unit-tested without omnidreams)."""

from __future__ import annotations

import math

import numpy as np


def quat_xyzw_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def rig_to_world(xyz: tuple[float, float, float], quat_xyzw) -> np.ndarray:
    """4x4 rig-to-world (FLU: x forward, y left, z up) from translation + quaternion."""
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = quat_xyzw_to_matrix(tuple(quat_xyzw))
    m[:3, 3] = xyz
    return m


def yaw_pitch_roll_from_matrix(r: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (yaw about z, pitch about y, roll about x) from a rotation."""
    yaw = math.atan2(r[1, 0], r[0, 0])
    pitch = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
    roll = math.atan2(r[2, 1], r[2, 2])
    return yaw, pitch, roll


def speed_from_poses(poses: np.ndarray, timestamps_us: np.ndarray) -> float:
    """Mean ground speed (m/s) over a pose sequence; 0 for a single pose."""
    if len(poses) < 2:
        return 0.0
    d = np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3])
    dt = (int(timestamps_us[-1]) - int(timestamps_us[0])) / 1e6
    return float(d / dt) if dt > 0 else 0.0

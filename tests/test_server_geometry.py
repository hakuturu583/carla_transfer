import math

import numpy as np

from carla_transfer.server.geometry import (
    rig_to_world,
    speed_from_poses,
    yaw_pitch_roll_from_matrix,
)


def test_rig_to_world_identity():
    m = rig_to_world((1, 2, 3), (0, 0, 0, 1))
    assert m.shape == (4, 4) and m.dtype == np.float32
    np.testing.assert_allclose(m[:3, :3], np.eye(3))
    np.testing.assert_allclose(m[:3, 3], [1, 2, 3])


def test_yaw_roundtrip():
    yaw = math.radians(37)
    q = (0, 0, math.sin(yaw / 2), math.cos(yaw / 2))
    m = rig_to_world((0, 0, 0), q)
    y, p, r = yaw_pitch_roll_from_matrix(m[:3, :3])
    assert math.isclose(y, yaw, abs_tol=1e-5)
    assert abs(p) < 1e-5 and abs(r) < 1e-5
    # x axis of the rig rotated by yaw
    np.testing.assert_allclose(m[:3, 0], [math.cos(yaw), math.sin(yaw), 0], atol=1e-6)


def test_speed_from_poses():
    poses = np.stack([rig_to_world((i * 0.5, 0, 0), (0, 0, 0, 1)) for i in range(8)])
    ts = np.arange(8) * 33_333
    assert math.isclose(speed_from_poses(poses, ts), 3.5 / (7 * 0.033333), rel_tol=1e-3)
    assert speed_from_poses(poses[:1], ts[:1]) == 0.0


def test_server_module_imports_without_omnidreams():
    # The heavy omnidreams imports are inside methods, so the module (and the
    # console script) can be imported on the host.
    from carla_transfer.server import omnidreams_server

    p = omnidreams_server.build_parser()
    a = p.parse_args(["--resolution", "512x288", "--no-offload-text-encoder"])
    assert a.resolution == "512x288" and a.offload is False

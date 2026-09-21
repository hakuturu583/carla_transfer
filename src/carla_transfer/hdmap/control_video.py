"""Render the HD-map control video for a scene archive.

Uses a raster-only OmniDreams session (``DebugOptions.skip_video_generation``)
so only the ludus rasterizer runs: the server can be started with
``--raster-only`` and no world model. The ego trajectory is the archive's own
``rig_trajectories.json`` (the roadgen drive along the route), resampled to
the Cosmos3 world-scenario transfer geometry (101 frames @ 10 fps by default).
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path

import grpc
import numpy as np

from carla_transfer.config import TransferConfig
from carla_transfer.frames import decode_image
from carla_transfer.protos.omnidreams import camera_pb2, common_pb2, video_model_pb2
from carla_transfer.protos.omnidreams import video_model_pb2_grpc as vm_grpc

log = logging.getLogger(__name__)

# Cosmos3 transfer, "World scenario (WSM)" row of the diffusers table.
WSM_FRAMES = 101
WSM_FPS = 10


def archive_trajectory(archive: Path) -> tuple[np.ndarray, np.ndarray]:
    """(poses [N,4,4], timestamps_us [N]) from the archive's rig_trajectories.json."""
    with zipfile.ZipFile(archive) as zf:
        doc = json.loads(zf.read("rig_trajectories.json"))
    traj = doc["rig_trajectories"][0]
    return (
        np.asarray(traj["T_rig_worlds"], dtype=np.float32),
        np.asarray(traj["T_rig_world_timestamps_us"], dtype=np.int64),
    )


def resample_trajectory(
    poses: np.ndarray, ts: np.ndarray, fps: float, num_frames: int, start_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Pick ``num_frames`` poses at ``fps`` from a denser trajectory (nearest sample)."""
    t0 = ts[start_index]
    wanted = t0 + np.arange(num_frames) * int(round(1_000_000 / fps))
    idx = np.searchsorted(ts, wanted).clip(0, len(ts) - 1)
    return poses[idx], ts[idx]


def _pose_pb(m: np.ndarray, t_us: int) -> common_pb2.PoseAtTime:
    from scipy.spatial.transform import Rotation  # optional dep; only here

    q = Rotation.from_matrix(m[:3, :3]).as_quat()  # x, y, z, w
    return common_pb2.PoseAtTime(
        pose=common_pb2.Pose(
            vec=common_pb2.Vec3(x=float(m[0, 3]), y=float(m[1, 3]), z=float(m[2, 3])),
            quat=common_pb2.Quat(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
        ),
        timestamp_us=int(t_us),
    )


def render_control_frames(
    cfg: TransferConfig,
    archive: Path,
    poses: np.ndarray,
    ts: np.ndarray,
    chunk: int = 16,
) -> list[np.ndarray]:
    """HD-map condition frames for ``poses`` via a raster-only session."""
    od = cfg.omnidreams
    mb = od.max_message_mb * 1024 * 1024
    channel = grpc.insecure_channel(
        f"{od.host}:{od.port}",
        options=[("grpc.max_send_message_length", mb), ("grpc.max_receive_message_length", mb)],
    )
    stub = vm_grpc.WorldModelServiceStub(channel)
    if od.scene_transfer == "inline":
        blob = archive.read_bytes()
    else:
        blob = b"scene://" + archive.name.encode()
    w, h = cfg.resolution
    sid = stub.start_session(
        video_model_pb2.SessionRequest(
            static_world_map=video_model_pb2.StaticWorldMap(hdmap_parquets=blob),
            camera_specs=[
                camera_pb2.CameraSpec(logical_id=od.camera_name, resolution_w=w, resolution_h=h)
            ],
            debug_options=video_model_pb2.DebugOptions(
                skip_video_generation=True, return_hdmap_frames=True
            ),
        )
    )
    frames: list[np.ndarray] = []
    try:
        for i in range(0, len(poses), chunk):
            traj = common_pb2.Trajectory(
                poses=[
                    _pose_pb(m, t)
                    for m, t in zip(poses[i : i + chunk], ts[i : i + chunk], strict=True)
                ]
            )
            ret = stub.render_video_chunk(
                video_model_pb2.VideoChunkRequest(session_id=sid, rig_trajectory=traj)
            )
            for cam in ret.camera_outputs:
                frames += [decode_image(img.data) for img in cam.hdmap_condition_frames]
    finally:
        stub.close_session(video_model_pb2.SessionCloseRequest(session_id=sid.session_id))
        channel.close()
    log.info("rendered %d control frames at %dx%d", len(frames), w, h)
    return frames


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack(frames)
    try:  # PyAV directly: imageio's pyav plugin trips over the stream size
        import av

        with av.open(str(path), mode="w") as out:
            stream = out.add_stream("libx264", rate=int(round(fps)))
            stream.width, stream.height = int(stack.shape[2]), int(stack.shape[1])
            stream.pix_fmt = "yuv420p"
            for frame in stack:
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
    except ImportError:  # imageio-ffmpeg backend
        iio.imwrite(path, stack, plugin="FFMPEG", fps=fps, codec="libx264", macro_block_size=1)
    return path


def render_control_video(
    cfg: TransferConfig,
    archive: Path,
    out_mp4: Path,
    fps: float = WSM_FPS,
    num_frames: int = WSM_FRAMES,
    start_index: int = 0,
) -> Path:
    poses, ts = archive_trajectory(archive)
    poses, ts = resample_trajectory(poses, ts, fps, num_frames, start_index)
    frames = render_control_frames(cfg, archive, poses, ts)
    return write_video(frames, out_mp4, fps)

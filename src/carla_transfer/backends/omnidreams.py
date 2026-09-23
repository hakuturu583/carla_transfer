"""gRPC client for the OmniDreams ``WorldModelService`` (flashdreams).

The server is ``python -m omnidreams.grpc.server`` inside the container
(see docker/Dockerfile). Wire format is the upstream proto, vendored under
``carla_transfer.protos.omnidreams``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import grpc

from carla_transfer.backends.base import WorldModelBackend
from carla_transfer.config import TransferConfig
from carla_transfer.frames import Frame, Pose, decode_image, encode_image
from carla_transfer.protos.omnidreams import camera_pb2, common_pb2, video_model_pb2
from carla_transfer.protos.omnidreams import video_model_pb2_grpc as vm_grpc

log = logging.getLogger(__name__)

CAMERA_ID = "front"  # server's output camera id; input camera is od.camera_name


class OmniDreamsBackend(WorldModelBackend):
    def __init__(self, config: TransferConfig):
        self.config = config
        self.od = config.omnidreams
        self.frames_per_chunk = self.od.frames_per_chunk
        mb = self.od.max_message_mb * 1024 * 1024
        self._channel = grpc.insecure_channel(
            f"{self.od.host}:{self.od.port}",
            options=[
                ("grpc.max_send_message_length", mb),
                ("grpc.max_receive_message_length", mb),
            ],
        )
        self._stub = vm_grpc.WorldModelServiceStub(self._channel)
        self.first_chunk_frames = self.od.first_chunk_frames
        self.accepts_partial_chunk = False  # the server enforces 5-then-8
        self._session_id: str | None = None
        # clipgt scene archive. OmniDreams is HD-map conditioned: the server
        # rasterizes this archive's lanes/objects from the ego poses we send.
        # Either the archive bytes or a scene://<file> reference into the
        # container's /scenes mount. TODO: export the CARLA map to a clipgt
        # archive instead of reusing an nvidia/omni-dreams-scenes one.
        self.scene_archive: bytes = b""
        if self.od.scene_archive:
            if self.od.scene_transfer == "inline":
                self.scene_archive = self.od.scene_archive.read_bytes()
            else:
                self.scene_archive = b"scene://" + self.od.scene_archive.name.encode()
        # The prompt the session runs under, and the one chunks build on.
        self.base_prompt = self.od.positive_prompt or _archive_prompt(self.od.scene_archive)
        self._sent_prompt = self.base_prompt
        # When a list: every frame sent to the model, as {t_us, xyz, q, prompt}
        # (run --dump-trajectory), to check two runs saw the same drive.
        self.sent: list[dict] | None = None

    # -- lifecycle -----------------------------------------------------------

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                v = self._stub.get_version(common_pb2.Empty(), timeout=5)
                log.info("OmniDreams server ready: %s", v)
                return
            except grpc.RpcError as e:
                if time.monotonic() > deadline:
                    raise TimeoutError("OmniDreams server did not come up") from e
                time.sleep(2)

    def open(self, initial_frame: Frame) -> None:
        if not self.scene_archive:
            raise RuntimeError("omnidreams.scene_archive is not set (clipgt .usdz required)")
        w, h = self.config.resolution
        frame = initial_frame.resized(w, h)
        req = video_model_pb2.SessionRequest(
            static_world_map=video_model_pb2.StaticWorldMap(hdmap_parquets=self.scene_archive),
            text_prompt=video_model_pb2.TextPrompt(
                positive=self.base_prompt, negative=self.od.negative_prompt
            ),
            camera_specs=[self._camera_spec(w, h, self.od.camera_name)],
            # With initial_image="archive" no frame is sent and the server
            # keeps the archive's first_image.png (the Cosmos3 render).
            initial_frames=(
                [
                    video_model_pb2.Image(
                        data=encode_image(frame.rgb, "PNG"), format=video_model_pb2.PNG
                    )
                ]
                if self.od.initial_image == "carla"
                else []
            ),
            rig_to_camera=[_pose_pb(Pose())],
            random_seed=self.od.random_seed,
            debug_options=video_model_pb2.DebugOptions(return_hdmap_frames=self.od.return_hdmap),
        )
        self._session_id = self._stub.start_session(req).session_id
        log.info("OmniDreams session %s opened at %dx%d", self._session_id, w, h)

    def generate(self, sim_frames: Sequence[Frame]) -> list[Frame]:
        if self._session_id is None:
            raise RuntimeError("generate() before open()")
        traj = common_pb2.Trajectory(
            poses=[
                common_pb2.PoseAtTime(pose=_pose_pb(f.pose or Pose()), timestamp_us=f.timestamp_us)
                for f in sim_frames
            ]
        )
        req = video_model_pb2.VideoChunkRequest(
            session_id=video_model_pb2.SessionId(session_id=self._session_id),
            rig_trajectory=traj,
            # TODO(details pending): populate from CARLA actors (vehicles,
            # walkers) so the model renders dynamic agents.
            dynamic_state=video_model_pb2.DynamicWorldState(),
        )
        prompt = self.chunk_prompt(sim_frames)
        if prompt != self._sent_prompt:
            req.text_prompt.positive = prompt
            self._sent_prompt = prompt
            log.info("prompt for chunk at t=%.2fs: %r", sim_frames[-1].timestamp_us / 1e6,
                     prompt[len(self.base_prompt):].strip() or "(base)")  # fmt: skip
        if self.sent is not None:
            self.sent.extend(
                {"t_us": f.timestamp_us, "xyz": list((f.pose or Pose()).xyz),
                 "q": list((f.pose or Pose()).quat_xyzw), "prompt": self._sent_prompt,
                 "carla_frame": f.meta.get("carla_frame")}
                for f in sim_frames
            )  # fmt: skip
        ret = self._stub.render_video_chunk(req)
        out: list[Frame] = []
        for cam in ret.camera_outputs:
            if cam.camera_logical_id not in (CAMERA_ID, ""):
                continue
            hdmaps = [decode_image(h.data) for h in cam.hdmap_condition_frames]
            for i, img in enumerate(cam.rgb_frames):
                ts = sim_frames[min(i, len(sim_frames) - 1)].timestamp_us
                meta = {"generated": True}
                if i < len(hdmaps):
                    meta["hdmap"] = hdmaps[i]
                out.append(Frame(decode_image(img.data), ts, None, meta))
        return out

    def chunk_prompt(self, sim_frames: Sequence[Frame]) -> str:
        """The prompt for a chunk: the base prompt plus, with
        ``sync_traffic_lights``, the sentence for the state of the light ahead
        at the chunk's last frame (none when no light is in range)."""
        if not self.od.sync_traffic_lights or not sim_frames:
            return self.base_prompt
        state = sim_frames[-1].meta.get("traffic_light")
        sentence = self.od.traffic_light_prompts.get(state or "", "")
        return f"{self.base_prompt.rstrip()} {sentence}".strip() if sentence else self.base_prompt

    def close(self) -> None:
        if self._session_id is not None:
            try:
                self._stub.close_session(
                    video_model_pb2.SessionCloseRequest(session_id=self._session_id), timeout=30
                )
            except grpc.RpcError as e:
                log.warning("close_session failed: %s", e)
            self._session_id = None
        self._channel.close()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _camera_spec(
        w: int, h: int, logical_id: str, fov_deg: float = 90.0
    ) -> camera_pb2.CameraSpec:
        import math

        f = (w / 2) / math.tan(math.radians(fov_deg) / 2)
        return camera_pb2.CameraSpec(
            opencv_pinhole_param=camera_pb2.OpenCVPinholeCameraParam(
                principal_point_x=w / 2,
                principal_point_y=h / 2,
                focal_length_x=f,
                focal_length_y=f,
            ),
            logical_id=logical_id,
            resolution_w=w,
            resolution_h=h,
            shutter_type=camera_pb2.GLOBAL,
        )


def _pose_pb(p: Pose) -> common_pb2.Pose:
    x, y, z = p.xyz
    qx, qy, qz, qw = p.quat_xyzw
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=x, y=y, z=z), quat=common_pb2.Quat(x=qx, y=qy, z=qz, w=qw)
    )


def _archive_prompt(archive) -> str:
    """The archive's own default prompt (prompt1.txt), which the server uses
    when the session sends none."""
    import zipfile

    if not archive:
        return ""
    try:
        with zipfile.ZipFile(archive) as zf:
            return zf.read("prompt1.txt").decode("utf-8").strip()
    except (OSError, KeyError, zipfile.BadZipFile):
        return ""

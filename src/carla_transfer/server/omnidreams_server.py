"""``carla-transfer-server``: OmniDreams ``WorldModelService`` over the
interactive-drive world model.

Why not flashdreams' own ``omnidreams.grpc.server``? That server only knows
the 480p/720p/704p pipeline configs and keeps the ~15 GB text encoder
resident, so it needs ~48 GB. The interactive-drive path (fork branch
``claude/omnidreams-rtx3090-ahncny``) adds a manifest with a free
``resolution_wh`` plus ``--offload-text-encoder``, which is what fits a 24 GB
RTX 3090 (512x288 measured at 29 fps / 16 GB peak). This module exposes that
path behind the same gRPC proto so the host client is unchanged.

Protocol mapping:
  start_session      StaticWorldMap.hdmap_parquets = a clipgt scene archive
                     (.usdz zip: metadata.yaml, calibration, map parquets),
                     or the UTF-8 reference ``scene://<file>`` naming an
                     archive under --scene-dir (the nvidia/omni-dreams-scenes
                     archives are 2.3 GB, far beyond a gRPC message).
                     initial_frames[0] and text_prompt override the archive's
                     own initial image and prompt. camera_specs[0].logical_id
                     names the clipgt camera.
  render_video_chunk rig_trajectory poses are rig-to-world FLU. The first
                     call after start_session must carry 5 poses (the model's
                     initial chunk), every later call 8.
  close_session      resets the rollout and frees the scene.

  DebugOptions.skip_video_generation starts a *raster-only* session: chunks
  return the HD-map condition frames (in hdmap_condition_frames) without
  touching the world model, any chunk size is accepted, and the server can
  run without the model at all (--raster-only). This is how the control
  video for the Cosmos3 initial image is produced.

One session at a time: there is one GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import tempfile
import threading
import uuid
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
from PIL import Image

from carla_transfer.frames import decode_image, encode_image
from carla_transfer.protos.omnidreams import common_pb2, video_model_pb2
from carla_transfer.protos.omnidreams import video_model_pb2_grpc as vm_grpc
from carla_transfer.server.geometry import (
    rig_to_world,
    speed_from_poses,
    yaw_pitch_roll_from_matrix,
)

log = logging.getLogger("carla_transfer.server")

INITIAL_CHUNK_FRAMES = 5


@dataclasses.dataclass
class ServerOptions:
    manifest: str = "example_world_model_rtx3090.yaml"
    offload_text_encoder: bool = True
    resolution_wh: tuple[int, int] | None = None
    fps: int | None = None
    camera_name: str = "camera_front_wide_120fov"
    variant: str = "default"
    jpeg_quality: int = 90
    # Directory that ``scene://<file>`` references resolve against.
    scene_dir: Path = Path("/scenes")
    # Where inline archives are spooled.
    spool_dir: Path = dataclasses.field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "carla_transfer_scenes"
    )


SCENE_REF_PREFIX = b"scene://"


class _Session:
    def __init__(
        self,
        session_id: str,
        scene_path: Path,
        owns_file: bool,
        raster_only: bool = False,
        return_hdmap: bool = False,
    ):
        self.session_id = session_id
        self.scene_path = scene_path
        self.owns_file = owns_file
        self.raster_only = raster_only
        self.return_hdmap = return_hdmap
        self.chunks_rendered = 0


class OmniDreamsWorldModelService(vm_grpc.WorldModelServiceServicer):
    def __init__(self, opts: ServerOptions):
        self.opts = opts
        self._lock = threading.Lock()
        self._session: _Session | None = None
        self._backend = None
        self._rasterizer = None  # raster-only sessions
        self._raster = None
        self._chunk = None
        self._manifest = None

    # -- model ---------------------------------------------------------------

    def load_model(self) -> None:
        """Build the world model and run its warmup (compile / CUDA graphs)."""
        from omnidreams.interactive_drive.backends.world_model import WorldModelRenderBackend

        self._configure()
        manifest = self._manifest
        w, h = manifest.resolution_wh
        log.info(
            "loading world model manifest=%s res=%dx%d fps=%d chunk=%d offload_text_encoder=%s",
            self.opts.manifest,
            w,
            h,
            manifest.fps,
            manifest.num_frames_per_block,
            self.opts.offload_text_encoder,
        )
        self._backend = WorldModelRenderBackend(
            manifest=manifest,
            chunk=self._chunk,
            raster=self._raster,
            offload_text_encoder=self.opts.offload_text_encoder,
        )
        self._backend.warmup_model()
        log.info("world model ready")

    def _configure(self) -> None:
        from omnidreams.interactive_drive.cli import resolve_manifest_path
        from omnidreams.interactive_drive.config import ChunkConfig, RasterConfig
        from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest

        manifest = load_world_model_manifest(resolve_manifest_path(self.opts.manifest))
        if self.opts.resolution_wh is not None:
            manifest = dataclasses.replace(manifest, resolution_wh=self.opts.resolution_wh)
        if self.opts.fps is not None:
            manifest = dataclasses.replace(manifest, fps=self.opts.fps)
        w, h = manifest.resolution_wh
        self._manifest = manifest
        self._raster = RasterConfig(width=w, height=h)
        self._chunk = ChunkConfig(
            fps=manifest.fps,
            initial_chunk_frames=INITIAL_CHUNK_FRAMES,
            chunk_frames=manifest.num_frames_per_block,
        )

    def load_rasterizer(self) -> None:
        """HD-map rasterizer only (no world model): serves raster-only sessions."""
        from omnidreams.interactive_drive.rasterizer import LudusConditionRasterizer

        if self._raster is None:
            self._configure()
        self._rasterizer = LudusConditionRasterizer(self._raster)
        log.info("rasterizer ready (%dx%d)", *self._manifest.resolution_wh)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self._manifest.resolution_wh

    # -- rpc -----------------------------------------------------------------

    def get_version(self, request, context):
        return common_pb2.VersionId()

    def start_session(self, request, context):
        from omnidreams.interactive_drive.scene_loader import load_scene_bundle

        with self._lock:
            if self._session is not None:
                self._close_locked()
            if not request.static_world_map.hdmap_parquets:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "static_world_map.hdmap_parquets must hold a clipgt scene archive",
                )
            session_id = uuid.uuid4().hex[:12]
            scene_path, owns_file = self._resolve_scene(
                session_id, request.static_world_map.hdmap_parquets, context
            )

            camera = self.opts.camera_name
            if request.camera_specs and request.camera_specs[0].logical_id:
                camera = request.camera_specs[0].logical_id
            prompt = request.text_prompt.positive or None
            scene = load_scene_bundle(scene_path, camera, self.opts.variant, prompt, self._raster)
            if request.initial_frames:
                scene = dataclasses.replace(
                    scene, initial_rgb=self._fit(decode_image(request.initial_frames[0].data))
                )
            raster_only = bool(request.debug_options.skip_video_generation)
            return_hdmap = raster_only or bool(request.debug_options.return_hdmap_frames)
            log.info(
                "session %s: scene=%s camera=%s raster_only=%s prompt=%r",
                session_id,
                scene.scene_id,
                camera,
                raster_only,
                scene.prompt[:60],
            )
            if raster_only:
                if self._rasterizer is None:
                    self.load_rasterizer()
                self._rasterizer.load_scene(scene)
            else:
                if self._backend is None:
                    context.abort(
                        grpc.StatusCode.FAILED_PRECONDITION,
                        "server runs --raster-only; set debug_options.skip_video_generation",
                    )
                self._backend.load_scene(scene)
            self._session = _Session(session_id, scene_path, owns_file, raster_only, return_hdmap)
            return video_model_pb2.SessionId(session_id=session_id)

    def render_video_chunk(self, request, context):
        from omnidreams.interactive_drive.types import TrajectoryChunk, VehicleState

        with self._lock:
            s = self._session
            if s is None or request.session_id.session_id != s.session_id:
                context.abort(grpc.StatusCode.NOT_FOUND, "no such session")
            poses_pb = request.rig_trajectory.poses
            expected = INITIAL_CHUNK_FRAMES if s.chunks_rendered == 0 else self._chunk.chunk_frames
            if not s.raster_only and len(poses_pb) != expected:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"chunk {s.chunks_rendered} needs {expected} poses, got {len(poses_pb)}",
                )
            poses = np.stack(
                [
                    rig_to_world(
                        (p.pose.vec.x, p.pose.vec.y, p.pose.vec.z),
                        (p.pose.quat.x, p.pose.quat.y, p.pose.quat.z, p.pose.quat.w),
                    )
                    for p in poses_pb
                ]
            )
            ts = np.array([p.timestamp_us for p in poses_pb], dtype=np.int64)
            yaw, pitch, roll = yaw_pitch_roll_from_matrix(poses[-1, :3, :3])
            traj = TrajectoryChunk(
                timestamps_us=ts,
                rig_poses_world=poses,
                boundary_state_after_chunk=VehicleState(
                    x_m=float(poses[-1, 0, 3]),
                    y_m=float(poses[-1, 1, 3]),
                    z_m=float(poses[-1, 2, 3]),
                    yaw_rad=yaw,
                    speed_mps=speed_from_poses(poses, ts),
                    steer_rad=0.0,
                    pitch_rad=pitch,
                    roll_rad=roll,
                ),
            )
            hdmap_frames: list = []
            if s.raster_only:
                raster = self._rasterizer.render_chunk(poses, ts)
                rgb_frames: list = []
                hdmap_frames = [np.asarray(f.rgb_host_uint8) for f in raster.frames]
            else:
                if s.chunks_rendered == 0:
                    out = self._backend.render_first_chunk(traj)
                else:
                    out = self._backend.render_next_chunk(traj)
                # In the world-model path PresentedFrame.rgb_host_uint8 is the
                # HD-map *condition* frame; the generated image is
                # model_rgb_host_uint8 (see backends/world_model.py _merge).
                rgb_frames = [np.asarray(f.model_rgb_host_uint8) for f in out.frames]
                if s.return_hdmap:
                    hdmap_frames = [np.asarray(f.rgb_host_uint8) for f in out.frames]
            s.chunks_rendered += 1

            return video_model_pb2.VideoChunkReturn(
                camera_outputs=[
                    video_model_pb2.CameraOutput(
                        camera_logical_id="front",
                        rgb_frames=[self._jpeg(f) for f in rgb_frames],
                        hdmap_condition_frames=[self._png(f) for f in hdmap_frames],
                    )
                ]
            )

    def close_session(self, request, context):
        with self._lock:
            if self._session is not None and request.session_id == self._session.session_id:
                self._close_locked()
        return common_pb2.Empty()

    # -- helpers -------------------------------------------------------------

    def _jpeg(self, rgb: np.ndarray) -> video_model_pb2.Image:
        return video_model_pb2.Image(
            data=encode_image(rgb, "JPEG", self.opts.jpeg_quality), format=video_model_pb2.JPEG
        )

    @staticmethod
    def _png(rgb: np.ndarray) -> video_model_pb2.Image:
        # Condition frames are flat-colour line art: PNG is small and exact.
        return video_model_pb2.Image(data=encode_image(rgb, "PNG"), format=video_model_pb2.PNG)

    def _close_locked(self) -> None:
        assert self._session is not None
        log.info(
            "session %s closed after %d chunks",
            self._session.session_id,
            self._session.chunks_rendered,
        )
        try:
            if not self._session.raster_only and self._backend is not None:
                self._backend.reset()
        finally:
            if self._session.owns_file:
                try:
                    self._session.scene_path.unlink()
                except OSError:
                    pass
            self._session = None

    def _resolve_scene(self, session_id: str, blob: bytes, context) -> tuple[Path, bool]:
        """Return (archive path, whether we own the file)."""
        if blob.startswith(SCENE_REF_PREFIX):
            name = blob[len(SCENE_REF_PREFIX) :].decode("utf-8")
            # Lexical check only: the archive is typically a symlink (HF cache
            # snapshots point into blobs/), so resolve() would leave scene_dir.
            if not name or "/" in name or name in (".", ".."):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"bad scene reference {name!r}")
            path = self.opts.scene_dir / name
            if not path.is_file():
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"scene {name!r} not found under {self.opts.scene_dir}",
                )
            return path, False
        self.opts.spool_dir.mkdir(parents=True, exist_ok=True)
        path = self.opts.spool_dir / f"{session_id}.usdz"
        path.write_bytes(blob)
        return path, True

    def _fit(self, rgb: np.ndarray) -> np.ndarray:
        w, h = self.resolution_wh
        if rgb.shape[:2] != (h, w):
            rgb = np.asarray(Image.fromarray(rgb).resize((w, h), Image.BILINEAR))
        return rgb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument(
        "--manifest",
        default="example_world_model_rtx3090.yaml",
        help="bundled interactive-drive manifest name or a path",
    )
    p.add_argument("--offload-text-encoder", dest="offload", action="store_true", default=True)
    p.add_argument("--no-offload-text-encoder", dest="offload", action="store_false")
    p.add_argument("--resolution", metavar="WxH", help="override the manifest resolution_wh")
    p.add_argument("--fps", type=int)
    p.add_argument("--camera", default="camera_front_wide_120fov")
    p.add_argument("--variant", default="default")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("/scenes"),
        help="directory that scene://<file> references resolve against",
    )
    p.add_argument("--max-message-mb", type=int, default=256)
    p.add_argument(
        "--raster-only",
        action="store_true",
        help="serve HD-map condition frames only; do not load the world model",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    res = None
    if args.resolution:
        w, h = args.resolution.lower().split("x")
        res = (int(w), int(h))
    opts = ServerOptions(
        manifest=args.manifest,
        offload_text_encoder=args.offload,
        resolution_wh=res,
        fps=args.fps,
        camera_name=args.camera,
        variant=args.variant,
        jpeg_quality=args.jpeg_quality,
        scene_dir=args.scene_dir,
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    svc = OmniDreamsWorldModelService(opts)
    if args.raster_only:
        svc.load_rasterizer()
    else:
        svc.load_model()

    mb = args.max_message_mb * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[("grpc.max_send_message_length", mb), ("grpc.max_receive_message_length", mb)],
    )
    vm_grpc.add_WorldModelServiceServicer_to_server(svc, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    log.info("WorldModelService listening on %s:%d", args.host, args.port)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

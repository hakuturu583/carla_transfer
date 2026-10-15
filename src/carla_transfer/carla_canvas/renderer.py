"""carla_transfer as a carla_canvas renderer.

carla_canvas runs a typesafe_carla program unchanged and has its CARLA
cameras rendered by a renderer service; this module makes carla_transfer's
world model (OmniDreams, in its Docker container) that renderer. carla_canvas
keeps everything CARLA-specific (sensor attributes, measurement layouts, gRPC,
the ROS 2 topics); this renderer turns each tick of an RGB camera into a pose
for the model and hands back the model's frames:

* poses: the CARLA camera's world pose (``pose: camera``, default: the rig is
  placed so that the archive's camera sits where the CARLA camera is) or its
  parent vehicle's (``pose: vehicle``: the archive's camera mount), in the
  OpenDRIVE frame of the scene archive (``carla-transfer prepare-scene``);
* chunks: OmniDreams renders 5 poses, then 8 per call, so the frame delivered
  at a tick is the model's frame of an earlier tick (``frames.delivery_delay``:
  7 ticks; until then the archive's first image);
* images: the model's f-theta camera resampled into the CARLA camera's
  pinhole fov and size (``camera_model: pinhole``, default) or only resized
  (``camera_model: ftheta``).

Configuration (carla_canvas's file, ``CARLA_CANVAS_CONFIG``)::

    renderer:
      backend: carla_transfer.carla_canvas:CarlaTransferRenderer
      timeout: 120                     # a chunk (and the first one's warm-up) per call
      startup_timeout: 300             # LoadWorld opens the model session
      assets:
        maps:
          "Town10HD*": {scene: scenes/town10.usdz}   # prepare-scene's archive
      options:
        world_model: "omnidreams:50051"   # default: $CARLA_TRANSFER_WORLD_MODEL, 127.0.0.1:50051
        transfer_config: omnidreams_rtx3090.yaml  # optional carla_transfer YAML
        omnidreams: {positive_prompt: "..."}      # over its omnidreams section
        pose: camera                   # camera | vehicle
        camera_model: pinhole          # pinhole | ftheta
        pipelined_chunks: 0            # >0: generate behind the ticks (8 more ticks of delay each)
        world_model_wait: 30           # seconds LoadWorld waits for the model server
        archive_camera: {position: [1.7, 0.0, 1.45], roll_pitch_yaw: [0, 0, 0], fov: 120}

One camera per renderer: the model runs one session on one GPU.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
import zipfile
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from carla_canvas import CameraFrame, CameraView, Renderer, WorldInfo
except ImportError:
    raise ImportError(
        "carla_transfer.carla_canvas requires carla_canvas.\n"
        "Install with: pip install 'carla-transfer[carla_canvas]'"
    ) from None

from carla_transfer.backends import WorldModelBackend, make_backend
from carla_transfer.carla_canvas.frames import (
    ArchiveCamera,
    FThetaToPinhole,
    delivery_delay,
    flu_from_carla,
    quat_xyzw,
    resize,
)
from carla_transfer.config import TransferConfig
from carla_transfer.frames import Frame, Pose, decode_image

log = logging.getLogger("carla_transfer.carla_canvas")

ENV_WORLD_MODEL = "CARLA_TRANSFER_WORLD_MODEL"
DEFAULT_WORLD_MODEL = "127.0.0.1:50051"


def _version() -> str:
    try:
        return metadata.version("carla-transfer")
    except metadata.PackageNotFoundError:
        return "0"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class _Chunk:
    start: int  # session frame index of its first pose
    size: int
    future: Future


@dataclass
class _Session:
    """One model session: the camera it renders and its frames in flight."""

    sensor_id: int
    backend: WorldModelBackend
    pending: list[Frame] = field(default_factory=list)  # poses of the chunk being filled
    chunks: list[_Chunk] = field(default_factory=list)  # submitted, oldest first
    ticks: int = 0  # camera ticks rendered
    t0: float | None = None  # sim time of its first tick: timestamps start at 0
    last: tuple[int, np.ndarray] | None = None  # (carla frame, model image) delivered last


class CarlaTransferRenderer(Renderer):
    name = "carla_transfer"
    version = _version()
    camera_outputs = frozenset({"rgb"})

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        #: builds the world-model client (tests inject a fake)
        self.backend_factory: Callable[[TransferConfig], WorldModelBackend] = make_backend
        self.config: TransferConfig | None = None
        self._session: _Session | None = None
        self._placeholder: np.ndarray | None = None
        self._resamplers: dict[tuple, FThetaToPinhole] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    # --- world -------------------------------------------------------------------

    def load_world(self, world: WorldInfo) -> None:
        self._close_session()
        self.config = self._transfer_config(world)
        self.archive_camera = ArchiveCamera.from_options(self.options.get("archive_camera"))
        self.pose_source = str(self.options.get("pose", "camera"))
        if self.pose_source not in ("camera", "vehicle"):
            raise ValueError("options.pose must be camera or vehicle")
        self.camera_model = str(self.options.get("camera_model", "pinhole"))
        if self.camera_model not in ("pinhole", "ftheta"):
            raise ValueError("options.camera_model must be pinhole or ftheta")
        od = self.config.omnidreams
        self.delay = delivery_delay(
            od.first_chunk_frames, od.frames_per_chunk, int(self.options.get("pipelined_chunks", 0))
        )
        self.generate_timeout = float(self.options.get("generate_timeout", 600.0))
        self._placeholder = _archive_first_image(od.scene_archive)
        # The session is opened here, under LoadWorld's (longer) deadline: the
        # model loads the scene and embeds the prompt on start_session.
        self._session = self._open_session(sensor_id=0)
        log.info(
            "map %s: scene archive %s, world model %s:%d, delay %d frames",
            world.map_name,
            od.scene_archive,
            od.host,
            od.port,
            self.delay,
        )

    def _transfer_config(self, world: WorldInfo) -> TransferConfig:
        """carla_transfer's configuration: ``options.transfer_config`` (a
        carla_transfer YAML) with the options on top, the map's archive and
        the world model's endpoint."""
        o = self.options
        raw: dict[str, Any] = {}
        if o.get("transfer_config"):
            raw = yaml.safe_load(world.resolve(o["transfer_config"]).read_text()) or {}
        for key in ("backend", "width", "height", "fps"):
            if key in o:
                raw[key] = o[key]
        raw.setdefault("backend", "omnidreams")
        if raw["backend"] not in ("omnidreams", "mock"):
            raise ValueError("carla_canvas renders with the omnidreams (or mock) backend")
        od = _merge(raw.get("omnidreams") or {}, dict(o.get("omnidreams") or {}))
        endpoint = o.get("world_model") or os.environ.get(ENV_WORLD_MODEL)
        if endpoint or "host" not in od:
            host, _, port = str(endpoint or DEFAULT_WORLD_MODEL).rpartition(":")
            od["host"], od["port"] = host, int(port)
        scene = self._scene_archive(world)
        if scene is not None:
            od["scene_archive"] = str(scene)
        elif od.get("scene_archive"):
            od["scene_archive"] = str(world.resolve(od["scene_archive"]))
        elif raw["backend"] == "omnidreams":
            raise ValueError(f"no scene archive for map {world.map_name} (renderer.assets.maps)")
        # A CARLA archive is a few hundred KB: send it with the session, so
        # the model container needs no mount.
        od.setdefault("scene_transfer", "inline")
        # There is no CARLA image to start from (this renderer replaces it).
        od["initial_image"] = "archive"
        raw["omnidreams"] = od
        return TransferConfig.from_dict(raw)

    @staticmethod
    def _scene_archive(world: WorldInfo) -> Path | None:
        for pattern, entry in ((world.config.get("assets") or {}).get("maps") or {}).items():
            if fnmatch.fnmatchcase(world.map_basename, pattern) or fnmatch.fnmatchcase(
                world.map_name, pattern
            ):
                scene = entry["scene"] if isinstance(entry, dict) else entry
                return world.resolve(scene)
        return None

    # --- session -----------------------------------------------------------------

    def _open_session(self, sensor_id: int) -> _Session:
        assert self.config is not None
        backend = self.backend_factory(self.config)
        # The model container may still be loading (minutes on a cold start).
        backend.wait_ready(float(self.options.get("world_model_wait", 30.0)))
        w, h = self.config.resolution
        backend.open(Frame(np.zeros((h, w, 3), np.uint8), 0))
        if self._executor is None:
            self._executor = ThreadPoolExecutor(1, thread_name_prefix="carla-transfer-gen")
        return _Session(sensor_id, backend)

    def _close_session(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        for chunk in session.chunks:
            chunk.future.cancel()
        for chunk in session.chunks:
            if not chunk.future.cancelled():
                try:
                    chunk.future.result()
                except Exception:  # noqa: BLE001 - the session is going away
                    pass
        session.backend.close()

    def _session_for(self, sensor_id: int, frame: int) -> _Session:
        session = self._session
        if session is None:
            raise RuntimeError("no world loaded (load_world has not run)")
        if session.sensor_id == 0:
            session.sensor_id = sensor_id
        elif session.sensor_id != sensor_id:
            # A camera is rendered every tick while it exists: one that was
            # rendered this tick or the last is still there, and two cameras
            # would interleave their poses in one rollout.
            if session.last is not None and frame - session.last[0] <= 1:
                raise ValueError(
                    f"carla_transfer renders one camera (sensor {session.sensor_id} is "
                    f"rendered too): list only one in the configuration's sensors "
                    f"(e.g. by role_name)"
                )
            log.info("camera %d replaces %d: new model session", sensor_id, session.sensor_id)
            self._close_session()
            session = self._session = self._open_session(sensor_id)
        return session

    # --- cameras -----------------------------------------------------------------

    def rig_pose(self, view: CameraView) -> np.ndarray:
        """4x4 rig-to-world (OpenDRIVE frame, FLU) for the model."""
        scene = view.scene
        if self.pose_source == "vehicle":
            info = scene.actors.get(view.sensor_id)
            parent = info.parent_id if info is not None else 0
            if parent and parent in scene.states:
                return flu_from_carla(scene.transform(parent).matrix())
        return self.archive_camera.rig_from_camera(flu_from_carla(view.camera.pose.matrix()))

    def render_camera(self, view: CameraView) -> CameraFrame:
        with self._lock:
            session = self._session_for(view.sensor_id, view.scene.frame)
            frame_id = view.scene.frame
            if session.last is not None and session.last[0] == frame_id:
                image = session.last[1]  # the same tick again
            else:
                image = self._tick(session, view)
                session.last = (frame_id, image)
        return CameraFrame(rgb=self._to_camera(image, view))

    def _tick(self, session: _Session, view: CameraView) -> np.ndarray:
        assert self.config is not None
        if session.t0 is None:
            session.t0 = view.scene.elapsed_seconds
        t_us = int(round((view.scene.elapsed_seconds - session.t0) * 1e6))
        rig = self.rig_pose(view)
        pose = Pose(tuple(float(v) for v in rig[:3, 3]), quat_xyzw(rig[:3, :3]))
        session.pending.append(Frame(_NO_IMAGE, t_us, pose, {"carla_frame": view.scene.frame}))
        index = session.ticks
        session.ticks += 1
        backend = session.backend
        if len(session.pending) >= backend.chunk_size(len(session.chunks)):
            frames, session.pending = session.pending, []
            start = index + 1 - len(frames)
            future = self._executor.submit(self._generate, backend, frames)
            session.chunks.append(_Chunk(start, len(frames), future))
        target = index - self.delay
        if target < 0:
            return self._placeholder_image()
        while session.chunks and session.chunks[0].start + session.chunks[0].size <= target:
            session.chunks.pop(0)  # delivered
        chunk = session.chunks[0]
        frames = chunk.future.result(timeout=self.generate_timeout)
        if not frames:
            raise RuntimeError(f"the world model returned no frames for chunk at {chunk.start}")
        return frames[min(target - chunk.start, len(frames) - 1)]

    def _generate(self, backend: WorldModelBackend, frames: list[Frame]) -> list[np.ndarray]:
        w, h = self.config.resolution
        # The mock (bring-up) backend echoes its inputs: give it images.
        if self.config.backend.value == "mock":
            frames = [Frame(np.zeros((h, w, 3), np.uint8), f.timestamp_us, f.pose, f.meta)
                      for f in frames]  # fmt: skip
        return [f.rgb for f in backend.generate(frames)]

    def _placeholder_image(self) -> np.ndarray:
        if self._placeholder is None:
            w, h = self.config.resolution
            self._placeholder = np.zeros((h, w, 3), np.uint8)
        return self._placeholder

    def _to_camera(self, image: np.ndarray, view: CameraView) -> np.ndarray:
        """The model's (f-theta) image as the CARLA camera's."""
        cam = view.camera
        if self.camera_model == "ftheta":
            return resize(image, cam.width, cam.height)
        sh, sw = image.shape[:2]
        key = (sw, sh, cam.width, cam.height, float(cam.fov))
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = self._resamplers[key] = FThetaToPinhole(
                sw, sh, self.archive_camera.fov_deg, cam.width, cam.height, float(cam.fov)
            )
            if resampler.coverage < 0.999:
                log.warning(
                    "camera fov %.0f deg reaches beyond the model's %.0f deg f-theta image: "
                    "%.0f%% of it is black",
                    cam.fov,
                    self.archive_camera.fov_deg,
                    100 * (1 - resampler.coverage),
                )
        return resampler(image)

    def close(self) -> None:
        with self._lock:
            self._close_session()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None


#: generate() needs poses only; Frame wants an image
_NO_IMAGE = np.zeros((1, 1, 3), np.uint8)


def _archive_first_image(archive: Path | None) -> np.ndarray | None:
    """The archive's first_image.png: the model's starting image, shown until
    its first frames are due."""
    if archive is None:
        return None
    try:
        with zipfile.ZipFile(archive) as zf:
            return decode_image(zf.read("first_image.png"))
    except (OSError, KeyError, zipfile.BadZipFile):
        return None

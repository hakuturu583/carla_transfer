"""``carla-transfer prepare-scene``: CARLA .xodr → ready-to-drive scene archive.

    1. roadgen reads the OpenDRIVE and writes the ClipGT layers; they are
       packed into an OmniDreams archive (hdmap/clipgt.py).
    2. The OmniDreams image runs --raster-only and renders the HD-map control
       video along the archive's ego route (hdmap/control_video.py).
    3. A photoreal first frame is generated from the map and written into the
       archive as first_image.png:
         first_image="cosmos3" (default): Cosmos3-Nano transfer with the
           world-scenario hint (server/cosmos3_transfer.py, fp8 + offload on
           a 24 GB card). The control video comes from cosmos-transfer2.5's
           world-scenario renderer when that checkout is available
           (transfer25.repo), else from the ludus raster-only server.
         first_image="transfer25": Cosmos-Transfer2.5-2B auto-multiview,
           front view only — runs, but the 7-view model degrades badly with
           one active view (verified on NVIDIA's own example). Kept for
           multi-GPU setups.
         first_image="none": keep the placeholder.

Each GPU step runs on its own; ``--local`` runs the cosmos3 raster/transfer
steps in the current Python environment instead of Docker.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from carla_transfer.config import TransferConfig
from carla_transfer.hdmap.clipgt import (
    SceneSpec,
    export_clipgt_dir,
    pack_scene_archive,
)
from carla_transfer.hdmap.control_video import WSM_FPS, WSM_FRAMES, render_control_video
from carla_transfer.hdmap.xodr import read_carla_xodr

log = logging.getLogger(__name__)

RASTER_CONTAINER = "carla-transfer-raster"


def replace_in_archive(archive: Path, member: str, data: bytes) -> None:
    """Rewrite ``archive`` with ``member`` replaced (zip has no in-place update)."""
    tmp = archive.with_suffix(archive.suffix + ".tmp")
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename != member:
                dst.writestr(info, src.read(info.filename))
        dst.writestr(member, data)
    shutil.move(tmp, archive)


def _inline(cfg: TransferConfig) -> TransferConfig:
    """``cfg`` with archives sent in the session request: one raster server
    then serves any number of (small, roadgen-exported) archives without a
    bind mount per archive."""
    import dataclasses

    return dataclasses.replace(
        cfg, omnidreams=dataclasses.replace(cfg.omnidreams, scene_transfer="inline")
    )


def render_control_videos_docker(
    cfg: TransferConfig, archives: list[Path], out_mp4s: list[Path]
) -> list[Path]:
    """Control video per archive, all from one raster-only container."""
    from carla_transfer.docker.manager import ContainerManager, stop_container

    cfg = _inline(cfg)
    mgr = ContainerManager(cfg)
    stop_container(RASTER_CONTAINER)
    args = mgr.raster_server_args(None, RASTER_CONTAINER)
    log.info("starting raster server: %s", " ".join(args))
    subprocess.run(args, check=True)
    try:
        _wait_grpc(cfg, timeout_s=600)
        return [
            render_control_video(cfg, a, o, WSM_FPS, WSM_FRAMES)
            for a, o in zip(archives, out_mp4s, strict=True)
        ]
    finally:
        stop_container(RASTER_CONTAINER)


def render_control_videos_local(
    cfg: TransferConfig, archives: list[Path], out_mp4s: list[Path]
) -> list[Path]:
    """Same as the docker path, with the raster server as a subprocess of the
    current interpreter (needs omnidreams importable, e.g. the flashdreams venv)."""
    import sys

    cfg = _inline(cfg)
    od = cfg.omnidreams
    cmd = [
        sys.executable, "-m", "carla_transfer.server.omnidreams_server",
        "--raster-only", "--port", str(od.port), "--manifest", od.manifest,
        "--resolution", f"{cfg.width}x{cfg.height}", "--fps", str(cfg.fps),
        "--camera", od.camera_name,
    ]  # fmt: skip
    log.info("starting local raster server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        _wait_grpc(cfg, timeout_s=600)
        return [
            render_control_video(cfg, a, o, WSM_FPS, WSM_FRAMES)
            for a, o in zip(archives, out_mp4s, strict=True)
        ]
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def _wait_grpc(cfg: TransferConfig, timeout_s: float) -> None:
    import grpc

    from carla_transfer.protos.omnidreams import common_pb2
    from carla_transfer.protos.omnidreams import video_model_pb2_grpc as vm_grpc

    od = cfg.omnidreams
    deadline = time.monotonic() + timeout_s
    with grpc.insecure_channel(f"{od.host}:{od.port}") as ch:
        stub = vm_grpc.WorldModelServiceStub(ch)
        while True:
            try:
                stub.get_version(common_pb2.Empty(), timeout=5)
                return
            except grpc.RpcError as e:
                if time.monotonic() > deadline:
                    raise TimeoutError("raster server did not come up") from e
                time.sleep(2)


def run_cosmos3_docker(cfg: TransferConfig, work_dirs: list[Path], job_args: list[str]):
    """One Cosmos3 container (one pipeline load) for every work dir; each
    holds control_wsm.mp4 and gets cosmos3/first_image.png."""
    import os

    from carla_transfer.docker.manager import ContainerManager

    dirs = [d.resolve() for d in work_dirs]
    root = Path(os.path.commonpath(dirs)) if len(dirs) > 1 else dirs[0]
    rel = [Path("/work") / d.relative_to(root) for d in dirs]
    args = ContainerManager(cfg).cosmos3_job_args(
        root,
        ["--control", *[str(r / "control_wsm.mp4") for r in rel],
         "--out-dir", *[str(r / "cosmos3") for r in rel], *job_args],
    )  # fmt: skip
    log.info("running cosmos3 transfer: %s", " ".join(args))
    subprocess.run(args, check=True)
    return [d / "cosmos3" / "first_image.png" for d in work_dirs]


def run_cosmos3_local(work_dirs: list[Path], job_args: list[str]) -> list[Path]:
    from carla_transfer.server.cosmos3_transfer import build_parser, run

    run(build_parser().parse_args(
        ["--control", *[str(d / "control_wsm.mp4") for d in work_dirs],
         "--out-dir", *[str(d / "cosmos3") for d in work_dirs], *job_args]
    ))  # fmt: skip
    return [d / "cosmos3" / "first_image.png" for d in work_dirs]


def cosmos3_first_images(
    cfg: TransferConfig,
    archives: list[Path],
    work_dirs: list[Path],
    local: bool = False,
    cosmos3_args: list[str] | None = None,
) -> list[Path]:
    """Control video of each archive (map + its obstacles) → Cosmos3 transfer;
    returns the generated first_image.png files (not yet written to the
    archives). One raster server and one Cosmos3 load serve the whole batch."""
    job_args = list(cosmos3_args or [])
    for d in work_dirs:
        d.mkdir(parents=True, exist_ok=True)
    controls = [d / "control_wsm.mp4" for d in work_dirs]
    # The control video must come from the *same camera model OmniDreams
    # rasterizes with* (the archive calibration through ludus): the first
    # image inherits its framing, and a control rendered with another
    # camera (cosmos-transfer2.5's renderer) left every lane and car
    # offset from the HD map for the whole rollout.
    if local:
        render_control_videos_local(cfg, archives, controls)
        return run_cosmos3_local(work_dirs, job_args)
    render_control_videos_docker(cfg, archives, controls)
    return run_cosmos3_docker(cfg, work_dirs, job_args)


def prepare_scene(
    cfg: TransferConfig,
    xodr: Path,
    archive: Path,
    spec: SceneSpec,
    work_dir: Path | None = None,
    first_image: str = "transfer25",
    local: bool = False,
    cosmos3_args: list[str] | None = None,
) -> Path:
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="carla_transfer_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("work dir %s", work_dir)

    # 1. map → clipgt dir (kept: Transfer2.5's renderer reads it) → archive
    spec.camera.width, spec.camera.height = cfg.resolution
    clip_dir = work_dir / "clipgt"
    m = read_carla_xodr(xodr)
    clip_id = export_clipgt_dir(m, spec, clip_dir)
    pack_scene_archive(clip_dir, clip_id, spec, archive)
    log.info("archive %s (%d bytes)", archive, archive.stat().st_size)

    if first_image == "none":
        return archive
    if first_image == "transfer25":
        from carla_transfer.hdmap.transfer25 import first_image_from_clipgt

        first = first_image_from_clipgt(cfg.transfer25, clip_dir, clip_id, work_dir)
    elif first_image == "cosmos3":
        (first,) = cosmos3_first_images(cfg, [archive], [work_dir], local, cosmos3_args)
    else:
        raise ValueError(f"unknown first_image {first_image!r}")
    replace_in_archive(archive, "first_image.png", first.read_bytes())
    log.info("archive %s now carries the %s first image", archive, first_image)
    return archive

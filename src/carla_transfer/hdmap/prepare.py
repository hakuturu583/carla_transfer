"""``carla-transfer prepare-scene``: CARLA .xodr → ready-to-drive scene archive.

    1. roadgen reads the OpenDRIVE and writes the ClipGT layers; they are
       packed into an OmniDreams archive (hdmap/clipgt.py).
    2. The OmniDreams image runs --raster-only and renders the HD-map control
       video along the archive's ego route (hdmap/control_video.py).
    3. The Cosmos3 image runs transfer (world-scenario hint) on that control
       video; the first frame of the result is written into the archive as
       first_image.png (server/cosmos3_transfer.py).

Steps 2-3 need the GPU one at a time; each container exits before the next
starts. ``--local`` runs steps 2-3 in the current Python environment instead
of Docker (development on a machine that has omnidreams / diffusers).
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
from carla_transfer.hdmap.clipgt import SceneSpec, export_scene_archive
from carla_transfer.hdmap.control_video import WSM_FPS, WSM_FRAMES, render_control_video

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


def render_control_video_docker(cfg: TransferConfig, archive: Path, out_mp4: Path) -> Path:
    from carla_transfer.docker.manager import ContainerManager, stop_container

    mgr = ContainerManager(cfg)
    stop_container(RASTER_CONTAINER)
    args = mgr.raster_server_args(archive, RASTER_CONTAINER)
    log.info("starting raster server: %s", " ".join(args))
    subprocess.run(args, check=True)
    try:
        _wait_grpc(cfg, timeout_s=600)
        return render_control_video(cfg, archive, out_mp4, WSM_FPS, WSM_FRAMES)
    finally:
        stop_container(RASTER_CONTAINER)


def render_control_video_local(cfg: TransferConfig, archive: Path, out_mp4: Path) -> Path:
    """Same as the docker path, with the raster server as a subprocess of the
    current interpreter (needs omnidreams importable, e.g. the flashdreams venv)."""
    import sys

    od = cfg.omnidreams
    cmd = [
        sys.executable, "-m", "carla_transfer.server.omnidreams_server",
        "--raster-only", "--port", str(od.port), "--manifest", od.manifest,
        "--resolution", f"{cfg.width}x{cfg.height}", "--fps", str(cfg.fps),
        "--camera", od.camera_name, "--scene-dir", str(archive.resolve().parent),
    ]  # fmt: skip
    log.info("starting local raster server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        _wait_grpc(cfg, timeout_s=600)
        return render_control_video(cfg, archive, out_mp4, WSM_FPS, WSM_FRAMES)
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


def run_cosmos3_docker(cfg: TransferConfig, work_dir: Path, job_args: list[str]) -> Path:
    from carla_transfer.docker.manager import ContainerManager

    mgr = ContainerManager(cfg)
    args = mgr.cosmos3_job_args(work_dir, ["--control", "/work/control_wsm.mp4",
                                           "--out-dir", "/work/cosmos3", *job_args])  # fmt: skip
    log.info("running cosmos3 transfer: %s", " ".join(args))
    subprocess.run(args, check=True)
    return work_dir / "cosmos3" / "first_image.png"


def run_cosmos3_local(work_dir: Path, job_args: list[str]) -> Path:
    from carla_transfer.server.cosmos3_transfer import build_parser, run

    out = work_dir / "cosmos3"
    run(build_parser().parse_args(
        ["--control", str(work_dir / "control_wsm.mp4"), "--out-dir", str(out), *job_args]
    ))  # fmt: skip
    return out / "first_image.png"


def prepare_scene(
    cfg: TransferConfig,
    xodr: Path,
    archive: Path,
    spec: SceneSpec,
    work_dir: Path | None = None,
    skip_cosmos3: bool = False,
    local: bool = False,
    cosmos3_args: list[str] | None = None,
) -> Path:
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="carla_transfer_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("work dir %s", work_dir)

    # 1. map → archive
    spec.camera.width, spec.camera.height = cfg.resolution
    export_scene_archive(xodr, archive, spec)
    log.info("archive %s (%d bytes)", archive, archive.stat().st_size)
    if skip_cosmos3:
        return archive

    # 2. control video
    control = work_dir / "control_wsm.mp4"
    if local:
        render_control_video_local(cfg, archive, control)
    else:
        render_control_video_docker(cfg, archive, control)

    # 3. Cosmos3 transfer → first image
    job_args = list(cosmos3_args or [])
    first = (
        run_cosmos3_local(work_dir, job_args)
        if local
        else run_cosmos3_docker(cfg, work_dir, job_args)
    )
    replace_in_archive(archive, "first_image.png", first.read_bytes())
    log.info("archive %s now carries the Cosmos3 first image", archive)
    return archive

"""Start / stop / inspect the model containers via the ``docker`` CLI.

The docker CLI is used rather than the docker SDK so the package has no
dependency on a matching daemon API version and so that ``--gpus`` (NVIDIA
Container Toolkit) works the same way it does from a shell.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from carla_transfer.config import BackendKind, TransferConfig

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"

COSMOS3_IMAGE = "carla-transfer/cosmos3:latest"


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker CLI not found on PATH")


def docker(*args: str, capture: bool = False, check: bool = True) -> str:
    r = subprocess.run(["docker", *args], check=check, capture_output=capture, text=True)
    return r.stdout if capture else ""


def container_running(name: str) -> bool:
    return bool(docker("ps", "-q", "--filter", f"name=^/{name}$", capture=True).strip())


def stop_container(name: str) -> None:
    if container_running(name):
        docker("stop", name)
    # A container started without --rm lingers stopped under the same name.
    docker("rm", "-f", name, capture=True, check=False)


def build_image(target: str, tag: str) -> None:
    """Build docker/Dockerfile.<target> with the repo root as context (the
    images install this package so the containers can run its servers)."""
    dockerfile = DOCKER_DIR / f"Dockerfile.{target}"
    if not dockerfile.exists():
        raise FileNotFoundError(dockerfile)
    subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", tag, str(REPO_ROOT)], check=True
    )


class ContainerManager:
    """The long-running model server for ``config.backend``."""

    def __init__(self, config: TransferConfig):
        self.config = config
        self.d = config.docker
        require_docker()

    # -- queries -------------------------------------------------------------

    def is_running(self) -> bool:
        return container_running(self.d.container_name)

    def logs(self, tail: int = 100) -> str:
        return docker("logs", "--tail", str(tail), self.d.container_name, capture=True)

    # -- argv builders -------------------------------------------------------

    def base_args(self, detached: bool = True, name: str | None = None) -> list[str]:
        """`docker run` prefix shared by the server and one-off jobs: GPU,
        caches, env. No image or command."""
        self.d.hf_cache.mkdir(parents=True, exist_ok=True)
        self.d.inductor_cache.mkdir(parents=True, exist_ok=True)
        args = ["docker", "run", "--rm"] + (["-d"] if detached else [])
        if name:
            args += ["--name", name]
        args += [
            "--gpus", self.d.gpus,
            "--ipc=host",
            "--shm-size", self.d.shm_size,
            "-v", f"{self.d.hf_cache}:/root/.cache/huggingface",
            "-v", f"{self.d.inductor_cache}:/root/.cache/torchinductor",
            "-e", "TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor",
            "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        ]  # fmt: skip
        # Forward the host HF token unless the config sets one explicitly.
        if "HF_TOKEN" in os.environ and "HF_TOKEN" not in self.d.env:
            args += ["-e", "HF_TOKEN"]
        for k, v in self.d.env.items():
            args += ["-e", f"{k}={v}"]
        args += self.d.extra_args
        return args

    def scene_mount_args(self, archive: Path | None = None) -> list[str]:
        """Bind the scene archive under its own name in /scenes. The file
        itself is mounted (not its directory): HF-cache snapshot entries are
        symlinks into blobs/, which a directory mount would not follow."""
        od = self.config.omnidreams
        archive = archive or od.scene_archive
        if not archive or od.scene_transfer != "mount":
            return []
        src = Path(archive).expanduser().resolve()
        return ["-v", f"{src}:/scenes/{Path(archive).name}:ro"]

    def run_args(self) -> list[str]:
        """The full `docker run` argv for the model server, exposed so
        `carla-transfer up --dry-run` can print it and tests can check it."""
        c = self.config
        args = self.base_args(name=self.d.container_name)
        if c.backend is BackendKind.OMNIDREAMS:
            args += ["-p", f"{c.omnidreams.port}:{c.omnidreams.port}"]
            args += self.scene_mount_args()
        elif c.backend is BackendKind.COSMOS3:
            port = _port_of(c.cosmos3.base_url)
            args += ["-p", f"{port}:{port}"]
        args.append(self.d.image)
        args += self._command()
        return args

    def _command(self) -> list[str]:
        c = self.config
        if c.backend is BackendKind.OMNIDREAMS:
            return self.omnidreams_server_command()
        if c.backend is BackendKind.COSMOS3:
            return [
                "vllm", "serve", c.cosmos3.model, "--omni",
                "--model-class-name", "Cosmos3OmniDiffusersPipeline",
                "--allowed-local-media-path", "/",
                "--port", str(_port_of(c.cosmos3.base_url)),
                "--init-timeout", "1800",
            ]  # fmt: skip
        return []

    def omnidreams_server_command(self, raster_only: bool = False) -> list[str]:
        c = self.config
        cmd = [
            "carla-transfer-server",
            "--host", "0.0.0.0",
            "--port", str(c.omnidreams.port),
            "--manifest", c.omnidreams.manifest,
            "--resolution", f"{c.width}x{c.height}",
            "--fps", str(c.fps),
            "--camera", c.omnidreams.camera_name,
        ]  # fmt: skip
        if not c.omnidreams.offload_text_encoder:
            cmd.append("--no-offload-text-encoder")
        if c.omnidreams.denoising_steps:
            cmd += ["--denoising-steps", ",".join(str(t) for t in c.omnidreams.denoising_steps)]
        if c.omnidreams.guide:
            cmd += ["--guide-mode", c.omnidreams.guide_mode,
                    "--guide-weight", str(c.omnidreams.guide_weight)]  # fmt: skip
        if raster_only:
            cmd.append("--raster-only")
        return cmd

    def raster_server_args(self, archive: Path, name: str) -> list[str]:
        """OmniDreams image in --raster-only mode (no world model) for
        rendering HD-map control videos."""
        c = self.config
        args = self.base_args(name=name)
        args += ["-p", f"{c.omnidreams.port}:{c.omnidreams.port}"]
        args += self.scene_mount_args(archive)
        args.append(self.d.image)
        args += self.omnidreams_server_command(raster_only=True)
        return args

    def cosmos3_job_args(self, work_dir: Path, job_args: list[str], image: str = COSMOS3_IMAGE):
        """One-off Cosmos3 transfer job; ``work_dir`` is mounted at /work."""
        args = self.base_args(detached=False)
        args += ["-v", f"{work_dir.resolve()}:/work"]
        args.append(image)
        args += job_args  # image ENTRYPOINT is carla-transfer-cosmos3
        return args

    # -- lifecycle -----------------------------------------------------------

    def up(self) -> None:
        if self.is_running():
            log.info("container %s already running", self.d.container_name)
            return
        args = self.run_args()
        log.info("starting: %s", " ".join(args))
        subprocess.run(args, check=True)

    def down(self) -> None:
        stop_container(self.d.container_name)

    def build(self, target: str | None = None, tag: str | None = None) -> None:
        target = target or self.config.backend.value
        build_image(target, tag or (COSMOS3_IMAGE if target == "cosmos3" else self.d.image))


def _port_of(url: str) -> int:
    from urllib.parse import urlparse

    p = urlparse(url)
    return p.port or (443 if p.scheme == "https" else 80)

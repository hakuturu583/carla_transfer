"""Start / stop / inspect the model container via the ``docker`` CLI.

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

DOCKER_DIR = Path(__file__).resolve().parents[3] / "docker"


class ContainerManager:
    def __init__(self, config: TransferConfig):
        self.config = config
        self.d = config.docker
        if shutil.which("docker") is None:
            raise RuntimeError("docker CLI not found on PATH")

    # -- queries -------------------------------------------------------------

    def is_running(self) -> bool:
        out = self._docker("ps", "-q", "--filter", f"name=^/{self.d.container_name}$", capture=True)
        return bool(out.strip())

    def logs(self, tail: int = 100) -> str:
        return self._docker("logs", "--tail", str(tail), self.d.container_name, capture=True)

    # -- lifecycle -----------------------------------------------------------

    def run_args(self) -> list[str]:
        """The full `docker run` argv, exposed so `carla-transfer up --dry-run`
        can print it and so tests can check it without a daemon."""
        c = self.config
        self.d.hf_cache.mkdir(parents=True, exist_ok=True)
        self.d.inductor_cache.mkdir(parents=True, exist_ok=True)
        args = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            self.d.container_name,
            "--gpus",
            self.d.gpus,
            "--ipc=host",
            "--shm-size",
            self.d.shm_size,
            "-v",
            f"{self.d.hf_cache}:/root/.cache/huggingface",
            "-v",
            f"{self.d.inductor_cache}:/root/.cache/torchinductor",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor",
            "-e",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        ]
        if c.backend is BackendKind.OMNIDREAMS:
            args += ["-p", f"{c.omnidreams.port}:{c.omnidreams.port}"]
        elif c.backend is BackendKind.COSMOS3:
            port = _port_of(c.cosmos3.base_url)
            args += ["-p", f"{port}:{port}"]
        # Forward the host HF token unless the config sets one explicitly.
        if "HF_TOKEN" in os.environ and "HF_TOKEN" not in self.d.env:
            args += ["-e", "HF_TOKEN"]
        for k, v in self.d.env.items():
            args += ["-e", f"{k}={v}"]
        args += self.d.extra_args
        args.append(self.d.image)
        args += self._command()
        return args

    def _command(self) -> list[str]:
        c = self.config
        if c.backend is BackendKind.OMNIDREAMS:
            return [
                "python",
                "-m",
                "omnidreams.grpc.server",
                "--host",
                "0.0.0.0",
                "--port",
                str(c.omnidreams.port),
                "--pipeline_config_name",
                c.omnidreams.pipeline_config,
                "--resolution",
                c.omnidreams.server_resolution,
                "--output_format",
                "jpeg",
            ]
        if c.backend is BackendKind.COSMOS3:
            return [
                "vllm",
                "serve",
                c.cosmos3.model,
                "--omni",
                "--model-class-name",
                "Cosmos3OmniDiffusersPipeline",
                "--allowed-local-media-path",
                "/",
                "--port",
                str(_port_of(c.cosmos3.base_url)),
                "--init-timeout",
                "1800",
            ]
        return []

    def up(self) -> None:
        if self.is_running():
            log.info("container %s already running", self.d.container_name)
            return
        args = self.run_args()
        log.info("starting: %s", " ".join(args))
        subprocess.run(args, check=True)

    def down(self) -> None:
        if self.is_running():
            self._docker("stop", self.d.container_name)

    def build(self, target: str | None = None, tag: str | None = None) -> None:
        """Build docker/Dockerfile.<backend> and tag it as config.docker.image."""
        target = target or self.config.backend.value
        dockerfile = DOCKER_DIR / f"Dockerfile.{target}"
        if not dockerfile.exists():
            raise FileNotFoundError(dockerfile)
        subprocess.run(
            ["docker", "build", "-f", str(dockerfile), "-t", tag or self.d.image, str(DOCKER_DIR)],
            check=True,
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _docker(*args: str, capture: bool = False) -> str:
        r = subprocess.run(["docker", *args], check=True, capture_output=capture, text=True)
        return r.stdout if capture else ""


def _port_of(url: str) -> int:
    from urllib.parse import urlparse

    p = urlparse(url)
    return p.port or (443 if p.scheme == "https" else 80)

"""CARLA simulator server in Docker, kept cheap on the GPU.

The world model needs most of the 24 GB card, so CARLA runs off-screen at
``-quality-level=Low`` with a small internal resolution. The camera sensor
the pipeline reads is spawned at the model resolution (512x288) regardless.
"""

from __future__ import annotations

import logging
import subprocess

from carla_transfer.config import CarlaServerConfig
from carla_transfer.docker.manager import container_running, docker, require_docker, stop_container

log = logging.getLogger(__name__)


class CarlaServerManager:
    def __init__(self, cfg: CarlaServerConfig):
        self.cfg = cfg
        require_docker()

    def is_running(self) -> bool:
        return container_running(self.cfg.container_name)

    def logs(self, tail: int = 100) -> str:
        return docker("logs", "--tail", str(tail), self.cfg.container_name, capture=True)

    def run_args(self) -> list[str]:
        c = self.cfg
        launcher = "CarlaUnreal.sh" if c.image.split(":")[-1].startswith("0.10") else "CarlaUE4.sh"
        args = [
            "docker", "run", "-d", "--rm",
            "--name", c.container_name,
            "--gpus", c.gpus,
            "--net=host",
            "--privileged",
            "-e", "SDL_VIDEODRIVER=offscreen",
            *c.extra_args,
            c.image,
            "bash", launcher,
            "-RenderOffScreen",
            f"-quality-level={c.quality_level}",
            f"-ResX={c.width}", f"-ResY={c.height}",
            "-nosound",
            f"-carla-rpc-port={c.rpc_port}",
        ]  # fmt: skip
        if c.fps:
            args += ["-benchmark", f"-fps={c.fps}"]
        return args

    def up(self) -> None:
        if self.is_running():
            log.info("CARLA container %s already running", self.cfg.container_name)
            return
        args = self.run_args()
        log.info("starting CARLA: %s", " ".join(args))
        subprocess.run(args, check=True)

    def down(self) -> None:
        stop_container(self.cfg.container_name)

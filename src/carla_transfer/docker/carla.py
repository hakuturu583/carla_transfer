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
            # No --rm: keep the container so `carla logs` can show a crash.
            "docker", "run", "-d",
            "--name", c.container_name,
            "--runtime", "nvidia",
            "--gpus", c.gpus,
            "--net=host",
            "--shm-size", "2g",
            # Graphics libs (libnvidia-glcore, libGLX_nvidia, Vulkan ICD) come
            # from the NVIDIA container toolkit's CDI spec; if CARLA dies with
            # "GameThread timed out waiting for RenderThread", regenerate it:
            #   sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
            "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
            # Persist UE's config/shader cache: the first -quality-level=Low
            # start of CARLA 0.10 tends to trip "GameThread timed out waiting
            # for RenderThread" while shaders compile; later starts are fine.
            "-v", f"{c.ue_config_dir}:/home/carla/.config/Epic",
            # No SDL_VIDEODRIVER=offscreen: with it CARLA exits immediately;
            # -RenderOffScreen alone is what works.
            *c.extra_args,
            c.image,
            "bash", launcher,
            "-RenderOffScreen",
            f"-quality-level={c.quality_level}",
            f"-ResX={c.width}", f"-ResY={c.height}",
            "-nosound",
            # UE kills itself when the render thread stalls > 60 s ("GameThread
            # timed out waiting for RenderThread"); that happens whenever the
            # world model hogs the GPU (session open, first-chunk compile).
            "-NoTimeouts",
            f"-carla-rpc-port={c.rpc_port}",
            *c.launcher_args,
        ]  # fmt: skip
        # -benchmark -fps=N is deliberately not passed: with it CARLA 0.10
        # off-screen hit the RenderThread timeout on start; the tick rate is
        # fixed by the client's synchronous mode instead.
        return args

    def up(self) -> None:
        self.cfg.ue_config_dir.mkdir(parents=True, exist_ok=True)
        if self.is_running():
            log.info("CARLA container %s already running", self.cfg.container_name)
            return
        args = self.run_args()
        log.info("starting CARLA: %s", " ".join(args))
        subprocess.run(args, check=True)

    def down(self) -> None:
        stop_container(self.cfg.container_name)

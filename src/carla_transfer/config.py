"""Configuration for the transfer pipeline and its Docker-hosted backend."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class BackendKind(str, enum.Enum):
    """Which world model produces the output video."""

    OMNIDREAMS = "omnidreams"
    COSMOS3 = "cosmos3"
    MOCK = "mock"


def _default_hf_cache() -> Path:
    return Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))


@dataclass
class DockerConfig:
    """How the model container is started. All fields map to `docker run` flags."""

    image: str = "carla-transfer/omnidreams:latest"
    container_name: str = "carla-transfer-model"
    gpus: str = "all"
    # Host directory holding the Hugging Face cache; mounted so models are
    # downloaded once and shared across container restarts.
    hf_cache: Path = field(default_factory=_default_hf_cache)
    # Persistent torch.compile/Inductor cache. Without it every restart pays the
    # ~7 min cold-compile cost on Ampere (see the RTX 3090 manifest notes).
    inductor_cache: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "carla_transfer" / "torchinductor"
    )
    shm_size: str = "16g"
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    # Seconds to wait for the server to accept connections after `docker run`.
    startup_timeout_s: float = 1800.0


@dataclass
class CarlaServerConfig:
    """CARLA simulator container. Low quality on purpose: the GPU belongs to
    the world model."""

    image: str = "carlasim/carla:0.10.0"
    container_name: str = "carla-transfer-carla"
    gpus: str = "device=0"
    rpc_port: int = 2000
    # Low | Epic. On this host CARLA 0.10 at Low dies at start (RenderThread
    # timeout) more often than not; Epic starts reliably and only costs
    # 6.9 GB vs 6.3 GB, so Epic + a tiny off-screen resolution is the default.
    quality_level: str = "Epic"
    # Host dir mounted as UE's ~/.config/Epic (shader cache survives restarts).
    ue_config_dir: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "carla_transfer" / "carla_config"
    )
    # Off-screen render target; the pipeline's camera is spawned at the model
    # resolution separately, so this only bounds CARLA's own GPU cost.
    width: int = 320
    height: int = 180
    extra_args: list[str] = field(default_factory=list)


@dataclass
class OmniDreamsConfig:
    """Client-side settings for the OmniDreams `WorldModelService` gRPC server."""

    host: str = "127.0.0.1"
    port: int = 50051
    # OmniDreams generates 8-frame chunks; this is the number of ego poses sent
    # per render_video_chunk call.
    frames_per_chunk: int = 8
    positive_prompt: str = ""
    negative_prompt: str = ""
    random_seed: int = 0
    max_message_mb: int = 256
    # The model's first chunk is always 5 frames, later ones `frames_per_chunk`.
    first_chunk_frames: int = 5
    # clipgt scene archive (.usdz) used as the static world map. Until the
    # CARLA map exporter exists this is one of the nvidia/omni-dreams-scenes
    # archives; the CARLA camera frame replaces its initial image.
    scene_archive: Path | None = None
    # How the archive reaches the server: "mount" bind-mounts its directory
    # into the container at /scenes and sends a scene:// reference (needed
    # for the 2.3 GB official archives); "inline" sends the bytes in the
    # start_session request (fine for small CARLA-exported archives).
    scene_transfer: str = "mount"
    # clipgt camera name inside the archive.
    camera_name: str = "camera_front_wide_120fov"
    # Initial image for the rollout: "carla" sends the first CARLA frame;
    # "archive" uses the archive's first_image.png, i.e. the Cosmos3-rendered
    # photo written by `carla-transfer prepare-scene`.
    initial_image: str = "archive"
    # Server-side settings, only used by `carla-transfer up` to build the
    # container command (carla_transfer.server.omnidreams_server). The manifest
    # is an interactive-drive manifest from the fork; the server resolution
    # overrides its resolution_wh and defaults to the client width/height.
    manifest: str = "example_world_model_rtx3090.yaml"
    offload_text_encoder: bool = True


@dataclass
class Cosmos3Config:
    """Client-side settings for a vLLM-Omni server hosting `nvidia/Cosmos3-Nano`."""

    base_url: str = "http://127.0.0.1:8000"
    model: str = "nvidia/Cosmos3-Nano"
    num_frames: int = 5
    num_inference_steps: int = 4
    guidance_scale: float = 6.0
    prompt: str = ""
    negative_prompt: str = ""
    request_timeout_s: float = 600.0


@dataclass
class Transfer25Config:
    """Cosmos-Transfer2.5-2B auto-multiview for the scene's first image. Runs
    as subprocesses of a cosmos-transfer2.5 checkout (its own uv venv)."""

    repo: Path = field(default_factory=lambda: Path.home() / "cosmos-transfer2.5")
    python: Path | None = None
    camera: str = "camera_front_wide_120fov"
    fps: int = 10
    guidance: float = 3.0
    seed: int = 0
    num_steps: int | None = None
    prompt: str | None = None
    evict_text_encoder: bool = True
    disable_guardrails: bool = True
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class TransferConfig:
    """Top-level config. Load from YAML with :meth:`from_yaml`."""

    backend: BackendKind = BackendKind.OMNIDREAMS
    width: int = 512
    height: int = 288
    fps: int = 30
    docker: DockerConfig = field(default_factory=DockerConfig)
    carla: CarlaServerConfig = field(default_factory=CarlaServerConfig)
    omnidreams: OmniDreamsConfig = field(default_factory=OmniDreamsConfig)
    cosmos3: Cosmos3Config = field(default_factory=Cosmos3Config)
    transfer25: Transfer25Config = field(default_factory=Transfer25Config)
    # Where the `record` sink writes mp4 files.
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TransferConfig:
        raw = dict(raw)
        docker_raw = _paths(raw.pop("docker", None) or {}, "hf_cache", "inductor_cache")
        # An `env:` block that only holds comments parses as None.
        docker_raw["env"] = docker_raw.get("env") or {}
        docker = DockerConfig(**docker_raw)
        omni = OmniDreamsConfig(**_paths(raw.pop("omnidreams", None) or {}, "scene_archive"))
        cosmos = Cosmos3Config(**(raw.pop("cosmos3", None) or {}))
        carla = CarlaServerConfig(**_paths(raw.pop("carla", None) or {}, "ue_config_dir"))
        t25_raw = _paths(raw.pop("transfer25", None) or {}, "repo", "python")
        t25_raw["env"] = t25_raw.get("env") or {}
        transfer25 = Transfer25Config(**t25_raw)
        if "backend" in raw:
            raw["backend"] = BackendKind(raw["backend"])
        if "output_dir" in raw:
            raw["output_dir"] = Path(raw["output_dir"])
        return cls(
            docker=docker,
            carla=carla,
            omnidreams=omni,
            cosmos3=cosmos,
            transfer25=transfer25,
            **raw,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TransferConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


def _paths(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    d = dict(d)
    for k in keys:
        if d.get(k) is not None:
            d[k] = Path(d[k]).expanduser()
    return d

"""Configuration for the transfer pipeline and its Docker-hosted backend."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class BackendKind(str, enum.Enum):
    """Which world model produces the output video."""

    OMNIDREAMS = "omnidreams"
    COSMOS3 = "cosmos3"
    MOCK = "mock"


@dataclass
class DockerConfig:
    """How the model container is started. All fields map to `docker run` flags."""

    image: str = "carla-transfer/omnidreams:latest"
    container_name: str = "carla-transfer-model"
    gpus: str = "all"
    # Host directory holding the Hugging Face cache; mounted so models are
    # downloaded once and shared across container restarts.
    hf_cache: Path = field(default_factory=lambda: Path.home() / ".cache" / "huggingface")
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
    # Server-side settings (only used by `carla-transfer up`). The gRPC server
    # accepts a registered pipeline config name and one of 480p/720p/704p; the
    # RTX 3090 interactive-drive manifest runs 512x288 but that path is not
    # exposed by the gRPC server yet, so on a 24 GB card 480p is the floor.
    pipeline_config: str = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
    server_resolution: str = "480p"
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
class TransferConfig:
    """Top-level config. Load from YAML with :meth:`from_yaml`."""

    backend: BackendKind = BackendKind.OMNIDREAMS
    width: int = 512
    height: int = 288
    fps: int = 30
    docker: DockerConfig = field(default_factory=DockerConfig)
    omnidreams: OmniDreamsConfig = field(default_factory=OmniDreamsConfig)
    cosmos3: Cosmos3Config = field(default_factory=Cosmos3Config)
    # Where the `record` sink writes mp4 files.
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TransferConfig:
        raw = dict(raw)
        docker_raw = _paths(raw.pop("docker", None) or {}, "hf_cache", "inductor_cache")
        # An `env:` block that only holds comments parses as None.
        docker_raw["env"] = docker_raw.get("env") or {}
        docker = DockerConfig(**docker_raw)
        omni = OmniDreamsConfig(**(raw.pop("omnidreams", None) or {}))
        cosmos = Cosmos3Config(**(raw.pop("cosmos3", None) or {}))
        if "backend" in raw:
            raw["backend"] = BackendKind(raw["backend"])
        if "output_dir" in raw:
            raw["output_dir"] = Path(raw["output_dir"])
        return cls(docker=docker, omnidreams=omni, cosmos3=cosmos, **raw)

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
        if k in d:
            d[k] = Path(d[k]).expanduser()
    return d

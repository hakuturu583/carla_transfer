from pathlib import Path

import pytest

from carla_transfer.config import BackendKind, TransferConfig
from carla_transfer.docker.manager import ContainerManager

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_yaml_roundtrip_omnidreams():
    cfg = TransferConfig.from_yaml(CONFIGS / "omnidreams_rtx3090.yaml")
    assert cfg.backend is BackendKind.OMNIDREAMS
    assert cfg.resolution == (512, 288)
    assert cfg.omnidreams.frames_per_chunk == 8
    assert cfg.omnidreams.first_chunk_frames == 5
    assert cfg.omnidreams.scene_archive is None
    assert cfg.docker.container_name == "carla-transfer-omnidreams"


def test_yaml_roundtrip_cosmos3():
    cfg = TransferConfig.from_yaml(CONFIGS / "cosmos3_nano.yaml")
    assert cfg.backend is BackendKind.COSMOS3
    assert cfg.cosmos3.num_frames == 5


@pytest.fixture
def fake_docker(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
    return tmp_path


def test_docker_run_args_omnidreams(fake_docker):
    cfg = TransferConfig.from_yaml(CONFIGS / "omnidreams_rtx3090.yaml")
    cfg.docker.hf_cache = fake_docker / "hf"
    cfg.docker.inductor_cache = fake_docker / "inductor"
    args = ContainerManager(cfg).run_args()
    assert args[:3] == ["docker", "run", "-d"]
    assert "--gpus" in args and args[args.index("--gpus") + 1] == "all"
    assert "-p" in args and "50051:50051" in args
    assert "carla-transfer/omnidreams:latest" in args
    assert "carla-transfer-server" in args
    assert args[args.index("--resolution") + 1] == "512x288"
    assert args[args.index("--manifest") + 1] == "example_world_model_rtx3090.yaml"
    assert "--no-offload-text-encoder" not in args
    assert not any(a.startswith("/scenes") or ":/scenes/" in a for a in args)


def test_docker_run_args_mounts_scene_archive(fake_docker):
    cfg = TransferConfig.from_yaml(CONFIGS / "omnidreams_rtx3090.yaml")
    cfg.docker.hf_cache = fake_docker / "hf"
    cfg.docker.inductor_cache = fake_docker / "inductor"
    blob = fake_docker / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"x")
    link = fake_docker / "scene.usdz"
    link.symlink_to(blob)
    cfg.omnidreams.scene_archive = link
    args = ContainerManager(cfg).run_args()
    assert f"{blob}:/scenes/scene.usdz:ro" in args
    assert (fake_docker / "hf").is_dir()


def test_docker_run_args_cosmos3(fake_docker):
    cfg = TransferConfig.from_yaml(CONFIGS / "cosmos3_nano.yaml")
    cfg.docker.hf_cache = fake_docker / "hf"
    cfg.docker.inductor_cache = fake_docker / "inductor"
    args = ContainerManager(cfg).run_args()
    assert "8000:8000" in args
    assert "vllm/vllm-omni:cosmos3" in args
    assert "nvidia/Cosmos3-Nano" in args

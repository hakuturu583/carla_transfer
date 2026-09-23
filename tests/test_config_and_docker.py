from pathlib import Path

import pytest

from carla_transfer.config import BackendKind, TransferConfig
from carla_transfer.docker.manager import COSMOS3_IMAGE, ContainerManager

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_yaml_roundtrip_omnidreams():
    cfg = TransferConfig.from_yaml(CONFIGS / "omnidreams_rtx3090.yaml")
    assert cfg.backend is BackendKind.OMNIDREAMS
    assert cfg.resolution == (512, 288)
    assert cfg.omnidreams.frames_per_chunk == 8
    assert cfg.omnidreams.first_chunk_frames == 5
    assert cfg.omnidreams.scene_archive == Path("outputs/scenes/town10.usdz")
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
    cfg.omnidreams.scene_archive = None
    args = ContainerManager(cfg).run_args()
    assert args[:3] == ["docker", "run", "--rm"] and "-d" in args
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


def test_raster_and_cosmos3_job_args(fake_docker):
    cfg = TransferConfig.from_yaml(CONFIGS / "omnidreams_rtx3090.yaml")
    cfg.docker.hf_cache = fake_docker / "hf"
    cfg.docker.inductor_cache = fake_docker / "inductor"
    archive = fake_docker / "town.usdz"
    archive.write_bytes(b"x")
    mgr = ContainerManager(cfg)
    args = mgr.raster_server_args(archive, "raster")
    assert "--raster-only" in args and f"{archive}:/scenes/town.usdz:ro" in args
    job = mgr.cosmos3_job_args(fake_docker, ["--revision", "fp8"])
    assert job[:3] == ["docker", "run", "--rm"] and "-d" not in job
    assert f"{fake_docker}:/work" in job
    assert job[-3:] == [COSMOS3_IMAGE, "--revision", "fp8"]


def test_cosmos3_batch_mounts_common_root(fake_docker, monkeypatch):
    from carla_transfer.hdmap import prepare

    cfg = TransferConfig.from_yaml(CONFIGS / "omnidreams_rtx3090.yaml")
    cfg.docker.hf_cache = fake_docker / "hf"
    cfg.docker.inductor_cache = fake_docker / "inductor"
    dirs = [fake_docker / "scenarios" / n / "first_image" for n in ("a", "b")]
    seen = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda args, check: seen.append(args))
    out = prepare.run_cosmos3_docker(cfg, dirs, ["--steps", "20"])
    (args,) = seen  # one container for the whole batch
    assert f"{(fake_docker / 'scenarios').resolve()}:/work" in args
    i = args.index("--control")
    assert args[i : i + 3] == [
        "--control",
        "/work/a/first_image/control_wsm.mp4",
        "/work/b/first_image/control_wsm.mp4",
    ]
    assert args[-2:] == ["--steps", "20"]
    assert out == [d / "cosmos3" / "first_image.png" for d in dirs]


def test_carla_server_args(fake_docker):
    from carla_transfer.docker.carla import CarlaServerManager

    cfg = TransferConfig()
    args = CarlaServerManager(cfg.carla).run_args()
    assert "-quality-level=Epic" in args and "-RenderOffScreen" in args
    assert "CarlaUnreal.sh" in args and "--net=host" in args
    cfg.carla.image = "carlasim/carla:0.9.16"
    assert "CarlaUE4.sh" in CarlaServerManager(cfg.carla).run_args()


def test_pipeline_config_replaces_manifest_and_build_args(fake_docker, monkeypatch):
    from carla_transfer.docker import manager

    cfg = TransferConfig.from_yaml(CONFIGS / "omnidreams_blackwell.yaml")
    cfg.docker.hf_cache = fake_docker / "hf"
    cfg.docker.inductor_cache = fake_docker / "inductor"
    cmd = ContainerManager(cfg).omnidreams_server_command()
    assert cmd[cmd.index("--pipeline-config") + 1] == "omnidreams-optimized-rtx-pro-6000"
    assert "--manifest" not in cmd and "--no-offload-text-encoder" not in cmd
    assert cfg.omnidreams.sync_traffic_lights

    seen = []
    monkeypatch.setattr(manager.subprocess, "run", lambda args, check: seen.append(args))
    ContainerManager(cfg).build()
    (args,) = seen
    assert "--build-arg" in args
    assert args[args.index("--build-arg") + 1] == (
        "FLASHDREAMS_REF=claude/omnidreams-native-text-replace"
    )
    assert args[args.index("-t") + 1] == "carla-transfer/omnidreams:main"
